"""Agent : compréhension -> recherche -> mémoire -> opération, avec les sources de chaque affirmation.

Aucune action n'est exécutée : les mails, tâches et événements sont des brouillons à confirmer.
Le destinataire d'un mail n'est retenu que s'il est désigné explicitement comme responsable
(champ owner) dans les réunions ; une simple mention ne suffit pas. Aucune adresse n'est inventée.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from sae.agent.understanding import ACTION_INTENTS, Understanding, norm, understand
from sae.data import TYPE_LABELS_FR, EvalQuery, gid
from sae.memory import Memory, apply_memory
from sae.retrieval.bm25 import BM25, FRENCH_STOPWORDS
from sae.schemas import (Claim, Clarification, Dossier, DossierSection, EmailDraft, EventDraft, Recipient,
                         Scenario, TaskDraft, Unit)

GENERIC = set("""mail e-mail email courriel message ecris ecrire redige rediger envoie envoyer prepare preparer
cree creer tache planifie reunion point bonne personne personnes quelqu quelqu'un responsable peux pourrais
merci salut bonjour stp svp dossier complet projet faire fais demander demande lui leur
envoie-lui envoie-leur ecris-lui ecris-leur dis-lui dis-leur demande-lui demande-leur
lundi mardi mercredi jeudi vendredi samedi dimanche janvier fevrier mars avril juin juillet aout septembre
octobre novembre decembre avant semaine prochaine""".split())
OWNER_TYPES = {"commitment", "action", "fact", "decision"}
# Abstention sur les questions d'information : « none » garde le comportement historique (toujours répondre).
ABSTENTION_STRATEGIES = ("none", "lexical", "score", "gap", "both", "either")
LEXICAL_K = 3   # nombre d'unités examinées pour la vérification lexicale, comme pour le choix du destinataire
# Seuils par stratégie et par intention, calibrés sur train + val (scripts/eval_abstention.py --calibrate),
# figés avant toute mesure sur test et test2. Pas de seuil pour compare : aucune question de comparaison
# sans réponse dans les données de calibrage.
ABSTENTION_THRESHOLDS: dict[str, dict[str, float]] = {
    "score": {"answer": 0.5121}, "gap": {"answer": 0.0031}, "both": {"answer": 0.5379}, "either": {"answer": 0.5121},
}
DEFAULT_ABSTENTION = "score"   # choisie sur train + val : aucune abstention à tort, 6/9 abstentions justifiées
TEMPORAL_TAIL = re.compile(
    r"\s+(?:avant|d'ici|d’ici|pour)?\s*(?:le |la |ce |cette )?(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
    r"\d{1,2}(?:er)? \w+|demain|la semaine prochaine|fin de (?:la )?semaine|fin du mois)\b.*$", re.I)


@dataclass
class Source:
    unit_id: str
    date: str
    meeting: str
    speaker: str | None
    statement: str
    evidence: str
    status: str

    def cite(self) -> str:
        who = f", {self.speaker}" if self.speaker else ""
        return f"[{self.unit_id}] réunion « {self.meeting} » du {self.date}{who} : « {self.evidence} »"


@dataclass
class AgentResult:
    kind: str           # answer, compare, no_answer, dossier, email_draft, task_draft, event_draft, clarify, no_action
    text: str
    understanding: Understanding
    sources: list[Source] = field(default_factory=list)
    payload: dict | None = None
    signals: dict | None = None   # indices de confiance de la recherche (questions d'information)


_STEMMER = BM25()   # sert seulement à la tokenisation et à la racinisation françaises


def stems(text: str) -> set[str]:
    return {t for t in _STEMMER.tokenize(norm(text)) if len(t) >= 3}


def should_abstain(strategy: str, thresholds: dict[str, float], intent: str, signals: dict) -> bool:
    """Décision d'abstention à partir des indices de la recherche (fonction pure, réutilisée au calibrage)."""
    if strategy == "none":
        return False
    miss = not signals["lexical"]
    t = thresholds.get(intent)
    low = t is not None and signals["gap" if strategy == "gap" else "score"] < t
    return {"lexical": miss, "score": low, "gap": low, "both": miss and low, "either": miss or low}[strategy]


class Agent:
    def __init__(self, scenarios: list[Scenario], memory: Memory, retriever, with_type: bool = True,
                 abstention: str = DEFAULT_ABSTENTION, thresholds: dict[str, float] | None = None) -> None:
        if abstention not in ABSTENTION_STRATEGIES:
            raise ValueError(f"stratégie d'abstention inconnue : {abstention}")
        self.abstention = abstention
        self.thresholds = ABSTENTION_THRESHOLDS.get(abstention, {}) if thresholds is None else thresholds
        self.scenarios = {s.project_id: s for s in scenarios}
        self.memory = memory
        self.with_type = with_type
        self.retriever = retriever.fit(memory.corpus.texts)
        self.index = memory.corpus.index()
        self.units = memory.corpus.units
        self.inactive = set(np.array(memory.corpus.ids)[memory.inactive_mask()])
        self.predecessors: dict[str, list[str]] = {}
        for old, new in memory.successor.items():
            self.predecessors.setdefault(new, []).append(old)
        self.meetings = {gid(s.project_id, m.meeting_id): m.title for s in scenarios for m in s.meetings}
        self.people = {p.name: p for s in scenarios for p in s.people}

    # ------------------------------------------------------------------ outils
    def source(self, g: str) -> Source:
        u = self.units[g]
        project = g.split("/", 1)[0]
        return Source(g, u.source_date.isoformat(), self.meetings.get(gid(project, u.source_id), u.source_id),
                      u.speaker, u.statement, u.evidence, "inactive" if g in self.inactive else u.status)

    def search(self, text: str, project: str | None, intent: str, k: int = 5) -> list[tuple[str, float]]:
        scores = self.retriever.score([text])
        q = EvalQuery(qid=f"{project or '?'}/agent", text=text, relevant=(), hard_negatives=(), answerable=True,
                      origin="agent", tags=(), intent=intent)
        scores = apply_memory(scores, [q], self.memory, status=True, project=project is not None)[0]
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.memory.corpus.ids[i], float(scores[i])) for i in order]

    def history(self, g: str) -> list[str]:
        """Chaîne complète de versions, de la plus ancienne à celle en vigueur."""
        current = self.memory.current(g)
        chain, frontier = [current], [current]
        while frontier:
            nxt = [p for x in frontier for p in self.predecessors.get(x, []) if p not in chain]
            chain += nxt
            frontier = nxt
        return sorted(chain, key=lambda x: (self.units[x].source_date, x))

    def signals(self, message: str, hits: list[tuple[str, float]]) -> dict:
        """Indices de confiance : score de la 1re unité, écart avec la 2e, racine commune avec le sujet."""
        topic = stems(" ".join(self.topic_words(message)))
        # sans mot de sujet exploitable, la vérification lexicale ne peut pas conclure : on ne la compte pas comme échec
        lexical = not topic or any(topic & stems(self.units[g].statement) for g, _ in hits[:LEXICAL_K])
        gap = hits[0][1] - hits[1][1] if len(hits) > 1 else hits[0][1]
        return {"score": hits[0][1], "gap": gap, "lexical": lexical}

    def abstains(self, intent: str, sig: dict) -> bool:
        return should_abstain(self.abstention, self.thresholds, intent, sig)

    def _no_answer(self, u: Understanding, hits: list[tuple[str, float]], sig: dict) -> AgentResult:
        """Information absente des réunions : on le dit, et on montre les unités proches sans les donner comme réponse."""
        why = []
        if not sig["lexical"]:
            why.append("aucun mot du sujet dans les unités trouvées")
        t = self.thresholds.get(u.intent)
        key = "gap" if self.abstention == "gap" else "score"
        if t is not None and sig[key] < t:
            why.append(f"{'écart' if key == 'gap' else 'similarité'} {sig[key]:.3f} sous le seuil {t:.3f}")
        near = [self.source(g) for g, _ in hits[:2]]
        text = "Je n'ai pas trouvé cette information dans les réunions."
        if near:
            text += ("\n\nSujets proches (ce ne sont pas des réponses à la question) :\n"
                     + "\n".join(f"- {s.statement} [{s.unit_id}, {s.date}]" for s in near))
        payload = {"reason": "; ".join(why), "related_unit_ids": [s.unit_id for s in near]}
        return AgentResult("no_answer", text, u, [], payload, sig)

    def topic_query(self, message: str) -> str:
        words = self.topic_words(message)
        return " ".join(words) if words else message

    @staticmethod
    def topic_words(message: str) -> list[str]:
        return [w for w in re.findall(r"[\w'-]+", norm(message))
                if len(w) > 3 and w not in FRENCH_STOPWORDS and w not in GENERIC]

    def recipient(self, u: Understanding, message: str) -> tuple[Recipient | None, list[str], str]:
        """Destinataire justifié par une attribution explicite (owner), ou explication du blocage."""
        if u.person:
            evidence = [g for g, x in self.units.items() if x.owner == u.person and g not in self.inactive
                        and (u.project is None or g.startswith(u.project + "/"))]
            person = self.people.get(u.person)
            email = person.email if person else None
            return (Recipient(name=u.person, email=email, evidence_unit_ids=evidence,
                              resolution_status="resolved" if email else "email_missing"), evidence,
                    "personne nommée dans la demande")
        hits = self.search(self.topic_query(message), u.project, "answer", k=5)
        owners: dict[str, str] = {}                      # responsable -> unité qui le désigne
        topic = stems(" ".join(self.topic_words(message)))
        for g, _ in hits[:3]:
            x = self.units[g]
            # l'unité doit parler du sujet demandé (au moins un mot en commun, après racinisation)
            if not (topic & stems(x.statement)):
                continue
            if x.owner and x.type in OWNER_TYPES and g not in self.inactive:
                # pour chaque responsable, on garde l'attribution la plus récente
                if x.owner not in owners or x.source_date > self.units[owners[x.owner]].source_date:
                    owners[x.owner] = g
        if not owners:
            return None, [g for g, _ in hits[:3]], "aucun responsable explicite pour ce sujet dans les réunions"
        # l'attribution trouvée est-elle encore valable ? on cherche une réattribution plus récente sur le même sujet
        for name_, g_ in list(owners.items()):
            later = self.reassignment(g_, u.project)
            if later and self.units[later].owner not in owners:
                owners[self.units[later].owner] = later
        ranked = sorted(owners.items(), key=lambda kv: self.units[kv[1]].source_date, reverse=True)
        name, g = ranked[0]
        if len(ranked) > 1 and self.units[ranked[1][1]].source_date == self.units[g].source_date:
            tied = [n for n, h in ranked if self.units[h].source_date == self.units[g].source_date]
            return (Recipient(name=None, resolution_status="ambiguous", evidence_unit_ids=list(owners.values())),
                    list(owners.values()), "plusieurs responsables désignés le même jour : " + ", ".join(tied))
        why = f"responsable désigné dans {g} (réunion du {self.units[g].source_date})"
        if len(ranked) > 1:
            why += "; attribution la plus récente, " + ", ".join(
                f"{n} était désigné(e) avant ({h})" for n, h in ranked[1:])
        email = self.people[name].email if name in self.people else None
        return (Recipient(name=name, email=email, evidence_unit_ids=[g],
                          resolution_status="resolved" if email else "email_missing"),
                [g] + [h for _, h in ranked[1:]], why)

    def reassignment(self, g: str, project: str | None) -> str | None:
        """Unité plus récente, très proche de g, qui désigne un autre responsable (réattribution)."""
        x = self.units[g]
        for h, _ in self.search(x.statement, project, "answer", k=3):
            y = self.units[h]
            if (h != g and y.owner and y.owner != x.owner and y.type in ("action", "commitment")
                    and y.source_date > x.source_date and h not in self.inactive):
                return h
        return None

    @staticmethod
    def request_text(message: str) -> str:
        """Ce qui est demandé, sans la consigne d'écriture (« Rédige un mail à X pour … »)."""
        m = re.sub(r"^\s*(r[ée]dige|[ée]cris|envoie|pr[ée]pare|cr[ée]e)[^,]*?\b(pour|afin de|en lui demandant de)\s+",
                   "", message, flags=re.I)
        return m.strip().rstrip(".?! ")

    @staticmethod
    def email_sentence(request: str) -> tuple[str, str]:
        """Transforme « lui demander … » en question directe ; renvoie (phrase, objet du mail)."""
        r = request.strip().rstrip(".?! ")
        m = re.match(r"(?:lui|leur) demander (?:de |d')(.+)", r, re.I)
        if m:
            return f"Pourrais-tu {m.group(1)} ?", m.group(1)
        m = re.match(r"(?:lui|leur) demander ((?:où|si|quand|comment|combien|qui|pourquoi|quel|quelle)\b.+)", r, re.I)
        if m:
            return f"Pourrais-tu me dire {m.group(1)} ?", m.group(1)
        m = re.match(r"(?:lui|leur) demander (.+)", r, re.I)
        if m:
            return f"Pourrais-tu me transmettre {m.group(1)} ?", m.group(1)
        return (r[0].upper() + r[1:] + ".") if r else r, r

    @staticmethod
    def title_from(message: str) -> str:
        t = message.split(":", 1)[1] if ":" in message else re.sub(
            r"^\s*(planifie|programme|organise|pr[ée]pare|cale|cr[ée]e|ajoute)\s+(une t[âa]che\s+(pour|à)\s+\S+\s*)?",
            "", message, flags=re.I)
        t = t.strip().rstrip(".?! ")
        return t[0].upper() + t[1:] if t else message

    # ------------------------------------------------------------------ point d'entrée
    def handle(self, message: str, project: str | None = None) -> AgentResult:
        u = understand(message, list(self.scenarios.values()), default_project=project)

        if u.hypothetical:
            return AgentResult("no_action", "C'est une piste évoquée, pas une demande : je ne crée aucune action. "
                               "Dis-moi si tu veux que je prépare quelque chose.", u)

        needs_project = u.intent in ACTION_INTENTS | {"build_dossier"}
        if needs_project and u.project is None:
            options = u.candidates_projects or list(self.scenarios)
            c = Clarification(question="Sur quel projet porte ta demande ?", reason="projet non identifié",
                              candidates=options)
            return AgentResult("clarify", f"{c.question} ({', '.join(options)})", u, payload=c.model_dump())

        if u.intent in ACTION_INTENTS and not self.topic_words(message) and not u.person:
            c = Clarification(question="De quel sujet s'agit-il, ou à qui veux-tu écrire ?",
                              reason="demande trop vague pour choisir un destinataire ou un contenu")
            return AgentResult("clarify", c.question, u, payload=c.model_dump())

        return getattr(self, f"_{u.intent}")(message, u)

    # ------------------------------------------------------------------ opérations
    def _answer(self, message: str, u: Understanding) -> AgentResult:
        hits = self.search(message, u.project, "answer", k=3)
        sig = self.signals(message, hits)
        if self.abstains("answer", sig):
            return self._no_answer(u, hits, sig)
        top = self.source(hits[0][0])
        text = top.statement
        older = [g for g in self.history(hits[0][0]) if g != hits[0][0]]
        if older:
            text += "\n\nCette information remplace : " + " ; ".join(
                f"{self.units[g].statement} ({self.units[g].source_date})" for g in older)
        text += "\n\nSource : " + top.cite()
        return AgentResult("answer", text, u, [top] + [self.source(g) for g in older], signals=sig)

    def _compare(self, message: str, u: Understanding) -> AgentResult:
        hits = self.search(message, u.project, "compare", k=3)
        sig = self.signals(message, hits)
        if self.abstains("compare", sig):
            return self._no_answer(u, hits, sig)
        chain = self.history(hits[0][0])
        if len(chain) == 1:
            top = self.units[hits[0][0]]
            same = [g for g, _ in hits if self.units[g].type == top.type]
            dates = {self.units[g].source_date for g in same}
            if len(same) > 1 and len(dates) > 1:
                srcs = [self.source(g) for g in sorted(same, key=lambda x: (self.units[x].source_date, x))]
                lines = [f"- {x.date} : {x.statement} [{x.unit_id}]" for x in srcs]
                return AgentResult("compare", "Je n'ai pas de lien de remplacement confirmé, mais voici les "
                                   "informations sur ce sujet, de la plus ancienne à la plus récente :\n"
                                   + "\n".join(lines) + f"\n\nLa plus récente : {srcs[-1].statement} "
                                   "(à vérifier : je ne peux pas confirmer qu'elle remplace les précédentes)", u, srcs,
                                   signals=sig)
            s = self.source(chain[0])
            return AgentResult("compare", "Je n'ai trouvé aucune modification enregistrée sur ce point. "
                               f"Dernière information : {s.statement}\n\nSource : {s.cite()}", u, [s], signals=sig)
        srcs = [self.source(g) for g in chain]
        lines = [f"- {s.date} : {s.statement} [{s.unit_id}]" for s in srcs]
        text = ("Évolution sur ce point, de la plus ancienne à la plus récente :\n" + "\n".join(lines)
                + f"\n\nVersion en vigueur : {srcs[-1].statement}")
        return AgentResult("compare", text, u, srcs, signals=sig)

    def _build_dossier(self, message: str, u: Understanding) -> AgentResult:
        p = u.project
        ids = [g for g in self.memory.corpus.ids if g.startswith(p + "/")]
        active = [g for g in ids if g not in self.inactive]

        def claims(gs, fmt=lambda x: x.statement):
            return [Claim(text=fmt(self.units[g]), unit_ids=[g]) for g in gs]

        def with_owner(x: Unit) -> str:
            extra = f" (responsable : {x.owner}" + (f", échéance : {x.due_date}" if x.due_date else "") + ")" \
                if x.owner else ""
            return x.statement + extra

        sections = [
            DossierSection(title="Décisions en vigueur", claims=claims([g for g in active if self.units[g].type == "decision"])),
            DossierSection(title="Engagements et actions", claims=claims(
                [g for g in active if self.units[g].type in ("commitment", "action")], with_owner)),
            DossierSection(title="Risques", claims=claims([g for g in active if self.units[g].type == "risk"])),
            DossierSection(title="Propositions et idées non tranchées", claims=claims(
                [g for g in active if self.units[g].type == "idea"])),
            DossierSection(title="Historique des changements", claims=[
                Claim(text=f"{self.units[o].statement} → remplacé par : {self.units[n].statement}", unit_ids=[o, n])
                for o, n in self.memory.successor.items() if o.startswith(p + "/")]),
        ]
        people = sorted({x for g in ids for x in (self.units[g].speaker, self.units[g].owner) if x})
        missing = [f"Adresse e-mail inconnue pour {n}" for n in people if n in self.people and not self.people[n].email]
        missing += [f"Pas d'échéance pour : {self.units[g].statement}" for g in active
                    if self.units[g].type in ("commitment", "action") and not self.units[g].due_date]
        name = self.scenarios[p].project_name
        dossier = Dossier(project_id=p, title=f"Dossier : {name}", sections=sections, missing_information=missing)

        md = [f"# {dossier.title}", ""]
        for sec in dossier.sections:
            md.append(f"## {sec.title}")
            md += [f"- {c.text} [{', '.join(c.unit_ids)}]" for c in sec.claims] or ["- (rien dans les réunions)"]
            md.append("")
        md.append("## Personnes impliquées")
        md += [f"- {n}" + (f" ({self.people[n].role})" if n in self.people and self.people[n].role else "") for n in people]
        md += ["", "## Informations manquantes"] + ([f"- {x}" for x in missing] or ["- aucune"])
        return AgentResult("dossier", "\n".join(md), u, [self.source(g) for g in active], dossier.model_dump(mode="json"))

    def _action_common(self, message: str, u: Understanding):
        rec, evidence, why = self.recipient(u, message)
        if rec is None or rec.resolution_status == "ambiguous":
            c = Clarification(question="À qui dois-je l'adresser ?", reason=why,
                              candidates=[] if rec is None else why.split(" : ", 1)[-1].split(", "))
            return None, AgentResult("clarify", f"{c.question} ({why})", u,
                                     [self.source(g) for g in evidence], c.model_dump())
        context = self.search(self.topic_query(message), u.project, "answer", k=2)
        sources = list(dict.fromkeys(evidence + [g for g, _ in context]))
        return (rec, sources, why), None

    def _draft_email(self, message: str, u: Understanding) -> AgentResult:
        ok, blocked = self._action_common(message, u)
        if blocked:
            return blocked
        rec, sources, why = ok
        ctx = self.units[sources[-1]] if sources else None
        sentence, obj = self.email_sentence(self.request_text(message))
        if u.deadline and u.deadline not in norm(sentence):
            sentence = sentence.rstrip(" ?.") + f" {u.deadline} ?"
        name = self.scenarios[u.project].project_name
        body = f"Bonjour {rec.name.split()[0]},\n\n{sentence}"
        if ctx:
            body += f"\n\nPour rappel, lors de la réunion du {ctx.source_date} : {ctx.statement}"
        body += "\n\nMerci d'avance,\n[Ton prénom]"
        subject_obj = TEMPORAL_TAIL.sub("", obj).strip()
        draft = EmailDraft(recipient=rec, subject=f"{name} : {subject_obj[:70]}", body=body,
                           source_unit_ids=sources)
        warn = "" if rec.email else "\n\nAttention : je n'ai pas son adresse e-mail, il faudra la compléter."
        text = (f"Brouillon de mail (non envoyé, à confirmer)\nÀ : {rec.name} <{rec.email or '?'}>\n"
                f"Objet : {draft.subject}\n\n{body}\n\nPourquoi ce destinataire : {why}.{warn}")
        return AgentResult("email_draft", text, u, [self.source(g) for g in sources], draft.model_dump(mode="json"))

    def _create_task(self, message: str, u: Understanding) -> AgentResult:
        ok, blocked = self._action_common(message, u)
        if blocked:
            return blocked
        rec, sources, why = ok
        title = self.title_from(message)
        if u.deadline and u.deadline not in norm(title):
            title += f" ({u.deadline})"
        draft = TaskDraft(title=title, assignee=rec, source_unit_ids=sources)
        text = f"Proposition de tâche (non créée, à confirmer)\nTitre : {title}\nAssignée à : {rec.name}\nPourquoi : {why}."
        return AgentResult("task_draft", text, u, [self.source(g) for g in sources], draft.model_dump(mode="json"))

    def _schedule_event(self, message: str, u: Understanding) -> AgentResult:
        hits = self.search(message, u.project, "answer", k=3)
        names = list(dict.fromkeys(x for g, _ in hits for x in (self.units[g].owner, self.units[g].speaker) if x))
        participants = [Recipient(name=n, email=self.people[n].email if n in self.people else None,
                                  resolution_status="resolved" if n in self.people and self.people[n].email
                                  else "email_missing") for n in names]
        title = self.title_from(message)
        if u.deadline and u.deadline not in norm(title):
            title += f" ({u.deadline})"
        draft = EventDraft(title=title, participants=participants, source_unit_ids=[g for g, _ in hits])
        text = (f"Proposition d'événement (non créé, à confirmer)\nTitre : {title}\n"
                f"Participants proposés : {', '.join(names) or 'à préciser'}\n"
                "Pourquoi : personnes impliquées dans les unités les plus proches du sujet.")
        return AgentResult("event_draft", text, u, [self.source(g) for g, _ in hits], draft.model_dump(mode="json"))
