"""Interface web de démonstration de l'agent (Streamlit).

    streamlit run app/streamlit_app.py

L'interface ne fait qu'appeler l'agent (scripts/agent.py : build_agent, DEMO) et afficher ses résultats ;
aucune action n'est exécutée, les mails, tâches et événements restent des brouillons.
"""
from __future__ import annotations

import json
import os
import re
import sys
from argparse import Namespace
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# mêmes chemins relatifs que la ligne de commande (data/scenarios/..., models/...)
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sae.agent.engine import ABSTENTION_STRATEGIES, DEFAULT_ABSTENTION, Agent, AgentResult  # noqa: E402
from sae.agent.writer import DEFAULT_HOST, WRITER_MODELS, Writer  # noqa: E402
from scripts.agent import DEMO, build_agent, free_gpu  # noqa: E402

# projets proposés (test et test2, plus atlas de train pour sa démo) : nom court et jeu de données
PROJECTS = {
    "clinique-tilleuls": ("Clinique des Tilleuls", "test"),
    "horizons-2026": ("Séminaire Horizons", "test"),
    "nordstock-relais": ("Relais NordStock", "test"),
    "grain-ocre-croix-rousse": ("Boutique Grain d'Ocre", "test2"),
    "pierre-angle-crm": ("CRM Pierre d'Angle", "test2"),
    "atlas": ("Atlas", "train"),
}
RELATIONS = {"predicted": "prédite par le modèle", "gold": "annotations (référence)", "none": "aucune"}
ABSTENTION_LABELS = {"score": "seuil de similarité", "none": "jamais (toujours répondre)",
                     "lexical": "vérification lexicale", "gap": "écart entre les deux premières unités",
                     "both": "lexicale et seuil", "either": "lexicale ou seuil"}
INTENTS = {"answer": "question d'information", "compare": "comparaison", "build_dossier": "dossier",
           "draft_email": "brouillon de mail", "create_task": "tâche", "schedule_event": "événement",
           "help": "aide"}
STATUSES = {"active": "en vigueur", "proposed": "proposée", "hypothetical": "hypothétique",
            "superseded": "remplacée", "cancelled": "annulée", "inactive": "remplacée (mémoire)"}
DRAFTS = {"email_draft": "Brouillon de mail", "task_draft": "Proposition de tâche",
          "event_draft": "Proposition d'événement"}
WRITERS = {m: f"modèle local {m}" for m in WRITER_MODELS} | {"none": "aucune (première unité trouvée)"}
# mode comparaison : système de gauche (encodeur non entraîné, mémoire ou non) ; texte des unités avec leur type,
# comme le système complet, pour que seule la différence d'entraînement compte
BASELINES = {
    "e5-base": ("e5-base d'origine, sans mémoire", "e5-base", False),
    "e5-base-memoire": ("e5-base d'origine, avec mémoire", "e5-base", True),
    "bge-m3": ("bge-m3, sans mémoire", "bge-m3", False),
    "bm25": ("BM25, sans mémoire", "bm25", False),
}
BASELINE_TITLES = {"e5-base": "e5-base", "bge-m3": "bge-m3", "bm25": "BM25"}
EXAMPLE_BUTTONS = {"Décision": "Une décision", "Comparaison": "Ce qui a changé", "Dossier": "Un dossier",
                   "Mail": "Un mail"}


@st.cache_resource(show_spinner=False, max_entries=2)
def get_agent(split: str, relations: str, abstention: str, writer: str) -> Agent:
    """Même construction que `python scripts/agent.py` avec ses options par défaut."""
    args = Namespace(split=f"data/scenarios/{split}", retriever="models/e5b-both-type-s1", no_type=False,
                     relations=relations, relations_model="models/relations-cross", threshold=0.45,
                     abstention=abstention, writer=writer, ollama_host=DEFAULT_HOST)
    return build_agent(args)


@st.cache_resource(show_spinner=False, max_entries=1)
def get_baseline(split: str, retriever: str, relations: str, abstention: str, writer: str) -> Agent:
    """Système de gauche du mode comparaison : même construction, encodeur non entraîné."""
    args = Namespace(split=f"data/scenarios/{split}", retriever=retriever, no_type=False, relations=relations,
                     relations_model="models/relations-cross", threshold=0.45, abstention=abstention, writer=writer,
                     ollama_host=DEFAULT_HOST)
    return build_agent(args)


def unload_writers() -> None:
    """Décharge les modèles de rédaction d'Ollama avant de charger un modèle PyTorch : sinon Ollama, chargé
    entre-temps, a pris la mémoire GPU libre. Il les recharge à la question suivante avec la place restante."""
    for m in WRITER_MODELS:
        Writer(m).unload()


def release_baseline() -> None:
    """Libère le second encodeur (et sa mémoire GPU) quand le mode comparaison est coupé ou change."""
    # l'exécution précédente du script garde encore une référence à l'agent : on lui retire son encodeur
    old = st.session_state.pop("baseline_agent", None)
    if old is not None:
        old.retriever = None
    get_baseline.clear()
    free_gpu()
    st.session_state.baseline_key = None


# ---------------------------------------------------------------------- affichage
def split_text(r: AgentResult) -> tuple[str, list[str]]:
    """Réponse visible, et paragraphes rangés dans « Détails » (historique, source, raison du destinataire)."""
    visible, details = [], []
    for par in r.text.split("\n\n"):
        hidden = par.startswith(("Cette information remplace", "Source :", "Sources :", "Pourquoi ce destinataire"))
        (details if hidden else visible).append(par)
    return "\n\n".join(visible), details


def show_details(r: AgentResult, extra: list[str]) -> None:
    with st.expander("Détails"):
        for par in extra:
            st.markdown(par)
        if r.kind == "no_answer" and (r.payload or {}).get("reason"):
            st.markdown(f"Raison de l'abstention : {r.payload['reason']}.")
        for s in r.sources:
            who = f" · {s.speaker}" if s.speaker else ""
            st.markdown(f"**{s.date}** · « {s.meeting} »{who} · {STATUSES.get(s.status, s.status)}\n\n> {s.evidence}")
        w = (r.payload or {}).get("writer")
        if w and w["status"] == "accepted":
            st.caption(f"Réponse rédigée par {w['model']} en {w['seconds']:.1f} s à partir de "
                       f"{len(w['given_unit_ids'])} unités ; sources : les unités qu'il cite.")
        elif w:
            st.caption(f"Rédaction écartée ({w['message']}) : affichage de la première unité trouvée.")
        u = r.understanding
        params = [f"{k} : {v}" for k, v in (("personne", u.person), ("projet", u.project), ("échéance", u.deadline))
                  if v]
        st.caption(f"Intention : {INTENTS.get(u.intent, u.intent)} ({u.reason}) · résultat : {r.kind}"
                   + (" · " + " · ".join(params) if params else ""))
        if r.kind in DRAFTS:
            st.code(json.dumps(r.payload, ensure_ascii=False, indent=2), language="json")


def justification(text: str) -> str | None:
    m = re.search(r"^Pourquoi(?: ce destinataire)? : (.+?)\.?$", text, re.M)
    return m.group(1) if m else None


def recipient_line(rec: dict) -> str:
    return f"{rec.get('name') or '?'} <{rec.get('email') or 'adresse inconnue'}>"


def show_draft(r: AgentResult) -> None:
    p = r.payload or {}
    with st.container(border=True):
        st.markdown(f"**{DRAFTS[r.kind]}** · non envoyé, à confirmer")
        if r.kind == "email_draft":
            rec = p["recipient"]
            st.markdown(f"**À :** {recipient_line(rec)}  \n**Objet :** {p['subject']}")
            st.text(p["body"])
            missing = [rec] if not rec.get("email") else []
        elif r.kind == "task_draft":
            st.markdown(f"**Titre :** {p['title']}  \n**Assignée à :** {recipient_line(p['assignee'])}")
            missing = [p["assignee"]] if not p["assignee"].get("email") else []
        else:
            st.markdown(f"**Titre :** {p['title']}  \n**Participants proposés :** "
                        + (", ".join(recipient_line(x) for x in p["participants"]) or "à préciser"))
            missing = [x for x in p["participants"] if not x.get("email")]
        why = justification(r.text)
        if why:
            st.caption(f"Pourquoi : {why}")
        if missing:
            st.warning("Adresse e-mail manquante pour " + ", ".join(x.get("name") or "?" for x in missing)
                       + " : elle n'est pas inventée, il faudra la compléter.")


def show_result(r: AgentResult, key: str) -> None:
    text, extra = split_text(r)
    if r.kind in DRAFTS:
        show_draft(r)
        extra = [x for x in extra if not x.startswith("Pourquoi")]   # déjà sur la carte
    elif r.kind == "dossier":
        st.markdown(text)
        st.download_button("Télécharger le dossier (.md)", r.text, file_name=f"dossier-{r.understanding.project}.md",
                           mime="text/markdown", key=f"dl-{key}")
    elif r.kind in ("clarify", "no_action", "no_answer"):
        st.info(text)
    else:
        st.markdown(text.replace("\n", "  \n"))
        if ((r.payload or {}).get("writer") or {}).get("disagreement"):
            st.caption("Les réunions ne disent pas toutes la même chose sur ce point.")
    if r.kind != "help":
        show_details(r, extra)


# ---------------------------------------------------------------------- page
# page large en mode comparaison (deux colonnes)
st.set_page_config(page_title="Agent de réunions", layout="wide" if st.session_state.get("compare") else "centered")
st.title("Agent de réunions")
st.caption("Réponses sourcées à partir de réunions de projet fictives. Aucune action n'est exécutée.")

with st.sidebar:
    project = st.selectbox("Projet", list(PROJECTS), format_func=lambda p: PROJECTS[p][0])
    name, split = PROJECTS[project]
    play_demo = st.button("Démonstration", disabled=project not in DEMO, use_container_width=True,
                          help="joue les 7 cas du sujet")
    if project not in DEMO:
        st.caption("Démonstration disponible pour la Clinique des Tilleuls et Atlas.")
    clear = st.button("Effacer la conversation", use_container_width=True)
    with st.expander("Options avancées"):
        relations = st.selectbox("Mémoire des remplacements", list(RELATIONS), format_func=RELATIONS.get)
        st.caption("Comment l'agent sait qu'une information en remplace une autre (une date reportée, "
                   "un responsable changé).")
        strategies = [DEFAULT_ABSTENTION, *(s for s in ABSTENTION_STRATEGIES if s != DEFAULT_ABSTENTION)]
        abstention = st.selectbox("Abstention", strategies, format_func=ABSTENTION_LABELS.get)
        st.caption("Quand l'agent préfère dire qu'il n'a pas trouvé plutôt que répondre à côté.")
        writer = st.selectbox("Rédaction des réponses", list(WRITERS), format_func=WRITERS.get)
        st.caption("Un modèle local (Ollama) lit les unités trouvées et rédige la réponse en citant ses sources ; "
                   "qwen3.5:4b est plus petit, mais n'utilise pas moins de mémoire GPU.")
        stats = st.empty()
    compare = st.toggle("Comparer avec un modèle non entraîné", key="compare")
    if compare:
        baseline = st.selectbox("Système de gauche", list(BASELINES), format_func=lambda b: BASELINES[b][0])
        left_writer = st.checkbox("Rédaction aussi à gauche", value=False)
        st.caption("Plus lent : chaque question passe par les deux systèmes"
                   + (", avec deux rédactions." if left_writer else "."))

agent_key = (split, relations, abstention, writer)
if st.session_state.get("agent_key") != agent_key:
    unload_writers()
with st.spinner("Chargement des modèles..."):
    agent = get_agent(*agent_key)
st.session_state.agent_key = agent_key
if writer != "none" and agent.writer is None:
    st.warning(f"Rédaction indisponible : {Writer(writer).unavailable() or 'modèle injoignable'}. "
               "Les réponses affichent la première unité trouvée.")
stats.caption(f"Mémoire : {len(agent.memory.corpus.ids)} unités, {len(agent.memory.successor)} remplacements.")

left = None
if compare:
    label, retriever, with_memory = BASELINES[baseline]
    key = (split, retriever, relations if with_memory else "none", abstention, writer if left_writer else "none")
    if st.session_state.get("baseline_key") not in (None, key):
        release_baseline()
    if st.session_state.get("baseline_key") != key:
        unload_writers()
    with st.spinner(f"Chargement du système de gauche ({label})..."):
        left = get_baseline(*key)
    st.session_state.baseline_key = key
    st.session_state.baseline_agent = left
    left_title = f"Sans entraînement : {BASELINE_TITLES[retriever]}"
    left_caption = label.split(", ", 1)[1] + ("" if left_writer else ", sans rédaction")
elif st.session_state.get("baseline_key"):
    release_baseline()

# changer de projet repart d'une conversation vide (les exemples dépendent du projet)
if clear or st.session_state.get("project") != project:
    st.session_state.history = []
st.session_state.project = project
history: list[dict] = st.session_state.setdefault("history", [])


def ask(message: str, label: str = "") -> None:
    turn = {"label": label, "message": message, "result": agent.handle(message, project=project)}
    if left is not None:
        turn |= {"left": left.handle(message, project=project), "left_title": left_title,
                 "left_caption": left_caption}
    history.append(turn)


if play_demo:
    with st.spinner("Démonstration en cours..."):
        for label, msg in DEMO[project]:
            ask(msg, label)
if message := st.chat_input("Pose une question sur les réunions"):
    with st.spinner("L'agent réfléchit..."):
        ask(message)

if not history:
    # message d'accueil, avec des exemples tirés des réunions du projet ; un clic envoie la question
    with st.chat_message("assistant"):
        st.markdown(f"Je réponds aux questions sur les réunions du projet {name}, en citant mes sources. "
                    "Je peux aussi montrer ce qui a changé, préparer un dossier ou rédiger un brouillon de mail, "
                    "jamais envoyé. Par exemple :")
        examples = agent.examples(project)
        for col, (label, question) in zip(st.columns(len(examples)), examples):
            if col.button(EXAMPLE_BUTTONS.get(label, label), key=f"ex-{label}", help=question,
                          use_container_width=True):
                with st.spinner("L'agent réfléchit..."):
                    ask(question)
                st.rerun()

for i, turn in enumerate(history):
    with st.chat_message("user"):
        if turn["label"]:
            st.caption(turn["label"])
        st.markdown(turn["message"])
    with st.chat_message("assistant"):
        if compare and "left" in turn:
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(f"**{turn['left_title']}**")
                st.caption(turn["left_caption"])
                show_result(turn["left"], f"{i}-g")
            with col_right:
                st.markdown("**Système complet**")
                st.caption("e5-base fine-tuné, mémoire et rédaction selon les options avancées")
                show_result(turn["result"], f"{i}-d")
        else:
            show_result(turn["result"], str(i))
