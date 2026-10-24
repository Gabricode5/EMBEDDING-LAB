"""Aide à la relecture : signale les incohérences probables que le validateur ne voit pas.

    python scripts/review_scenarios.py                      # tous les splits
    python scripts/review_scenarios.py data/scenarios/test  # un split
    python scripts/review_scenarios.py data/scenarios/train/immobilis.json --show

Les signalements sont des pistes à vérifier, pas des erreurs certaines.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from sae.schemas import Scenario

STOP = set("""le la les l un une des de du d et à au aux en pour par sur dans avec ce cette ces qui que
qu est sont a ont été être il elle ils on se sa son ses leur leurs pas ne plus si mais ou donc""".split())
HYPOTHESIS = re.compile(r"\b(pourrait|pourrions|pourraient|peut-être|éventuellement|envisage|évoque|suggère|idée)\b", re.I)
PROPOSAL = re.compile(r"\b(propose|proposé|proposée|suggère)\b", re.I)
COMMITMENT = re.compile(r"\bs'engage\b", re.I)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 3 and w not in STOP}


def review(s: Scenario, reference_texts: set[str]) -> list[str]:
    issues: list[str] = []
    units = {u.unit_id: u for u in s.units}

    for u in s.units:
        if COMMITMENT.search(u.statement) and u.type != "commitment":
            issues.append(f"{u.unit_id} dit « s'engage » mais type={u.type}")
        if HYPOTHESIS.search(u.statement) and u.status == "active" and u.type in ("decision", "commitment"):
            issues.append(f"{u.unit_id} ressemble à une hypothèse mais type={u.type}, status=active")
        if PROPOSAL.search(u.statement) and u.type == "decision" and u.status == "active":
            issues.append(f"{u.unit_id} ressemble à une proposition mais est une décision active")
        if u.type in ("commitment", "action") and not u.owner:
            issues.append(f"{u.unit_id} ({u.type}) sans responsable (owner)")
        if u.statement.strip().lower() == u.evidence.strip().lower():
            issues.append(f"{u.unit_id} : statement identique à evidence")

    for r in s.relations:
        a, b = units.get(r.from_id), units.get(r.to_id)
        if not a or not b:
            continue
        tag = f"{r.from_id} --{r.relation}--> {r.to_id}"
        if r.relation == "supersedes":
            if not (words(a.statement) & words(b.statement)):
                issues.append(f"{tag} : aucun mot commun, sujets différents ?\n      « {b.statement} »\n      « {a.statement} »")
        if r.relation == "answers":
            if b.type != "question":
                issues.append(f"{tag} : la cible n'est pas une question (type={b.type})")
            if a.source_date < b.source_date:
                issues.append(f"{tag} : la réponse est antérieure à la question")
            if a.speaker and a.speaker == b.speaker:
                issues.append(f"{tag} : même personne pose la question et répond")
        if r.relation == "creates_action":
            if a.source_date > b.source_date:
                issues.append(f"{tag} : la cause est postérieure à l'action")
            if b.type not in ("action", "commitment", "decision"):
                issues.append(f"{tag} : la cible n'est pas une action (type={b.type})")

    for q in s.queries:
        if q.text.strip().lower() in reference_texts:
            issues.append(f"{q.query_id} recopiée de l'exemple : « {q.text} »")
        for uid in q.relevant_unit_ids:
            if units[uid].status == "superseded" and "compare" not in q.tags and q.intent != "compare":
                issues.append(f"{q.query_id} : réponse attendue {uid} est remplacée (superseded)")
        if q.answerable:
            q_words = words(q.text)
            best = max((len(q_words & words(units[u].statement)) for u in q.relevant_unit_ids), default=0)
            if best == 0:
                issues.append(f"{q.query_id} : aucun mot commun avec sa réponse, vérifier la réponse attendue")
    return issues


def show(s: Scenario) -> None:
    for m in s.meetings:
        print(f"\n== {m.meeting_id} {m.date} · {m.title}")
        for u in m.units:
            owner = f" (resp. {u.owner})" if u.owner else ""
            print(f"  {u.unit_id} [{u.type}/{u.status}] {u.statement}{owner}")
    print("\n== relations")
    for r in s.relations:
        print(f"  {r.from_id} --{r.relation}--> {r.to_id}")
    print("\n== requêtes")
    for q in s.queries:
        print(f"  {q.query_id} {q.text}\n     pertinent={q.relevant_unit_ids} négatifs={q.hard_negative_unit_ids}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    targets = [Path(a) for a in args] or [Path("data/scenarios") / s for s in ("train", "val", "test")]
    files = sorted(p for t in targets for p in ([t] if t.suffix == ".json" else t.glob("*.json")))

    # Scénarios servant d'exemples au générateur : leurs requêtes ne doivent pas être recopiées.
    references = [Path("data/scenarios/train/atlas.json"), Path("data/scenarios/train/immobilis.json")]
    ref_texts: set[str] = set()
    for ref_path in references:
        if ref_path.exists():
            ref = Scenario.model_validate_json(ref_path.read_text(encoding="utf-8"))
            ref_texts |= {q.text.strip().lower() for q in ref.queries}

    total = 0
    for f in files:
        s = Scenario.model_validate_json(f.read_text(encoding="utf-8"))
        issues = [] if f in references else review(s, ref_texts)
        total += len(issues)
        print(f"\n### {f}  ({len(issues)} point(s) à vérifier)")
        for i in issues:
            print(f"  - {i}")
        if "--show" in sys.argv:
            show(s)
    print(f"\nTotal : {total} point(s) à vérifier.")


if __name__ == "__main__":
    main()
