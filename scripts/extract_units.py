"""Extrait automatiquement les unités des réunions d'un split, à partir de la transcription reconstituée.

    python scripts/extract_units.py --split val
    python scripts/extract_units.py --split train --projects atlas      # un seul projet
    python scripts/extract_units.py --split val --force                  # ignore le cache

Sorties (data/scenarios n'est jamais modifié) :
    data/extracted/<split>/<projet>.json             même format que les scénarios : mêmes réunions et personnes,
                                                     unités extraites, relations vides, questions réécrites
    data/extracted/<split>/alignment/<projet>.json   unité annotée -> unité extraite, questions perdues
    data/extracted/<split>/.cache/<projet>/<réunion>.json   réponse brute du modèle (réutilisée si le prompt
                                                     et le modèle n'ont pas changé)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from sae.data import gid, load_scenarios
from sae.extraction import align, extract_raw, parse, prompt_hash, rewrite_queries, to_units
from sae.schemas import Meeting, Scenario


def cached_raw(path: Path, meeting: Meeting, args, key: str) -> tuple[str, bool]:
    """Réponse brute du modèle, depuis le cache si possible ; (réponse, vient du cache)."""
    if path.exists() and not args.force:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("key") == key:
            return entry["raw"], True
    raw = extract_raw(meeting, args.model, args.ollama_host)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "model": args.model, "raw": raw}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return raw, False


def extract_scenario(s: Scenario, out: Path, args, key: str, progress: list[int]) -> tuple[Scenario, dict]:
    meetings, stats = [], {"proposed": 0, "regrounded": 0, "dropped": 0}
    for m in s.meetings:
        t0 = time.perf_counter()
        raw, hit = cached_raw(out / ".cache" / s.project_id / f"{m.meeting_id}.json", m, args, key)
        units, st = to_units(parse(raw), m, s.people, start=1 + sum(len(x.units) for x in meetings))
        stats = {k: stats[k] + st[k] for k in stats}
        meetings.append(m.model_copy(update={"units": units}))
        progress[0] += 1
        print(f"  [{progress[0]}/{progress[1]}] {s.project_id}/{m.meeting_id} : {len(m.units)} annotées -> "
              f"{len(units)} extraites ({'cache' if hit else f'{time.perf_counter() - t0:.1f} s'})", flush=True)
    extracted = [u for m in meetings for u in m.units]
    mapping = align(s.units, extracted)
    queries, lost = rewrite_queries(s.queries, mapping)
    result = Scenario(project_id=s.project_id, project_name=s.project_name, description=s.description,
                      people=s.people, meetings=meetings, relations=[], queries=queries)
    report = {"units": {gid(s.project_id, a): gid(s.project_id, e) for a, e in mapping.items()},
              "lost_queries": lost, "grounding": stats, "key": key}
    return result, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="nom du split (train, val, test, test2)")
    ap.add_argument("--projects", nargs="*", help="limiter à certains projets")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--force", action="store_true", help="ignore le cache et relance le modèle")
    args = ap.parse_args()

    scenarios = [s for s in load_scenarios(f"data/scenarios/{args.split}")
                 if not args.projects or s.project_id in args.projects]
    out = Path("data/extracted") / args.split
    (out / "alignment").mkdir(parents=True, exist_ok=True)
    key = prompt_hash(args.model)
    progress = [0, sum(len(s.meetings) for s in scenarios)]
    print(f"Extraction {args.split} : {len(scenarios)} projets, {progress[1]} réunions, modèle {args.model} "
          f"(prompt {key})")
    for s in scenarios:
        result, report = extract_scenario(s, out, args, key, progress)
        (out / f"{s.project_id}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (out / "alignment" / f"{s.project_id}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        g = report["grounding"]
        print(f"  {s.project_id} : {len(s.units)} annotées, {len(result.units)} extraites, "
              f"{len(report['units'])} alignées · evidence recalées {g['regrounded']}, écartées {g['dropped']} · "
              f"questions perdues {len(report['lost_queries'])}")
    print(f"\nRésultats : {out}/")


if __name__ == "__main__":
    main()
