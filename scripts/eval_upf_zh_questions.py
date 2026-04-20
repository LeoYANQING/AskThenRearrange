"""Pull a representative sample of UPF English questions from the Study2 logs,
translate each via the actual Study 2 translator, and print side-by-side so we
can evaluate naturalness + conformance to the PE definition.

Output format:
  [idx] (room/episode) kind=RC|CAT hypothesis=...
    EN: ...
    ZH: ...
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from study2_app.backend.translate import build_name_mapping, translate_en_to_zh


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).open()]


def pick_samples():
    """Return a mixed sample: 6 RC + 8 CAT questions across rooms."""
    records = []
    for p in [
        "logs/sim_upf_user_study_study2.jsonl",
        "logs/sim_upf_user_study_study2_offset7.jsonl",
    ]:
        records.extend(load_jsonl(ROOT / p))

    rc, cat = [], []
    for r in records:
        for t in r["turns"]:
            if t["pattern"] != "preference_eliciting":
                continue
            entry = {
                "room": r["room"],
                "episode_id": r["episode_id"],
                "hypothesis": t["hypothesis"],
                "covered_objects": t.get("covered_objects") or [],
                "receptacles": None,
                "seen_objects": None,
                "question": t["question"],
            }
            if not t.get("covered_objects"):
                rc.append(entry)
            else:
                cat.append(entry)
    # Spread across rooms for diversity
    def spread(pool, n):
        by_room = {}
        for e in pool:
            by_room.setdefault(e["room"], []).append(e)
        out = []
        while len(out) < n and any(by_room.values()):
            for room in list(by_room.keys()):
                if by_room[room]:
                    out.append(by_room[room].pop(0))
                    if len(out) >= n:
                        break
        return out

    return spread(rc, 6), spread(cat, 8)


def main():
    rc_samples, cat_samples = pick_samples()
    # Build an ad-hoc name mapping covering the vocab in picked questions.
    # We need receptacle + object translations so the glossary isn't empty.
    # Load full episode list to get canonical vocab.
    from data import load_episodes
    eps = load_episodes(ROOT / "data" / "scenarios_three_rooms_102.json")
    names = set()
    for e in eps[:30]:
        names.update(e.receptacles)
        names.update(e.seen_objects)
        names.update(e.unseen_objects)
    print(f"Building name map for {len(names)} English names...", flush=True)
    name_map = build_name_mapping(sorted(names))

    print(f"\n{'=' * 78}\n RECEPTACLE-CENTRIC QUESTIONS\n{'=' * 78}")
    for i, s in enumerate(rc_samples, 1):
        zh = translate_en_to_zh(s["question"], name_map)
        print(f"\n[RC-{i}] room={s['room']}  episode={s['episode_id']}")
        print(f"  hyp: {s['hypothesis']}")
        print(f"  EN: {s['question']}")
        print(f"  ZH: {zh}")

    print(f"\n{'=' * 78}\n CATEGORY-CENTRIC QUESTIONS\n{'=' * 78}")
    for i, s in enumerate(cat_samples, 1):
        zh = translate_en_to_zh(s["question"], name_map)
        print(f"\n[CAT-{i}] room={s['room']}  episode={s['episode_id']}")
        print(f"  hyp: {s['hypothesis']}")
        print(f"  covered: {s['covered_objects']}")
        print(f"  EN: {s['question']}")
        print(f"  ZH: {zh}")


if __name__ == "__main__":
    main()
