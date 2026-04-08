"""Create hard split v4: no gap receptacles, expanded objects, rule-based difficulty.

Design principles:
  1. NO gap receptacles — every receptacle has at least 1 seen object
  2. Expand objects by ~30% (24 → ~31), uneven per rule (2-7 objects each)
  3. Annotations cover ALL objects (seen + unseen)
  4. Difficulty: unseen objects require rule generalization beyond CA analogy
     - seen objects are "typical" examples of a rule
     - unseen objects include "edge cases" that belong to the same rule
       but are harder to match by simple semantic similarity

Uses qwen3.5 for object expansion and annotation generation.
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_ollama import ChatOllama

random.seed(42)

SRC = Path(__file__).resolve().parent / "data" / "scenarios_three_rooms_102.json"
DST = Path(__file__).resolve().parent / "data" / "scenarios_three_rooms_102_hard_v4.json"

MODEL = "qwen3.5"
BASE_URL = "http://127.0.0.1:11434"
TARGET_SEEN = 12
EXPAND_RATIO = 1.3  # 30% more objects


def _llm(temp=0.3):
    return ChatOllama(model=MODEL, base_url=BASE_URL, temperature=temp, reasoning=False, timeout=180)


def _match_receptacle(note: str, receptacles: List[str]) -> str:
    note_lower = note.lower()
    matches = [r for r in receptacles if r.lower().replace("_", " ") in note_lower]
    return max(matches, key=len) if matches else ""


# ---------------------------------------------------------------------------
# Step 1: Expand objects per rule (add ~30% more, uneven)
# ---------------------------------------------------------------------------

def expand_objects(
    ep: Dict[str, Any],
    llm: Any,
) -> Dict[str, str]:
    """Generate additional objects. Return full {object: receptacle} map."""
    all_placements = {obj: rec for obj, rec in ep["seen_placements"] + ep["unseen_placements"]}
    by_rec = defaultdict(list)
    for obj, rec in all_placements.items():
        by_rec[rec].append(obj)

    total_original = len(all_placements)
    target_total = int(total_original * EXPAND_RATIO)
    needed = target_total - total_original
    if needed <= 0:
        return dict(all_placements)

    # Distribute new objects unevenly: larger groups get more
    used_recs = [(rec, objs) for rec, objs in by_rec.items() if objs]
    # Weight by current size
    total_size = sum(len(objs) for _, objs in used_recs)
    alloc = {}
    remaining = needed
    for rec, objs in used_recs:
        share = max(1, round(needed * len(objs) / total_size))
        alloc[rec] = min(share, remaining)
        remaining -= alloc[rec]
        if remaining <= 0:
            break
    # Distribute leftover
    for rec, _ in used_recs:
        if remaining <= 0:
            break
        if rec in alloc:
            alloc[rec] += 1
            remaining -= 1

    # Generate new objects per receptacle
    rec_blocks = []
    for rec, count in alloc.items():
        if count <= 0:
            continue
        existing = by_rec[rec]
        rec_blocks.append(
            f"Receptacle: {rec}\n"
            f"Existing objects: {existing}\n"
            f"Generate {count} NEW objects that belong here but are DIFFERENT from the existing ones.\n"
            f"Include some edge cases — objects that a person might not immediately associate with this spot "
            f"but still belong there based on the organizing rule."
        )

    if not rec_blocks:
        return dict(all_placements)

    prompt = f"""Generate new household objects for a {ep["room"]}.

Each object name: 2-5 words, specific and concrete.
Objects must belong at the specified receptacle.
Include some "edge case" objects that still fit the rule but are less obvious.

{"".join(rec_blocks)}

Reply with one line per object: receptacle_name -> object_name"""

    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        return dict(all_placements)

    expanded = dict(all_placements)
    existing_names = {obj.lower() for obj in all_placements}
    recs_lower = {r.lower(): r for r in ep["receptacles"]}

    for line in text.split("\n"):
        line = line.strip().lstrip("- ").lstrip("* ").lstrip("0123456789. ")
        sep = "->" if "->" in line else ("→" if "→" in line else None)
        if not sep:
            continue
        parts = line.split(sep, 1)
        if len(parts) != 2:
            continue
        rec_text = parts[0].strip().strip('"').strip("'").lower()
        obj_text = parts[1].strip().strip('"').strip("'")
        if not obj_text or obj_text.lower() in existing_names:
            continue
        matched_rec = recs_lower.get(rec_text)
        if not matched_rec:
            for r_low, r_orig in recs_lower.items():
                if rec_text in r_low or r_low in rec_text:
                    matched_rec = r_orig
                    break
        if matched_rec:
            expanded[obj_text] = matched_rec
            existing_names.add(obj_text.lower())

    return expanded


# ---------------------------------------------------------------------------
# Step 2: Generate diverse PE-friendly annotations
# ---------------------------------------------------------------------------

def generate_annotations(
    ep: Dict[str, Any],
    all_placements: Dict[str, str],
    llm: Any,
) -> List[str]:
    """Generate one annotation note per used receptacle."""
    by_rec = defaultdict(list)
    for obj, rec in all_placements.items():
        by_rec[rec].append(obj)

    used_recs = [r for r in ep["receptacles"] if by_rec[r]]
    rec_blocks = []
    for rec in used_recs:
        rec_blocks.append(f"Receptacle: {rec}\nObjects: {by_rec[rec]}")

    prompt = f"""Write one organizing rule per receptacle for a {ep["room"]}.

Each rule must:
- Be one natural sentence: "[description] go to the [receptacle name]."
- Describe WHAT KIND of items belong there using simple, common words
- Be broad enough to cover ALL listed objects, including edge cases
- Be specific enough that someone could decide if a NEW object fits
- Vary phrasing between rules — don't repeat the same structure

Good: "Books, magazines, and reading materials go to the bookshelf."
Good: "Small portable electronics and gadgets go to the side drawer."
Bad: "Durable loose organizers and synthetic accessories" — unnatural
Bad: "Catchall items" — too vague

{chr(10).join(rec_blocks)}

Reply with exactly {len(used_recs)} lines: receptacle_name -> rule"""

    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        return ep["annotator_notes"]

    new_notes = {}
    for line in text.split("\n"):
        line = line.strip().lstrip("- ").lstrip("* ").lstrip("0123456789. ")
        sep = "->" if "->" in line else ("→" if "→" in line else None)
        if not sep:
            continue
        parts = line.split(sep, 1)
        if len(parts) != 2:
            continue
        rec_name = parts[0].strip().strip('"').strip("'")
        rule = parts[1].strip().strip('"').strip("'")
        matched = None
        for r in ep["receptacles"]:
            if r.lower() == rec_name.lower() or r.lower().replace("_", " ") == rec_name.lower():
                matched = r
                break
        if not matched:
            for r in ep["receptacles"]:
                if rec_name.lower() in r.lower() or r.lower() in rec_name.lower():
                    matched = r
                    break
        if matched and matched not in new_notes:
            if matched.lower().replace("_", " ") not in rule.lower():
                rule = rule.rstrip(".") + f" go to the {matched}."
            new_notes[matched] = rule

    if len(new_notes) >= len(used_recs) - 1:
        notes = []
        for rec in used_recs:
            if rec in new_notes:
                notes.append(new_notes[rec])
            else:
                notes.append(f"Items for the {rec} go to the {rec}.")
        return notes
    return ep["annotator_notes"]


# ---------------------------------------------------------------------------
# Step 3: Smart split — NO gap receptacles
# ---------------------------------------------------------------------------

def build_split(
    all_placements: Dict[str, str],
    receptacles: List[str],
) -> Tuple[List[str], List[str]]:
    """Split into seen/unseen ensuring EVERY receptacle with objects has at least 1 seen object."""
    by_rec = defaultdict(list)
    for obj, rec in all_placements.items():
        by_rec[rec].append(obj)

    used_recs = [r for r in receptacles if by_rec[r]]

    # Phase 1: guarantee at least 1 seen per receptacle
    must_seen = []
    remaining_pool = []
    for rec in used_recs:
        objs = list(by_rec[rec])
        random.shuffle(objs)
        must_seen.append(objs[0])  # at least 1 seen per receptacle
        remaining_pool.extend(objs[1:])

    # Phase 2: fill seen up to TARGET_SEEN from remaining pool
    random.shuffle(remaining_pool)
    seen_set = set(must_seen)
    for obj in remaining_pool:
        if len(seen_set) >= TARGET_SEEN:
            break
        seen_set.add(obj)

    # If too many must_seen (>12), trim (keep at least 1 per rec)
    if len(must_seen) > TARGET_SEEN:
        # Can't guarantee all recs covered with only 12 seen
        # Prioritize receptacles with most unseen objects
        rec_unseen_count = {r: len(objs) - 1 for r, objs in by_rec.items() if objs}
        sorted_recs = sorted(used_recs, key=lambda r: -rec_unseen_count.get(r, 0))
        seen_set = set()
        for rec in sorted_recs[:TARGET_SEEN]:
            objs = by_rec[rec]
            seen_set.add(objs[0])

    seen = list(seen_set)[:TARGET_SEEN]
    unseen = [obj for obj in all_placements if obj not in set(seen)]

    return seen, unseen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = json.loads(SRC.read_text())
    print(f"Loaded {len(data)} episodes")

    llm = _llm()
    result = []
    stats = {"new_objs": 0, "total_seen": 0, "total_unseen": 0, "gap_episodes": 0}

    for idx, ep in enumerate(data):
        t0 = time.perf_counter()

        # Step 1: Expand objects
        all_placements = expand_objects(ep, llm)
        new_count = len(all_placements) - 24
        stats["new_objs"] += new_count

        # Step 2: Generate annotations
        notes = generate_annotations(ep, all_placements, llm)

        # Step 3: Build split (no gaps)
        seen, unseen = build_split(all_placements, ep["receptacles"])
        stats["total_seen"] += len(seen)
        stats["total_unseen"] += len(unseen)

        # Verify no gap
        seen_recs = set(all_placements[o] for o in seen)
        unseen_recs = set(all_placements[o] for o in unseen)
        has_gap = bool(unseen_recs - seen_recs)
        if has_gap:
            stats["gap_episodes"] += 1

        # Build episode
        new_ep = {
            "annotator_notes": notes,
            "receptacles": ep["receptacles"],
            "room": ep["room"],
            "seen_objects": seen,
            "unseen_objects": unseen,
            "seen_placements": [[obj, all_placements[obj]] for obj in seen],
            "unseen_placements": [[obj, all_placements[obj]] for obj in unseen],
            "tags": ep.get("tags", []),
        }
        result.append(new_ep)

        elapsed = time.perf_counter() - t0
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1:>3}/{len(data)}] {elapsed:.1f}s | +{new_count} objs, seen={len(seen)}, unseen={len(unseen)}, gap={has_gap}")

    DST.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    n = len(result)
    print(f"\nSaved to {DST}")
    print(f"New objects: {stats['new_objs']} total ({stats['new_objs']/n:.1f}/ep)")
    print(f"Avg seen: {stats['total_seen']/n:.1f}, Avg unseen: {stats['total_unseen']/n:.1f}")
    print(f"Episodes with gap receptacles: {stats['gap_episodes']}/{n}")

    # Verify coverage
    all_notes = [n for ep in result for n in ep["annotator_notes"]]
    print(f"Unique notes: {len(set(all_notes))}/{len(all_notes)}")

    # Verify all receptacles covered in seen
    total_recs = 0
    covered_recs = 0
    for ep in result:
        seen_recs = set(r for _, r in ep["seen_placements"])
        unseen_recs = set(r for _, r in ep["unseen_placements"])
        for r in unseen_recs:
            total_recs += 1
            if r in seen_recs:
                covered_recs += 1
    print(f"Unseen receptacles covered by seen: {covered_recs}/{total_recs} ({100*covered_recs/total_recs:.1f}%)")

    # Sample
    ep = result[3]
    print(f"\nSample ep3 ({ep['room']}):")
    print(f"  Seen({len(ep['seen_objects'])}): {ep['seen_objects'][:6]}...")
    print(f"  Unseen({len(ep['unseen_objects'])}): {ep['unseen_objects'][:6]}...")
    by_rec = defaultdict(lambda: {"seen": [], "unseen": []})
    for o, r in ep["seen_placements"]:
        by_rec[r]["seen"].append(o)
    for o, r in ep["unseen_placements"]:
        by_rec[r]["unseen"].append(o)
    for rec in ep["receptacles"]:
        if by_rec[rec]["seen"] or by_rec[rec]["unseen"]:
            print(f"  {rec}: seen={len(by_rec[rec]['seen'])} unseen={len(by_rec[rec]['unseen'])}")
    print(f"  Notes:")
    for note in ep["annotator_notes"]:
        print(f"    - {note}")


if __name__ == "__main__":
    main()
