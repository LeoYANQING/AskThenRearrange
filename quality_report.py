#!/usr/bin/env python3
"""
Comprehensive quality report for the v4 dataset and oracle.

Checks:
1. Coverage: every receptacle has objects AND a matching annotation note
2. Consistency: every object's placement matches at least one annotation note's receptacle
3. Note quality: can qwen3.5 match objects at a receptacle to the note? (10 episodes)
4. Oracle quality: PE questions get useful oracle answers (5 episodes)
5. Split balance: seen/unseen distribution per receptacle
6. Diversity: unique note count, vocabulary stats
"""

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent / "data" / "scenarios_three_rooms_102_hard_v4.json"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5")
BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data() -> List[Dict[str, Any]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_used_receptacles(ep: Dict[str, Any]) -> Set[str]:
    recs: Set[str] = set()
    for _, rec in ep["seen_placements"]:
        recs.add(rec)
    for _, rec in ep["unseen_placements"]:
        recs.add(rec)
    return recs


def get_objects_at_receptacle(ep: Dict[str, Any], rec: str) -> Tuple[List[str], List[str]]:
    """Return (seen_objects, unseen_objects) at a given receptacle."""
    seen = [obj for obj, r in ep["seen_placements"] if r == rec]
    unseen = [obj for obj, r in ep["unseen_placements"] if r == rec]
    return seen, unseen


def find_note_for_receptacle(notes: List[str], rec: str) -> str | None:
    for note in notes:
        if rec.lower() in note.lower():
            return note
    return None


# ---------------------------------------------------------------------------
# Check 1: Coverage
# ---------------------------------------------------------------------------

def check_coverage(data: List[Dict[str, Any]]) -> Tuple[bool, str]:
    issues = []
    for i, ep in enumerate(data):
        used_recs = get_used_receptacles(ep)
        listed_recs = set(ep["receptacles"])

        # Every listed receptacle should have objects
        for rec in listed_recs:
            seen, unseen = get_objects_at_receptacle(ep, rec)
            if len(seen) + len(unseen) == 0:
                issues.append(f"  ep {i}: '{rec}' has 0 objects")

        # Every listed receptacle should have a matching note
        for rec in listed_recs:
            note = find_note_for_receptacle(ep["annotator_notes"], rec)
            if note is None:
                issues.append(f"  ep {i}: '{rec}' has no matching annotation note")

        # Every used receptacle should be in the list
        missing = used_recs - listed_recs
        if missing:
            issues.append(f"  ep {i}: used but not listed: {missing}")

    passed = len(issues) == 0
    detail = f"All {len(data)} episodes pass coverage check." if passed else f"{len(issues)} issues found:\n" + "\n".join(issues[:20])
    return passed, detail


# ---------------------------------------------------------------------------
# Check 2: Consistency
# ---------------------------------------------------------------------------

def check_consistency(data: List[Dict[str, Any]]) -> Tuple[bool, str]:
    issues = []
    for i, ep in enumerate(data):
        noted_recs: Set[str] = set()
        for note in ep["annotator_notes"]:
            for rec in ep["receptacles"]:
                if rec.lower() in note.lower():
                    noted_recs.add(rec)

        all_placements = list(ep["seen_placements"]) + list(ep["unseen_placements"])
        for obj, rec in all_placements:
            if rec not in noted_recs:
                issues.append(f"  ep {i}: '{obj}' -> '{rec}' but no annotation note mentions '{rec}'")

    passed = len(issues) == 0
    detail = f"All placements consistent with annotation notes." if passed else f"{len(issues)} issues:\n" + "\n".join(issues[:20])
    return passed, detail


# ---------------------------------------------------------------------------
# Check 3: Note quality (LLM matching test)
# ---------------------------------------------------------------------------

class NoteMatchResult(BaseModel):
    matches: bool = Field(description="True if all objects at the receptacle can plausibly be explained by the note.")
    reasoning: str = Field(description="Brief explanation.")


def _parse_yes_no(text: str) -> bool:
    """Fallback parser: look for yes/no signal in free text."""
    lower = text.lower()
    # Check for explicit "yes" at the start or "all ... match" patterns
    if lower.startswith("yes") or "all of the objects plausibly belong" in lower or "every item" in lower:
        return True
    if lower.startswith("no") or "not all" in lower or "does not" in lower:
        return False
    # Default: if "yes" appears more than "no", treat as yes
    return lower.count("yes") >= lower.count("no")


def check_note_quality(data: List[Dict[str, Any]], num_episodes: int = 10) -> Tuple[bool, str]:
    llm = ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0.0, reasoning=False, timeout=120)
    structured = llm.with_structured_output(NoteMatchResult)
    raw_llm = ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0.0, reasoning=False, timeout=120)

    total_tests = 0
    total_pass = 0
    failures = []

    for i in range(min(num_episodes, len(data))):
        ep = data[i]
        for rec in ep["receptacles"]:
            note = find_note_for_receptacle(ep["annotator_notes"], rec)
            if note is None:
                continue
            seen, unseen = get_objects_at_receptacle(ep, rec)
            all_objs = seen + unseen
            if not all_objs:
                continue

            total_tests += 1
            prompt = f"""Given this household organizing rule:
"{note}"

Can ALL of these objects plausibly belong at that location according to the rule?
Objects: {all_objs}

Answer whether ALL objects match the rule. Be lenient — if an object reasonably fits the described category, count it as matching."""

            messages = [{"role": "user", "content": prompt}]

            try:
                result = structured.invoke(messages)
                if result.matches:
                    total_pass += 1
                else:
                    failures.append(f"  ep {i}, {rec}: {result.reasoning}")
            except Exception:
                # Fallback: use raw LLM and parse yes/no from text
                try:
                    raw_resp = raw_llm.invoke(messages)
                    text = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
                    if _parse_yes_no(text):
                        total_pass += 1
                    else:
                        failures.append(f"  ep {i}, {rec}: raw LLM said no: {text[:150]}")
                except Exception as e2:
                    failures.append(f"  ep {i}, {rec}: LLM error: {e2}")

    pass_rate = total_pass / total_tests if total_tests > 0 else 0
    passed = pass_rate >= 0.80
    detail = f"Note quality: {total_pass}/{total_tests} ({pass_rate:.0%}) receptacle-note matches pass."
    if failures:
        detail += f"\n  Failures ({len(failures)}):\n" + "\n".join(failures[:10])
    return passed, detail


# ---------------------------------------------------------------------------
# Check 4: Oracle quality (PE question test)
# ---------------------------------------------------------------------------

def check_oracle_quality(data: List[Dict[str, Any]], num_episodes: int = 5) -> Tuple[bool, str]:
    from oracle import NaturalUserOracle

    oracle = NaturalUserOracle(model=MODEL, base_url=BASE_URL, temperature=0.0)
    total_tests = 0
    total_correct = 0
    details = []

    for i in range(min(num_episodes, len(data))):
        ep = data[i]
        # Normalize placements to dict
        seen_dict = {obj: rec for obj, rec in ep["seen_placements"]}

        for rec in ep["receptacles"]:
            note = find_note_for_receptacle(ep["annotator_notes"], rec)
            if note is None:
                continue

            total_tests += 1
            question = f"What kinds of items do you usually keep on the {rec}?"

            try:
                resp = oracle.answer(
                    question=question,
                    room=ep["room"],
                    receptacles=ep["receptacles"],
                    seen_objects=ep["seen_objects"],
                    annotator_notes=ep["annotator_notes"],
                    gt_seen_placements=seen_dict,
                    qa_history=[],
                )
                # Check if oracle names the correct receptacle
                if resp.referenced_receptacle == rec:
                    total_correct += 1
                else:
                    details.append(
                        f"  ep {i}, asked about '{rec}': got ref='{resp.referenced_receptacle}', answer='{resp.answer[:100]}...'"
                    )
            except Exception as e:
                details.append(f"  ep {i}, {rec}: error: {e}")

    rate = total_correct / total_tests if total_tests > 0 else 0
    passed = rate >= 0.80
    summary = f"Oracle PE quality: {total_correct}/{total_tests} ({rate:.0%}) correct receptacle references."
    if details:
        summary += f"\n  Mismatches ({len(details)}):\n" + "\n".join(details[:10])
    return passed, summary


# ---------------------------------------------------------------------------
# Check 5: Split balance
# ---------------------------------------------------------------------------

def check_split_balance(data: List[Dict[str, Any]]) -> Tuple[bool, str]:
    rec_seen_counts: Dict[str, List[int]] = defaultdict(list)
    rec_unseen_counts: Dict[str, List[int]] = defaultdict(list)
    gaps = 0
    total_recs = 0

    for ep in data:
        for rec in ep["receptacles"]:
            seen, unseen = get_objects_at_receptacle(ep, rec)
            rec_seen_counts[rec].append(len(seen))
            rec_unseen_counts[rec].append(len(unseen))
            total_recs += 1
            if len(seen) == 0 or len(unseen) == 0:
                gaps += 1

    lines = []
    unique_recs = sorted(rec_seen_counts.keys())
    for rec in unique_recs:
        s = rec_seen_counts[rec]
        u = rec_unseen_counts[rec]
        avg_s = sum(s) / len(s) if s else 0
        avg_u = sum(u) / len(u) if u else 0
        zero_seen = sum(1 for x in s if x == 0)
        zero_unseen = sum(1 for x in u if x == 0)
        lines.append(f"  {rec:25s}: avg_seen={avg_s:.1f}, avg_unseen={avg_u:.1f}, zero_seen={zero_seen}, zero_unseen={zero_unseen}")

    gap_rate = gaps / total_recs if total_recs > 0 else 0
    passed = gap_rate < 0.30  # less than 30% of rec-instances have a seen/unseen gap
    summary = f"Split balance: {gaps}/{total_recs} ({gap_rate:.0%}) receptacle-instances have a seen or unseen gap.\n"
    summary += f"Unique receptacle names across dataset: {len(unique_recs)}\n"
    summary += "\n".join(lines[:30])
    if len(lines) > 30:
        summary += f"\n  ... and {len(lines) - 30} more"
    return passed, summary


# ---------------------------------------------------------------------------
# Check 6: Diversity
# ---------------------------------------------------------------------------

def check_diversity(data: List[Dict[str, Any]]) -> Tuple[bool, str]:
    all_notes = []
    for ep in data:
        all_notes.extend(ep["annotator_notes"])

    unique_notes = set(all_notes)
    total_notes = len(all_notes)

    # Vocabulary stats
    words: List[str] = []
    for note in all_notes:
        tokens = re.findall(r"[a-z]+", note.lower())
        words.extend(tokens)

    word_counts = Counter(words)
    unique_words = len(word_counts)
    top_words = word_counts.most_common(15)

    # Notes per episode
    notes_per_ep = [len(ep["annotator_notes"]) for ep in data]
    avg_notes = sum(notes_per_ep) / len(notes_per_ep) if notes_per_ep else 0
    min_notes = min(notes_per_ep) if notes_per_ep else 0
    max_notes = max(notes_per_ep) if notes_per_ep else 0

    # Room distribution
    rooms = Counter(ep["room"] for ep in data)

    passed = len(unique_notes) >= len(data)  # at least as many unique notes as episodes
    summary = (
        f"Total notes: {total_notes}, unique notes: {len(unique_notes)}\n"
        f"  Notes per episode: avg={avg_notes:.1f}, min={min_notes}, max={max_notes}\n"
        f"  Vocabulary: {unique_words} unique words, {len(words)} total tokens\n"
        f"  Top words: {top_words}\n"
        f"  Room distribution: {dict(rooms)}"
    )
    return passed, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  QUALITY REPORT: scenarios_three_rooms_102_hard_v4.json")
    print("=" * 70)
    print(f"  Model: {MODEL}  |  Ollama: {BASE_URL}")
    print(f"  Data: {DATA_PATH}")
    print("=" * 70)
    print()

    data = load_data()
    print(f"Loaded {len(data)} episodes.\n")

    checks = [
        ("1. Coverage", lambda: check_coverage(data)),
        ("2. Consistency", lambda: check_consistency(data)),
        ("3. Note Quality (LLM, 10 eps)", lambda: check_note_quality(data, 10)),
        ("4. Oracle Quality (PE, 5 eps)", lambda: check_oracle_quality(data, 5)),
        ("5. Split Balance", lambda: check_split_balance(data)),
        ("6. Diversity", lambda: check_diversity(data)),
    ]

    results = []
    for name, check_fn in checks:
        print(f"--- {name} ---")
        t0 = time.time()
        try:
            passed, detail = check_fn()
        except Exception as e:
            passed, detail = False, f"ERROR: {e}"
        elapsed = time.time() - t0
        status = "PASS" if passed else "FAIL"
        results.append((name, passed, detail))
        print(f"[{status}] ({elapsed:.1f}s)")
        print(detail)
        print()

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    total = len(results)
    passed_count = sum(1 for _, p, _ in results if p)
    for name, passed, _ in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print()
    print(f"  Overall: {passed_count}/{total} checks passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
