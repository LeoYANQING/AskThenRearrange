"""
Verification test for BeliefEstimator.

Checks:
1. LLM returns valid structured output (parsing works)
2. Probabilities are non-trivial (not all uniform)
3. Entropy decreases after adding confirmed evidence
4. Objects covered by preferences have lower entropy

Usage:
    python -m test_belief_estimator [--num-episodes 3]
"""
from __future__ import annotations

import argparse
import json
import os
from time import perf_counter
from typing import Dict, List

from agent_schema import AgentState
from belief_estimator import BeliefEstimator, max_entropy, shannon_entropy
from data import DEFAULT_DATA_PATH, get_episode, load_episodes
from state_init import build_initial_state


OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def print_belief_table(entropies: Dict[str, float], h_max: float) -> None:
    """Print sorted entropy table."""
    sorted_items = sorted(entropies.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Object':<35} {'Entropy':>8} {'% of max':>10}")
    print(f"  {'─' * 35} {'─' * 8} {'─' * 10}")
    for obj, h in sorted_items:
        pct = (h / h_max * 100) if h_max > 0 else 0
        print(f"  {obj:<35} {h:8.3f} {pct:9.1f}%")
    mean_h = sum(entropies.values()) / max(1, len(entropies))
    print(f"\n  Mean entropy: {mean_h:.3f} / {h_max:.3f} ({mean_h / h_max * 100:.1f}%)")


def print_detailed_beliefs(estimator: BeliefEstimator, state: AgentState) -> None:
    """Print full belief breakdown per object."""
    result = estimator.estimate_detailed(state)
    if result is None:
        print("  [ERROR] LLM returned no structured output")
        return
    for b in result.beliefs:
        pairs = list(zip(b.top_receptacles, b.probabilities))
        dist_str = ", ".join(f"{r}: {p:.2f}" for r, p in pairs)
        print(f"  {b.object_name:<35} → {dist_str}")


def test_episode(
    estimator: BeliefEstimator,
    episode_index: int,
    data_path: str,
) -> Dict:
    """Run belief estimation verification on one episode."""
    episode = get_episode(data_path, index=episode_index)
    state = build_initial_state(episode, strategy="direct", budget_total=6)
    h_max = max_entropy(len(state["receptacles"]))

    print(f"\n{'=' * 70}")
    print(f"Episode {episode.episode_id} | room={episode.room}")
    print(f"  receptacles ({len(state['receptacles'])}): {state['receptacles']}")
    print(f"  unresolved_objects ({len(state['unresolved_objects'])}): {state['unresolved_objects']}")
    print(f"  H_max = {h_max:.3f} bits")

    # ── Test 1: Initial beliefs (no evidence) ──
    print(f"\n── Test 1: Initial beliefs (zero evidence) ──")
    t0 = perf_counter()
    entropies_initial = estimator.estimate(state)
    t1 = perf_counter()
    print(f"  LLM call: {t1 - t0:.2f}s")

    if not entropies_initial:
        print("  [FAIL] No entropies returned")
        return {"episode": episode.episode_id, "pass": False, "reason": "no output"}

    print_belief_table(entropies_initial, h_max)
    print(f"\n  Detailed beliefs:")
    print_detailed_beliefs(estimator, state)

    # Check: not all uniform
    h_values = list(entropies_initial.values())
    h_range = max(h_values) - min(h_values)
    uniform_check = h_range > 0.1
    print(f"\n  Entropy range: {h_range:.3f} (non-uniform: {'PASS' if uniform_check else 'WARN — all similar'})")

    # ── Test 2: Add fake confirmed evidence, re-estimate ──
    print(f"\n── Test 2: After adding confirmed evidence ──")
    # Simulate: 2 confirmed actions + 1 preference
    obj_0 = state["unresolved_objects"][0]
    obj_1 = state["unresolved_objects"][1] if len(state["unresolved_objects"]) > 1 else obj_0
    receptacle_0 = state["receptacles"][0]

    state_with_evidence = dict(state)
    state_with_evidence["confirmed_actions"] = [
        {"object_name": obj_0, "receptacle": receptacle_0},
        {"object_name": obj_1, "receptacle": receptacle_0},
    ]
    # Remove confirmed objects from unresolved
    state_with_evidence["unresolved_objects"] = [
        o for o in state["unresolved_objects"] if o not in (obj_0, obj_1)
    ]
    # Add a preference covering some remaining objects
    remaining = state_with_evidence["unresolved_objects"]
    if len(remaining) >= 2:
        pref_objects = remaining[:2]
        state_with_evidence["confirmed_preferences"] = [
            {
                "hypothesis": f"Items like {pref_objects[0]} and {pref_objects[1]} go together",
                "covered_objects": pref_objects,
                "receptacle": state["receptacles"][1] if len(state["receptacles"]) > 1 else receptacle_0,
            }
        ]
    else:
        pref_objects = []
        state_with_evidence["confirmed_preferences"] = []

    t0 = perf_counter()
    entropies_after = estimator.estimate(state_with_evidence)
    t1 = perf_counter()
    print(f"  Evidence: confirmed_actions={len(state_with_evidence['confirmed_actions'])}, "
          f"confirmed_preferences={len(state_with_evidence['confirmed_preferences'])}")
    print(f"  LLM call: {t1 - t0:.2f}s")
    print_belief_table(entropies_after, h_max)

    # Check: objects covered by preference should have lower entropy
    if pref_objects:
        pref_entropies = [entropies_after[o] for o in pref_objects if o in entropies_after]
        other_entropies = [h for o, h in entropies_after.items() if o not in pref_objects]
        if pref_entropies and other_entropies:
            mean_pref = sum(pref_entropies) / len(pref_entropies)
            mean_other = sum(other_entropies) / len(other_entropies)
            pref_check = mean_pref < mean_other
            print(f"\n  Preference-covered objects mean entropy: {mean_pref:.3f}")
            print(f"  Other objects mean entropy:              {mean_other:.3f}")
            print(f"  Preference reduces entropy: {'PASS' if pref_check else 'WARN'}")
        else:
            pref_check = None
    else:
        pref_check = None

    # ── Test 3: Mean entropy should decrease with evidence ──
    mean_initial = sum(entropies_initial.values()) / max(1, len(entropies_initial))
    mean_after = sum(entropies_after.values()) / max(1, len(entropies_after))
    # Note: remaining objects might not decrease (different set), so this is informational
    print(f"\n── Summary ──")
    print(f"  Mean entropy (initial, all objects):      {mean_initial:.3f}")
    print(f"  Mean entropy (after evidence, remaining): {mean_after:.3f}")

    return {
        "episode": episode.episode_id,
        "num_objects": len(state["unresolved_objects"]),
        "h_max": h_max,
        "mean_initial": mean_initial,
        "mean_after": mean_after,
        "entropy_range": h_range,
        "non_uniform": uniform_check,
        "preference_reduces_entropy": pref_check,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify BeliefEstimator output quality")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL)
    args = parser.parse_args()

    estimator = BeliefEstimator(model=args.model, base_url=args.base_url)

    results = []
    for i in range(args.num_episodes):
        r = test_episode(estimator, episode_index=i, data_path=args.data)
        results.append(r)

    # ── Final verdict ──
    print(f"\n{'=' * 70}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        non_uni = "PASS" if r.get("non_uniform") else "WARN"
        pref = r.get("preference_reduces_entropy")
        pref_str = "PASS" if pref is True else ("WARN" if pref is False else "N/A")
        print(
            f"  {r['episode']:<20} "
            f"H_range={r.get('entropy_range', 0):.3f} ({non_uni})  "
            f"pref_effect={pref_str}  "
            f"mean_H={r.get('mean_initial', 0):.3f}→{r.get('mean_after', 0):.3f}"
        )

    all_non_uniform = all(r.get("non_uniform", False) for r in results)
    pref_checks = [r["preference_reduces_entropy"] for r in results if r.get("preference_reduces_entropy") is not None]
    pref_pass_rate = sum(1 for p in pref_checks if p) / max(1, len(pref_checks))

    print(f"\n  Non-uniform beliefs: {'ALL PASS' if all_non_uniform else 'SOME WARN'}")
    print(f"  Preference effect: {pref_pass_rate:.0%} pass ({len(pref_checks)} tested)")
    print(f"\n  Verdict: {'READY for entropy-driven policy' if all_non_uniform and pref_pass_rate >= 0.5 else 'NEEDS investigation'}")


if __name__ == "__main__":
    main()
