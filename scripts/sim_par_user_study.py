"""Simulate PAR (Parallel Exploration) trials for Study 2 question-quality eval.

PAR mix is AO + PI. Mirrors sim_upf_user_study.py structure but uses mode=
"parallel_exploration" at every controller call so we get a realistic stream
of AO + PI questions to inspect + feed to the translator.

Output tagged `<SIM_TAG>`; default 10 episodes, B=6.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import load_episodes
from evaluation import FinalPlacementPlanner, evaluate_episode_state
from oracle import NaturalUserOracle
from proposers import (
    ActionProposer,
    PreferenceElicitingProposer,
    PreferenceInductionProposer,
)
from question_policy import QuestionDecision, QuestionPolicyController
from state_init import build_initial_state
from state_update import StateUpdate

BUDGET = 6
ROOMS = ["living room", "bedroom", "kitchen"]
MODE = "parallel_exploration"

DEFAULT_OFFSET = int(os.environ.get("SIM_OFFSET", "0"))


def pick_episodes(offset: int = 0):
    eps = load_episodes(ROOT / "data" / "scenarios_three_rooms_102.json")
    by_room = {r: [e for e in eps if e.room == r] for r in ROOMS}
    picked = []
    picked.extend(by_room["living room"][offset:offset + 4])
    picked.extend(by_room["bedroom"][offset:offset + 3])
    picked.extend(by_room["kitchen"][offset:offset + 3])
    return picked


def propose(state, decision, ao, pe, pi):
    if decision.question_pattern == "action_oriented":
        return ao.propose(state=state, guidance=decision.guidance)
    if decision.question_pattern == "preference_eliciting":
        return pe.propose(state=state, guidance=decision.guidance)
    if decision.question_pattern == "preference_induction":
        intents = pi.propose(state=state, guidance=decision.guidance, max_intents=3)
        return intents[0] if intents else None
    return None


def _fallback_to_ao(state, ao):
    decision = QuestionDecision(
        question_pattern="action_oriented",
        guidance="Primary proposer returned nothing; ask a direct placement question.",
    )
    return decision, ao.propose(state=state, guidance=decision.guidance)


def run_episode(episode, controller, ao, pe, pi, oracle, updater):
    state = build_initial_state(episode=episode, strategy="parallel_exploration", budget_total=BUDGET)
    turns = []
    while len(state["qa_history"]) < BUDGET:
        decision = controller.plan_next_question(state=state, mode=MODE)
        if decision is None:
            break
        intent = propose(state, decision, ao, pe, pi)
        if intent is None:
            decision, intent = _fallback_to_ao(state, ao)
            if intent is None:
                break

        question = intent.question if hasattr(intent, "question") else str(intent.get("question", ""))
        hypothesis = None
        covered = []
        if decision.question_pattern in ("preference_eliciting", "preference_induction"):
            hypothesis = str(intent.get("hypothesis", ""))
            covered = list(intent.get("covered_objects", []))

        oracle_resp = oracle.answer(
            question=question,
            room=state["room"],
            receptacles=state["receptacles"],
            seen_objects=state["seen_objects"],
            annotator_notes=episode.annotator_notes,
            gt_seen_placements=episode.seen_placements,
            qa_history=state["qa_history"],
        )
        answer = oracle_resp.answer

        if decision.question_pattern == "action_oriented":
            state = updater.update_state_from_action_answer(
                state=state, target=intent.object_name,
                answer=answer, question=intent.question,
                action_mode=intent.action_mode,
            )
        elif decision.question_pattern == "preference_eliciting":
            state = updater.update_state_from_preference_eliciting_answer(
                state=state, hypothesis=hypothesis, covered_objects=covered,
                answer=answer, question=question,
                oracle_receptacle=oracle_resp.referenced_receptacle,
            )
        elif decision.question_pattern == "preference_induction":
            state = updater.update_state_from_preference_induction_answer(
                state=state, hypothesis=hypothesis, covered_objects=covered,
                answer=answer, question=question,
            )

        turns.append({
            "turn": len(state["qa_history"]),
            "pattern": decision.question_pattern,
            "question": question,
            "hypothesis": hypothesis,
            "covered_objects": covered,
            "answer": answer,
            "oracle_receptacle": oracle_resp.referenced_receptacle,
        })
    return state, turns


def _mean_stderr(values):
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        stderr = (var ** 0.5) / (n ** 0.5)
    else:
        stderr = 0.0
    return mean, stderr


def main():
    offset = DEFAULT_OFFSET
    out_tag = os.environ.get("SIM_TAG", "par_run1")
    episodes = pick_episodes(offset)
    print(f"Running {len(episodes)} PAR episodes, budget={BUDGET}, offset={offset}, tag={out_tag}")
    controller = QuestionPolicyController(selection_method="rule")
    ao = ActionProposer()
    pe = PreferenceElicitingProposer()  # PAR doesn't use PE, but keep for completeness
    pi = PreferenceInductionProposer()
    oracle = NaturalUserOracle()
    updater = StateUpdate()
    planner = FinalPlacementPlanner()

    pattern_counter = Counter()
    all_results = []
    seen_scores, unseen_scores, total_scores = [], [], []

    for idx, ep in enumerate(episodes, 1):
        print(f"\n{'=' * 80}")
        print(f"Episode {idx}/{len(episodes)}  id={ep.episode_id}  room={ep.room}")
        print(f"{'=' * 80}", flush=True)
        try:
            state, turns = run_episode(ep, controller, ao, pe, pi, oracle, updater)
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            continue
        for t in turns:
            pattern_counter[t["pattern"]] += 1
            print(f"\n[T{t['turn']}] pattern={t['pattern']}")
            if t["hypothesis"]:
                print(f"  hypothesis: {t['hypothesis']}")
                print(f"  covered:    {t['covered_objects']}")
            print(f"  Q: {t['question']}")
            print(f"  A: {t['answer']}")
        try:
            ev = evaluate_episode_state(ep, state, planner=planner)
        except Exception as e:
            print(f"[EVAL ERROR] {type(e).__name__}: {e}")
            ev = {"seen_accuracy": 0.0, "unseen_accuracy": 0.0}
        seen_acc = float(ev.get("seen_accuracy", 0.0))
        unseen_acc = float(ev.get("unseen_accuracy", 0.0))
        n_seen = len(ep.seen_objects) or 1
        n_unseen = len(ep.unseen_objects) or 1
        total_acc = (seen_acc * n_seen + unseen_acc * n_unseen) / (n_seen + n_unseen)
        print(f"\n[EVAL] seen={seen_acc:.3f}  unseen={unseen_acc:.3f}  total={total_acc:.3f}", flush=True)
        seen_scores.append(seen_acc); unseen_scores.append(unseen_acc); total_scores.append(total_acc)

        all_results.append({
            "episode_id": ep.episode_id, "room": ep.room, "turns": turns,
            "seen_psr": seen_acc, "unseen_psr": unseen_acc, "total_psr": total_acc,
        })

    print(f"\n{'=' * 80}\nPattern distribution across all turns:")
    total_turns = sum(pattern_counter.values())
    for pat, cnt in pattern_counter.most_common():
        pct = 100.0 * cnt / total_turns if total_turns else 0
        print(f"  {pat:<24} {cnt:>4}  ({pct:.1f}%)")

    seen_m, seen_se = _mean_stderr(seen_scores)
    unseen_m, unseen_se = _mean_stderr(unseen_scores)
    total_m, total_se = _mean_stderr(total_scores)
    print(f"\nAggregate PSR (n={len(seen_scores)}, B={BUDGET}):")
    print(f"  seen_psr    = {seen_m:.3f} ± {seen_se:.3f}")
    print(f"  unseen_psr  = {unseen_m:.3f} ± {unseen_se:.3f}")
    print(f"  total_psr   = {total_m:.3f} ± {total_se:.3f}")

    out_path = ROOT / "logs" / f"sim_par_user_study_{out_tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nFull transcript -> {out_path}")


if __name__ == "__main__":
    main()
