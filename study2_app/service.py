from __future__ import annotations

import copy
import csv
import json
from typing import Any

from study2_app.config import (
    DEFAULT_BASE_URL,
    DEFAULT_BUDGET_TOTAL,
    DEFAULT_EVALUATION_MODEL,
    DEFAULT_PROPOSER_MODEL,
    DEFAULT_QUESTIONNAIRE_URL,
    DEFAULT_SELECTION_METHOD,
    DEFAULT_UPDATER_MODEL,
    EXPORTS_DIR,
    SCENE_ORDER_INDICES,
    STRATEGY_ORDERS,
)
from study2_app.runtime import Study2Runtime, utc_now
from study2_app.scenes import SceneLibrary
from study2_app.storage import Study2Storage
from study2_app.types import (
    FailureRecord,
    ParticipantRecord,
    PendingTurnRecord,
    PreferenceFormRecord,
    TrialAssignment,
    TrialRecord,
    TrialResultRecord,
    TrialStatus,
)


def _item_accuracy(predicted: dict[str, str], reference: dict[str, str], objects: list[str]) -> float:
    if not objects:
        return 1.0
    correct = sum(1 for obj in objects if predicted.get(obj) == reference.get(obj))
    return correct / len(objects)


def _detect_discussed_items(turns: list[dict[str, Any]], all_items: list[str]) -> list[str]:
    discussed: list[str] = []
    for item in all_items:
        needle = item.lower()
        for turn in turns:
            haystacks = [
                str(turn.get("question", "")),
                str(turn.get("participant_answer_raw", "")),
            ]
            if any(needle in haystack.lower() for haystack in haystacks):
                discussed.append(item)
                break
    return discussed
def _format_runtime_error(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


class Study2ExperimentService:
    def __init__(self) -> None:
        self.storage = Study2Storage()
        self.scenes = SceneLibrary()
        self.runtime = Study2Runtime(
            proposer_model=DEFAULT_PROPOSER_MODEL,
            updater_model=DEFAULT_UPDATER_MODEL,
            evaluation_model=DEFAULT_EVALUATION_MODEL,
            base_url=DEFAULT_BASE_URL,
            budget_total=DEFAULT_BUDGET_TOTAL,
            selection_method=DEFAULT_SELECTION_METHOD,
        )

    def _new_participant_id(self) -> str:
        return f"P{len(self.storage.list_participants()) + 1:03d}"

    def _build_assignments(self, participant_id: str, strategy_group: int, scene_group: int) -> list[TrialAssignment]:
        strategy_order = STRATEGY_ORDERS[strategy_group]
        scene_catalog = self.scenes.list_scenes()
        scene_order = [scene_catalog[index] for index in SCENE_ORDER_INDICES[scene_group]]
        assignments: list[TrialAssignment] = []
        for idx, strategy in enumerate(strategy_order):
            scene = scene_order[idx]
            assignments.append(
                TrialAssignment(
                    trial_id=f"{participant_id}_t{idx + 1}",
                    trial_index=idx,
                    strategy=strategy,  # type: ignore[arg-type]
                    scene_id=scene["scene_id"],
                    scene_label=scene["label"],
                    scene_episode_index=scene["episode_index"],
                )
            )
        return assignments

    def create_participant(self) -> ParticipantRecord:
        count = len(self.storage.list_participants())
        strategy_group = count % len(STRATEGY_ORDERS)
        scene_group = (count // len(STRATEGY_ORDERS)) % len(SCENE_ORDER_INDICES)
        participant_id = self._new_participant_id()
        assignments = self._build_assignments(participant_id, strategy_group, scene_group)
        now = utc_now()
        participant: ParticipantRecord = ParticipantRecord(
            participant_id=participant_id,
            counterbalance_group=f"S{strategy_group + 1}-C{scene_group + 1}",
            strategy_group=strategy_group,
            scene_group=scene_group,
            trial_order=[assignment["trial_id"] for assignment in assignments],
            current_trial_index=0,
            status="active",
            created_at=now,
            updated_at=now,
            completed_at=None,
        )
        self.storage.save_participant(participant)
        for assignment in assignments:
            trial: TrialRecord = TrialRecord(
                trial_id=assignment["trial_id"],
                participant_id=participant_id,
                trial_index=assignment["trial_index"],
                strategy=assignment["strategy"],
                scene_id=assignment["scene_id"],
                scene_label=assignment["scene_label"],
                scene_episode_index=assignment["scene_episode_index"],
                budget_total=DEFAULT_BUDGET_TOTAL,
                questionnaire_url=DEFAULT_QUESTIONNAIRE_URL,
                status="assigned",
                created_at=now,
                updated_at=now,
                started_at=None,
                completed_at=None,
                agent_state=None,
                pending_turn=None,
                turns=[],
                failure=None,
                failure_history=[],
                preference_form=None,
                result=None,
            )
            self.storage.save_trial(trial)
        return participant

    def list_participants(self) -> list[dict[str, Any]]:
        participants = []
        for participant in self.storage.list_participants():
            current_trial = self.get_current_trial(participant["participant_id"])
            participants.append(
                {
                    **participant,
                    "current_trial": current_trial,
                }
            )
        return participants

    def get_participant_dashboard(self, participant_id: str) -> dict[str, Any]:
        participant = self.storage.load_participant(participant_id)
        trials = self.storage.list_trials_for_participant(participant_id)
        current_trial = self.get_current_trial(participant_id)
        return {
            "participant": participant,
            "trials": trials,
            "current_trial": current_trial,
        }

    def get_current_trial(self, participant_id: str) -> TrialRecord | None:
        participant = self.storage.load_participant(participant_id)
        if participant["current_trial_index"] >= len(participant["trial_order"]):
            return None
        trial_id = participant["trial_order"][participant["current_trial_index"]]
        return self.storage.load_trial(trial_id)

    def get_trial(self, trial_id: str) -> TrialRecord:
        return self.storage.load_trial(trial_id)

    def start_trial(self, trial_id: str) -> TrialRecord:
        trial = self.storage.load_trial(trial_id)
        if trial["status"] not in ("assigned", "trial_interrupted"):
            return trial
        scene = self.scenes.get_scene(trial["scene_id"])
        if trial["agent_state"] is None:
            trial["agent_state"] = self.runtime.initialize_state(scene["episode"])
        trial["status"] = "dialogue_active"
        trial["started_at"] = trial["started_at"] or utc_now()
        trial["updated_at"] = utc_now()
        self.storage.save_trial(trial)

        try:
            pending = self.runtime.prepare_next_turn(
                state=trial["agent_state"],
                mode=trial["strategy"],
            )
            trial["pending_turn"] = pending
            trial["failure"] = None
            trial["status"] = "dialogue_waiting_for_answer" if pending else "preference_form_active"
        except Exception as exc:
            failure: FailureRecord = FailureRecord(
                stage="prepare_next_turn",
                message=_format_runtime_error(exc),
                turn_index=len(trial["turns"]) + 1,
                failed_at=utc_now(),
                retry_count=0,
                pending_turn=None,
                answer_text="",
                state_before_turn=copy.deepcopy(trial["agent_state"]),
            )
            trial["failure"] = failure
            trial["failure_history"].append(copy.deepcopy(failure))
            trial["status"] = "dialogue_failed"
        trial["updated_at"] = utc_now()
        self.storage.save_trial(trial)
        return trial

    def submit_answer(self, trial_id: str, answer_text: str) -> TrialRecord:
        trial = self.storage.load_trial(trial_id)
        if trial["status"] != "dialogue_waiting_for_answer" or not trial["pending_turn"]:
            return trial

        state_before = copy.deepcopy(trial["agent_state"])
        pending_turn = copy.deepcopy(trial["pending_turn"])
        trial["status"] = "dialogue_processing"
        trial["updated_at"] = utc_now()
        self.storage.save_trial(trial)

        try:
            next_state, turn_record, next_pending = self.runtime.apply_answer_and_advance(
                state=state_before,
                mode=trial["strategy"],
                pending_turn=pending_turn,
                answer_text=answer_text,
                retry_count=pending_turn["retry_count"],
            )
            trial["agent_state"] = next_state
            trial["turns"].append(turn_record)
            trial["pending_turn"] = next_pending
            trial["failure"] = None
            if next_pending is None:
                trial["status"] = "dialogue_complete"
                trial["updated_at"] = utc_now()
                self.storage.save_trial(trial)
                trial["status"] = "preference_form_active"
            else:
                trial["status"] = "dialogue_waiting_for_answer"
        except Exception as exc:
            failure: FailureRecord = FailureRecord(
                stage="apply_answer",
                message=_format_runtime_error(exc),
                turn_index=pending_turn["turn_index"],
                failed_at=utc_now(),
                retry_count=pending_turn["retry_count"] + 1,
                pending_turn=pending_turn,
                answer_text=answer_text,
                state_before_turn=state_before,
            )
            trial["failure"] = failure
            trial["failure_history"].append(copy.deepcopy(failure))
            trial["status"] = "dialogue_failed"
            trial["agent_state"] = state_before
            trial["pending_turn"] = pending_turn

        trial["updated_at"] = utc_now()
        self.storage.save_trial(trial)
        return trial

    def retry_failed_turn(self, trial_id: str) -> TrialRecord:
        trial = self.storage.load_trial(trial_id)
        failure = trial["failure"]
        if not failure:
            return trial

        if failure["stage"] == "prepare_next_turn":
            trial["agent_state"] = copy.deepcopy(failure["state_before_turn"])
            trial["status"] = "dialogue_active"
            trial["updated_at"] = utc_now()
            self.storage.save_trial(trial)
            return self.start_trial(trial_id)

        if failure["stage"] == "apply_answer":
            pending_turn = copy.deepcopy(failure["pending_turn"])
            if pending_turn is None:
                return trial
            pending_turn["retry_count"] = failure["retry_count"]
            try:
                next_state, turn_record, next_pending = self.runtime.apply_answer_and_advance(
                    state=copy.deepcopy(failure["state_before_turn"]),
                    mode=trial["strategy"],
                    pending_turn=pending_turn,
                    answer_text=failure["answer_text"],
                    retry_count=failure["retry_count"],
                )
                trial["agent_state"] = next_state
                trial["turns"].append(turn_record)
                trial["pending_turn"] = next_pending
                trial["failure"] = None
                trial["status"] = "preference_form_active" if next_pending is None else "dialogue_waiting_for_answer"
            except Exception as exc:
                failure["message"] = _format_runtime_error(exc)
                failure["failed_at"] = utc_now()
                failure["retry_count"] = failure["retry_count"] + 1
                trial["failure"] = failure
                trial["failure_history"].append(copy.deepcopy(failure))
                trial["status"] = "dialogue_failed"
                trial["pending_turn"] = pending_turn
            trial["updated_at"] = utc_now()
            self.storage.save_trial(trial)
        return trial

    def interrupt_trial(self, trial_id: str) -> TrialRecord:
        trial = self.storage.load_trial(trial_id)
        trial["status"] = "trial_interrupted"
        trial["updated_at"] = utc_now()
        self.storage.save_trial(trial)
        participant = self.storage.load_participant(trial["participant_id"])
        participant["status"] = "interrupted"
        participant["updated_at"] = utc_now()
        self.storage.save_participant(participant)
        return trial

    def resume_trial(self, trial_id: str) -> TrialRecord:
        trial = self.storage.load_trial(trial_id)
        participant = self.storage.load_participant(trial["participant_id"])
        participant["status"] = "active"
        participant["updated_at"] = utc_now()
        self.storage.save_participant(participant)

        if trial["status"] != "trial_interrupted":
            return trial

        if trial["pending_turn"]:
            trial["status"] = "dialogue_waiting_for_answer"
            trial["updated_at"] = utc_now()
            self.storage.save_trial(trial)
            return trial

        if trial["preference_form"]:
            trial["status"] = "results_computed" if trial["result"] else "preference_form_active"
            trial["updated_at"] = utc_now()
            self.storage.save_trial(trial)
            return trial

        return self.start_trial(trial_id)

    def submit_preference_form(self, trial_id: str, placements: dict[str, str]) -> TrialRecord:
        trial = self.storage.load_trial(trial_id)
        scene = self.scenes.get_scene(trial["scene_id"])
        episode = scene["episode"]
        allowed_receptacles = set(episode.receptacles)
        all_items = list(episode.seen_objects) + list(episode.unseen_objects)

        if set(placements.keys()) != set(all_items):
            missing = sorted(set(all_items) - set(placements.keys()))
            extra = sorted(set(placements.keys()) - set(all_items))
            raise ValueError(f"Preference form keys mismatch. Missing={missing}, Extra={extra}")
        for obj, receptacle in placements.items():
            if receptacle not in allowed_receptacles:
                raise ValueError(f"Invalid receptacle '{receptacle}' for item '{obj}'.")

        trial["preference_form"] = PreferenceFormRecord(
            trial_id=trial_id,
            placements=placements,
            submitted_at=utc_now(),
        )
        trial["status"] = "preference_form_active"
        trial["updated_at"] = utc_now()
        self.storage.save_trial(trial)
        return self.compute_trial_results(trial_id)

    def compute_trial_results(self, trial_id: str) -> TrialRecord:
        trial = self.storage.load_trial(trial_id)
        if trial["agent_state"] is None or trial["preference_form"] is None:
            return trial

        scene = self.scenes.get_scene(trial["scene_id"])
        episode = scene["episode"]
        predictions = self.runtime.finalize_trial(state=trial["agent_state"])
        all_items = list(episode.seen_objects) + list(episode.unseen_objects)
        discussed_items = _detect_discussed_items(trial["turns"], all_items)
        undiscussed_items = [item for item in all_items if item not in set(discussed_items)]
        reference = trial["preference_form"]["placements"]

        result: TrialResultRecord = TrialResultRecord(
            trial_id=trial_id,
            predicted_placements=predictions,
            participant_reference_placements=reference,
            discussed_items=discussed_items,
            undiscussed_items=undiscussed_items,
            discussed_item_accuracy=_item_accuracy(predictions, reference, discussed_items),
            undiscussed_item_accuracy=_item_accuracy(predictions, reference, undiscussed_items),
            overall_accuracy=_item_accuracy(predictions, reference, all_items),
            confirmed_actions_summary=list(trial["agent_state"]["confirmed_actions"]),
            confirmed_preferences_summary=list(trial["agent_state"]["confirmed_preferences"]),
            computed_at=utc_now(),
        )
        trial["result"] = result
        trial["status"] = "results_computed"
        trial["updated_at"] = utc_now()
        self.storage.save_trial(trial)
        return trial

    def mark_questionnaire_pending(self, trial_id: str) -> TrialRecord:
        trial = self.storage.load_trial(trial_id)
        trial["status"] = "questionnaire_pending"
        trial["updated_at"] = utc_now()
        self.storage.save_trial(trial)
        return trial

    def complete_trial(self, trial_id: str) -> TrialRecord:
        trial = self.storage.load_trial(trial_id)
        trial["status"] = "trial_complete"
        trial["completed_at"] = utc_now()
        trial["updated_at"] = utc_now()
        self.storage.save_trial(trial)

        participant = self.storage.load_participant(trial["participant_id"])
        if participant["current_trial_index"] < len(participant["trial_order"]):
            participant["current_trial_index"] += 1
        participant["updated_at"] = utc_now()
        if participant["current_trial_index"] >= len(participant["trial_order"]):
            participant["status"] = "completed"
            participant["completed_at"] = utc_now()
        else:
            participant["status"] = "active"
        self.storage.save_participant(participant)
        return trial

    def export_participant(self, participant_id: str) -> list[str]:
        participant = self.storage.load_participant(participant_id)
        trials = self.storage.list_trials_for_participant(participant_id)
        participant_export_path = EXPORTS_DIR / f"{participant_id}_bundle.json"
        payload = {
            "participant": participant,
            "trials": trials,
        }
        participant_export_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return [participant_export_path.name]

    def export_all_data(self) -> list[str]:
        participant_rows: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        participants = self.storage.list_participants()
        trials = self.storage.list_trials()
        turns: list[dict[str, Any]] = []
        preference_forms: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []

        for participant in participants:
            participant_rows.append(
                {
                    "participant_id": participant["participant_id"],
                    "counterbalance_group": participant["counterbalance_group"],
                    "strategy_group": participant["strategy_group"],
                    "scene_group": participant["scene_group"],
                    "current_trial_index": participant["current_trial_index"],
                    "status": participant["status"],
                    "created_at": participant["created_at"],
                    "updated_at": participant["updated_at"],
                    "completed_at": participant["completed_at"],
                    "trial_order_json": json.dumps(participant["trial_order"], ensure_ascii=False),
                }
            )

        for trial in trials:
            trial_rows.append(
                {
                    "trial_id": trial["trial_id"],
                    "participant_id": trial["participant_id"],
                    "trial_index": trial["trial_index"],
                    "strategy": trial["strategy"],
                    "scene_id": trial["scene_id"],
                    "scene_label": trial["scene_label"],
                    "scene_episode_index": trial["scene_episode_index"],
                    "budget_total": trial["budget_total"],
                    "status": trial["status"],
                    "started_at": trial["started_at"],
                    "completed_at": trial["completed_at"],
                    "num_turns": len(trial["turns"]),
                    "has_preference_form": bool(trial["preference_form"]),
                    "has_result": bool(trial["result"]),
                    "failure_stage": trial["failure"]["stage"] if trial["failure"] else "",
                    "failure_message": trial["failure"]["message"] if trial["failure"] else "",
                }
            )
            for turn in trial["turns"]:
                turns.append(
                    {
                        "trial_id": trial["trial_id"],
                        "participant_id": trial["participant_id"],
                        **turn,
                    }
                )
            if trial["preference_form"]:
                preference_forms.append(
                    {
                        "trial_id": trial["trial_id"],
                        "participant_id": trial["participant_id"],
                        "submitted_at": trial["preference_form"]["submitted_at"],
                        "placements_json": json.dumps(trial["preference_form"]["placements"], ensure_ascii=False),
                    }
                )
            if trial["result"]:
                results.append(
                    {
                        "trial_id": trial["trial_id"],
                        "participant_id": trial["participant_id"],
                        "discussed_item_accuracy": trial["result"]["discussed_item_accuracy"],
                        "undiscussed_item_accuracy": trial["result"]["undiscussed_item_accuracy"],
                        "overall_accuracy": trial["result"]["overall_accuracy"],
                        "discussed_items_json": json.dumps(trial["result"]["discussed_items"], ensure_ascii=False),
                        "undiscussed_items_json": json.dumps(trial["result"]["undiscussed_items"], ensure_ascii=False),
                        "predicted_placements_json": json.dumps(trial["result"]["predicted_placements"], ensure_ascii=False),
                    }
                )

        exports = {
            "participants.csv": participant_rows,
            "trials.csv": trial_rows,
            "turns.csv": turns,
            "preference_forms.csv": preference_forms,
            "trial_results.csv": results,
        }
        for filename, rows in exports.items():
            path = EXPORTS_DIR / filename
            fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if fieldnames:
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
        return list(exports.keys())
