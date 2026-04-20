from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_schema import AgentState  # noqa: E402
from data import Episode, load_episodes  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "scenarios_three_rooms_102.json"

# Strategy display name → policy mode
STRATEGY_TO_MODE: Dict[str, str] = {
    "DQ": "direct_querying",
    "UPF": "user_preference_first",
    "PAR": "parallel_exploration",
}

# 6-row Latin square: each row is 3 trials of (strategy, room_type)
# Each strategy and each room appears exactly once per row.
LATIN_SQUARE: List[List[tuple]] = [
    [("DQ", "living room"),   ("UPF", "bedroom"),    ("PAR", "kitchen")],
    [("UPF", "kitchen"),      ("PAR", "living room"), ("DQ", "bedroom")],
    [("PAR", "bedroom"),      ("DQ", "kitchen"),      ("UPF", "living room")],
    [("DQ", "kitchen"),       ("UPF", "living room"), ("PAR", "bedroom")],
    [("UPF", "bedroom"),      ("PAR", "kitchen"),     ("DQ", "living room")],
    [("PAR", "living room"),  ("DQ", "bedroom"),      ("UPF", "kitchen")],
]

UNLIMITED_BUDGET = 9999


def _load_episodes_by_room() -> Dict[str, List[Episode]]:
    all_episodes = load_episodes(DATA_PATH)
    by_room: Dict[str, List[Episode]] = {}
    for ep in all_episodes:
        by_room.setdefault(ep.room, []).append(ep)
    return by_room


EPISODES_BY_ROOM: Dict[str, List[Episode]] = _load_episodes_by_room()


def get_episode_for_room(room_type: str, episode_index: int) -> Episode:
    episodes = EPISODES_BY_ROOM.get(room_type, [])
    if not 0 <= episode_index < len(episodes):
        raise ValueError(
            f"episode_index {episode_index} out of range for '{room_type}' "
            f"({len(episodes)} episodes available)"
        )
    return _shrink_episode(episodes[episode_index])


# Test-time scene shrink. Until experiment-scenario design is finalized, halve
# the object set so each trial finishes faster. Override with
# STUDY2_OBJECT_FRACTION=1.0 to run full scenes. Deterministic (takes the first
# N // k items) so every session sees the same objects.
_OBJECT_FRACTION = float(os.environ.get("STUDY2_OBJECT_FRACTION", "0.5"))


def _shrink_episode(ep: Episode) -> Episode:
    if _OBJECT_FRACTION >= 1.0:
        return ep
    n_seen = max(1, int(len(ep.seen_objects) * _OBJECT_FRACTION))
    n_unseen = max(1, int(len(ep.unseen_objects) * _OBJECT_FRACTION))
    seen = list(ep.seen_objects[:n_seen])
    unseen = list(ep.unseen_objects[:n_unseen])
    return replace(
        ep,
        seen_objects=seen,
        unseen_objects=unseen,
        seen_placements={o: r for o, r in ep.seen_placements.items() if o in seen},
        unseen_placements={o: r for o, r in ep.unseen_placements.items() if o in unseen},
    )


@dataclass
class PendingQA:
    pattern: str          # "action_oriented" | "preference_eliciting" | "preference_induction"
    question: str         # English — passed to state_update as context
    target: str           # object_name (AO) or hypothesis text (PE/PI)
    covered_objects: List[str] = field(default_factory=list)
    action_mode: Optional[str] = None
    turn_index: int = 0
    question_zh: str = ""  # Chinese — shown to the participant; stored in trial.dialogue


class SessionState:
    def __init__(
        self,
        session_id: str,
        participant_id: str,
        latin_square_row: int,
        notes: str,
        budget_total: int = 6,
    ):
        self.session_id = session_id
        self.participant_id = participant_id
        self.latin_square_row = latin_square_row  # 1-indexed
        self.notes = notes
        self.budget_total = budget_total

        row = LATIN_SQUARE[latin_square_row - 1]
        self.trial_order: List[Dict[str, str]] = [
            {"strategy": s, "room_type": r} for s, r in row
        ]

        self.current_trial_index = 0
        self.trials: List[Dict[str, Any]] = []
        self.phase = "created"  # global session phase

        self.agent_state: Optional[AgentState] = None
        self.current_episode: Optional[Episode] = None
        self.pending_qa: Optional[PendingQA] = None
        self.strategy_ranking: Optional[List[str]] = None
        self.final_comment: str = ""
        self.log_path = LOGS_DIR / f"{participant_id}_{session_id[:8]}.jsonl"

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "event": event_type,
            "trial_index": self.current_trial_index,
            "data": data,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def current_trial_config(self) -> Dict[str, str]:
        return self.trial_order[self.current_trial_index]

    def current_trial_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.current_trial_index < len(self.trials):
            return self.trials[self.current_trial_index]
        return None

    def start_trial(
        self,
        episode_index: int,
        name_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        from state_init import build_initial_state  # lazy import after sys.path setup

        config = self.current_trial_config()
        room_type = config["room_type"]
        strategy = config["strategy"]

        episode = get_episode_for_room(room_type, episode_index)
        self.current_episode = episode

        self.agent_state = build_initial_state(
            episode,
            strategy="direct",  # not used — we drive via policy mode; field is ignored
            budget_total=self.budget_total,
        )

        trial: Dict[str, Any] = {
            "trial_index": self.current_trial_index,
            "strategy": strategy,
            "room_type": room_type,
            "episode_index": episode_index,
            "receptacles": list(episode.receptacles),
            "seen_objects": list(episode.seen_objects),
            "unseen_objects": list(episode.unseen_objects),
            "name_mapping": dict(name_mapping or {}),  # en -> zh for Study 2 display layer only
            "dialogue": [],
            "turns_used": 0,
            "stop_reason": None,
            "preference_assignments": None,
            "predicted_placements": None,
            "psr": None,
            "phase": "scene_intro",
        }
        if self.current_trial_index < len(self.trials):
            self.trials[self.current_trial_index] = trial
        else:
            self.trials.append(trial)

        self.phase = "scene_intro"
        self.pending_qa = None

        self.log_event("trial_started", {
            "strategy": strategy,
            "room_type": room_type,
            "episode_index": episode_index,
            "episode_id": episode.episode_id,
            "receptacles": list(episode.receptacles),
            "seen_objects": list(episode.seen_objects),
            "unseen_objects": list(episode.unseen_objects),
            "name_mapping": dict(name_mapping or {}),
        })
        return trial

    def agent_state_snapshot(self) -> Optional[Dict[str, Any]]:
        """Serialize the live AgentState for the experimenter monitor panel."""
        if self.agent_state is None:
            return None
        s = self.agent_state
        return {
            "room": s.get("room"),
            "receptacles": list(s.get("receptacles") or []),
            "seen_objects": list(s.get("seen_objects") or []),
            "unseen_objects": list(s.get("unseen_objects") or []),
            "qa_turns": len(s.get("qa_history") or []),
            "confirmed_actions": list(s.get("confirmed_actions") or []),
            "negative_actions": list(s.get("negative_actions") or []),
            "confirmed_preferences": list(s.get("confirmed_preferences") or []),
            "negative_preferences": list(s.get("negative_preferences") or []),
            "unresolved_objects": list(s.get("unresolved_objects") or []),
        }

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "latin_square_row": self.latin_square_row,
            "trial_order": self.trial_order,
            "current_trial_index": self.current_trial_index,
            "trials": self.trials,
            "phase": self.phase,
            "notes": self.notes,
            "agent_state": self.agent_state_snapshot(),
            "strategy_ranking": list(self.strategy_ranking) if self.strategy_ranking else None,
            "final_comment": self.final_comment,
            "budget_total": self.budget_total,
        }


_sessions: Dict[str, SessionState] = {}


def create_session(
    participant_id: str,
    latin_square_row: int,
    notes: str,
    budget_total: int = 6,
) -> SessionState:
    if not 1 <= latin_square_row <= 6:
        raise ValueError(f"latin_square_row must be 1–6, got {latin_square_row}")
    if not 1 <= budget_total <= 100:
        raise ValueError(f"budget_total must be 1–100, got {budget_total}")
    session_id = uuid.uuid4().hex[:8]
    session = SessionState(session_id, participant_id, latin_square_row, notes, budget_total)
    _sessions[session_id] = session
    session.log_event("session_created", {
        "participant_id": participant_id,
        "latin_square_row": latin_square_row,
        "budget_total": budget_total,
        "trial_order": session.trial_order,
    })
    return session


def get_session(session_id: str) -> Optional[SessionState]:
    return _sessions.get(session_id)
