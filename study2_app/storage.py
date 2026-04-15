from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from study2_app.config import EXPORTS_DIR, PARTICIPANTS_DIR, TRIALS_DIR
from study2_app.types import ParticipantRecord, TrialRecord


class Study2Storage:
    def __init__(self) -> None:
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        PARTICIPANTS_DIR.mkdir(parents=True, exist_ok=True)
        TRIALS_DIR.mkdir(parents=True, exist_ok=True)
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def participant_path(self, participant_id: str) -> Path:
        return PARTICIPANTS_DIR / f"{participant_id}.json"

    def trial_path(self, trial_id: str) -> Path:
        return TRIALS_DIR / f"{trial_id}.json"

    def save_participant(self, participant: ParticipantRecord) -> None:
        self._write_json(self.participant_path(participant["participant_id"]), participant)

    def save_trial(self, trial: TrialRecord) -> None:
        self._write_json(self.trial_path(trial["trial_id"]), trial)

    def load_participant(self, participant_id: str) -> ParticipantRecord:
        return self._read_json(self.participant_path(participant_id))  # type: ignore[return-value]

    def load_trial(self, trial_id: str) -> TrialRecord:
        return self._read_json(self.trial_path(trial_id))  # type: ignore[return-value]

    def list_participants(self) -> list[ParticipantRecord]:
        participants: list[ParticipantRecord] = []
        for path in sorted(PARTICIPANTS_DIR.glob("*.json")):
            participants.append(self._read_json(path))  # type: ignore[arg-type]
        return participants

    def list_trials(self) -> list[TrialRecord]:
        trials: list[TrialRecord] = []
        for path in sorted(TRIALS_DIR.glob("*.json")):
            trials.append(self._read_json(path))  # type: ignore[arg-type]
        return trials

    def list_trials_for_participant(self, participant_id: str) -> list[TrialRecord]:
        trials = [
            trial for trial in self.list_trials()
            if trial["participant_id"] == participant_id
        ]
        return sorted(trials, key=lambda item: item["trial_index"])

