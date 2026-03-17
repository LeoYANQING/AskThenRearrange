from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


PlacementMap = Dict[str, str]
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "scenarios_aug_tiny.json"


@dataclass
class Episode:
    episode_id: str
    room: str
    receptacles: List[str]
    seen_objects: List[str]
    unseen_objects: List[str]
    seen_placements: PlacementMap
    unseen_placements: PlacementMap
    annotator_notes: List[str]
    tags: List[str]


def _to_str_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list, got {type(value)}")
    return [str(item) for item in value]


def _normalize_placements(value: Any, field_name: str) -> PlacementMap:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(obj): str(receptacle) for obj, receptacle in value.items()}
    if isinstance(value, list):
        normalized: PlacementMap = {}
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"{field_name} contains invalid placement item: {item!r}")
            obj, receptacle = item
            normalized[str(obj)] = str(receptacle)
        return normalized
    raise ValueError(f"{field_name} must be a dict or list of pairs, got {type(value)}")


def _check_no_duplicates(values: List[str], field_name: str) -> None:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{field_name} contains duplicates: {duplicates}")


def _require_exact_keys(
    placements: PlacementMap,
    expected_objects: List[str],
    field_name: str,
) -> None:
    placement_keys = set(placements.keys())
    expected_keys = set(expected_objects)
    if placement_keys != expected_keys:
        missing = sorted(expected_keys - placement_keys)
        extra = sorted(placement_keys - expected_keys)
        raise ValueError(
            f"{field_name} keys do not match object list. Missing={missing}, Extra={extra}"
        )


def _validate_receptacles(placements: PlacementMap, receptacles: List[str], field_name: str) -> None:
    allowed = set(receptacles)
    invalid = {obj: receptacle for obj, receptacle in placements.items() if receptacle not in allowed}
    if invalid:
        raise ValueError(f"{field_name} contains invalid receptacles: {invalid}")


def _validate_episode(episode: Episode) -> None:
    _check_no_duplicates(episode.receptacles, "receptacles")
    _check_no_duplicates(episode.seen_objects, "seen_objects")
    _check_no_duplicates(episode.unseen_objects, "unseen_objects")

    overlap = sorted(set(episode.seen_objects) & set(episode.unseen_objects))
    if overlap:
        raise ValueError(f"seen_objects and unseen_objects overlap: {overlap}")

    _require_exact_keys(episode.seen_placements, episode.seen_objects, "seen_placements")
    _require_exact_keys(episode.unseen_placements, episode.unseen_objects, "unseen_placements")
    _validate_receptacles(episode.seen_placements, episode.receptacles, "seen_placements")
    _validate_receptacles(episode.unseen_placements, episode.receptacles, "unseen_placements")


def _episode_from_record(record: Dict[str, Any], index: int) -> Episode:
    seen_key = "seen_placements"
    unseen_key = "unseen_placements"
    
    episode = Episode(
        episode_id=str(record.get("episode_id", f"episode_{index}")),
        room=str(record["room"]),
        receptacles=_to_str_list(record["receptacles"], "receptacles"),
        seen_objects=_to_str_list(record["seen_objects"], "seen_objects"),
        unseen_objects=_to_str_list(record.get("unseen_objects", []), "unseen_objects"),
        seen_placements=_normalize_placements(record.get(seen_key), seen_key),
        unseen_placements=_normalize_placements(record.get(unseen_key), unseen_key),
        annotator_notes=_to_str_list(record.get("annotator_notes", []), "annotator_notes"),
        tags=_to_str_list(record.get("tags", []), "tags"),
    )
    _validate_episode(episode)
    return episode


def load_episodes(json_path: str | Path) -> List[Episode]:
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of episodes or a single episode record")

    return [_episode_from_record(record, index) for index, record in enumerate(data)]


def get_episode(json_path: str | Path, index: int = 0) -> Episode:
    episodes = load_episodes(json_path)
    if index < 0 or index >= len(episodes):
        raise IndexError(f"Episode index {index} out of range: [0, {len(episodes) - 1}]")
    return episodes[index]


def _smoke_snapshot(episode: Episode) -> Dict[str, Any]:
    return {
        "episode": {
            "episode_id": episode.episode_id,
            "room": episode.room,
            "num_receptacles": len(episode.receptacles),
            "num_seen_objects": len(episode.seen_objects),
            "num_unseen_objects": len(episode.unseen_objects),
        },
        "sanity_checks": {
            "seen_unseen_overlap": sorted(set(episode.seen_objects) & set(episode.unseen_objects)),
            "seen_keys_match": sorted(episode.seen_placements.keys()) == sorted(episode.seen_objects),
            "unseen_keys_match": sorted(episode.unseen_placements.keys()) == sorted(episode.unseen_objects),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for dataset loading.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to dataset JSON.",
    )
    parser.add_argument("--index", type=int, default=0, help="Episode index to inspect.")
    args = parser.parse_args()

    episode = get_episode(args.data, args.index)
    print(json.dumps(_smoke_snapshot(episode), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
