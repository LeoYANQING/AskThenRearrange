from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data import Episode, load_episodes

from study2_app.config import REPO_ROOT, SCENE_MANIFEST_PATH


class SceneLibrary:
    def __init__(self, manifest_path: Path = SCENE_MANIFEST_PATH) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_path = REPO_ROOT / manifest["dataset_path"]
        episodes = load_episodes(dataset_path)
        self.dataset_path = dataset_path
        self._scenes: dict[str, dict[str, Any]] = {}

        for raw_scene in manifest["scenes"]:
            episode_index = int(raw_scene["episode_index"])
            episode = episodes[episode_index]
            self._scenes[raw_scene["scene_id"]] = {
                "scene_id": raw_scene["scene_id"],
                "label": raw_scene["label"],
                "episode_index": episode_index,
                "episode": episode,
            }

    def list_scenes(self) -> list[dict[str, Any]]:
        return [
            {
                "scene_id": scene["scene_id"],
                "label": scene["label"],
                "episode_index": scene["episode_index"],
            }
            for scene in self._scenes.values()
        ]

    def get_scene(self, scene_id: str) -> dict[str, Any]:
        if scene_id not in self._scenes:
            raise KeyError(f"Unknown scene_id: {scene_id}")
        return self._scenes[scene_id]

    def get_episode(self, scene_id: str) -> Episode:
        return self.get_scene(scene_id)["episode"]

