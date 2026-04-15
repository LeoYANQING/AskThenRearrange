from __future__ import annotations

import os
from pathlib import Path

from llm_factory import DEFAULT_BASE_URL, DEFAULT_MODEL


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent

APP_NAME = "Study 2 Experiment App"

SCENE_MANIFEST_PATH = APP_ROOT / "scene_manifest.json"
DATA_ROOT = REPO_ROOT / "study2_data"
PARTICIPANTS_DIR = DATA_ROOT / "participants"
TRIALS_DIR = DATA_ROOT / "trials"
EXPORTS_DIR = DATA_ROOT / "exports"

DEFAULT_BUDGET_TOTAL = int(os.environ.get("STUDY2_BUDGET", "6"))
DEFAULT_QUESTIONNAIRE_URL = os.environ.get("STUDY2_QUESTIONNAIRE_URL", "")
DEFAULT_SELECTION_METHOD = os.environ.get("STUDY2_SELECTION_METHOD", "rule")
DEFAULT_PROPOSER_MODEL = os.environ.get("STUDY2_PROPOSER_MODEL", DEFAULT_MODEL)
DEFAULT_UPDATER_MODEL = os.environ.get("STUDY2_UPDATER_MODEL", DEFAULT_MODEL)
DEFAULT_EVALUATION_MODEL = os.environ.get("STUDY2_EVALUATION_MODEL", DEFAULT_MODEL)
DEFAULT_BASE_URL = os.environ.get("STUDY2_BASE_URL", DEFAULT_BASE_URL)
AUTO_REFRESH_SECONDS = int(os.environ.get("STUDY2_AUTO_REFRESH_SECONDS", "2"))

STRATEGY_ORDERS = [
    ["direct_querying", "user_preference_first", "parallel_exploration"],
    ["user_preference_first", "parallel_exploration", "direct_querying"],
    ["parallel_exploration", "direct_querying", "user_preference_first"],
]

SCENE_ORDER_INDICES = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
]
