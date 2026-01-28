# %% [cell 0]
import json
from typing import List, Dict, Union
import tqdm

def load_and_split_samples(json_path: str) -> List[Dict]:
    """
    Load scenarios.json and split into individual samples (dicts).

    Supports:
    1) List[Dict] format
    2) Dict with key 'scenarios' -> List[Dict]

    Returns:
        samples: List of scenario dicts
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: already a list of samples
    if isinstance(data, list):
        samples = data

    # Case 2: wrapped in a dict
    elif isinstance(data, dict):
        if "scenarios" in data and isinstance(data["scenarios"], list):
            samples = data["scenarios"]
        else:
            # Single scenario dict
            samples = [data]
    else:
        raise ValueError("Unsupported JSON format for scenarios")

    # Basic sanity check
    required_keys = {
        "annotator_notes",
        "receptacles",
        "room",
        "seen_objects",
        "seen_placements",
        "tags",
        "unseen_objects",
        "unseen_placements",
    }

    for i, s in enumerate(samples):
        missing = required_keys - set(s.keys())
        if missing:
            raise KeyError(f"Sample {i} missing keys: {missing}")

    return samples


def append_json_list(json_path: str, item: Dict) -> None:
    """
    Append one item to a top-level JSON list (create file if missing).
    Writes immediately so progress is not lost.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError("Output JSON must be a top-level list.")
        else:
            data = []
    except FileNotFoundError:
        data = []

    data.append(item)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(data)} items to: {json_path}")

# %% [cell 1]
json_path = "scenarios.json"

samples = load_and_split_samples(json_path)

print("Loaded samples:", len(samples))
print("Keys of first sample:", samples[0].keys())
samples[0]

# %% [cell 2]
import sys
sys.path.append("../")  # Adjust the path as needed
from ollama_call import VLMAPI

import json
from typing import Tuple, List, Dict, Any

REQUIRED_KEYS = [
    "annotator_notes", "receptacles", "room",
    "seen_objects", "seen_placements",
    "tags",
    "unseen_objects", "unseen_placements"
]

# Preference-type tag set (CLOSED set; no other tags allowed)
ALLOWED_TAGS = [
    "category",
    "attribute",
    "function",
    "subcategory",
    "multiple categories"
]

FEWSHOT_EXAMPLE: Dict[str, Any] = {
  "annotator_notes": "Keep fresh and ready-to-eat fruits visible on the coffee table for easy access. Bruised, overripe, or cut fruits should be placed in the lidded bin to avoid mess and pests. Small hard toys and toys with loose parts should be stored in the storage box. Soft plush toys that are frequently used can stay on the sofa. Battery-powered toys should be kept in the side drawer to prevent accidental activation. Items that do not clearly fit these categories should be placed based on their dominant material and usage.",
  "receptacles": [
    "coffee table",
    "lidded bin",
    "storage box",
    "sofa",
    "side drawer"
  ],
  "room": "living room",
  "seen_objects": [
    "fresh banana",
    "overripe banana",
    "fresh apple",
    "cut apple slices",
    "small Lego brick",
    "large Lego brick",
    "toy car with batteries",
    "toy car without batteries",
    "plush teddy bear",
    "stuffed rabbit",
    "plastic action figure",
    "metal toy robot"
  ],
  "seen_placements": [
    ["fresh banana", "coffee table"],
    ["fresh apple", "coffee table"],
    ["overripe banana", "lidded bin"],
    ["cut apple slices", "lidded bin"],
    ["small Lego brick", "storage box"],
    ["large Lego brick", "storage box"],
    ["plastic action figure", "storage box"],
    ["metal toy robot", "storage box"],
    ["plush teddy bear", "sofa"],
    ["stuffed rabbit", "sofa"],
    ["toy car with batteries", "side drawer"],
    ["toy car without batteries", "storage box"]
  ],
  # IMPORTANT: tags MUST be ONLY from the 5 preference types
  # This example uses attribute-based rules and sorts multiple categories (fruits + toys)
  "tags": [
    "attribute",
    "multiple categories"
  ],
  "unseen_objects": [
    "fresh orange",
    "bruised orange",
    "fresh peach",
    "overripe peach",
    "tiny puzzle piece",
    "wooden toy block",
    "battery-powered toy train",
    "wind-up toy train",
    "plush dog",
    "plush dinosaur",
    "electronic game controller",
    "metal spinning top"
  ],
  "unseen_placements": [
    ["fresh orange", "coffee table"],
    ["fresh peach", "coffee table"],
    ["bruised orange", "lidded bin"],
    ["overripe peach", "lidded bin"],
    ["tiny puzzle piece", "storage box"],
    ["wooden toy block", "storage box"],
    ["metal spinning top", "storage box"],
    ["plush dog", "sofa"],
    ["plush dinosaur", "sofa"],
    ["battery-powered toy train", "side drawer"],
    ["electronic game controller", "side drawer"],
    ["wind-up toy train", "storage box"]
  ]
}

def build_generation_prompt(sample: dict) -> Tuple[str, str]:
    """
    Build a system prompt + user prompt for generating ONE augmented scenario.
    Includes a few-shot example to anchor style and constraint satisfaction.
    Tags are restricted to a closed set of 5 preference types:
    ["category", "attribute", "function", "subcategory", "multiple categories"]
    """
    systext = (
        "You are a data generation engine for household rearrangement scenarios.\n"
        "Return STRICT JSON only (no markdown, no explanation).\n"
        "Do not output any extra keys or commentary."
    )

    usertext = f"""
        You will be given ONE input scenario JSON.
        Generate ONE NEW augmented scenario JSON that satisfies ALL constraints.

        ====================
        HARD OUTPUT FORMAT
        ====================
        - Output MUST be a single JSON object with EXACTLY these keys (no extras):
        {REQUIRED_KEYS}

        ====================
        CONSTRAINTS
        ====================
        - room MUST be "living room"
        - receptacles: 4–6 realistic living-room receptacles; every receptacle must be used at least once
        - seen_objects: 10–20 items; unseen_objects: 10–20 items; no overlap between seen/unseen
        - Each object appears EXACTLY ONCE in its placements

        - annotator_notes:
          - Write 4–7 concise preference rules
          - Rules must be ABSTRACT and GENERALIZABLE (long-term user preferences)
          - Rules MUST depend on OBJECT ATTRIBUTES (e.g., clean/dirty, wet/dry, fragile/not, leaking/not, battery-powered/not, fresh/spoiled)
          - Rules must deterministically explain ALL placements
          - Do NOT enumerate object names
          - Do NOT restate placements in sentence form
          - Do NOT include fallback rules
          - Do NOT mention receptacle-on-receptacle relations

        - Object design requirements:
          - seen_objects MUST include at least 3 pairs of same-category objects differing only by attributes
            (and the attribute difference MUST change placement)
          - Use at least 2 different attribute dimensions that affect placement

        - tags (IMPORTANT):
          - tags MUST be a list of preference types describing HOW objects are sorted
          - tags MUST be chosen ONLY from this closed set:
            {ALLOWED_TAGS}
          - tags MUST be semantically consistent with annotator_notes
            Examples:
              * If rules are mainly about object attributes -> include "attribute"
              * If rules are about seasonal / frequency-of-use / purpose -> include "function"
              * If rules separate a subordinate group from a superordinate group -> include "subcategory"
              * If rules mix multiple categories into one receptacle -> include "multiple categories"
              * If rules are purely category-to-location -> include "category"
          - Multiple tags are allowed if multiple sorting criteria are present
          - Do NOT invent new tags
          - Do NOT include meta tags like "augmented" or "long-term"

        - Avoid trivial solutions: no single receptacle gets >70% objects
        - unseen_objects MUST follow the SAME attribute dimensions and rules (no new rules)

        ====================
        FEW-SHOT EXAMPLE (STYLE + FORMAT)
        ====================
        Example output JSON:
        {json.dumps(FEWSHOT_EXAMPLE, ensure_ascii=False)}

        ====================
        INPUT (REFERENCE ONLY)
        ====================
        Input scenario JSON (do NOT copy; generate a NEW scenario that still satisfies all constraints):
        {json.dumps(sample, ensure_ascii=False)}

        ====================
        OUTPUT
        ====================
        Return ONLY the augmented JSON object.
        """.strip()

    return systext, usertext

# %% [cell 3]
import re

def parse_json_strict(text: str) -> dict:
    # 尽量从返回中截取第一个 JSON 对象（防止模型偶尔带前后废话）
    text = text.strip()
    # 若直接就是 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 尝试截取 {...} 的最大段
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Model output does not contain a JSON object.")
    return json.loads(m.group(0))


def augment_one_sample_with_vlm(
    sample: dict,
    model_name: str = "qwen3:32b",
    max_rounds: int = 5,
    max_tokens: int = 5000
) -> dict:
    llm = VLMAPI(model_name)

    systext, usertext = build_generation_prompt(sample)
    critique = ""

    for r in range(max_rounds):
        prompt_user = usertext if r == 0 else (usertext + "\n\nPrevious attempt problems:\n" + critique + "\n\nRegenerate a corrected JSON only.")
        raw = llm.vlm_request(systext, prompt_user, max_tokens=max_tokens)

        try:
            cand = parse_json_strict(raw)
        except Exception as e:
            critique = f"Failed to parse JSON: {e}. Output was: {raw[:300]}..."
            continue
        return cand

    raise RuntimeError(
        f"Failed to generate a valid augmented sample after {max_rounds} rounds. "
        f"Last issues:\n{critique}"
    )

# %% [cell 4]
# 假设你已经有 samples 列表
output_json = "scenarios_aug.json"

from tqdm import tqdm

for i, sample_in in enumerate(tqdm(samples)):
    aug_sample = augment_one_sample_with_vlm(sample_in, model_name="qwen3:32b", max_rounds=5)

    print(f"VALID augmented sample generated! index={i}")
    print(json.dumps(aug_sample, ensure_ascii=False, indent=2))

    append_json_list(output_json, aug_sample)


