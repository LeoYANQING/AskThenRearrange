"""Study-2-only translation utilities.

Used ONLY at the dialogue API boundary to display Chinese to the participant
while keeping the entire backend pipeline (proposers, state_update, evaluation)
running in English exactly as in Study 1.

Flow:
  1. At trial start: build_name_mapping() returns a stable {en: zh} dict for all
     object and receptacle names in the episode. Stored on the trial.
  2. When a question leaves the backend: translate_en_to_zh(question, name_map)
     -> Chinese, with glossary-pinned term translations for semantic alignment.
  3. When an answer arrives: translate_zh_to_en(answer, name_map) -> English,
     using the reversed mapping so state_update sees the exact English names
     from the original episode.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_factory import create_chat_model  # noqa: E402

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = create_chat_model(temperature=0.0)
    return _llm


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_name_mapping(english_names: List[str]) -> Dict[str, str]:
    """Batch-translate object/receptacle names once per episode. Returns {en: zh}.

    Falls back to the original English string for any name the LLM fails on,
    so the pipeline never breaks on translation failure.
    """
    if not english_names:
        return {}
    unique = list(dict.fromkeys(english_names))

    system_msg = (
        "You are a precise translator for household objects and furniture.\n"
        "Translate English names to Simplified Chinese.\n"
        "Rules:\n"
        "- Use natural, common Chinese household vocabulary\n"
        "- Keep translations concise (2-8 characters preferred)\n"
        "- Same English name must always produce the same Chinese translation\n"
        "- Return ONLY a JSON object, no markdown fences, no extra text"
    )
    user_msg = (
        "Translate each name to Chinese. Return only a JSON object.\n\n"
        f"Names: {json.dumps(unique, ensure_ascii=False)}\n\n"
        'Format: {"bookshelf": "书架", "remote control": "遥控器"}'
    )

    try:
        llm = _get_llm()
        response = llm.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])
        text = response.content if hasattr(response, "content") else str(response)
        text = _strip_think(text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            mapping = json.loads(match.group())
            return {name: str(mapping.get(name, name)) for name in unique}
    except Exception:
        pass

    return {name: name for name in unique}


def _glossary_lines(pairs: Dict[str, str]) -> str:
    return "\n".join(f'- "{src}" -> "{dst}"' for src, dst in pairs.items() if src != dst)


def translate_en_to_zh(text: str, name_map: Dict[str, str]) -> str:
    """Translate English question text to natural Simplified Chinese.

    The `name_map` ({en: zh}) is provided as a required glossary so that object
    and receptacle names come back with the exact Chinese spellings chosen at
    trial start — guaranteeing round-trip alignment.
    """
    if not text.strip():
        return text
    glossary = _glossary_lines(name_map)
    system_msg = (
        "把英文问句翻成自然的简体中文口语问句。只输出译文，不要引号、解释、Markdown。\n"
        "\n"
        "规则：\n"
        "- 简短、直接。保留问句（'?' -> '？'）。不要加原文里没有的信息。\n"
        "- 固定句式映射：\n"
        "  'How do you usually like to organize your X?' -> '你平时喜欢怎么整理X？'\n"
        "  'How do you usually organize X?' -> '你是如何整理X的？'\n"
        "  'How do you usually organize X, like A or B?' -> '你是如何整理X的？比如A、B'\n"
        "  'How do you usually organize X, like A?' -> '你是如何整理X的？比如A'\n"
        "  'What kinds of items do you typically keep in the X?' -> 'X里你一般放什么类型的物品？'\n"
        "  'Where do you usually put X?' -> 'X你一般放在哪？'\n"
        "  'Where do you usually put X, like A or B?' -> 'X你一般放在哪？比如A、B'\n"
        "  'Where do you usually put X, like A?' -> 'X你一般放在哪？比如A'\n"
        "  'What do you usually put in the X?' -> 'X里你一般放什么？'\n"
        "  'Where should X go?' -> 'X应该放在哪？'\n"
        "  room词（kitchen/bedroom/living room）必须翻成（厨房/卧室/客厅）。\n"
        "- 以下术语必须使用对应中文：\n"
        f"{glossary}"
    )
    try:
        llm = _get_llm()
        response = llm.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text},
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        return _strip_think(raw).strip().strip('"').strip()
    except Exception:
        return text


def translate_zh_to_en(text: str, name_map: Dict[str, str]) -> str:
    """Translate Chinese answer text to natural English.

    Uses the reversed `name_map` so Chinese names are converted back to the
    exact English strings in the episode data, which state_update needs for
    matching against receptacles / seen_objects lists.
    """
    if not text.strip():
        return text
    reverse = {zh: en for en, zh in name_map.items() if en != zh}
    glossary = _glossary_lines(reverse)
    system_msg = (
        "Translate the given Simplified Chinese text to natural English.\n"
        "Output only the translation. No quotes. No explanations. No markdown.\n"
        "When you encounter any of the following Chinese terms, you MUST use the exact English translation:\n"
        f"{glossary}"
    )
    try:
        llm = _get_llm()
        response = llm.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text},
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        return _strip_think(raw).strip().strip('"').strip()
    except Exception:
        return text
