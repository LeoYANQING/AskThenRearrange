"""
MVP LangGraph for preference-first / parallel / direct strategies.

Includes:
- State (AgentState)
- Tools (oracle_answer_tool using a separate Ollama model)
- Nodes
- Graph wiring
- Test runner that loads scenarios_aug_tiny.json and prints qa_history

Requirements:
  pip install langgraph langchain-ollama langchain-core

Ollama:
  - Make sure `ollama serve` is running
  - Make sure models exist: `ollama list`
"""

from __future__ import annotations

import argparse
import json
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict
from tqdm import tqdm

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from eval import evaluate_episode, plot_strategy_tradeoff


# =========================
# Config: LLMs (Ollama)
# =========================
# Question generator model (agent-side)
QUESTION_MODEL = "qwen3.5"
# Oracle answerer model (user simulator)
ANSWER_MODEL = "qwen3.5"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

question_llm = ChatOllama(
    model=QUESTION_MODEL,  # 哪怕是VL模型，也可以只传纯文字
    base_url=OLLAMA_BASE_URL,
    temperature=0
)
answer_llm = ChatOllama(
    model=ANSWER_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0
)


# =========================
# Types: State
# =========================
QType = Literal["preference", "action", "summary"]
Strategy = Literal["preference-first", "parallel", "direct"]


class QAItem(TypedDict):
    question: str
    answer: str
    q_type: QType


class AgentState(TypedDict):
    # --- control ---
    strategy: Strategy
    budget_total: int
    budget_used: int

    # --- task input (immutable per episode) ---
    room: str
    receptacles: List[str]
    seen_objects: List[str]
    unseen_objects: List[str]
    seen_placements: Dict[str, str]
    unseen_placements: Dict[str, str]
    annotator_notes: List[str]  # oracle preference source

    # --- interaction log ---
    qa_history: List[QAItem]
    asked_questions: List[str]

    # --- intermediate artifacts ---
    preference: str

    # --- outputs ---
    predicted_placements_seen: Dict[str, str]
    predicted_placements_unseen: Dict[str, str]

    # --- transient fields (node-to-node) ---
    current_q_type: Optional[QType]
    current_question: Optional[str]
    current_answer: Optional[str]


# =========================
# Scenario helper (oracle)
# =========================
@dataclass
class Scenario:
    room: str
    receptacles: List[str]
    seen_objects: List[str]
    unseen_objects: List[str]
    seen_placements: Dict[str, str]
    unseen_placements: Dict[str, str]
    annotator_notes: List[str]


def build_answer_prompt(scenario: Scenario, question: str) -> str:
    annotator_notes = "\n".join(f"- {x}" for x in scenario.annotator_notes)
    seen_placements = "\n".join(f"- {k} -> {v}" for k, v in scenario.seen_placements.items())
    receptacles = ", ".join(scenario.receptacles)

    return (
        "You are the user in a household organization task.\n"
        "Below are your personal organization preferences (follow them strictly):\n"
        f"{annotator_notes}\n\n"
        "Below are the ground-truth placements for objects that are currently present (seen objects):\n"
        f"{seen_placements}\n\n"
        f"Valid receptacles: [{receptacles}]\n\n"
        f"Question: {question}\n"
        "Answer naturally and concisely. If you mention locations, use ONLY the receptacle names above. "
        "Start your response immediately with the result. Output only the answer."
    )


def extract_answer(text: str) -> str:
    # MVP: just strip. You can harden later.
    return text.strip().strip('"').strip("'").strip()


# =========================
# Tool: Oracle Answerer
# =========================
@tool
def oracle_answer_tool(
    question: str,
    room: str,
    receptacles: list[str],
    seen_objects: list[str],
    unseen_objects: list[str],
    seen_placements: dict,
    unseen_placements: dict,
    annotator_notes: list[str],
) -> str:
    """
    Simulated user/oracle that answers household organization questions
    based on annotator preferences and ground-truth placements.
    """
    scenario = Scenario(
        room=room,
        receptacles=receptacles,
        seen_objects=seen_objects,
        unseen_objects=unseen_objects,
        seen_placements=seen_placements,
        unseen_placements=unseen_placements,
        annotator_notes=annotator_notes,
    )

    answer_prompt = build_answer_prompt(scenario, question)

    msg = answer_llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are the user answering a household organization question. "
                    "Output only the answer. Do NOT include reasoning or analysis."
                ),
            },
            {"role": "user", "content": answer_prompt},
        ]
    )

    text = msg.content if hasattr(msg, "content") else str(msg)
    return extract_answer(text)


# =========================
# Nodes
# =========================
def init_node(state: AgentState) -> dict:
    return {
        "budget_used": 0,
        "qa_history": [],
        "asked_questions": [],
        "preference": "",
        "predicted_placements_seen": {},
        "predicted_placements_unseen": {},
        "current_q_type": None,
        "current_question": None,
        "current_answer": None,
    }


def _gate_decision_schema(allowed: List[QType]) -> dict:
    return {
        "type": "object",
        "properties": {
            "q_type": {
                "type": "string",
                "enum": allowed,
                "description": "Next question type.",
            }
        },
        "required": ["q_type"],
        "additionalProperties": False,
    }


def _planner_output_schema(
    *,
    seen_objects: List[str],
    unseen_objects: List[str],
    receptacles: List[str],
) -> dict:
    seen_props = {obj: {"type": "string", "enum": receptacles} for obj in seen_objects}
    unseen_props = {obj: {"type": "string", "enum": receptacles} for obj in unseen_objects}
    return {
        "type": "object",
        "properties": {
            "predicted_placements_seen": {
                "type": "object",
                "properties": seen_props,
                "required": seen_objects,
                "additionalProperties": False,
            },
            "predicted_placements_unseen": {
                "type": "object",
                "properties": unseen_props,
                "required": unseen_objects,
                "additionalProperties": False,
            },
        },
        "required": ["predicted_placements_seen", "predicted_placements_unseen"],
        "additionalProperties": False,
    }


def _gate_prompt(state: AgentState, *, strategy: Strategy, allowed: List[QType]) -> str:
    used = state["budget_used"]
    total = state["budget_total"]
    remaining = total - used
    qa_history = state["qa_history"]
    num_pref = sum(1 for x in qa_history if x["q_type"] == "preference")
    num_action = sum(1 for x in qa_history if x["q_type"] == "action")
    num_summary = sum(1 for x in qa_history if x["q_type"] == "summary")
    recent = qa_history[-3:]

    return f"""
You are choosing the next question type for a household organization agent.
Return a JSON object that matches the schema and choose q_type from: {allowed}.

Strategy: {strategy}
Budget: total={total}, used={used}, remaining={remaining}
Counts: preference={num_pref}, action={num_action}, summary={num_summary}
Known preference summary (may be empty): {state["preference"]}
Recent QA history (last 3 turns max): {recent}

Strategy guidance:
- preference-first: prioritize preference questions early; ask action only if preferences seem sufficient or budget is low; use summary only near the end.
- parallel: mix preference and action; ask summary once when there is enough evidence and budget is nearly done.
- direct: prioritize action questions; use summary only if budget is nearly done and there is some evidence.

Return ONLY the JSON object with key "q_type".
""".strip()


def _decide_q_type_with_llm(state: AgentState, *, strategy: Strategy) -> QType:
    used = state["budget_used"]
    total = state["budget_total"]
    remaining = total - used
    qa_history = state["qa_history"]
    num_summary = sum(1 for x in qa_history if x["q_type"] == "summary")

    allow_summary = remaining > 0 and num_summary < 1
    allowed: List[QType] = ["preference", "action"]
    if allow_summary:
        allowed.append("summary")

    prompt = _gate_prompt(state, strategy=strategy, allowed=allowed)
    structured_gate_llm = question_llm.with_structured_output(
        _gate_decision_schema(allowed),
        method="json_schema",
    )
    decision = structured_gate_llm.invoke([{"role": "user", "content": prompt}])
    if not isinstance(decision, dict):
        raise ValueError(f"Gate model returned non-dict decision: {decision!r}")

    q_type = decision.get("q_type")
    if not isinstance(q_type, str):
        raise ValueError(f"Gate model decision missing string q_type: {decision!r}")
    if q_type not in allowed:
        raise ValueError(f"Gate model returned invalid q_type={q_type!r}, allowed={allowed}")
    return q_type  # type: ignore[return-value]


def gate_node(state: AgentState) -> dict:
    strategy = state["strategy"]
    return {"current_q_type": _decide_q_type_with_llm(state, strategy=strategy)}


def _question_prompt_base(state: AgentState, *, q_type: QType) -> str:
    strategy = state["strategy"]
    remaining = state["budget_total"] - state["budget_used"]
    return f"""
You are a household rearrangement robot.
Strategy: {strategy}
Question type: {q_type}

Room: {state["room"]}
Available receptacles: {state["receptacles"]}
Seen objects: {state["seen_objects"]}

Budget:
- total: {state["budget_total"]}
- used: {state["budget_used"]}
- remaining: {remaining}

Conversation so far (qa_history):
{state["qa_history"]}

Known preference summary (may be empty):
    {state["preference"]}
    """.strip()


def _normalize_question_text(raw: Any) -> str:
    text = raw.content if hasattr(raw, "content") else str(raw)
    if isinstance(text, list):
        parts: List[str] = []
        for item in text:
            if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    text = str(text).strip()
    if not text:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.strip("\"' ")
    return ""


def _fallback_question(state: AgentState, *, q_type: QType) -> str:
    if q_type == "action":
        obj = state["seen_objects"][0] if state["seen_objects"] else "object"
        return f"Where should I place the {obj}?"
    if q_type == "preference":
        return "What are your main organizing preferences for this room?"
    return "Based on our conversation, is this summary of your preferences correct?"


def question_gen_action_node(state: AgentState) -> dict:
    """
    Generate one action question: ask where to place ONE seen object.
    """
    prompt = (
        _question_prompt_base(state, q_type="action")
        + "\n\nGenerate ONE concise action question: ask where to place ONE specific object "
        "(choose from seen_objects). Use the remaining budget to pick the most informative object. "
        "Return only the question text. Do not include any explanation."
    )
    msg = question_llm.invoke([{"role": "user", "content": prompt}])
    q_text = _normalize_question_text(msg)
    if not q_text:
        q_text = _fallback_question(state, q_type="action")
        print("[WARN] Empty action question from model; using fallback question.")
    return {"current_question": q_text}


def question_gen_preference_node(state: AgentState) -> dict:
    """
    Generate one preference-eliciting question: ask about organizing principles.
    """
    prompt = (
        _question_prompt_base(state, q_type="preference")
        + "\n\nGenerate ONE concise preference question about general organizing principles "
        "(category/function/attributes/separation/grouping). Use the remaining budget to ask the "
        "most impactful preference question. Return only the question text. Do not include any explanation."
    )
    msg = question_llm.invoke([{"role": "user", "content": prompt}])
    q_text = _normalize_question_text(msg)
    if not q_text:
        q_text = _fallback_question(state, q_type="preference")
        print("[WARN] Empty preference question from model; using fallback question.")
    return {"current_question": q_text}


def question_gen_summary_node(state: AgentState) -> dict:
    """
    Generate one preference-summary question: summarize inferred prefs and ask to confirm/correct.
    """
    prompt = (
        _question_prompt_base(state, q_type="summary")
        + "\n\nGenerate ONE concise summary question: summarize inferred preferences from the "
        "conversation and ask the user to confirm/correct. Keep it short and high confidence "
        "given the remaining budget. Return only the question text. Do not include any explanation."
    )
    msg = question_llm.invoke([{"role": "user", "content": prompt}])
    q_text = _normalize_question_text(msg)
    if not q_text:
        q_text = _fallback_question(state, q_type="summary")
        print("[WARN] Empty summary question from model; using fallback question.")
    return {"current_question": q_text}


def oracle_node(state: AgentState) -> dict:
    """
    Call oracle tool (LLM-based answerer).
    """
    q = state["current_question"] or ""
    a = oracle_answer_tool.invoke(
        {
            "question": q,
            "room": state["room"],
            "receptacles": state["receptacles"],
            "seen_objects": state["seen_objects"],
            "unseen_objects": state.get("unseen_objects", []),
            "seen_placements": state["seen_placements"],
            "unseen_placements": state.get("unseen_placements", {}),
            "annotator_notes": state["annotator_notes"],
        }
    )
    return {"current_answer": a}


def state_update_node(state: AgentState) -> dict:
    q_type = state["current_q_type"]
    q = state["current_question"] or ""
    a = state["current_answer"] or ""

    qa_item: QAItem = {"question": q, "answer": a, "q_type": q_type}  # type: ignore

    updates: dict = {
        "qa_history": state["qa_history"] + [qa_item],
        "asked_questions": state["asked_questions"] + [q],
        "budget_used": state["budget_used"] + 1,
    }

    # clear transient fields (optional)
    updates["current_question"] = None
    updates["current_answer"] = None
    updates["current_q_type"] = None

    return updates


def should_continue(state: AgentState) -> bool:
    return state["budget_used"] < state["budget_total"]


def route_question_node(state: AgentState) -> str:
    q_type = state["current_q_type"]
    if q_type == "preference":
        return "qgen_preference"
    if q_type == "summary":
        return "qgen_summary"
    return "qgen_action"


def route_after_update(state: AgentState) -> str:
    if should_continue(state):
        return "gate"
    return "summarize_pref"


def summarize_pref_node(state: AgentState) -> dict:
    """
    Summarize overall preferences from qa_history after the question budget is used.
    Uses all question types (action + preference + summary) as evidence.
    """
    system = (
        "You are summarizing a user's household organization preferences. "
        "Only output the final preference summary. Do not include reasoning."
    )
    user = f"""
        Room: {state["room"]}
        Receptacles: {state["receptacles"]}

        qa_history (all question types):
        {state["qa_history"]}

        Summarize the user's overall organization preferences into concise rules.
        Output only the summary text.
        """.strip()

    msg = question_llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    text = msg.content if hasattr(msg, "content") else str(msg)
    return {"preference": text.strip()}


def planner_node(state: AgentState) -> dict:
    """
    LLM-based planner (no extra parsing/validation helpers).
    Uses qa_history + preference to assign receptacles.

    Output must be strict JSON:
      {
        "predicted_placements_seen": {...},
        "predicted_placements_unseen": {...}
      }

    """
    receptacles = state["receptacles"]
    fallback = receptacles[0] if receptacles else "unknown"

    seen_objects = state["seen_objects"]
    unseen_objects = state.get("unseen_objects", [])

    system = (
        "You are a household rearrangement planner. "
        "Assign each object to exactly one receptacle. "
        "Output ONLY valid JSON (no markdown, no extra text)."
    )

    user = f"""
        Room: {state["room"]}

        Allowed receptacles (choose ONLY from this list):
        {receptacles}

        Seen objects (must output ALL of them):
        {seen_objects}

        Unseen objects (must output ALL of them):
        {unseen_objects}

        qa_history:
        {state["qa_history"]}

        preference (learned summary, may be empty):
        {state["preference"]}

        Return ONLY this JSON object:
        {{
        "predicted_placements_seen": {{ "<object>": "<receptacle>", ... }},
        "predicted_placements_unseen": {{ "<object>": "<receptacle>", ... }}
        }}

        Rules:
        - Every seen object must appear exactly once in predicted_placements_seen.
        - Every unseen object must appear exactly once in predicted_placements_unseen.
        - Do NOT add any objects not in the lists.
        - Receptacle values must match exactly one of the allowed receptacles.
        """.strip()

    structured_planner_llm = question_llm.with_structured_output(
        _planner_output_schema(
            seen_objects=seen_objects,
            unseen_objects=unseen_objects,
            receptacles=receptacles,
        ),
        method="json_schema",
    )
    try:
        parsed = structured_planner_llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
    except Exception as e:
        print(f"[WARN] planner structured output failed, using fallback placements: {e}")
        parsed = {}

    parsed_seen = parsed.get("predicted_placements_seen", {}) if isinstance(parsed, dict) else {}
    parsed_unseen = parsed.get("predicted_placements_unseen", {}) if isinstance(parsed, dict) else {}

    allowed = set(receptacles)
    pred_seen: Dict[str, str] = {}
    pred_unseen: Dict[str, str] = {}

    for obj in seen_objects:
        rec = parsed_seen.get(obj, fallback) if isinstance(parsed_seen, dict) else fallback
        pred_seen[obj] = rec if rec in allowed else fallback

    for obj in unseen_objects:
        rec = parsed_unseen.get(obj, fallback) if isinstance(parsed_unseen, dict) else fallback
        pred_unseen[obj] = rec if rec in allowed else fallback

    return {
        "predicted_placements_seen": pred_seen,
        "predicted_placements_unseen": pred_unseen,
    }



# =========================
# Graph Builder
# =========================
def build_mvp_graph():
    g = StateGraph(AgentState)

    g.add_node("init", init_node)
    g.add_node("gate", gate_node)
    g.add_node("qgen_action", question_gen_action_node)
    g.add_node("qgen_preference", question_gen_preference_node)
    g.add_node("qgen_summary", question_gen_summary_node)
    g.add_node("oracle", oracle_node)
    g.add_node("update", state_update_node)
    g.add_node("summarize_pref", summarize_pref_node)
    g.add_node("plan", planner_node)

    g.set_entry_point("init")
    g.add_edge("init", "gate")
    g.add_conditional_edges(
        "gate",
        route_question_node,
        {
            "qgen_action": "qgen_action",
            "qgen_preference": "qgen_preference",
            "qgen_summary": "qgen_summary",
        },
    )
    g.add_edge("qgen_action", "oracle")
    g.add_edge("qgen_preference", "oracle")
    g.add_edge("qgen_summary", "oracle")
    g.add_edge("oracle", "update")

    g.add_conditional_edges(
        "update",
        route_after_update,
        {
            "gate": "gate",
            "summarize_pref": "summarize_pref",
        },
    )

    g.add_edge("summarize_pref", "plan")
    g.add_edge("plan", END)
    return g.compile()


# =========================
# Dataset Loader
# =========================
def load_scenarios(path: str) -> List[dict]:
    """
    Expected JSON: list[scenario]
    Each scenario should contain at least:
      room, receptacles, seen_objects, annotator_notes,
      seen_placements (dict or list of [obj, receptacle])
    Optional:
      unseen_objects, unseen_placements
    """
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as f:
        scenarios = json.load(f)

    if not isinstance(scenarios, list):
        raise ValueError("Expected top-level JSON to be a list of scenarios.")
    return scenarios


def normalize_placements(placements) -> Dict[str, str]:
    """
    Accept either:
      - dict: {obj: receptacle}
      - list: [[obj, receptacle], ...]
    """
    if placements is None:
        return {}
    if isinstance(placements, dict):
        return dict(placements)
    if isinstance(placements, list):
        out: Dict[str, str] = {}
        for item in placements:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out[str(item[0])] = str(item[1])
        return out
    raise ValueError(f"Unknown placements format: {type(placements)}")


def make_initial_state(sample: dict, *, strategy: Strategy, budget_total: int) -> AgentState:
    seen_placements = normalize_placements(sample["seen_placements"])
    unseen_placements = normalize_placements(sample.get("unseen_placements"))

    state: AgentState = {
        "strategy": strategy,
        "budget_total": budget_total,
        "budget_used": 0,
        "room": sample["room"],
        "receptacles": list(sample["receptacles"]),
        "seen_objects": list(sample["seen_objects"]),
        "unseen_objects": list(sample.get("unseen_objects", [])),
        "seen_placements": seen_placements,
        "unseen_placements": unseen_placements,
        "annotator_notes": list(sample.get("annotator_notes", [])),
        "qa_history": [],
        "asked_questions": [],
        "preference": "",
        "predicted_placements_seen": {},
        "predicted_placements_unseen": {},
        "current_q_type": None,
        "current_question": None,
        "current_answer": None,
    }
    return state


# =========================
# Test Runner
# =========================
def run_one_episode(
    scenarios: List[dict],
    graph: Any,
    *,
    index: int,
    strategy: Strategy,
    budget_total: int,
) -> dict:
    if index < 0 or index >= len(scenarios):
        raise IndexError(f"index {index} out of range (0..{len(scenarios)-1})")

    sample = scenarios[index]

    init_state = make_initial_state(sample, strategy=strategy, budget_total=budget_total)
    final_state = graph.invoke(init_state)

    print("\n==============================")
    print(f"Episode index: {index}")
    print(f"Strategy: {strategy}")
    print(f"Budget: {budget_total}")
    print("==============================\n")

    print("===== QA HISTORY =====")
    for i, qa in enumerate(final_state["qa_history"]):
        print(f"\n[{i+1}] ({qa['q_type']})")
        print("Q:", qa["question"])
        print("A:", qa["answer"])

    print("\n===== PREDICTIONS =====")
    print("Predicted placements (seen):", final_state["predicted_placements_seen"])
    print("Predicted placements (unseen):", final_state["predicted_placements_unseen"])
    print("Preference:", final_state["preference"])

    print("\n===== EVAL =====")
    metrics = evaluate_episode(
        seen_objects=final_state["seen_objects"],
        unseen_objects=final_state.get("unseen_objects", []),
        predicted_seen=final_state["predicted_placements_seen"],
        predicted_unseen=final_state["predicted_placements_unseen"],
        gt_seen=final_state["seen_placements"],
        gt_unseen=final_state.get("unseen_placements", {}),
    )
    print("Seen acc:", metrics["seen_satisfaction"])
    print("Unseen acc:", metrics["unseen_satisfaction"])

    # Save qa_history with original dataset fields + predictions + preference
    qa_record = dict(sample)
    qa_record["strategy"] = strategy
    qa_record["budget_total"] = budget_total
    qa_record["budget_used"] = final_state["budget_used"]
    qa_record["qa_history"] = final_state["qa_history"]
    qa_record["predicted_placements_seen"] = final_state["predicted_placements_seen"]
    qa_record["predicted_placements_unseen"] = final_state["predicted_placements_unseen"]
    qa_record["preference"] = final_state["preference"]

    log_dir = Path("test_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    out_path = log_dir / f"{strategy}_ep{index}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(qa_record, f, indent=2, ensure_ascii=False)
    print(f"\nSaved qa_history -> {out_path}")
    return qa_record

    # Optional: show planner outputs (MVP)
    # print("\nPredicted placements (seen):", final_state["predicted_placements_seen"])
    # print("Predicted placements (unseen):", final_state["predicted_placements_unseen"])


def main():
    parser = argparse.ArgumentParser(description="Run LangGraph episodes with configurable strategies/budgets.")
    parser.add_argument(
        "--dataset",
        default="scenarios_aug_tiny.json",
        help="Path to dataset JSON (list of scenarios).",
    )
    parser.add_argument(
        "--strategies",
        default="preference-first,parallel,direct",
        help="Comma-separated strategies to test.",
    )
    parser.add_argument(
        "--budgets",
        default="3",
        help="Comma-separated budgets to test.",
    )
    parser.add_argument(
        "--indices",
        default=0,
        help="Comma-separated scenario indices to test. Use ALL or omit to evaluate all scenarios.",
    )
    parser.add_argument(
        "--plot-output",
        default="test_logs/strategy_tradeoff.png",
        help="Output path for the strategy tradeoff plot.",
    )

    args = parser.parse_args()

    dataset_path = args.dataset
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    scenarios = load_scenarios(dataset_path)
    indices_arg = "" if args.indices is None else str(args.indices).strip()
    if not indices_arg or indices_arg.upper() == "ALL":
        indices = list(range(len(scenarios)))
    else:
        indices = [int(x.strip()) for x in indices_arg.split(",") if x.strip()]

    graph = build_mvp_graph()
    with open("agent_graph.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())
        print("Saved graph visualization -> agent_graph.png")


    tasks: List[tuple[int, Strategy, int]] = []
    for index in indices:
        for strategy in strategies:
            for budget in budgets:
                tasks.append((index, strategy, budget))  # type: ignore

    iterator = tqdm(tasks, desc="Running episodes") if tqdm else tasks

    episodes: List[dict] = []
    for index, strategy, budget in iterator:
        episodes.append(
            run_one_episode(
                scenarios,
                graph,
                index=index,
                strategy=strategy,  # type: ignore
                budget_total=budget,
            )
        )

    if episodes:
        plot_strategy_tradeoff(episodes, output_path=args.plot_output)


if __name__ == "__main__":
    main()
