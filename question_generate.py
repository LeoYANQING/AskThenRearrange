
#def generate_question_general(problem: str, history: str = "") -> str:

def generate_question_user_preference(problem: str, history: str = "") -> str:
    """
    Generate a question for the user preference.
    """
    user_preference_prompt_template = f"""You are an AI agent controlling a home service robot in a Goal-Oriented task (e.g., refrigerator organization). Your goal is to use the User-Preference-First strategy to minimize uncertainty before acting.

    Context:
    History of Q&A:
    {history}

    Instructions: 
    * Review the "Remaining objects to place".
    * Phase 1 (Prioritize Preferences): If you don't know the user's general grouping logic for these types of items, ask a K11 (User Preferences) question. Inquire about habits or grouping rules that could apply to multiple items (e.g., "How do you organize fruits?").
    * Phase 2 (Task Execution): If you have enough information to determine the placement of one or more objects, DO NOT ASK. Instead, output one or more placement commands.
    * Question Form: Use Q14 (Judging) or Q1 (Confirmation).

    Output Format:
    - If asking a question: Output ONLY the question sentence.
    - If placing objects: Output "PLACEMENT: <object_name> -> <receptacle_name>" (one per line).
    - Do NOT output any internal thoughts, reasoning, or analysis. Start your response directly.

    Example:
    (If history is empty)
    Do you have a preferred way to group items in the fridge?

    (If history contains "I like fruits in the drawer")
    PLACEMENT: apple -> bottom drawer
    PLACEMENT: banana -> bottom drawer

    problem: {problem}
    """
    return user_preference_prompt_template.format(problem=problem, history=history)


def generate_question_ParallelExploration(problem: str, history: str = "") -> str:
    """
    Generate a question for the parallel exploration.
    """
    parallel_exploration_prompt_template = f"""
    You are an AI agent controlling a home service robot in a Process-Oriented task (e.g., cocktail mixing). Your goal is to use the Parallel Exploration strategy.

    Context:
    History of Q&A:
    {history}

    Instructions: 
    * Step-by-Step Execution: Focus on the immediate next step. Ask specific, operational questions first (e.g., K4 Quantities, K8 Procedure) to keep the task moving.
    * Interleaved Preferences: While executing, occasionally ask about preferences (K11) to guide future steps, but do not start with abstract preference questions.
    * Inference: If you know the next step from history, perform it (PLACEMENT).

    Output Format:
    - please do not return analysis and reasoning. Output ONLY the question sentence. Do not include any internal thoughts, reasoning, or analysis.


    problem: {problem}
    """
    return parallel_exploration_prompt_template.format(problem=problem, history=history)

def generate_question_direct_question(problem: str, history: str = "") -> str: 
    """
    Generate a question for the direct question.
    """
    direct_question_prompt_template = f"""
    You are an AI agent controlling a home service robot. Your goal is to use the Direct Querying strategy.

    Context:
    History of Q&A:
    {history}

    Instructions:
    * Operational Focus: Direct all questions toward immediate task execution.
    * Avoid Preferences: Do not proactively ask about general habits.
    * Reactive Logic: Ask about specific objects if you don't know where they go.
    * Inference: If you know where an object goes, place it.
    
    Output Format:
    - Output ONLY the question sentence.
    - Do NOT output any internal thoughts, reasoning, or analysis. Start your response directly.

    problem: {problem}
    """
    return direct_question_prompt_template.format(problem=problem, history=history)
