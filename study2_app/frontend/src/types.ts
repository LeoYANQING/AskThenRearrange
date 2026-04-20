export interface AgentStateSnapshot {
  room?: string | null
  receptacles: string[]
  seen_objects: string[]
  unseen_objects: string[]
  qa_turns: number
  confirmed_actions: Array<{ object_name: string; receptacle: string }>
  negative_actions: Array<{ object_name: string; receptacle: string }>
  confirmed_preferences: Array<{
    hypothesis: string
    covered_objects?: string[]
    receptacle?: string | null
  }>
  negative_preferences: Array<{
    hypothesis: string
    covered_objects?: string[]
    receptacle?: string | null
  }>
  unresolved_objects: string[]
}

export interface QATurn {
  turn_index: number
  pattern: string
  question: string
  answer: string
  state_after?: AgentStateSnapshot | null
}

export interface TrialSnapshot {
  trial_index: number
  strategy: string
  room_type: string
  episode_index: number
  receptacles: string[]
  seen_objects: string[]
  unseen_objects: string[]
  name_mapping?: Record<string, string>  // en -> zh display labels
  dialogue: QATurn[]
  turns_used: number
  stop_reason: string | null
  preference_assignments: Record<string, string> | null
  predicted_placements: Record<string, string> | null
  psr: {
    seen_psr: number
    unseen_psr: number
    total_psr: number
    item_scores: Record<string, boolean>
  } | null
  phase: string
}

export interface SessionSnapshot {
  session_id: string
  participant_id: string
  latin_square_row: number
  trial_order: Array<{ strategy: string; room_type: string }>
  current_trial_index: number
  trials: TrialSnapshot[]
  phase: string
  notes: string
  agent_state?: AgentStateSnapshot | null
  strategy_ranking?: string[] | null
  final_comment?: string
  budget_total?: number
}

export interface NextQuestionResponse {
  question: string
  pattern: string
  turn_index: number
  dialogue_complete: boolean
}

export interface ScoreResponse {
  seen_psr: number
  unseen_psr: number
  total_psr: number
  item_scores: Record<string, boolean>
}

export const STRATEGY_LABELS: Record<string, string> = {
  DQ: '直接提问（DQ）',
  UPF: '用户偏好优先（UPF）',
  PAR: '并行探索（PAR）',
}

export const PATTERN_LABELS: Record<string, string> = {
  action_oriented: '动作导向',
  preference_eliciting: '偏好探询',
  preference_induction: '偏好归纳',
}

export const ROOM_LABELS: Record<string, string> = {
  'living room': '客厅',
  bedroom: '卧室',
  kitchen: '厨房',
}

export const PHASE_LABELS: Record<string, string> = {
  created: '待载入场景',
  scene_intro: '场景介绍',
  dialogue: '对话进行中',
  dialogue_complete: '对话已结束',
  preference_form: '偏好分配中',
  final_ranking: '策略排名',
  completed: '实验完成',
}
