import axios from 'axios'
import type { NextQuestionResponse, ScoreResponse, SessionSnapshot } from './types'

const api = axios.create({ baseURL: '/' })

export async function createSession(
  participantId: string,
  latinSquareRow: number,
  notes: string,
  budgetTotal: number,
): Promise<SessionSnapshot> {
  const res = await api.post('/sessions', {
    participant_id: participantId,
    latin_square_row: latinSquareRow,
    notes,
    budget_total: budgetTotal,
  })
  return res.data
}

export async function getSession(sessionId: string): Promise<SessionSnapshot> {
  const res = await api.get(`/sessions/${sessionId}`)
  return res.data
}

export async function startTrial(
  sessionId: string,
  roomType: string,
  episodeIndex: number,
): Promise<SessionSnapshot> {
  const res = await api.post(`/sessions/${sessionId}/trial`, {
    room_type: roomType,
    episode_index: episodeIndex,
  })
  return res.data
}

export async function startDialogue(sessionId: string): Promise<NextQuestionResponse> {
  const res = await api.post(`/dialogue/${sessionId}/start`)
  return res.data
}

export async function submitAnswer(
  sessionId: string,
  answer: string,
): Promise<NextQuestionResponse> {
  const res = await api.post(`/dialogue/${sessionId}/answer`, { answer })
  return res.data
}

export async function stopDialogue(sessionId: string): Promise<SessionSnapshot> {
  const res = await api.post(`/dialogue/${sessionId}/stop`)
  return res.data
}

export async function submitPreferenceForm(
  sessionId: string,
  assignments: Record<string, string>,
): Promise<SessionSnapshot> {
  const res = await api.post(`/sessions/${sessionId}/preference_form`, { assignments })
  return res.data
}

export async function computeScore(sessionId: string): Promise<ScoreResponse> {
  const res = await api.post(`/sessions/${sessionId}/score`)
  return res.data
}

export async function submitFinalRanking(
  sessionId: string,
  strategyRanking: string[],
  comment: string,
): Promise<SessionSnapshot> {
  const res = await api.post(`/sessions/${sessionId}/final_ranking`, {
    strategy_ranking: strategyRanking,
    comment,
  })
  return res.data
}
