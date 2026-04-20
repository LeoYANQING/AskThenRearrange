import React, { useState } from 'react'
import { useSession } from '../App'
import * as api from '../api'
import { ROOM_LABELS, STRATEGY_LABELS, PHASE_LABELS } from '../types'
import SessionCreate from './SessionCreate'
import AgentStatePanel from './AgentStatePanel'

export default function ExperimenterPanel() {
  const { session, setSession, setCurrentQuestion, setLastScore, setLoading, setError, loading, error } =
    useSession()
  const [episodeIndex, setEpisodeIndex] = useState(0)

  if (!session) {
    return (
      <div className="exp-panel">
        <h2>实验员控制台</h2>
        <SessionCreate />
      </div>
    )
  }

  const phase = session.phase
  const trialIdx = session.current_trial_index
  const trialConfig = session.trial_order[trialIdx]

  async function handleStartTrial() {
    if (!trialConfig) return
    setLoading(true)
    setError(null)
    try {
      const s = await api.startTrial(session!.session_id, trialConfig.room_type, episodeIndex)
      setSession(s)
      setCurrentQuestion(null)
      setLastScore(null)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e))
    } finally {
      setLoading(false)
    }
  }

  async function handleStartDialogue() {
    setLoading(true)
    setError(null)
    try {
      const q = await api.startDialogue(session!.session_id)
      setCurrentQuestion(q)
      const s = await api.getSession(session!.session_id)
      setSession(s)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e))
    } finally {
      setLoading(false)
    }
  }

  async function handleStopDialogue() {
    setLoading(true)
    setError(null)
    try {
      const s = await api.stopDialogue(session!.session_id)
      setSession(s)
      setCurrentQuestion(null)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e))
    } finally {
      setLoading(false)
    }
  }

  async function handleComputeScore() {
    setLoading(true)
    setError(null)
    try {
      const score = await api.computeScore(session!.session_id)
      setLastScore(score)
      const s = await api.getSession(session!.session_id)
      setSession(s)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="exp-panel">
      <h2>实验员控制台</h2>

      <div className="exp-section">
        <div className="exp-label">会话 ID</div>
        <div className="exp-value" style={{ fontFamily: 'monospace' }}>
          {session.session_id}
        </div>
        <div className="exp-label">参与者 ID</div>
        <div className="exp-value">{session.participant_id}</div>
        <div className="exp-label">拉丁方行</div>
        <div className="exp-value">第 {session.latin_square_row} 行</div>
      </div>

      <div className="exp-section">
        <div className="exp-label">当前阶段</div>
        <div>
          <span className="phase-badge">{PHASE_LABELS[phase] ?? phase}</span>
        </div>
        <div className="exp-label">当前试验</div>
        <div className="exp-value">
          {trialIdx < 3 ? `第 ${trialIdx + 1} / 3 轮` : '全部完成'}
        </div>
      </div>

      <div className="exp-section">
        <div className="exp-label">试验顺序</div>
        <ul className="trial-order-list">
          {session.trial_order.map((t, i) => (
            <li key={i} className={i === trialIdx ? 'active' : i < trialIdx ? 'done' : ''}>
              <span>{STRATEGY_LABELS[t.strategy] ?? t.strategy}</span>
              <span>{ROOM_LABELS[t.room_type] ?? t.room_type}</span>
            </li>
          ))}
        </ul>
      </div>

      {phase === 'created' && trialIdx < 3 && trialConfig && (
        <div className="exp-section">
          <div className="exp-label">
            选择场景（{ROOM_LABELS[trialConfig.room_type] ?? trialConfig.room_type}，0–33）
          </div>
          <input
            type="number"
            min={0}
            max={33}
            value={episodeIndex}
            onChange={(e) => setEpisodeIndex(Number(e.target.value))}
            className="form-input"
          />
          <button className="btn btn-primary mt-8" onClick={handleStartTrial} disabled={loading}>
            {loading ? '加载中…' : '载入场景'}
          </button>
        </div>
      )}

      {phase === 'scene_intro' && (
        <button
          className="btn btn-success"
          onClick={handleStartDialogue}
          disabled={loading}
        >
          {loading ? '请稍候…' : '开始对话'}
        </button>
      )}

      {phase === 'dialogue' && (
        <button
          className="btn btn-danger"
          onClick={handleStopDialogue}
          disabled={loading}
        >
          {loading ? '请稍候…' : '结束对话'}
        </button>
      )}

      {phase === 'dialogue_complete' && (
        <div className="exp-section" style={{ color: '#8fb3d0', fontSize: 13 }}>
          ↓ 等待参与者完成偏好分配表…
        </div>
      )}

      {phase === 'preference_form' && (
        <button
          className="btn btn-primary"
          onClick={handleComputeScore}
          disabled={loading}
        >
          {loading ? '计算中…' : '开始评分预测'}
        </button>
      )}

      {error && (
        <div style={{ color: '#ff6b6b', fontSize: 12, wordBreak: 'break-word' }}>
          ⚠ {error}
        </div>
      )}

      <AgentStatePanel />
    </div>
  )
}
