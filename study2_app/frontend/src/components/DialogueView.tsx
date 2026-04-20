import React, { useRef, useState } from 'react'
import { useSession } from '../App'
import * as api from '../api'
import { Recorder, transcribe } from '../voice'
import type { QATurn, TrialSnapshot } from '../types'
import { PATTERN_LABELS } from '../types'

interface Props {
  trial: TrialSnapshot
}

function patternClass(pattern: string) {
  if (pattern === 'action_oriented') return 'ao'
  if (pattern === 'preference_eliciting') return 'pe'
  if (pattern === 'preference_induction') return 'pi'
  return ''
}

type Action = { object_name: string; receptacle: string }

function actionKey(a: Action): string {
  return `${a.object_name}→${a.receptacle}`
}

// New actions confirmed in turn i, relative to turn i-1.
function newActionsForTurn(turn: QATurn, prev: QATurn | undefined): Action[] {
  const now = turn.state_after?.confirmed_actions ?? []
  const before = prev?.state_after?.confirmed_actions ?? []
  const seen = new Set(before.map(actionKey))
  return now.filter((a) => !seen.has(actionKey(a)))
}

function displayName(name: string, map?: Record<string, string>): string {
  if (!map) return name
  return map[name] ?? name
}

export default function DialogueView({ trial }: Props) {
  const { session, currentQuestion, setCurrentQuestion, setSession, loading, setLoading, setError } =
    useSession()
  const [answer, setAnswer] = useState('')
  const [sttState, setSttState] = useState<'idle' | 'recording' | 'transcribing'>('idle')
  const [sttNote, setSttNote] = useState<string>('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const recorderRef = useRef<Recorder | null>(null)

  async function handleRecord() {
    if (sttState === 'recording') {
      const rec = recorderRef.current
      if (!rec) return
      setSttState('transcribing')
      try {
        const { blob, durationMs, sampleRate, peak } = await rec.stop()
        if (peak < 0.02) {
          setSttNote(`⚠️ 录音信号极低 (peak=${peak.toFixed(3)})，请检查麦克风`)
        }
        const text = await transcribe(blob, 'zh', sampleRate)
        setAnswer((prev) => (prev ? prev + text : text))
        setSttNote(`识别完成 · ${(durationMs / 1000).toFixed(1)}s`)
        setTimeout(() => textareaRef.current?.focus(), 50)
      } catch (e: any) {
        setSttNote('识别失败: ' + (e?.message ?? String(e)))
      } finally {
        setSttState('idle')
        recorderRef.current = null
      }
      return
    }
    try {
      const rec = new Recorder()
      await rec.start()
      recorderRef.current = rec
      setSttState('recording')
      setSttNote('录音中…再次点击停止')
    } catch (e: any) {
      setSttNote('麦克风权限被拒绝: ' + (e?.message ?? String(e)))
    }
  }

  async function handleSubmitAnswer() {
    if (!answer.trim() || !session) return
    setLoading(true)
    setError(null)
    try {
      const next = await api.submitAnswer(session.session_id, answer.trim())
      setCurrentQuestion(next.dialogue_complete ? null : next)
      setAnswer('')
      setSttNote('')
      const s = await api.getSession(session.session_id)
      setSession(s)
      if (!next.dialogue_complete) {
        setTimeout(() => textareaRef.current?.focus(), 100)
      }
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e))
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmitAnswer()
    }
  }

  const isActive = session?.phase === 'dialogue'

  return (
    <div className="participant-card">
      <h2>对话记录</h2>

      {trial.dialogue.length === 0 && !currentQuestion && (
        <p style={{ color: '#999', fontSize: 13 }}>等待开始…</p>
      )}

      {trial.dialogue.length > 0 && (
        <div className="dialogue-list">
          {trial.dialogue.map((turn, i) => {
            const newActions = newActionsForTurn(turn, trial.dialogue[i - 1])
            return (
              <div key={turn.turn_index} className={`dialogue-turn ${patternClass(turn.pattern)}`}>
                <div className="turn-meta">
                  <span>#{turn.turn_index + 1}</span>
                  <span>{PATTERN_LABELS[turn.pattern] ?? turn.pattern}</span>
                </div>
                <div className="turn-question">{turn.question}</div>
                {turn.answer && <div className="turn-answer">{turn.answer}</div>}
                {newActions.length > 0 && (
                  <div className="action-hint">
                    <div className="action-hint-title">🔔 需要放置的物体</div>
                    <ul>
                      {newActions.map((a) => (
                        <li key={actionKey(a)}>
                          <strong>{displayName(a.object_name, trial.name_mapping)}</strong>
                          <span className="action-arrow"> → </span>
                          <strong>{displayName(a.receptacle, trial.name_mapping)}</strong>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {currentQuestion && !currentQuestion.dialogue_complete && isActive && (
        <div className="current-question-box">
          <div className="turn-meta">
            <span>#{currentQuestion.turn_index + 1}</span>
            <span>{PATTERN_LABELS[currentQuestion.pattern] ?? currentQuestion.pattern}</span>
          </div>
          <div className="question-text">{currentQuestion.question}</div>

          <textarea
            ref={textareaRef}
            className="answer-area"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="请输入参与者的回答（Ctrl+Enter 提交，或点击 🎙️ 语音输入）"
            disabled={loading}
          />
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8 }}>
            <button
              type="button"
              className="btn"
              onClick={handleRecord}
              disabled={loading || sttState === 'transcribing'}
              style={{
                background: sttState === 'recording' ? '#e53935' : '#f0f4f9',
                color: sttState === 'recording' ? '#fff' : '#1a3a5c',
                border: '1px solid #b3d4f5',
              }}
            >
              {sttState === 'recording'
                ? '⏹ 停止录音'
                : sttState === 'transcribing'
                ? '识别中…'
                : '🎙️ 语音输入'}
            </button>
            <button
              className="btn btn-primary"
              onClick={handleSubmitAnswer}
              disabled={loading || !answer.trim()}
            >
              {loading ? '处理中…' : '提交回答'}
            </button>
            {sttNote && <span style={{ color: '#666', fontSize: 12 }}>{sttNote}</span>}
          </div>
        </div>
      )}

      {session?.phase === 'dialogue_complete' && (
        <div style={{ marginTop: 12, fontSize: 13, color: '#666' }}>
          对话已结束。共 <strong>{trial.turns_used}</strong> 轮问答。
          {trial.stop_reason === 'user_terminated' && ' （实验员手动终止）'}
          {trial.stop_reason === 'all_resolved' && ' （所有物体已明确归属，提前结束）'}
        </div>
      )}
    </div>
  )
}
