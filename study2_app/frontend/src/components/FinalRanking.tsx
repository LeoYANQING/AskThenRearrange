import React, { useState } from 'react'
import { useSession } from '../App'
import * as api from '../api'
import { STRATEGY_LABELS, ROOM_LABELS } from '../types'

const STRATEGIES = ['DQ', 'UPF', 'PAR']

export default function FinalRanking() {
  const { session, setSession, setLoading, setError, loading } = useSession()
  const [ranking, setRanking] = useState<string[]>([...STRATEGIES])
  const [comment, setComment] = useState('')

  function moveUp(index: number) {
    if (index === 0) return
    setRanking((prev) => {
      const next = [...prev]
      ;[next[index - 1], next[index]] = [next[index], next[index - 1]]
      return next
    })
  }

  function moveDown(index: number) {
    if (index === ranking.length - 1) return
    setRanking((prev) => {
      const next = [...prev]
      ;[next[index], next[index + 1]] = [next[index + 1], next[index]]
      return next
    })
  }

  async function handleSubmit() {
    if (!session) return
    setLoading(true)
    setError(null)
    try {
      const s = await api.submitFinalRanking(session.session_id, ranking, comment)
      setSession(s)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="participant-card">
      <h2>策略排名</h2>
      <p style={{ fontSize: 13, color: '#666', marginBottom: 16 }}>
        请按照您的偏好对三种对话策略进行排名（从最喜欢到最不喜欢）。
        使用上下按钮调整顺序。
      </p>

      {/* Show per-trial results summary */}
      {session && session.trials.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <h3 style={{ marginBottom: 10, fontSize: 14 }}>各轮结果回顾</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {session.trials.map((t) => (
              <div
                key={t.trial_index}
                style={{
                  background: '#f8f9fa',
                  borderRadius: 6,
                  padding: '8px 12px',
                  fontSize: 13,
                  display: 'flex',
                  justifyContent: 'space-between',
                }}
              >
                <span>
                  {STRATEGY_LABELS[t.strategy] ?? t.strategy} — {ROOM_LABELS[t.room_type] ?? t.room_type}
                </span>
                <span style={{ color: '#555' }}>
                  {t.turns_used} 轮 ·{' '}
                  {t.psr ? `PSR ${(t.psr.total_psr * 100).toFixed(1)}%` : '—'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="ranking-list">
        {ranking.map((strategy, i) => (
          <div key={strategy} className="ranking-item">
            <div className="rank-number">{i + 1}</div>
            <div className="ranking-item-name" style={{ flex: 1 }}>
              {STRATEGY_LABELS[strategy] ?? strategy}
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              <button
                onClick={() => moveUp(i)}
                disabled={i === 0}
                style={{
                  background: 'none',
                  border: '1px solid #ccc',
                  borderRadius: 4,
                  padding: '4px 8px',
                  cursor: i === 0 ? 'not-allowed' : 'pointer',
                  opacity: i === 0 ? 0.4 : 1,
                }}
              >
                ↑
              </button>
              <button
                onClick={() => moveDown(i)}
                disabled={i === ranking.length - 1}
                style={{
                  background: 'none',
                  border: '1px solid #ccc',
                  borderRadius: 4,
                  padding: '4px 8px',
                  cursor: i === ranking.length - 1 ? 'not-allowed' : 'pointer',
                  opacity: i === ranking.length - 1 ? 0.4 : 1,
                }}
              >
                ↓
              </button>
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 16 }}>
        <label style={{ fontSize: 13, color: '#666', display: 'block', marginBottom: 6 }}>
          补充说明（可选）
        </label>
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="请输入您对实验的任何补充意见…"
          style={{
            width: '100%',
            minHeight: 80,
            padding: '8px 10px',
            border: '1px solid #ddd',
            borderRadius: 6,
            fontSize: 13,
            fontFamily: 'inherit',
            resize: 'vertical',
          }}
        />
      </div>

      <button
        className="btn btn-primary mt-16"
        onClick={handleSubmit}
        disabled={loading}
      >
        {loading ? '提交中…' : '提交最终排名'}
      </button>
    </div>
  )
}
