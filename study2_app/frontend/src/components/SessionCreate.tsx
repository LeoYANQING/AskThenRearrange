import React, { useState } from 'react'
import { useSession } from '../App'
import * as api from '../api'

export default function SessionCreate() {
  const { setSession, setLoading, setError, loading, error } = useSession()
  const [participantId, setParticipantId] = useState('')
  const [row, setRow] = useState(1)
  const [notes, setNotes] = useState('')
  const [budget, setBudget] = useState(6)

  async function handleCreate() {
    if (!participantId.trim()) {
      setError('请输入参与者 ID')
      return
    }
    if (!Number.isInteger(budget) || budget < 1 || budget > 100) {
      setError('Budget 必须是 1–100 的整数')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const s = await api.createSession(participantId.trim(), row, notes.trim(), budget)
      setSession(s)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div className="form-group">
        <label className="form-label">参与者 ID</label>
        <input
          className="form-input"
          value={participantId}
          onChange={(e) => setParticipantId(e.target.value)}
          placeholder="例如：P01"
        />
      </div>
      <div className="form-group">
        <label className="form-label">拉丁方行（1–6）</label>
        <input
          type="number"
          min={1}
          max={6}
          className="form-input"
          value={row}
          onChange={(e) => setRow(Number(e.target.value))}
        />
      </div>
      <div className="form-group">
        <label className="form-label">每 trial 提问预算 B</label>
        <input
          type="number"
          min={1}
          max={100}
          className="form-input"
          value={budget}
          onChange={(e) => setBudget(Number(e.target.value))}
        />
      </div>
      <div className="form-group">
        <label className="form-label">备注（可选）</label>
        <input
          className="form-input"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="实验备注"
        />
      </div>
      {error && (
        <div style={{ color: '#ff6b6b', fontSize: 12 }}>{error}</div>
      )}
      <button className="btn btn-primary" onClick={handleCreate} disabled={loading}>
        {loading ? '创建中…' : '创建会话'}
      </button>
    </div>
  )
}
