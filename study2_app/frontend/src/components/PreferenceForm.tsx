import React, { useState } from 'react'
import { useSession } from '../App'
import * as api from '../api'
import type { TrialSnapshot } from '../types'

interface Props {
  trial: TrialSnapshot
}

export default function PreferenceForm({ trial }: Props) {
  const { session, setSession, setLoading, setError, loading } = useSession()
  const [assignments, setAssignments] = useState<Record<string, string>>({})

  const allObjects = [...trial.seen_objects, ...trial.unseen_objects]

  function assign(obj: string, receptacle: string) {
    setAssignments((prev) => ({ ...prev, [obj]: receptacle }))
  }

  function unassign(obj: string) {
    setAssignments((prev) => {
      const next = { ...prev }
      delete next[obj]
      return next
    })
  }

  async function handleSubmit() {
    if (!session) return
    const unassigned = allObjects.filter((o) => !assignments[o])
    if (unassigned.length > 0) {
      setError(`以下物品还未分配：${unassigned.map(zh).join('、')}`)
      return
    }
    setLoading(true)
    setError(null)
    try {
      const s = await api.submitPreferenceForm(session.session_id, assignments)
      setSession(s)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e))
    } finally {
      setLoading(false)
    }
  }

  const unassignedObjects = allObjects.filter((o) => !assignments[o])
  const assignedCount = allObjects.length - unassignedObjects.length
  const zh = (name: string) => trial.name_mapping?.[name] ?? name

  return (
    <div className="participant-card">
      <h2>偏好分配表</h2>
      <p style={{ fontSize: 13, color: '#666', marginBottom: 16 }}>
        请将所有物品拖放至对应的收纳位置。（{assignedCount}/{allObjects.length} 已分配）
      </p>

      <div className="pref-form-container">
        <div className="pref-objects-column">
          <h4 style={{ fontSize: 14, color: '#888', marginBottom: 8 }}>待分配物品</h4>
          {unassignedObjects.length === 0 ? (
            <p style={{ fontSize: 14, color: '#27ae60' }}>全部已分配 ✓</p>
          ) : (
            unassignedObjects.map((obj) => (
              <div
                key={obj}
                className="pref-object-item"
                style={{
                  background: trial.unseen_objects.includes(obj) ? '#e8fce8' : '#e8f0fe',
                  borderColor: trial.unseen_objects.includes(obj) ? '#b3f5b3' : '#b3c8f5',
                }}
              >
                {zh(obj)}
                {trial.unseen_objects.includes(obj) && (
                  <span style={{ fontSize: 12, marginLeft: 4, color: '#1a8a2a' }}>[隐藏]</span>
                )}
                <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {trial.receptacles.map((r) => (
                    <button
                      key={r}
                      onClick={() => assign(obj, r)}
                      style={{
                        fontSize: 13,
                        padding: '3px 7px',
                        border: '1px solid #ccc',
                        borderRadius: 3,
                        background: '#fff',
                        cursor: 'pointer',
                      }}
                    >
                      {zh(r)}
                    </button>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>

        <div className="pref-receptacles-column">
          {trial.receptacles.map((r) => {
            const items = allObjects.filter((o) => assignments[o] === r)
            return (
              <div key={r} className="receptacle-zone">
                <h5>{zh(r)}</h5>
                <div className="assigned-items">
                  {items.map((obj) => (
                    <span key={obj} className="assigned-chip">
                      {zh(obj)}
                      <button onClick={() => unassign(obj)} title="移除">×</button>
                    </span>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <button
        className="btn btn-success mt-16"
        onClick={handleSubmit}
        disabled={loading || assignedCount < allObjects.length}
      >
        {loading ? '提交中…' : '确认提交偏好分配'}
      </button>
    </div>
  )
}
