import React from 'react'
import { useSession } from '../App'
import type { TrialSnapshot } from '../types'

interface Props {
  trial: TrialSnapshot
  showNext?: boolean
}

export default function PredictionView({ trial, showNext }: Props) {
  const { session } = useSession()

  const psr = trial.psr
  if (!psr) return null

  const allObjects = [...trial.seen_objects, ...trial.unseen_objects]
  const zh = (name: string) => trial.name_mapping?.[name] ?? name

  return (
    <div className="participant-card">
      <h2>预测结果 — 第 {trial.trial_index + 1} 轮</h2>

      <div className="psr-grid">
        <div className="psr-card">
          <div className="psr-value">{(psr.seen_psr * 100).toFixed(1)}%</div>
          <div className="psr-label">可见物品 PSR</div>
        </div>
        <div className="psr-card">
          <div className="psr-value">{(psr.unseen_psr * 100).toFixed(1)}%</div>
          <div className="psr-label">隐藏物品 PSR</div>
        </div>
        <div className="psr-card">
          <div className="psr-value">{(psr.total_psr * 100).toFixed(1)}%</div>
          <div className="psr-label">整体 PSR</div>
        </div>
      </div>

      <h3 style={{ marginBottom: 10 }}>逐项结果</h3>
      <div className="item-score-list">
        {allObjects.map((obj) => {
          const correct = psr.item_scores[obj]
          const predicted = trial.predicted_placements?.[obj] ?? '—'
          return (
            <span key={obj} className={`item-score ${correct ? 'correct' : 'wrong'}`}>
              {correct ? '✓' : '✗'} {zh(obj)} → {zh(predicted)}
            </span>
          )
        })}
      </div>

      <div style={{ marginTop: 12, fontSize: 13, color: '#666' }}>
        提问轮次：<strong>{trial.turns_used}</strong>
      </div>

      {showNext && session && session.current_trial_index < 3 && (
        <div style={{ marginTop: 16, fontSize: 13, color: '#555' }}>
          请在左侧选择下一轮场景（第 {session.current_trial_index + 1} / 3 轮）。
        </div>
      )}
    </div>
  )
}
