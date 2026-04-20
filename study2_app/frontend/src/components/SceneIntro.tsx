import React from 'react'
import type { TrialSnapshot } from '../types'
import { ROOM_LABELS, STRATEGY_LABELS } from '../types'

interface Props {
  trial: TrialSnapshot
}

export default function SceneIntro({ trial }: Props) {
  const zh = (name: string) => trial.name_mapping?.[name] ?? name
  return (
    <div className="participant-card">
      <h2>
        场景介绍 — 第 {trial.trial_index + 1} 轮 /{' '}
        {ROOM_LABELS[trial.room_type] ?? trial.room_type}
      </h2>
      <div style={{ fontSize: 12, color: '#888', marginBottom: 16 }}>
        策略：{STRATEGY_LABELS[trial.strategy] ?? trial.strategy}
      </div>

      <div className="scene-grid">
        <div className="scene-section">
          <h4>容器（收纳位置）</h4>
          <div className="tag-list">
            {trial.receptacles.map((r) => (
              <span key={r} className="tag receptacle">
                {zh(r)}
              </span>
            ))}
          </div>
        </div>

        <div className="scene-section">
          <h4>可见物品（共 {trial.seen_objects.length} 件）</h4>
          <div className="tag-list">
            {trial.seen_objects.map((o) => (
              <span key={o} className="tag">
                {zh(o)}
              </span>
            ))}
          </div>
        </div>

        <div className="scene-section">
          <h4>隐藏物品（共 {trial.unseen_objects.length} 件）</h4>
          <div className="tag-list">
            {trial.unseen_objects.map((o) => (
              <span key={o} className="tag unseen">
                {zh(o)}
              </span>
            ))}
          </div>
        </div>
      </div>

      {trial.turns_used > 0 && (
        <div style={{ fontSize: 13, color: '#666', marginTop: 8 }}>
          已完成提问轮次：<strong>{trial.turns_used}</strong>
        </div>
      )}
    </div>
  )
}
