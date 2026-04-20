import React from 'react'
import { useSession } from '../App'
import { STRATEGY_LABELS, ROOM_LABELS } from '../types'

function pct(v: number | undefined | null): string {
  if (v === undefined || v === null) return '—'
  return `${(v * 100).toFixed(1)}%`
}

export default function SessionReport() {
  const { session } = useSession()
  if (!session) return null

  function handleDownload() {
    if (!session) return
    const payload = JSON.stringify(session, null, 2)
    const blob = new Blob([payload], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `session_${session.participant_id}_${session.session_id}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const trials = session.trials
  const mean = (fn: (t: (typeof trials)[number]) => number | undefined) => {
    const vs = trials.map(fn).filter((v): v is number => typeof v === 'number')
    if (vs.length === 0) return undefined
    return vs.reduce((a, b) => a + b, 0) / vs.length
  }

  return (
    <div className="participant-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <h2>实验报告</h2>
        <button className="btn" onClick={handleDownload}>
          ⬇ 下载 JSON
        </button>
      </div>

      <div style={{ marginTop: 12, fontSize: 13, color: '#555' }}>
        参与者 <strong>{session.participant_id}</strong> · 拉丁方行{' '}
        <strong>{session.latin_square_row}</strong> · Session{' '}
        <code style={{ fontSize: 12 }}>{session.session_id}</code>
      </div>

      {/* Per-trial PSR table */}
      <div style={{ marginTop: 20 }}>
        <h3 style={{ fontSize: 14, marginBottom: 10 }}>各 trial 结果</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f4f6f8', textAlign: 'left' }}>
              <th style={{ padding: '8px 10px' }}>#</th>
              <th style={{ padding: '8px 10px' }}>策略</th>
              <th style={{ padding: '8px 10px' }}>场景</th>
              <th style={{ padding: '8px 10px', textAlign: 'right' }}>轮次</th>
              <th style={{ padding: '8px 10px', textAlign: 'right' }}>seen PSR</th>
              <th style={{ padding: '8px 10px', textAlign: 'right' }}>unseen PSR</th>
              <th style={{ padding: '8px 10px', textAlign: 'right' }}>total PSR</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((t) => (
              <tr key={t.trial_index} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '8px 10px' }}>{t.trial_index + 1}</td>
                <td style={{ padding: '8px 10px' }}>
                  {STRATEGY_LABELS[t.strategy] ?? t.strategy}
                </td>
                <td style={{ padding: '8px 10px' }}>
                  {ROOM_LABELS[t.room_type] ?? t.room_type}
                </td>
                <td style={{ padding: '8px 10px', textAlign: 'right' }}>{t.turns_used}</td>
                <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                  {pct(t.psr?.seen_psr)}
                </td>
                <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                  {pct(t.psr?.unseen_psr)}
                </td>
                <td style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600 }}>
                  {pct(t.psr?.total_psr)}
                </td>
              </tr>
            ))}
            <tr style={{ background: '#f9fbfc', fontWeight: 600 }}>
              <td colSpan={3} style={{ padding: '8px 10px', textAlign: 'right' }}>
                平均
              </td>
              <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                {trials.length > 0
                  ? (trials.reduce((a, t) => a + t.turns_used, 0) / trials.length).toFixed(1)
                  : '—'}
              </td>
              <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                {pct(mean((t) => t.psr?.seen_psr))}
              </td>
              <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                {pct(mean((t) => t.psr?.unseen_psr))}
              </td>
              <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                {pct(mean((t) => t.psr?.total_psr))}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Strategy ranking */}
      {session.strategy_ranking && session.strategy_ranking.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ fontSize: 14, marginBottom: 10 }}>参与者策略排名</h3>
          <ol style={{ paddingLeft: 20, fontSize: 13 }}>
            {session.strategy_ranking.map((s, i) => (
              <li key={s} style={{ marginBottom: 4 }}>
                <strong>{STRATEGY_LABELS[s] ?? s}</strong>
                {i === 0 && <span style={{ color: '#4a7', marginLeft: 8 }}>← 最喜欢</span>}
                {i === session.strategy_ranking!.length - 1 && (
                  <span style={{ color: '#a66', marginLeft: 8 }}>← 最不喜欢</span>
                )}
              </li>
            ))}
          </ol>
          {session.final_comment && (
            <div style={{ marginTop: 10, fontSize: 13, color: '#555' }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>补充说明：</div>
              <div style={{ whiteSpace: 'pre-wrap', background: '#f8f9fa', padding: 10, borderRadius: 6 }}>
                {session.final_comment}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
