import React from 'react'
import { useSession } from '../App'
import type { AgentStateSnapshot } from '../types'

function displayName(name: string, map?: Record<string, string>): string {
  if (!map) return name
  return map[name] ?? name
}

function renderAction(
  a: { object_name: string; receptacle: string },
  map?: Record<string, string>,
): string {
  return `${displayName(a.object_name, map)} → ${displayName(a.receptacle, map)}`
}

function renderPreference(
  p: { hypothesis: string; covered_objects?: string[]; receptacle?: string | null },
  map?: Record<string, string>,
): string {
  const parts: string[] = [p.hypothesis]
  if (p.receptacle) parts.push(`→ ${displayName(p.receptacle, map)}`)
  if (p.covered_objects && p.covered_objects.length > 0) {
    const objs = p.covered_objects.map((o) => displayName(o, map)).join(', ')
    parts.push(`[${objs}]`)
  }
  return parts.join(' ')
}

function Section({
  title,
  count,
  children,
}: {
  title: string
  count: number
  children: React.ReactNode
}) {
  return (
    <div className="agent-section">
      <div className="agent-section-title">
        <span>{title}</span>
        <span className="agent-count">{count}</span>
      </div>
      {count === 0 ? <div className="agent-empty">—</div> : children}
    </div>
  )
}

export default function AgentStatePanel() {
  const { session } = useSession()
  if (!session) return null

  const agent: AgentStateSnapshot | null | undefined = session.agent_state
  const currentIdx = session.current_trial_index
  const currentTrial = session.trials[currentIdx]
  const nameMap = currentTrial?.name_mapping

  if (!agent) {
    return (
      <div className="agent-panel">
        <h3>AgentState 监控</h3>
        <div className="agent-empty">尚未载入场景</div>
      </div>
    )
  }

  return (
    <div className="agent-panel">
      <h3>
        AgentState 监控
        <span className="agent-turns">已对话 {agent.qa_turns} 轮</span>
      </h3>

      <Section title="已确认动作（confirmed_actions）" count={agent.confirmed_actions.length}>
        <ul>
          {agent.confirmed_actions.map((a, i) => (
            <li key={i} className="agent-confirmed">{renderAction(a, nameMap)}</li>
          ))}
        </ul>
      </Section>

      <Section title="已排除动作（negative_actions）" count={agent.negative_actions.length}>
        <ul>
          {agent.negative_actions.map((a, i) => (
            <li key={i} className="agent-negative">{renderAction(a, nameMap)}</li>
          ))}
        </ul>
      </Section>

      <Section
        title="已确认偏好（confirmed_preferences）"
        count={agent.confirmed_preferences.length}
      >
        <ul>
          {agent.confirmed_preferences.map((p, i) => (
            <li key={i} className="agent-confirmed">{renderPreference(p, nameMap)}</li>
          ))}
        </ul>
      </Section>

      <Section
        title="已否认偏好（negative_preferences）"
        count={agent.negative_preferences.length}
      >
        <ul>
          {agent.negative_preferences.map((p, i) => (
            <li key={i} className="agent-negative">{renderPreference(p, nameMap)}</li>
          ))}
        </ul>
      </Section>

      <Section title="未解决物体（unresolved_objects）" count={agent.unresolved_objects.length}>
        <ul>
          {agent.unresolved_objects.map((o, i) => (
            <li key={i}>{displayName(o, nameMap)}</li>
          ))}
        </ul>
      </Section>
    </div>
  )
}
