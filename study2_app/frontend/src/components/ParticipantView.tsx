import React, { useState } from 'react'
import { useSession } from '../App'
import SceneIntro from './SceneIntro'
import DialogueView from './DialogueView'
import PreferenceForm from './PreferenceForm'
import PredictionView from './PredictionView'
import FinalRanking from './FinalRanking'
import SessionReport from './SessionReport'

export default function ParticipantView() {
  const { session, error } = useSession()
  const [showReport, setShowReport] = useState(false)

  if (!session) {
    return (
      <div className="empty-state">
        <p>请在左侧创建会话以开始实验。</p>
      </div>
    )
  }

  const phase = session.phase
  const currentIdx = session.current_trial_index
  const currentTrial = session.trials[currentIdx]
  const lastCompletedTrial =
    currentIdx > 0 ? session.trials[currentIdx - 1] : null

  return (
    <div>
      {error && (
        <div
          style={{
            background: '#fce8e8',
            border: '1px solid #f5c6c6',
            borderRadius: 8,
            padding: '10px 14px',
            marginBottom: 16,
            color: '#7a1a1a',
            fontSize: 13,
          }}
        >
          ⚠ {error}
        </div>
      )}

      {/* Show last trial results while waiting for next trial */}
      {phase === 'created' && lastCompletedTrial?.psr && (
        <PredictionView trial={lastCompletedTrial} showNext />
      )}

      {/* Scene is loaded — show scene info */}
      {(phase === 'scene_intro' || phase === 'dialogue' || phase === 'dialogue_complete') &&
        currentTrial && <SceneIntro trial={currentTrial} />}

      {/* Active or ended dialogue */}
      {(phase === 'dialogue' || phase === 'dialogue_complete') && currentTrial && (
        <DialogueView trial={currentTrial} />
      )}

      {/* Participant fills preference form — shown as soon as dialogue ends */}
      {(phase === 'dialogue_complete' || phase === 'preference_form') && currentTrial && (
        <PreferenceForm trial={currentTrial} />
      )}

      {/* Final ranking after all 3 trials */}
      {phase === 'final_ranking' && <FinalRanking />}

      {/* Session complete */}
      {phase === 'completed' && (
        <>
          <div className="participant-card">
            <h2>实验完成</h2>
            <p style={{ color: '#555', marginTop: 8 }}>感谢参与！所有数据已记录。</p>
            <button
              className="btn btn-primary mt-16"
              onClick={() => setShowReport((v) => !v)}
            >
              {showReport ? '隐藏实验报告' : '查看实验报告'}
            </button>
          </div>
          {showReport && <SessionReport />}
        </>
      )}

      {/* Waiting to load first trial */}
      {phase === 'created' && currentIdx === 0 && (
        <div className="empty-state">
          <p>请在左侧选择场景，然后点击"载入场景"。</p>
        </div>
      )}
    </div>
  )
}
