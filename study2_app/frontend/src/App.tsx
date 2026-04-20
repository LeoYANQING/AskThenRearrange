import React, { createContext, useCallback, useContext, useState } from 'react'
import type { NextQuestionResponse, ScoreResponse, SessionSnapshot } from './types'
import * as api from './api'
import ExperimenterPanel from './components/ExperimenterPanel'
import ParticipantView from './components/ParticipantView'

interface SessionCtx {
  session: SessionSnapshot | null
  currentQuestion: NextQuestionResponse | null
  lastScore: ScoreResponse | null
  loading: boolean
  error: string | null
  setSession: (s: SessionSnapshot) => void
  setCurrentQuestion: (q: NextQuestionResponse | null) => void
  setLastScore: (s: ScoreResponse | null) => void
  setLoading: (v: boolean) => void
  setError: (e: string | null) => void
  refreshSession: () => Promise<void>
}

export const SessionContext = createContext<SessionCtx>({} as SessionCtx)

export function useSession() {
  return useContext(SessionContext)
}

export default function App() {
  const [session, setSession] = useState<SessionSnapshot | null>(null)
  const [currentQuestion, setCurrentQuestion] = useState<NextQuestionResponse | null>(null)
  const [lastScore, setLastScore] = useState<ScoreResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refreshSession = useCallback(async () => {
    if (!session) return
    try {
      const s = await api.getSession(session.session_id)
      setSession(s)
    } catch {}
  }, [session])

  return (
    <SessionContext.Provider
      value={{
        session,
        currentQuestion,
        lastScore,
        loading,
        error,
        setSession,
        setCurrentQuestion,
        setLastScore,
        setLoading,
        setError,
        refreshSession,
      }}
    >
      <div className="app-layout">
        <div className="panel-left">
          <ExperimenterPanel />
        </div>
        <div className="panel-right">
          <ParticipantView />
        </div>
      </div>
    </SessionContext.Provider>
  )
}
