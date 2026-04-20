from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from study2_app.backend.routers import session, dialogue, evaluation, log  # noqa: E402
from study2_app.backend.voice import router as voice_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize LLM components once at startup
    from question_policy import QuestionPolicyController
    from proposers import (
        ActionProposer,
        PreferenceInductionProposer,
    )
    from state_update import StateUpdate
    from evaluation import FinalPlacementPlanner

    from study2_app.backend.pe_proposer_study2 import Study2PreferenceElicitingProposer

    app.state.policy = QuestionPolicyController(selection_method="rule")
    app.state.ao_proposer = ActionProposer()
    app.state.pe_proposer = Study2PreferenceElicitingProposer()
    app.state.pi_proposer = PreferenceInductionProposer()
    app.state.state_updater = StateUpdate()
    app.state.planner = FinalPlacementPlanner()

    yield


app = FastAPI(title="PrefQuest Study 2 Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router)
app.include_router(dialogue.router)
app.include_router(evaluation.router)
app.include_router(log.router)
app.include_router(voice_router)


@app.get("/health")
def health():
    return {"status": "ok"}
