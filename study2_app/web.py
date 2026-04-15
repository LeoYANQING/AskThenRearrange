from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from study2_app.config import APP_NAME, AUTO_REFRESH_SECONDS, EXPORTS_DIR
from study2_app.service import Study2ExperimentService


app = FastAPI(title=APP_NAME)
service = Study2ExperimentService()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
    name="static",
)


def render(request: Request, template_name: str, **context):
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context={
            "app_name": APP_NAME,
            "auto_refresh_seconds": AUTO_REFRESH_SECONDS,
            **context,
        },
    )


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/experimenter", status_code=303)


@app.get("/experimenter")
def experimenter_index(request: Request, message: str = ""):
    return render(
        request,
        "experimenter_index.html",
        participants=service.list_participants(),
        message=message,
    )


@app.post("/experimenter/participants")
def create_participant() -> RedirectResponse:
    participant = service.create_participant()
    return RedirectResponse(
        url=f"/experimenter/participants/{participant['participant_id']}?message=Participant%20created",
        status_code=303,
    )


@app.get("/experimenter/participants/{participant_id}")
def experimenter_participant(request: Request, participant_id: str, message: str = ""):
    dashboard = service.get_participant_dashboard(participant_id)
    return render(request, "experimenter_participant.html", message=message, **dashboard)


@app.post("/experimenter/participants/{participant_id}/export")
def export_participant(participant_id: str) -> RedirectResponse:
    created = service.export_participant(participant_id)
    message = f"Exported: {', '.join(created)}"
    return RedirectResponse(
        url=f"/experimenter/participants/{participant_id}?message={message}",
        status_code=303,
    )


@app.post("/experimenter/export/all")
def export_all() -> RedirectResponse:
    created = service.export_all_data()
    message = f"Exported: {', '.join(created)}"
    return RedirectResponse(url=f"/experimenter?message={message}", status_code=303)


@app.get("/experimenter/exports/{filename}")
def get_export(filename: str) -> FileResponse:
    path = EXPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file not found.")
    return FileResponse(path)


@app.get("/experimenter/trials/{trial_id}")
def experimenter_trial(request: Request, trial_id: str, message: str = ""):
    trial = service.get_trial(trial_id)
    dashboard = service.get_participant_dashboard(trial["participant_id"])
    return render(
        request,
        "experimenter_trial.html",
        trial=trial,
        participant=dashboard["participant"],
        trials=dashboard["trials"],
        message=message,
    )


@app.post("/experimenter/trials/{trial_id}/start")
def start_trial(trial_id: str) -> RedirectResponse:
    service.start_trial(trial_id)
    return RedirectResponse(url=f"/experimenter/trials/{trial_id}", status_code=303)


@app.post("/experimenter/trials/{trial_id}/retry")
def retry_trial(trial_id: str) -> RedirectResponse:
    service.retry_failed_turn(trial_id)
    return RedirectResponse(url=f"/experimenter/trials/{trial_id}", status_code=303)


@app.post("/experimenter/trials/{trial_id}/interrupt")
def interrupt_trial(trial_id: str) -> RedirectResponse:
    service.interrupt_trial(trial_id)
    return RedirectResponse(url=f"/experimenter/trials/{trial_id}", status_code=303)


@app.post("/experimenter/trials/{trial_id}/resume")
def resume_trial(trial_id: str) -> RedirectResponse:
    service.resume_trial(trial_id)
    return RedirectResponse(url=f"/experimenter/trials/{trial_id}", status_code=303)


@app.get("/participant/{participant_id}/current")
def participant_current(request: Request, participant_id: str):
    trial = service.get_current_trial(participant_id)
    if trial is None:
        return render(request, "participant_result.html", trial=None, participant_complete=True)

    if trial["status"] in ("assigned", "dialogue_active", "dialogue_processing", "dialogue_failed", "trial_interrupted"):
        return render(request, "participant_current.html", trial=trial)
    if trial["status"] in ("dialogue_waiting_for_answer",):
        return render(request, "participant_current.html", trial=trial)
    if trial["status"] in ("dialogue_complete", "preference_form_active"):
        return RedirectResponse(
            url=f"/participant/trials/{trial['trial_id']}/preference-form",
            status_code=303,
        )
    return RedirectResponse(url=f"/participant/trials/{trial['trial_id']}/result", status_code=303)


@app.post("/participant/trials/{trial_id}/answer")
def submit_answer(trial_id: str, answer_text: str = Form(...)) -> RedirectResponse:
    trial = service.submit_answer(trial_id, answer_text.strip())
    return RedirectResponse(
        url=f"/participant/{trial['participant_id']}/current",
        status_code=303,
    )


@app.get("/participant/trials/{trial_id}/preference-form")
def participant_preference_form(request: Request, trial_id: str):
    trial = service.get_trial(trial_id)
    scene = service.scenes.get_scene(trial["scene_id"])
    episode = scene["episode"]
    if trial["status"] not in ("dialogue_complete", "preference_form_active"):
        return RedirectResponse(url=f"/participant/{trial['participant_id']}/current", status_code=303)
    return render(
        request,
        "participant_preference_form.html",
        trial=trial,
        items=list(episode.seen_objects) + list(episode.unseen_objects),
        receptacles=episode.receptacles,
    )


@app.post("/participant/trials/{trial_id}/preference-form")
async def submit_preference_form(request: Request, trial_id: str) -> RedirectResponse:
    trial = service.get_trial(trial_id)
    scene = service.scenes.get_scene(trial["scene_id"])
    episode = scene["episode"]
    form = await request.form()
    placements = {}
    for item in list(episode.seen_objects) + list(episode.unseen_objects):
        value = str(form.get(f"placement__{item}", "")).strip()
        placements[item] = value
    service.submit_preference_form(trial_id, placements)
    return RedirectResponse(url=f"/participant/trials/{trial_id}/result", status_code=303)


@app.get("/participant/trials/{trial_id}/result")
def participant_result(request: Request, trial_id: str):
    trial = service.get_trial(trial_id)
    if trial["status"] in ("results_computed", "questionnaire_pending", "trial_complete"):
        return render(request, "participant_result.html", trial=trial, participant_complete=False)
    return RedirectResponse(url=f"/participant/{trial['participant_id']}/current", status_code=303)


@app.get("/participant/trials/{trial_id}/questionnaire")
def participant_questionnaire(trial_id: str):
    trial = service.mark_questionnaire_pending(trial_id)
    if trial["questionnaire_url"]:
        return RedirectResponse(url=trial["questionnaire_url"], status_code=303)
    return RedirectResponse(url=f"/participant/trials/{trial_id}/result", status_code=303)


@app.post("/participant/trials/{trial_id}/complete")
def participant_complete_trial(trial_id: str) -> RedirectResponse:
    trial = service.complete_trial(trial_id)
    return RedirectResponse(
        url=f"/participant/{trial['participant_id']}/current",
        status_code=303,
    )
