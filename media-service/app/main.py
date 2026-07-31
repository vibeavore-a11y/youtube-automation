from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .core import opportunity_score, quality_gate
from .render import render_thumbnail, render_video
from .production import produce

app = FastAPI(title="YouTube Automation Media Service", version="1.0.0")
WORK_ROOT = Path(os.getenv("WORK_ROOT", "/data/jobs"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(WORK_ROOT)), name="files")


class Candidate(BaseModel):
    platform: str
    source_url: str
    source_account: str = ""
    title: str
    views: int = Field(ge=0)
    likes: int = Field(ge=0, default=0)
    comments: int = Field(ge=0, default=0)
    age_hours: float = Field(gt=0)
    channel_median_views: int = Field(gt=0, default=1)
    search_demand: float = Field(ge=0, le=100, default=50)
    competition: float = Field(ge=0, le=100, default=50)
    localization_fit: float = Field(ge=0, le=100, default=50)
    rights_verified: bool = False


class Manifest(BaseModel):
    job_id: str
    channel_profile: str
    language_code: str
    source_url: str
    rights_verified: bool
    original_script: str
    hook: str
    payoff: str
    scene_count: int = Field(ge=1)
    duration_seconds: int = Field(gt=0)
    subtitle_coverage: float = Field(ge=0, le=1)
    audio_lufs: float
    audio_true_peak_db: float
    silence_ratio: float = Field(ge=0, le=1)
    visual_changes_per_minute: float = Field(ge=0)
    duplicate_scene_ratio: float = Field(ge=0, le=1)
    thumbnail_readability: float = Field(ge=0, le=100)
    child_safety_passed: bool = False
    human_reviewed: bool = False


class RenderRequest(BaseModel):
    video_path: str
    voice_path: str
    output_path: str
    subtitle_path: str | None = None


class ThumbnailRequest(BaseModel):
    background_path: str
    output_path: str
    headline: str = Field(min_length=1, max_length=80)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "dry_run": os.getenv("DRY_RUN", "true").lower() == "true"}


@app.post("/v1/score")
def score(candidate: Candidate) -> dict[str, Any]:
    return opportunity_score(candidate.model_dump())


@app.post("/v1/quality-gate")
def gate(manifest: Manifest) -> dict[str, Any]:
    return quality_gate(manifest.model_dump())


@app.get("/v1/probe")
def probe(path: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    work_root = Path(os.getenv("WORK_ROOT", "/data/jobs")).resolve()
    if work_root not in resolved.parents:
        raise HTTPException(400, "path must be below WORK_ROOT")
    if not resolved.is_file():
        raise HTTPException(404, "media file not found")
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height",
        "-of", "json", str(resolved),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise HTTPException(422, result.stderr[-500:])
    return json.loads(result.stdout)


@app.post("/v1/render")
def render(request: RenderRequest) -> dict[str, Any]:
    try:
        return render_video(
            request.video_path, request.voice_path, request.output_path,
            os.getenv("WORK_ROOT", "/data/jobs"), request.subtitle_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/v1/thumbnail")
def thumbnail(request: ThumbnailRequest) -> dict[str, Any]:
    try:
        return render_thumbnail(
            request.background_path, request.output_path,
            os.getenv("WORK_ROOT", "/data/jobs"), request.headline,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc


class DocumentaryJobRequest(BaseModel):
    request_id: str
    production_mode: str = "LIVE"
    documentary: dict[str, Any]
    acquisition: dict[str, Any] = {}
    timeline: dict[str, Any] = {}
    render: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    youtube: dict[str, Any] = {}


DOCUMENTARY_JOBS: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_documentary_job(job_id: str) -> None:
    job = DOCUMENTARY_JOBS[job_id]
    try:
        job.update({"status": "PROCESSING", "progress": 10, "updated_at": _now()})
        result = produce(
            job,
            WORK_ROOT / job_id,
            os.getenv("PUBLIC_BASE_URL", "https://youtube-automation-media.onrender.com"),
        )
        job.update({
            "status": "COMPLETED",
            "progress": 100,
            **result,
            "quality_gate_passed": True,
            "delivery_ready": True,
            "message": "Documentary production completed.",
            "updated_at": _now(),
        })
    except Exception as exc:
        job.update({"status": "FAILED", "error": str(exc), "updated_at": _now()})


def _authorize(authorization: str | None) -> None:
    expected = os.getenv("PRODUCTION_API_TOKEN", "").strip()
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(401, "invalid production token")


@app.post("/v1/documentary/jobs")
@app.post("/yt-factory-production")
def create_documentary_job(
    request: DocumentaryJobRequest,
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _authorize(authorization)
    job_id = str(uuid4())
    DOCUMENTARY_JOBS[job_id] = {
        "accepted": True,
        "job_id": job_id,
        "request_id": request.request_id,
        "status": "QUEUED",
        "progress": 0,
        "payload": request.model_dump(),
        "created_at": _now(),
        "updated_at": _now(),
    }
    background_tasks.add_task(process_documentary_job, job_id)
    return {"accepted": True, "job_id": job_id, "status": "QUEUED"}


@app.get("/v1/documentary/jobs/{job_id}")
def get_documentary_job(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    job = DOCUMENTARY_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Documentary job not found")
    job["production_poll_count"] = int(job.get("production_poll_count", 0)) + 1
    return job
