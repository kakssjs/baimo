"""HTTP API. Run: uv run uvicorn server.app:app --port 8000

One worker thread processes jobs one at a time (Blender and the GPU must not run concurrently).
"""
import queue
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import job
from pipeline.config import JOBS, LIMITS, ROOT

DOWNLOADS = ("final.mp4", "white.mp4", "source.mp4", "scene.zip")
CHUNK = 1 << 20

_queue = queue.Queue()


def _worker():
    while True:
        d, src = _queue.get()
        job.process(d, src)


@asynccontextmanager
async def lifespan(app):
    # Jobs that were queued or running when the server last stopped start again from the top.
    for state_file in sorted(JOBS.glob("*/job.json")):
        d = state_file.parent
        if job.load(d)["stage"] not in job.FINISHED:
            src = next(d.glob("input.*"), None)
            if src:
                _queue.put((d, src))
            else:
                job.fail(d, "The uploaded file is missing.")
    threading.Thread(target=_worker, daemon=True).start()
    yield


app = FastAPI(title="Baimo", lifespan=lifespan)


def _find(job_id):
    d = job.job_dir(job_id)
    if d is None:
        raise HTTPException(404, "Job not found.")
    return d


@app.get("/api/limits")
def get_limits():
    return LIMITS


@app.post("/api/jobs", status_code=201)
async def create_job(file: UploadFile):
    """Upload a video and queue it. Returns the new job's state."""
    d = job.create_job(file.filename or "upload")
    src = d / ("input" + (Path(file.filename or "").suffix.lower() or ".bin"))
    size, limit = 0, LIMITS["max_upload_mb"] * 2**20
    with open(src, "wb") as out:
        while chunk := await file.read(CHUNK):
            size += len(chunk)
            if size > limit:
                break
            out.write(chunk)
    if size > limit:
        shutil.rmtree(d)
        raise HTTPException(413, f"File is larger than the {LIMITS['max_upload_mb']} MB limit.")
    _queue.put((d, src))
    return job.load(d)


@app.get("/api/jobs")
def list_jobs():
    """All jobs, newest first."""
    dirs = sorted(JOBS.glob("*/job.json"), reverse=True) if JOBS.exists() else []
    return [job.load(p.parent) for p in dirs]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return job.load(_find(job_id))


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str):
    d = _find(job_id)
    path = job.pack_scene(d) if name == "scene.zip" else d / name
    if name not in DOWNLOADS or not path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(path)


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    d = _find(job_id)
    if job.load(d)["stage"] not in job.FINISHED:
        raise HTTPException(409, "The job is still running.")
    shutil.rmtree(d)


# The web page; mounted last so it never shadows /api.
app.mount("/", StaticFiles(directory=ROOT / "web", html=True))
