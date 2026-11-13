"""Creates jobs and runs their stages; progress is recorded in jobs/<id>/job.json."""
import json
import re
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from . import media
from .config import JOBS

ID_RE = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
FINISHED = ("done", "failed")


def job_dir(job_id):
    """Directory of an existing job, or None if the id is malformed or unknown."""
    d = JOBS / job_id
    return d if ID_RE.fullmatch(job_id) and (d / "job.json").exists() else None


def load(d):
    return json.loads((d / "job.json").read_text(encoding="utf-8"))


def _save(d, state):
    (d / "job.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def create_job(input_name):
    """Make jobs/<id>/ in the 'queued' state and return its directory."""
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    d = JOBS / job_id
    d.mkdir(parents=True)
    _save(d, {"id": job_id, "input": Path(input_name).name, "stage": "queued", "error": None})
    return d


def fail(d, message):
    _save(d, {**load(d), "stage": "failed", "error": message})


def pack_scene(d):
    """Build scene.zip (the .blend plus the depth frames it points at) if it is not there yet.
    It can be hundreds of megabytes, so it is only made when someone asks for it."""
    target = d / "scene.zip"
    if not target.exists() and (d / "scene.blend").exists():
        partial = d / "scene.zip.part"
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as z:    # PNGs are already compressed
            z.write(d / "scene.blend", "scene.blend")
            for png in sorted((d / "depth").glob("*.png")):
                z.write(png, f"depth/{png.name}")
        partial.replace(target)
    return target


def process(d, src, crop=None):
    """Run every stage for one job. Never raises; the outcome is in job.json."""
    state = load(d)

    def enter(stage):
        state["stage"] = stage
        _save(d, state)

    try:
        enter("ingest")
        state["source"] = media.ingest(src, d, crop)
        enter("analyse")
        media.analyse(d)
        enter("build")
        size = media.output_size(state["source"])
        media.build_scene(d, size)
        enter("render")
        media.render(d / "scene.blend", d / "white.mp4", size)
        enter("compose")
        state["final"] = media.compose(d)
        enter("done")
    except Exception as e:
        state["error"] = str(e)
        enter("failed")


def run_job(src, crop=None):
    """Create and process a job in one call (command-line use). Returns the job directory."""
    d = create_job(src)
    process(d, src, crop)
    return d
