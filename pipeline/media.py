"""Stage wrappers: probe, ingest (1), analyse (2), Blender scene build (3) and render (4), compose (5)."""
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .config import LIMITS, ROOT, WHITEBOX_SCRIPTS, tool

H264 = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
BLENDER_SCRIPTS = Path(__file__).parent / "blender"
BLENDER_FLAGS = ["--factory-startup", "--python-exit-code", "1"]
SAMPLES = 48            # frames looked at to find the picture area


def _run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode:
        log = (p.stderr.strip() or p.stdout.strip()).splitlines()[-8:]
        raise RuntimeError(f"{Path(cmd[0]).stem} failed:\n" + "\n".join(log))
    return p.stdout


def _ffmpeg(*args):
    _run([tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)])


def probe(path):
    """Return width, height, fps (fraction string), frames, duration and has_audio for a video file."""
    info = json.loads(_run([tool("ffprobe"), "-v", "error", "-show_streams", "-show_format",
                            "-of", "json", str(path)]))
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("The file has no video stream.")
    avg = Fraction(video["avg_frame_rate"]) if video["avg_frame_rate"] != "0/0" else Fraction(0)
    nominal = Fraction(video["r_frame_rate"])
    # r_frame_rate is the nominal rate (24 for a 23.75 fps average clip); on variable-rate
    # files it can be a meaningless timebase value, so fall back to the rounded average.
    fps = nominal if avg and nominal <= avg * 2 else Fraction(round(avg) or 24)
    return {
        "width": video["width"],
        "height": video["height"],
        "fps": str(fps),
        "frames": int(video.get("nb_frames", 0)),
        "duration": float(info["format"].get("duration", 0)),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def _longest_run(flags, gap):
    """(start, end) of the longest stretch of True values, bridging gaps of up to `gap` False values."""
    idx = np.flatnonzero(flags)
    if not len(idx):
        return 0, 0
    runs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > gap + 1:
            runs.append((start, a + 1))
            start = b
    runs.append((start, idx[-1] + 1))
    return max(runs, key=lambda r: r[1] - r[0])


def _seam(lum, axis):
    """Index where a comparison video's two halves meet along `axis` (1 = rows, 2 = columns), or None.

    The halves show different pictures, so the line between them is a hard edge in every frame;
    in an ordinary video no edge sits in the same place all the time."""
    step = np.abs(np.diff(lum, axis=axis)).mean(axis=3 - axis)      # (frames, lines - 1)
    profile = np.median(step, axis=0)
    lo, hi = int(0.4 * len(profile)), int(0.6 * len(profile)) + 1
    i = lo + int(np.argmax(profile[lo:hi]))
    return i + 1 if profile[i] > max(10.0, 8.0 * float(np.median(profile))) else None


def _film_score(part):
    """How much a picture looks like filmed footage rather than a render: filmed frames use many
    more distinct colours and have far fewer flat, untextured areas."""
    q = (part // 16).astype(np.int32)
    colours = np.mean([len(np.unique(c)) for c in q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]])
    flat = []
    for lum in part.mean(axis=3):
        mean = cv2.blur(lum, (5, 5))
        spread = np.sqrt(np.maximum(cv2.blur(lum * lum, (5, 5)) - mean * mean, 0))
        lit = lum > 30
        flat.append((spread[lit] < 1.5).mean() if lit.any() else 0.0)
    return colours * (1.0 - float(np.mean(flat)))


def find_picture(src, meta):
    """Where the picture is inside the frame, as an ffmpeg crop "w:h:x:y", or None if it fills it.

    Black bars (letterbox, pillarbox, the empty top and bottom of portrait uploads) are left out.
    In a comparison video (an original above, below or beside its white model) only the original
    half is kept.
    """
    w, h = meta["width"], meta["height"]
    k = 320 / max(w, h)
    sw, sh = max(2, round(w * k / 2) * 2), max(2, round(h * k / 2) * 2)
    rate = SAMPLES / max(meta["duration"], 1.0)
    raw = subprocess.run([tool("ffmpeg"), "-v", "error", "-i", str(src), "-vf",
                          f"fps={rate:.4f},scale={sw}:{sh}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8)[: len(raw) // (sw * sh * 3) * sw * sh * 3]
    frames = frames.reshape(-1, sh, sw, 3).astype(np.float32)
    if len(frames) < 2:
        return None
    lum = frames.mean(axis=3)
    content = (lum.max(axis=0) > 24) & (lum.std(axis=0) > 3)       # ever lit, and changing
    if content.mean() < 0.05:                                       # a still picture: anything not black
        content = lum.max(axis=0) > 24
    y0, y1 = _longest_run(content.mean(axis=1) > 0.2, gap=max(1, sh // 100))
    x0, x1 = _longest_run(content[y0:y1].mean(axis=0) > 0.2, gap=max(1, sw // 100))
    if y1 - y0 < 0.1 * sh or x1 - x0 < 0.1 * sw:
        return None
    pic = frames[:, y0:y1, x0:x1]
    for axis in (1, 2):
        split = _seam(pic.mean(axis=3), axis)
        if split is None:
            continue
        first, second = np.split(pic, [split], axis=axis)
        a, b = _film_score(first), _film_score(second)
        if max(a, b) < 1.8 * min(a, b):     # both halves look filmed: one picture with a hard line in it
            continue
        # Keep the filmed half, minus the line where the halves meet.
        lo, hi = (y0, y1) if axis == 1 else (x0, x1)
        lo, hi = (lo, lo + split - 1) if a > b else (lo + split + 1, hi)
        if axis == 1:
            y0, y1 = lo, hi
        else:
            x0, x1 = lo, hi
        break
    # Back to source pixels, inside the picture, with even sizes.
    left, top = int(np.ceil(x0 / k)), int(np.ceil(y0 / k))
    cw = (min(w, int(x1 / k)) - left) // 2 * 2
    ch = (min(h, int(y1 / k)) - top) // 2 * 2
    if cw >= 0.97 * w and ch >= 0.97 * h:
        return None
    return f"{cw}:{ch}:{left}:{top}"


def ingest(src, job_dir, crop=None):
    """Stage 1: validate the upload and write source.mp4 (constant fps, even size, H.264 + AAC).

    crop is an ffmpeg crop expression "w:h:x:y" applied before anything else; when it is not
    given, the picture area is found automatically (see find_picture). The crop used is returned
    with the probed source.
    """
    src = Path(src)
    size_mb = src.stat().st_size / 2**20
    if size_mb > LIMITS["max_upload_mb"]:
        raise ValueError(f"File is {size_mb:.0f} MB; the limit is {LIMITS['max_upload_mb']} MB.")
    try:
        meta = probe(src)
    except RuntimeError:
        raise ValueError("This file could not be read as a video.") from None
    if meta["duration"] > LIMITS["max_duration_s"]:
        raise ValueError(f"Video is {meta['duration']:.0f} s; the limit is {LIMITS['max_duration_s']} s.")
    # Compared long side to long side so portrait videos get the same allowance as landscape.
    size, cap = sorted((meta["width"], meta["height"])), sorted((LIMITS["max_width"], LIMITS["max_height"]))
    if size[0] > cap[0] or size[1] > cap[1]:
        raise ValueError(f"Video is {meta['width']}x{meta['height']}; the limit is "
                         f"{LIMITS['max_width']}x{LIMITS['max_height']}.")
    if crop is None:
        crop = find_picture(src, meta)
    filters = ([f"crop={crop}"] if crop else []) + ["scale=trunc(iw/2)*2:trunc(ih/2)*2", "setsar=1"]
    out = Path(job_dir) / "source.mp4"
    _ffmpeg("-i", src, "-map", "0:v:0", "-map", "0:a:0?", "-vf", ",".join(filters),
            "-r", meta["fps"], "-fps_mode", "cfr", *H264, "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", out)
    return {**probe(out), "crop": crop}


def output_size(meta):
    """Size of each half of the final video: the source size, enlarged so the long side is at
    least 1280 pixels. Small uploads would otherwise give a coarse white-model render."""
    k = max(1.0, 1280 / max(meta["width"], meta["height"]))
    return {**meta, "width": round(meta["width"] * k / 2) * 2, "height": round(meta["height"] * k / 2) * 2}


def analyse(job_dir):
    """Stage 2: write sets/NNN.npy and scene.json for source.mp4 (separate process, see analyse.py)."""
    _run([sys.executable, "-m", "pipeline.analyse", str(job_dir)], cwd=ROOT)


def build_scene(job_dir, meta):
    """Stage 3: build scene.blend: the sets, a figure per person and the moving camera."""
    job_dir = Path(job_dir)
    _run([tool("blender"), "-b", *BLENDER_FLAGS, "-P", str(BLENDER_SCRIPTS / "build_scene.py"), "--",
          "--scene", str(job_dir / "scene.json"), "--sets-dir", str(job_dir / "sets"),
          "--width", str(meta["width"]), "--height", str(meta["height"]),
          "--whitebox", WHITEBOX_SCRIPTS, "--out", str(job_dir / "scene.blend")])


def render(blend, out, meta):
    """Stage 4: render a .blend with Blender at meta's width, height, fps and frame count."""
    _run([tool("blender"), "-b", str(blend), *BLENDER_FLAGS,
          "-P", str(BLENDER_SCRIPTS / "render_shot.py"), "--", "--out", str(out), "--width", str(meta["width"]),
          "--height", str(meta["height"]), "--fps", meta["fps"], "--frames", str(meta["frames"])])


def compose(job_dir):
    """Stage 5: stack source.mp4 (scaled to the render's size) over white.mp4, keep the audio -> final.mp4."""
    job_dir = Path(job_dir)
    top, bottom = probe(job_dir / "source.mp4"), probe(job_dir / "white.mp4")
    for key in ("fps", "frames"):
        if top[key] != bottom[key]:
            raise RuntimeError(f"white.mp4 does not match source.mp4: {key} {bottom[key]} != {top[key]}")
    out = job_dir / "final.mp4"
    _ffmpeg("-i", job_dir / "source.mp4", "-i", job_dir / "white.mp4", "-filter_complex",
            f"[0:v]scale={bottom['width']}:{bottom['height']}:flags=lanczos,setsar=1[t];[t][1:v]vstack=inputs=2[v]",
            "-map", "[v]", "-map", "0:a?", *H264, "-c:a", "copy", "-movflags", "+faststart", out)
    return probe(out)
