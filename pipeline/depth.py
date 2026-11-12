"""Stage 3: source.mp4 -> depth/0001.png ... (16-bit greyscale, white = near) and figures.json
(tracked people with 2D poses and their depth).

Run as its own process so GPU memory is released before Blender starts:
    python -m pipeline.depth JOB_DIR
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from . import figures
from .config import tool
from .media import probe

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"  # Apache-2.0; the Base/Large weights are non-commercial
STORE_WIDTH = 512   # depth maps are kept at this width; the relief mesh is coarser than that anyway
BATCH = 8
RANGE_WINDOW = 31   # frames over which the near/far range is averaged (adapts to each shot)
SMOOTH = 0.5        # weight of the previous frame in the temporal average (reduces flicker)


def lift_out(d, pose, scale):
    """Return the person's depth and press their bump in the depth map d down to the background,
    so the relief shows the set and the Blender figure stands where the person was."""
    x0, y0, x1, y1 = (int(round(v * scale)) for v in pose["box"])
    region = d[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1]
    if region.size == 0:
        return 0.0
    body = [d[min(max(int(y * scale), 0), d.shape[0] - 1), min(max(int(x * scale), 0), d.shape[1] - 1)]
            for x, y, score in pose["kp"][5:13] if score >= figures.MIN_KEYPOINT]
    z = float(np.median(body)) if body else float(np.percentile(region, 75))
    # Background level = what surrounds the box at the top and sides (the bottom is usually floor).
    background = float(np.median(np.concatenate([region[0], region[:, 0], region[:, -1]])))
    if z > background + 0.03:
        region[region > background + 0.25 * (z - background)] = background
    return z


def main(job_dir):
    src = job_dir / "source.mp4"
    meta = probe(src)
    w, h = meta["width"], meta["height"]
    sw = min(STORE_WIDTH, w)
    sh = max(2, round(h * sw / w))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(device).eval()
    find_people = figures.PoseEstimator(device)

    # Pass 1: raw model output per frame (relative inverse depth: larger = nearer).
    reader = subprocess.Popen([tool("ffmpeg"), "-v", "error", "-i", str(src),
                               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    raw, batch, people = [], [], []

    def flush():
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.inference_mode():
            pred = model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(pred[:, None], size=(sh, sw), mode="bicubic",
                                               align_corners=False)[:, 0]
        raw.extend(pred.float().cpu().numpy().astype(np.float16))
        people.extend(find_people(batch))
        batch.clear()

    while len(buf := reader.stdout.read(w * h * 3)) == w * h * 3:
        batch.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()
    if reader.wait() or not raw:
        raise RuntimeError("Could not decode frames from source.mp4.")

    # Pass 2: normalise with a near/far range averaged over neighbouring frames (a per-frame
    # range would pulse; one range for the whole clip flattens shots with little depth),
    # smooth over time, then write the PNG sequence.
    ranges = np.array([np.percentile(f.astype(np.float32), (1, 99)) for f in raw])
    pad = RANGE_WINDOW // 2
    kernel = np.ones(RANGE_WINDOW) / RANGE_WINDOW
    lows, highs = (np.convolve(np.pad(col, pad, mode="edge"), kernel, mode="valid") for col in ranges.T)
    out_dir = job_dir / "depth"
    out_dir.mkdir(exist_ok=True)
    writer = subprocess.Popen([tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo",
                               "-pix_fmt", "gray16le", "-s", f"{sw}x{sh}", "-i", "-",
                               "-start_number", "1", str(out_dir / "%04d.png")], stdin=subprocess.PIPE)
    tracks = figures.track(people)
    on_frame = [[] for _ in raw]
    for t in tracks:
        for offset, pose in enumerate(t["poses"]):
            if pose:
                on_frame[t["start"] + offset].append(pose)
    prev = None
    for frame, lo, hi, poses in zip(raw, lows, highs, on_frame):
        d = np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        for pose in poses:
            pose["z"] = lift_out(d, pose, sw / w)
        prev = d if prev is None else SMOOTH * prev + (1 - SMOOTH) * d
        writer.stdin.write((prev * 65535).astype("<u2").tobytes())
    writer.stdin.close()
    if writer.wait():
        raise RuntimeError("Could not write the depth frames.")
    for t in tracks:    # steady each figure's depth over time, then make it JSON-friendly
        z = None
        for pose in filter(None, t["poses"]):
            z = pose["z"] if z is None else SMOOTH * z + (1 - SMOOTH) * pose["z"]
            pose.update(z=round(float(z), 4), kp=np.round(pose["kp"], 2).tolist())
            del pose["box"]
        t["start"] += 1  # Blender frames start at 1
    (job_dir / "figures.json").write_text(json.dumps({"min_keypoint": figures.MIN_KEYPOINT, "tracks": tracks}), encoding="utf-8")
    print(f"depth: {len(raw)} frames at {sw}x{sh} on {device}, {len(tracks)} figure(s)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
