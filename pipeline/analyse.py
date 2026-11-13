"""Stage 2: understand the video. source.mp4 -> scene.json

scene.json describes the white model in metres, in the camera's own space (x right, y forward,
z up): for every frame the box room that stands in for the set, and the 3D joints of every
tracked person.

Run as its own process so GPU memory is released before Blender starts:
    python -m pipeline.analyse JOB_DIR
"""
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from . import depth as dp
from . import figures, room
from .config import tool
from .media import probe

BATCH = 8
MIN_SHOT = 8            # frames; shorter stretches are merged into the previous shot
SMOOTH = 0.3            # weight of the previous frame's depth (reduces shimmer)
LENS = 43.0             # assumed lens in mm on a 36 mm wide sensor (about 45 degrees across)
TORSO_METRES = 0.52     # shoulder line to hip line of an adult; sets the scale of each shot
DEFAULT_DEPTH = 4.0     # metres to the typical surface when no person gives the scale


def read_frames(src, width, height):
    """Yield the frames of src as (height, width, 3) uint8 arrays."""
    proc = subprocess.Popen([tool("ffmpeg"), "-v", "error", "-i", str(src), "-vf", f"scale={width}:{height}",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = width * height * 3
    while len(buf := proc.stdout.read(size)) == size:
        yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    proc.wait()


def find_cuts(src):
    """Frame indices where a new shot starts (always includes 0).

    A cut is a frame that no longer correlates with the one before it. Plain brightness
    difference is not enough: it also fires on fast motion and misses cuts between dark shots.
    """
    t = np.stack([f.mean(axis=2) for f in read_frames(src, 64, 36)]).astype(np.float32)
    z = t - t.mean(axis=(1, 2), keepdims=True)
    norm = np.sqrt((z * z).sum(axis=(1, 2))) + 1e-6
    ncc = (z[1:] * z[:-1]).sum(axis=(1, 2)) / (norm[1:] * norm[:-1])
    change = np.abs(t[1:] - t[:-1]).mean(axis=(1, 2))
    cuts = [0]
    for i in np.flatnonzero((ncc < 0.3) & (change > 8)) + 1:
        if i - cuts[-1] >= MIN_SHOT and len(t) - i >= MIN_SHOT:
            cuts.append(int(i))
    return cuts, len(t)


def shot_figures(people, width, height):
    """Tracks for one shot -> (scale to metres, figures with camera-space joints in metres)."""
    focal = LENS / 36.0 * width                       # pixels
    tracks = figures.track(people)
    torsos, depths = [], []
    for t in tracks:
        z = None
        for pose in filter(None, t["poses"]):         # steady each person's depth over time
            z = pose["z"] if z is None else 0.7 * z + 0.3 * pose["z"]
            pose["z"] = z
            kp = pose["kp"]
            depths.append(z)
            if (kp[[5, 6, 11, 12], 2] >= figures.MIN_KEYPOINT).all():
                torso = np.linalg.norm((kp[5, :2] + kp[6, :2]) / 2 - (kp[11, :2] + kp[12, :2]) / 2)
                torsos.append(torso / focal * z)
    scale = TORSO_METRES / np.median(torsos) if len(torsos) >= 3 else None
    out = []
    for t in tracks:
        joints, sizes = [], []
        for pose in t["poses"]:
            if not pose:
                joints.append(None)
                continue
            kp, z = pose["kp"], pose["z"] * (scale or 1.0)
            cam = np.stack([(kp[:, 0] - width / 2) / focal * z, np.full(len(kp), z),
                            -(kp[:, 1] - height / 2) / focal * z], axis=1)
            seen = kp[:, 2] >= figures.MIN_KEYPOINT
            joints.append([p if ok else None for p, ok in zip(np.round(cam, 4).tolist(), seen)])
            if seen[5] and seen[6]:
                face = cam[:5][seen[:5]]
                spread = max((np.linalg.norm(a - b) for a in face for b in face), default=0.0)
                torso = (0.55 * np.linalg.norm((cam[5] + cam[6]) / 2 - (cam[11] + cam[12]) / 2)
                         if seen[11] and seen[12] else 0.0)
                sizes.append(max(np.linalg.norm(cam[5] - cam[6]), torso, 2.2 * spread))
        if sizes:
            out.append({"start": t["start"], "size": float(np.median(sizes)), "joints": joints})
    return scale, out


def main(job_dir):
    src = job_dir / "source.mp4"
    meta = probe(src)
    width, height = meta["width"], meta["height"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuts, total = find_cuts(src)
    if not total:
        raise RuntimeError("Could not decode frames from source.mp4.")

    find_people = figures.PoseEstimator(device)
    estimate = dp.DepthEstimator(device)
    people, fits, medians, batch, previous = [], [], [], [], None

    def flush():
        nonlocal previous
        for frame, found in zip(batch, find_people(batch)):
            i = len(people)
            depth = estimate(frame)
            if previous is not None and i not in cuts and previous.shape == depth.shape:
                depth = SMOOTH * previous + (1 - SMOOTH) * depth
            previous = depth.copy()
            sy, sx = depth.shape[0] / height, depth.shape[1] / width
            for person in sorted(found, key=lambda p: -p["box"][3]):     # nearest (lowest feet) first
                person["z"] = dp.lift_out(depth, person, sx, sy)
            fits.append(room.fit(depth, LENS / 36.0))
            medians.append(float(np.median(depth)))
            people.append(found)
        batch.clear()

    for frame in read_frames(src, width, height):
        batch.append(frame)
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()

    shots, all_figures, rooms = [], [], {key: [] for key in ("theta", *room.KEYS)}
    fps = float(Fraction(meta["fps"]))
    for s, e in zip(cuts, cuts[1:] + [len(people)]):
        scale, figs = shot_figures(people[s:e], width, height)
        if scale is None:       # nobody to measure against: put the typical surface at a set distance
            scale = DEFAULT_DEPTH / float(np.median(medians[s:e]))
            for f in figs:
                f["size"] *= scale
                f["joints"] = [j and [p and [c * scale for c in p] for p in j] for j in f["joints"]]
        shots.append({"start": s + 1, "end": e})     # Blender frames start at 1
        shot_room = room.smooth(fits[s:e], scale, fps)
        per_frame = [[] for _ in range(e - s)]
        for f in figs:
            for i, pose in enumerate(f["joints"], f["start"]):
                per_frame[i] += [p for p in pose or [] if p]
        room.contain(shot_room, [np.array(pts, dtype=float).reshape(-1, 3) for pts in per_frame], fps)
        for key, series in shot_room.items():
            rooms[key] += series if series is not None else [None] * (e - s)
        all_figures += [{**f, "start": f["start"] + s + 1} for f in figs]
    scene = {"lens": LENS, "frames": len(people), "shots": shots, "room": rooms, "figures": all_figures}
    (job_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    print(f"analyse: {len(people)} frames, {len(shots)} shot(s), {len(all_figures)} figure(s) on {device}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
