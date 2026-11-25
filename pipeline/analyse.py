"""Stage 2: understand the video. source.mp4 -> sets/NNN.npy and scene.json

The camera. Features on the background are followed from frame to frame, and the camera's
position and direction in every frame of a shot are found from them (camera.py).

The set. The depth (Depth Anything 3) of many frames, placed where the camera was, is laid
together into a few still sets per shot, turned into flat walls, floors and steps with the
people taken out, and left empty where too dark to see (planes.py). Blender shapes each set from
sets/NNN.npy (inverse depth on a picture seen from one camera), and the camera moves through them.

The people. Everyone is detected, outlined and followed through the shot, and one mannequin of
fixed proportions is posed in every frame to match them (mannequin.py). scene.json holds, in
metres in each shot's space (its first camera's: x right, y forward, z up), where the camera,
each set and each part of each mannequin is and how it is turned in every frame.

Run as its own process so GPU memory is released before Blender starts:
    python -m pipeline.analyse JOB_DIR
"""
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

from . import camera, figures, mannequin, planes
from . import depth as dp
from .config import tool
from .media import probe

BATCH = 8               # frames per batch for the person detector
MIN_SHOT = 8            # frames; shorter stretches are merged into the previous shot
LENS = 43.0             # assumed lens in mm on a 36 mm wide sensor (about 45 degrees across)
DEFAULT_DEPTH = 4.0     # metres to the typical surface when no person gives the scale
OPEN_EDGE = 0.01        # a person whose box ends this near the frame's bottom goes on out of view
BELOW = (0.01, 0.05)    # what the band this far below a person's box (shares of the frame height) shows
OCCLUDER = 0.8          # is the floor they stand on, unless nearer than this share of their depth (and,
                        # to stand on it, they are as deep as it is, unless it lies further back than
                        # this share of their depth again: a floor far below a ledge)
BRINK = 0.12            # a jump in depth between neighbouring rows of the band below someone's box is the edge of
                        # what they stand on
GROUND_SPAN = 2.0       # the set under someone's feet is what they stand on only within this factor of their own
                        # depth: further back, it is the floor beyond what hides their feet (the dark, a ledge)
GROUND_STEADY = 4       # frames: how deep the set is under someone's feet is the median of what was found this many
                        # frames either side (one frame's can be well off: a band below a box at the frame's edge)
FLOOR_SPAN = 25         # frames: the floor is taken as hidden where most of the frames around that can tell say a
                        # desk hides it (a desk's or counter's top just below someone behind it can pass for a floor)
TOUCH = 0.02            # boxes this near each other (share of the frame) count as overlapping; and someone whose box
                        # reaches this much further down the picture than another's is in front of them (where the
                        # depth model cannot tell)
NEARER = 0.1            # someone this share nearer than another (by the depth model) is in front of them
PAST = 0.03             # a box's edge is not where the body ends where the person's own keypoints lie this share of the
                        # box's size past it: a box can stop short of a blurred arm flung out
SURE = 0.5              # keypoints the pose model scores at least this are where the picture shows the person ...
FRONT = 0.05            # ... and the set there is behind them: they stand at least this share of its depth in front
BEYOND = 0.1            # of it, and a floor more than this share further back than it is no floor they stand on
PROP_STEADY = 5         # frames: the path of a thing someone carries is the median of where it was found over this
                        # many (its depth is measured afresh in every frame; a hold stays a hold)
PROP_JUMP = 1.5         # a thing found this many times nearer or further than the median of where it was found
                        # around it is something else boxed with it (the floor beyond a bag): as if not found
# From the camera's space (x right, y down, z forward) to the scene's (x right, y forward, z up).
TO_SCENE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def read_frames(src, width, height):
    """Yield the frames of src as (height, width, 3) uint8 arrays."""
    proc = subprocess.Popen([tool("ffmpeg"), "-v", "error", "-i", str(src), "-vf", f"scale={width}:{height}",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = width * height * 3
    try:
        while len(buf := proc.stdout.read(size)) == size:
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:                # also when the caller stops early
        proc.kill()
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


class Store:
    """Per-frame depth and masks, kept on disk while the video is analysed (too big for memory)."""

    def __init__(self, folder, frames, h, w):
        self.w = w
        self.depth = np.lib.format.open_memmap(folder / "depth.npy", "w+", np.float32, (frames, h, w))
        self.hole = np.lib.format.open_memmap(folder / "hole.npy", "w+", np.uint8, (frames, h, (w + 7) // 8))
        self.sky = np.lib.format.open_memmap(folder / "sky.npy", "w+", np.uint8, (frames, h, (w + 7) // 8))
        self.dark = np.lib.format.open_memmap(folder / "dark.npy", "w+", np.uint8, (frames, h, (w + 7) // 8))
        self.grey = np.lib.format.open_memmap(folder / "grey.npy", "w+", np.uint8, (frames, h, w))

    def put(self, i, depth=None, hole=None, sky=None, dark=None, grey=None):
        if depth is not None:
            self.depth[i] = depth
        if grey is not None:
            self.grey[i] = grey
        if hole is not None:
            self.hole[i] = np.packbits(hole, axis=1)
        if sky is not None:
            self.sky[i] = np.packbits(sky, axis=1)
        if dark is not None:
            self.dark[i] = np.packbits(dark, axis=1)

    def get(self, i):
        """(depth, hole, sky) of frame i."""
        return (np.array(self.depth[i]), np.unpackbits(self.hole[i], axis=1, count=self.w).astype(bool),
                np.unpackbits(self.sky[i], axis=1, count=self.w).astype(bool))

    def get_dark(self, i):
        """Where frame i is too dark to see (see depth.unlit)."""
        return np.unpackbits(self.dark[i], axis=1, count=self.w).astype(bool)

    def get_grey(self, i):
        """Frame i in grey levels (uint8), at the depth maps' size."""
        return np.array(self.grey[i])

    def close(self):
        del self.depth, self.hole, self.sky, self.dark, self.grey
        gc.collect()


def measure(person, mask, depth, width, height):
    """Adds the person's (or thing's) depth ("z", median over their outline) and the depth at each
    of their keypoints ("jz", nan where unknown), in the depth model's units. mask is frame-sized."""
    h, w = depth.shape
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    core = cv2.erode(m, np.ones((7, 7), np.uint8)).astype(bool)     # depth maps blur across outlines
    if core.sum() < 50:
        core = m.astype(bool)
    person["z"] = float(np.median(depth[core])) if core.any() else float("nan")
    jz = np.full(17, np.nan, np.float32)
    for j, (x, y, score) in enumerate(person["kp"] if person.get("kp") is not None else []):
        if score >= mannequin.MIN_SCORE:
            x, y = int(x * w / width), int(y * h / height)
            window = (slice(max(0, y - 4), min(h, y + 5)), slice(max(0, x - 4), min(w, x + 5)))
            values = depth[window][core[window]]
            if values.size:
                jz[j] = np.median(values)
    person["jz"] = jz


def floor_line(pose, depth, others, width, height):
    """How far down the picture the person's body may reach (y in pixels, or None for no limit),
    the depth of the floor they stand on and where their feet are in the picture (each None
    where not seen), and whether a desk hides the floor (True; False where the floor is seen,
    None where it cannot be told). Where another person's box covers the band below their box,
    that person is in front and hides it, even where its depth would pass for the floor (their
    own depth, blurred with that person's, can be too near): their body reaches no further down
    than that person's box (being farther away, their own feet are higher up the picture). Where
    the band shows the floor, nothing hides their feet: they are at the box's bottom, and they
    stand as deep as that floor. Where their box reaches the frame's edge, or something else in
    front of them hides what is below it (a desk), their legs may be anywhere down there.
    others: the other people's boxes."""
    x0, y0, x1, y1 = pose["box"]
    if y1 >= (1 - OPEN_EDGE) * height:
        return None, None, None, None
    x, y = (x0 + x1) / 2, y1 + BELOW[0] * height
    front = [b[3] for b in others if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
    if front:
        return float(max(front)), None, None, None
    z = pose.get("z", float("nan"))
    if not np.isfinite(z):
        return None, None, None, None
    h, w = depth.shape
    r0, r1 = int((y1 + BELOW[0] * height) * h / height), int(np.ceil((y1 + BELOW[1] * height) * h / height))
    c0, c1 = int((x0 + 0.25 * (x1 - x0)) * w / width), int(np.ceil((x1 - 0.25 * (x1 - x0)) * w / width))
    band = np.asarray(depth[max(0, r0):min(h, r1), max(0, c0):min(w, c1)])
    band = band[np.isfinite(band) & (band > 0)]
    if not band.size:
        return None, None, None, None
    floor = float(np.median(band))
    if floor >= OCCLUDER * z:
        return float(y1), (floor if floor * OCCLUDER <= z else None), float(y1), False
    return None, None, None, True


def steady_floor(lines):
    """floor_line's findings over one track's frames, with the floor taken as hidden by a desk
    where most of the frames around it (FLOOR_SPAN) that can tell say so: the top of a desk or
    counter seen just below someone standing behind it can pass for the floor for a moment."""
    desk = np.array([np.nan if ln[3] is None else float(ln[3]) for ln in lines])
    half = FLOOR_SPAN // 2
    out = []
    for k, ln in enumerate(lines):
        around = desk[max(0, k - half):k + half + 1]
        around = around[np.isfinite(around)]
        out.append((None, None, None, True) if ln[3] is False and around.mean() > 0.5 else ln)
    return out


def in_front(other, pose, height):
    """Whether the other person is in front of this one: nearer by the depth model (see NEARER), or
    where it has no depth for either, reaching further down the picture (see TOUCH)."""
    if np.isfinite(other.get("z", np.nan)) and np.isfinite(pose.get("z", np.nan)):
        return other["z"] < (1 - NEARER) * pose["z"]
    return other["box"][3] > pose["box"][3] + TOUCH * height


def own_keypoints(pose, ahead):
    """The person's keypoints, with those inside the box of anyone in front of them (ahead: their
    boxes) scored 0: given a box that takes in part of that person, the pose model can put joints
    on them."""
    if pose.get("kp") is None:
        return None
    kp = np.array(pose["kp"], np.float32)
    for b in ahead:
        kp[(kp[:, 0] >= b[0]) & (kp[:, 0] <= b[2]) & (kp[:, 1] >= b[1]) & (kp[:, 1] <= b[3]), 2] = 0.0
    return kp


def box_edges(pose, others, width, height):
    """Where the person's body ends in the picture, at the top and to either side ("top", "left",
    "right": y or x in pixels): their box's edges, each unless the frame's edge cuts it off, or
    another person's box overlaps it there, or nearly (who may hide them there, or whom the box
    may have taken in: then both boxes can end alike), or their own keypoints (those the mannequin
    follows) lie more than PAST past it; then None. others: the other people's boxes."""
    x0, y0, x1, y1 = pose["box"]
    m, n = TOUCH * height, TOUCH * width
    across = [b for b in others if b[0] < x1 and x0 < b[2]]
    down = [b for b in others if b[1] < y1 and y0 < b[3]]
    kp = np.zeros((0, 3)) if pose.get("kp") is None else np.asarray(pose["kp"], np.float64)
    kp = kp[kp[:, 2] >= mannequin.MIN_SCORE]
    out_x, out_y = PAST * (x1 - x0), PAST * (y1 - y0)
    return {"top": None if y0 <= OPEN_EDGE * height or any(b[1] - m <= y0 <= b[3] + m for b in across)
            or (kp[:, 1] < y0 - out_y).any() else float(y0),
            "left": None if x0 <= OPEN_EDGE * width or any(b[0] - n <= x0 <= b[2] + n for b in down)
            or (kp[:, 0] < x0 - out_x).any() else float(x0),
            "right": None if x1 >= (1 - OPEN_EDGE) * width or any(b[0] - n <= x1 <= b[2] + n for b in down)
            or (kp[:, 0] > x1 + out_x).any() else float(x1)}


def _sight(xs, ys, z, canvas, centre, rel, f):
    """How deep the set is where the frame's lines of sight through (xs, ys) first meet it (in the
    frame's camera, shot units; nan where none does), looked for from a quarter to four times z.
    xs, ys: the lines' directions, (pixel - middle of the depth map) / f; f: the depth maps' focal
    length in their pixels. canvas, centre, rel: as for on_set."""
    rays = np.stack((xs, ys, np.ones_like(xs)), -1).reshape(-1, 1, 3)
    d = 1.0 / np.linspace(4.0 / z, 0.25 / z, 400)        # along each line of sight, near to far
    R, T, _ = rel
    p = (rays * d[None, :, None] - np.asarray(T)) @ np.asarray(R)          # in the keyframe's camera
    ahead = p[..., 2] > 1e-6
    pz = np.where(ahead, p[..., 2], 1.0)
    i = np.round(f * p[..., 1] / pz + centre[1] - 0.5).astype(np.int64)
    j = np.round(f * p[..., 0] / pz + centre[0] - 0.5).astype(np.int64)
    inside = ahead & (i >= 0) & (i < canvas.shape[0]) & (j >= 0) & (j < canvas.shape[1])
    q = np.where(inside, canvas[np.clip(i, 0, canvas.shape[0] - 1), np.clip(j, 0, canvas.shape[1] - 1)], 0.0)
    behind = np.where(q > 0, q * pz - 1.0, -1.0)    # >= 0 where the line of sight is at or behind the set
    cross = (behind[:, 1:] >= 0) & (behind[:, :-1] < 0)
    k = np.argmax(cross, axis=1)
    n = np.arange(len(k))
    a, b = behind[n, k], behind[n, k + 1]
    meet = np.where(cross[n, k], d[k] + (d[k + 1] - d[k]) * -a / np.where(b > a, b - a, 1.0), np.nan)
    return meet.reshape(np.shape(xs))


def set_behind(pose, z, canvas, centre, rel, size, width, height):
    """How deep the set lies behind a person where the picture shows them (in the frame's camera,
    shot units), or None where it cannot be told: the median of where the lines of sight through
    their sure keypoints (SURE) meet it. Their outline was cut out of the set and closed with what
    is around them, so whatever the set shows there is behind them. z, canvas, centre, rel, size:
    as for on_set."""
    kp = np.asarray(pose["kp"], np.float64)
    sure = kp[:, 2] >= SURE
    if sure.sum() < 3:
        return None
    h, w = size
    f = LENS / 36.0 * w
    meet = _sight((kp[sure, 0] * w / width - w / 2) / f, (kp[sure, 1] * h / height - h / 2) / f,
                  z, canvas, centre, rel, f)
    meet = meet[np.isfinite(meet)]
    return float(np.median(meet)) if meet.size >= 3 else None


def on_set(box, z, canvas, centre, rel, size, width, height):
    """How deep the set lies under a person's feet (in the frame's camera, shot units), or None
    where it cannot be told: the set as the frame's camera sees it through the band below their
    box (see floor_line), carried on up to the box's bottom, where their feet are (on a floor
    exactly, on a stair along its slope), but not across an edge (see BRINK). Figures stand on
    the set they are shown in: the frame's own depth can put the floor well in front of it (the
    steps below someone on a stair). But not where it lies more than GROUND_SPAN times deeper
    (or nearer) than they are: there it is no floor they stand on, but one beyond what hides
    their feet (in the dark, the floor far down a corridor).
    z: about how deep the person is (looked for from a quarter to four times that); canvas: the
    set's inverse depth (see build_sets), centre its principal point; rel: (R, T, factor) of the
    frame relative to the set's keyframe (see camera.relative); size: (h, w) of the depth maps,
    whose pixels the set's picture has."""
    h, w = size
    f = LENS / 36.0 * w
    x0, _, x1, y1 = box
    rows = np.linspace(y1 + BELOW[0] * height, y1 + BELOW[1] * height, 6) * h / height
    cols = np.linspace(x0 + 0.25 * (x1 - x0), x1 - 0.25 * (x1 - x0), 6) * w / width
    ys, xs = np.meshgrid((rows - h / 2) / f, (cols - w / 2) / f, indexing="ij")
    inv = 1.0 / _sight(xs, ys, z, canvas, centre, rel, f)
    # Down each column the surface runs on from row to row until an edge (a jump in depth of more
    # than BRINK). Of the pieces that leaves, the nearest is what they stand on (a step's riser
    # just below their tread, the top of a step seen from above); what lies further back is the
    # floor beyond the step's edge, or the hole they left in the set, closed with what was behind.
    for c in range(inv.shape[1]):
        col = inv[:, c]
        piece = np.cumsum(np.r_[0, ~(np.abs(np.log(col[1:] / col[:-1])) <= np.log(1 + BRINK))])
        seen = np.isfinite(col)
        if seen.any():
            inv[piece != piece[seen][np.argmax(col[seen])], c] = np.nan
    per_row = np.array([np.median(r[np.isfinite(r)]) if np.isfinite(r).sum() >= 3 else np.nan for r in inv])
    ok = np.isfinite(per_row)
    if ok.sum() < 3:
        return None
    slope, at = np.polyfit(rows[ok], per_row[ok], 1)
    feet = slope * y1 * h / height + at
    if feet <= 0 or abs(np.log(feet * z)) > np.log(GROUND_SPAN):
        return None
    return float(1.0 / feet)


def look(src, meta, cuts, total, work, device):
    """First pass: depth, sky, people, the things they carry and background features for every
    frame.

    Depth and masks (people, and things where someone holds them, are taken out of the set) go to a
    Store. Also returns the people and the things (see figures.PROPS) found in each frame, the
    background features followed through the video (camera.Tracker) and the outlining model, which
    is needed once more."""
    width, height = meta["width"], meta["height"]
    find_people = figures.PoseEstimator(device)
    outline = figures.Segmenter(device)
    estimate = dp.DepthEstimator(device)
    store, people, things, batch = None, [], [], []
    tracker = camera.Tracker()

    def flush():
        nonlocal store
        for frame, found, items in zip(batch, *find_people(batch)):
            i = len(people)
            depth, sky = estimate(frame)
            h, w = depth.shape
            if store is None:
                store = Store(work, total, h, w)
            outline(frame, found + items)
            masks = [p.pop("mask") for p in found + items]
            for p, m in zip(found + items, masks):
                measure(p, m, depth, width, height)
            # (a thing goes out of the set where someone holds it, as it is drawn where it goes; one that
            # nobody holds stays part of the set)
            hole = planes.hole_mask(masks[:len(found)] + [m for p, m in zip(items, masks[len(found):])
                                                         if figures.held(p["box"], found)], depth)
            tracker(planes.motion_view(frame, hole, sky), i in cuts)
            store.put(i, depth, hole, sky, dp.unlit(frame, depth.shape), dp.grey(frame, depth.shape))
            people.append(found)
            things.append(items)
        batch.clear()

    for frame in read_frames(src, width, height):
        if len(people) + len(batch) == total:
            break
        batch.append(frame)
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()
    return store, people, things, tracker, outline


def follow(people, shots, width):
    """Tracks of people (see figures.track), shot by shot so none crosses a cut; "start" is
    the frame number in the whole video."""
    tracks = []
    for s, e in shots:
        tracks += [{**t, "start": s + t["start"]} for t in figures.track(people[s:e], width)]
    return tracks


def follow_things(things, people, shots):
    """Tracks of the things people carry (see figures.track_things), shot by shot as follow's,
    each with the "item" it is (from 1): a thing found again after it was lost (out of the
    picture, or in the dark) or after a cut is the one of its kind (figures.KINDS) last seen,
    unless that one is still in view, so it is drawn as one thing, in one colour."""
    tracks = [{**t, "start": s + t["start"]} for s, e in shots for t in figures.track_things(things[s:e], people[s:e])]
    last = {}                   # item -> its kind, and the last frame it was seen in
    for t in tracks:            # (in the order they start)
        kind = figures.KINDS.get(t["label"], t["label"])
        gone = [k for k, (other, end) in last.items() if other == kind and end < t["start"]]
        t["item"] = max(gone, key=lambda k: last[k][1]) if gone else len(last) + 1
        last[t["item"]] = kind, t["start"] + len(t["poses"]) - 1
    return tracks


def place_props(tracks, factors, meta):
    """Where each thing's track (see follow_things) is in every frame, in set depth units in the
    camera's space (x right, y down, z forward): the middle of its box, as deep as measured inside
    its outline (see measure; between detections, and where one is PROP_JUMP off the depth of
    those around it, box and depth bridged from either side); and how big it is, across and up
    (what its box typically spans at that depth). Returns per track its start, label, item,
    centres (N, 3) and size."""
    width, height = meta["width"], meta["height"]
    focal = LENS / 36.0 * width
    placed = []
    for t in tracks:
        n = len(t["poses"])
        z = np.array([p.get("z", np.nan) for p in t["poses"]], np.float64) / factors[t["start"]:t["start"] + n]
        seen = np.flatnonzero(np.isfinite(z) & (z > 0))
        if not len(seen):
            continue
        around = ndimage.median_filter(z[seen], size=PROP_STEADY, mode="nearest")
        ok = seen[np.abs(np.log(z[seen] / around)) <= np.log(PROP_JUMP)]     # (never empty: see the first)
        box = np.array([p["box"] for p in t["poses"]], np.float64)
        box = np.stack([np.interp(np.arange(n), ok, side) for side in box[ok].T], axis=1)
        z = np.interp(np.arange(n), ok, z[ok])
        u, v = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
        span = (box[:, 2:] - box[:, :2]) * z[:, None] / focal
        placed.append({"start": t["start"], "label": t["label"], "item": t["item"], "size": np.median(span[ok], axis=0),
                       "centres": np.stack(((u - width / 2) / focal * z, (v - height / 2) / focal * z, z), axis=1)})
    return placed


def outline_missed(src, meta, tracks, store, outline):
    """Where the detector missed someone it found before and after, outline them inside the box
    in between, so they are taken out of the set there too, and measure their depth."""
    need = {}
    for t in tracks:
        for k, pose in enumerate(t["poses"]):
            if pose.get("missed"):
                need.setdefault(t["start"] + k, []).append(pose)
    if not need:
        return 0
    width, height = meta["width"], meta["height"]
    for i, frame in enumerate(read_frames(src, width, height)):
        if i > max(need):
            break
        if i not in need:
            continue
        outline(frame, need[i])
        depth, hole, _ = store.get(i)
        found = []
        for pose in need[i]:
            mask = pose.pop("mask")
            x0, y0, x1, y1 = pose["box"]
            if 0.15 <= mask.sum() / max((x1 - x0) * (y1 - y0), 1.0) <= 1.2:    # a person-sized outline
                found.append(mask)
                measure(pose, mask, depth, width, height)
        if found:
            store.put(i, hole=hole | planes.hole_mask(found, depth))
    return sum(map(len, need.values()))


def build_sets(store, shots, tracker, sets_dir, device):
    """Second pass: the camera's path through each shot, and the sets it moves through.

    The camera is placed in every frame (camera.path). Each shot is split into stretches over
    which it stays near one place and direction (camera.stretches); each stretch gets one set,
    built from the depth of frames across it and seen from its keyframe (planes.SetDepth.splat),
    which stands still while the camera moves through it. A set is written as sets_dir/NNN.npy:
    inverse depth in the shot's units on a picture in its keyframe's view, 0 where there is none
    (sky) and -1 where it is left empty for being too dark to see; sets_dir/NNN_edges.npz:
    where the pixels along its outlines lie on the straight lines the outlines were drawn as, and
    which surface each pixel is on (planes.outline_shifts); and sets_dir/NNN_tone.png: the set
    painted in the footage's light and dark (planes.set_tones), in one spread of greys for the shot.
    Returns per shot its camera path (camera poses), the typical distance to what the camera
    sees and its sets; and per frame the factor that took the depth model's scale to the shot's
    (shot depth = model depth / factor)."""
    cleaner = planes.SetDepth(device, LENS / 36.0)
    factors = np.ones(shots[-1][1], np.float32)
    info, number = [], 0
    for s, e in shots:
        n = e - s
        poses, dists = camera.path(tracker, s, e, store.get, LENS / 36.0)
        factors[s:e] = [f for _, _, f in poses]
        sets, greys = [], []
        for a, b in camera.stretches(poses, dists):
            key = camera.keyframe(poses, a, b, lambda t: store.get(s + t), LENS / 36.0)
            # Built from BLEND / 2 frames more on either side, so it matches its neighbours where they meet.
            lo, hi = max(0, a - planes.BLEND // 2), min(n, b + planes.BLEND // 2)
            got = cleaner.splat(lambda t: store.get(s + lo + t), [camera.relative(poses[t], poses[key])
                                                                 for t in range(lo, hi)], key - lo,
                                lambda t: store.get_dark(s + lo + t), lambda t: store.get_grey(s + lo + t))
            if got is None:         # no set could be built here (all people, or a whip pan): none shown
                continue
            depth, centre, grey = got
            np.save(sets_dir / f"{number:03d}.npy", depth)
            at, across, down, moved, piece = planes.outline_shifts(depth)
            np.savez_compressed(sets_dir / f"{number:03d}_edges.npz", at=at, across=across, down=down, q=moved,
                                piece=piece)
            greys.append((number, grey, depth, piece))
            sets.append({"file": f"{number:03d}.npy", "first": s + a, "last": s + b - 1, "centre": centre,
                         "key": poses[key][:2]})
            number += 1
        spread = planes.tone_spread([g for _, g, _, _ in greys])      # (one for the shot's sets)
        for k, grey, q, piece in greys:
            cv2.imwrite(str(sets_dir / f"{k:03d}_tone.png"), planes.set_tones(grey, q, piece, spread))
        info.append({"poses": [p[:2] for p in poses], "typical": float(np.median(dists)), "sets": sets})
    return info, factors


def pose_people(tracks, factors, meta, device, store, shots, sets, sets_dir):
    """Third pass: a mannequin fitted to every track, in set depth units and camera space,
    standing on the set where the floor under it is seen (see floor_line, with the frames' depth
    from store, and on_set, with the sets from build_sets) and filling the person's box (see
    box_edges). Everyone in a shot is an adult of about the same size, as all their keypoints
    together say (see mannequin.sizes): one person's alone can mislead (someone crawling side on
    looks narrow); where a box says otherwise, the body bends to fit it. Returns per track its
    start, scale (see mannequin.fit) and parts (see mannequin.parts)."""
    width, height = meta["width"], meta["height"]
    focal = LENS / 36.0 * width
    seen = {}                   # frame -> everyone in it
    for t in tracks:
        for k, p in enumerate(t["poses"]):
            seen.setdefault(t["start"] + k, []).append(p)
    views, pictures = {}, {}    # frame -> (its set's picture and principal point, the frame's pose relative to it)
    for (s, _), shot in zip(shots, sets):
        for st in shot["sets"]:
            picture = pictures.setdefault(st["file"], np.load(sets_dir / st["file"], mmap_mode="r"))
            for t in range(st["first"], st["last"] + 1):
                views[t] = (picture, st["centre"], camera.relative((*shot["poses"][t - s], factors[t]), (*st["key"], 1.0)))
    ready, found = [], {}       # (track, its frames for mannequin.fit, its shot); shot -> sizes
    for t in tracks:
        n = len(t["poses"])
        f = factors[t["start"]:t["start"] + n]
        others = [[q["box"] for q in seen[t["start"] + k] if q is not p] for k, p in enumerate(t["poses"])]
        ahead = [[q["box"] for q in seen[t["start"] + k] if q is not p and in_front(q, p, height)]
                 for k, p in enumerate(t["poses"])]
        lines = steady_floor([floor_line(p, store.depth[t["start"] + k], others[k], width, height)
                              for k, p in enumerate(t["poses"])])
        # They stand on the set where the floor under them is seen (the frame's own depth puts
        # people a little in front of what they stand on, and the floor below them can lie well in
        # front of the set's: they would float above it), as deep as it is over the frames around
        # (GROUND_STEADY), and where it is not seen, as much deeper than their outline as they stand
        # elsewhere in the track.
        z = np.array([p.get("z", np.nan) for p in t["poses"]], np.float64) / f
        ground = np.array([on_set(p["box"], z[k], *views[t["start"] + k], store.depth.shape[1:], width, height)
                           if lines[k][1] is not None and t["start"] + k in views else None
                           for k, p in enumerate(t["poses"])], np.float64)
        # Where the picture shows them, the set is behind them (see set_behind; where the detector
        # missed them, as it was in the frames around): a floor further back than that is not what
        # they stand on but one far below (someone in mid-air), and they are no deeper than it.
        back = np.array([set_behind(p, z[k], *views[t["start"] + k], store.depth.shape[1:], width, height)
                         if not p.get("missed") and p.get("kp") is not None and np.isfinite(z[k]) and z[k] > 0
                         and t["start"] + k in views else None for k, p in enumerate(t["poses"])], np.float64)
        known = np.flatnonzero(np.isfinite(back))
        if known.size:
            gap = np.arange(n)
            gap = gap[(gap > known[0]) & (gap < known[-1]) & ~np.isfinite(back)]
            back[gap] = np.interp(gap, known, back[known])
        ground[ground > back * (1 + BEYOND)] = np.nan
        ground = np.array([np.nanmedian(ground[max(0, k - GROUND_STEADY):k + GROUND_STEADY + 1])
                           if np.isfinite(g) else np.nan for k, g in enumerate(ground)])
        both = np.isfinite(ground) & np.isfinite(z) & (z > 0)
        if both.any():
            z = np.where(np.isfinite(ground), ground, z * np.median(ground[both] / z[both]))
        z = np.where(~np.isfinite(ground) & (z > back * (1 - FRONT)), back * (1 - FRONT), z)
        ok = np.isfinite(z) & (z > 0)
        if not ok.any():
            continue
        z = np.interp(np.arange(n), np.flatnonzero(ok), z[ok])     # frames where the depth was not measured
        track = [{"kp": own_keypoints(p, ahead[k]), "box": p["box"], "z": z[k],
                  "grounded": bool(np.isfinite(ground[k])),
                  "jz": None if p.get("jz") is None else p["jz"] / f[k], "floor": lines[k][0],
                  "feet": lines[k][2], **box_edges(p, others[k], width, height)}
                 for k, p in enumerate(t["poses"])]
        shot = next(k for k, (s, e) in enumerate(shots) if s <= t["start"] < e)
        ready.append((t, track, shot))
        found.setdefault(shot, []).extend(mannequin.sizes(track, focal, width, height))
    fitted = []
    for t, track, shot in ready:
        size = np.percentile(found[shot], 75) if found[shot] else None      # (see mannequin.fit)
        scale, *pose = mannequin.fit(track, focal, width, height, device, size)
        fitted.append({"start": t["start"], "scale": scale, "parts": mannequin.parts(scale, *pose)})
    return fitted


def quaternions(m):
    """Rotation matrices (N, 3, 3) -> unit quaternions (N, 4) as w, x, y, z, each one on the
    same side as the one before so that nothing turns the long way round between frames."""
    d0, d1, d2 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    root = lambda v: 2.0 * np.sqrt(np.maximum(v, 1e-12))    # noqa: E731
    sw, sx, sy, sz = root(1 + d0 + d1 + d2), root(1 + d0 - d1 - d2), root(1 - d0 + d1 - d2), root(1 - d0 - d1 + d2)
    a, b, c = m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]
    p, q, r = m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1]
    options = np.stack([np.stack((sw / 4, a / sw, b / sw, c / sw), 1), np.stack((a / sx, sx / 4, p / sx, q / sx), 1),
                        np.stack((b / sy, p / sy, sy / 4, r / sy), 1), np.stack((c / sz, q / sz, r / sz, sz / 4), 1)])
    pick = np.argmax(np.stack((d0 + d1 + d2, d0, d1, d2), 1), 1)
    out = options[pick, np.arange(len(m))]
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    for k in range(1, len(out)):
        if out[k] @ out[k - 1] < 0:
            out[k] = -out[k]
    return out


def assemble(shots, info, fitted, depth_shape, frames, props=()):
    """scene.json's contents: everything in metres, in the scene's space (see TO_SCENE), which
    for each shot is the space of its first frame's camera. Blender frames start at 1. props:
    the things people carry (see place_props), each kept upright and turned as the camera is in
    the middle of its track, its path the median of where it was found over PROP_STEADY frames."""
    out_sets, out_figures, out_props, cam_loc, cam_rot = [], [], [], [], []
    focal = LENS / 36.0 * depth_shape[1]                    # in pixels of the sets' pictures
    for number, ((s, e), shot) in enumerate(zip(shots, info)):
        mine = [f for f in fitted if s <= f["start"] < e]
        # The median person in the shot is HEIGHT tall; with nobody to go by, the typical surface
        # is DEFAULT_DEPTH away.
        metres = 1.0 / np.median([f["scale"] for f in mine]) if mine else DEFAULT_DEPTH / shot["typical"]
        turns = np.array([R.T for R, _ in shot["poses"]])          # camera axes -> shot axes
        centres = np.array([-R.T @ T for R, T in shot["poses"]])
        cam_loc += np.round(centres @ TO_SCENE.T * metres, 4).tolist()
        # A Blender camera looks along its -z with y up: OpenCV's camera axes with y and z turned round.
        cam_rot += np.round(quaternions(TO_SCENE @ turns @ np.diag([1.0, -1.0, -1.0])), 5).tolist()
        for st in shot["sets"]:
            R, T = st["key"]
            out_sets.append({"file": st["file"], "shot": number, "start": st["first"] + 1, "end": st["last"] + 1,
                             "cx": float(st["centre"][0]), "cy": float(st["centre"][1]), "focal": focal,
                             "loc": np.round(TO_SCENE @ (-R.T @ T) * metres, 4).tolist(),
                             "rot": np.round(quaternions((TO_SCENE @ R.T @ TO_SCENE.T)[None])[0], 5).tolist(),
                             "scale": float(metres)})
        shifts = np.array([T for _, T in shot["poses"]])
        for f in mine:
            # The figure is fitted in the camera's space of each frame: into the shot's space.
            k = f["start"] - s + np.arange(len(next(iter(f["parts"].values()))[0]))
            parts = {}
            for name, (centre, axes) in f["parts"].items():
                world = np.einsum("nij,nj->ni", turns[k], centre - shifts[k])
                parts[name] = {"loc": np.round(world @ TO_SCENE.T * metres, 4).tolist(),
                               "rot": np.round(quaternions(TO_SCENE @ turns[k] @ axes), 5).tolist()}
            out_figures.append({"start": f["start"] + 1, "size": float(f["scale"] * metres), "parts": parts})
        for p in props:
            if not s <= p["start"] < e:
                continue
            k = p["start"] - s + np.arange(len(p["centres"]))
            world = np.einsum("nij,nj->ni", turns[k], p["centres"] - shifts[k]) @ TO_SCENE.T * metres
            world = ndimage.median_filter(world, size=(PROP_STEADY, 1), mode="nearest")
            mid = k[len(k) // 2]
            out_props.append({"label": p["label"], "item": p["item"], "start": p["start"] + 1,
                              "size": np.round(np.asarray(p["size"]) * metres, 4).tolist(),
                              "loc": np.round(world, 4).tolist(),
                              "rot": np.round(quaternions((TO_SCENE @ turns[mid] @ TO_SCENE.T)[None])[0], 5).tolist()})
    return {"lens": LENS, "frames": frames, "camera": {"loc": cam_loc, "rot": cam_rot}, "sets": out_sets,
            "part_sizes": mannequin.PART_SIZES, "figures": out_figures, "props": out_props}


def main(job_dir):
    src = job_dir / "source.mp4"
    meta = probe(src)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clock = time.time()
    cuts, total = find_cuts(src)
    if not total:
        raise RuntimeError("Could not decode frames from source.mp4.")
    work, sets_dir = job_dir / "work", job_dir / "sets"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    sets_dir.mkdir(exist_ok=True)
    for old in [*sets_dir.glob("*.npy"), *sets_dir.glob("*.npz"), *sets_dir.glob("*.png")]:
        old.unlink()
    store = None
    try:
        store, people, things, tracker, outline = look(src, meta, cuts, total, work, device)
        n = len(people)
        cuts = [c for c in cuts if c < n]
        shots = list(zip(cuts, cuts[1:] + [n]))
        tracks = follow(people, shots, meta["width"])
        carried = follow_things(things, people, shots)
        missed = outline_missed(src, meta, tracks, store, outline)
        del outline
        torch.cuda.empty_cache()
        seen = time.time()
        sets, factors = build_sets(store, shots, tracker, sets_dir, device)
        built = time.time()
        fitted = pose_people(tracks, factors, meta, device, store, shots, sets, sets_dir)
        scene = assemble(shots, sets, fitted, store.depth.shape[1:], n, place_props(carried, factors, meta))
    finally:
        if store is not None:
            store.close()
        shutil.rmtree(work, ignore_errors=True)
    (job_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    print(f"analyse: {n} frames, {len(shots)} shot(s) ({len(scene['sets'])} sets), {len(fitted)} figure(s), "
          f"{len(scene['props'])} prop(s), {missed} missed detections outlined; on {device}: people and depth "
          f"{seen - clock:.0f} s, set {built - seen:.0f} s, figures {time.time() - built:.0f} s")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
