"""People for the white model: detect persons, estimate their 2D pose, follow them across frames."""
import numpy as np
import torch
from transformers import (AutoProcessor, RTDetrForObjectDetection, Sam2Model, Sam2Processor,
                          VitPoseForPoseEstimation)

DETECTOR = "PekingU/rtdetr_r50vd_coco_o365"   # Apache-2.0
POSE = "usyd-community/vitpose-base-simple"   # Apache-2.0, 17 COCO keypoints
SEGMENTER = "facebook/sam2.1-hiera-small"     # Apache-2.0
MIN_PERSON_SCORE = 0.3
MIN_HEIGHT = 0.08     # ignore people shorter than this fraction of the frame height
SAME_BOX = 0.6        # two detections overlapping this much are the same person
MATCH_IOU = 0.2       # a detection continues a track when it overlaps the track's expected box this much
MAX_GAP = 24          # frames a person may go undetected before the track ends
MIN_TRACK = 8         # tracks with fewer detections than this are treated as false detections
TRACK_SCORE = 0.45    # ... and so are tracks whose detections are typically less sure than this
TRACK_KEYPOINTS = 0.4  # ... or whose body keypoints are (coats on a rack, statues, posters)
STITCH_GAP = 48       # a track that ends and another that starts at most this many frames later ...
STITCH_REACH = 1.0    # ... within this many body heights of where it ended are one person
SAME_TRACK = 0.5      # two tracks at the same time whose boxes typically overlap this much ...
SAME_POSE = 0.15      # ... or whose keypoints lie this close (in body heights) are one person seen twice
SAME_PART = 0.6       # a box mostly (this share) inside another, with keypoints that close, is a second
                      # box on the same person (all of them, and again just their upper half)


class PoseEstimator:
    def __init__(self, device):
        self.device = device
        self.det_proc = AutoProcessor.from_pretrained(DETECTOR)
        self.det = RTDetrForObjectDetection.from_pretrained(DETECTOR).to(device).eval()
        self.pose_proc = AutoProcessor.from_pretrained(POSE)
        self.pose = VitPoseForPoseEstimation.from_pretrained(POSE).to(device).eval()

    def __call__(self, frames):
        """frames: list of HxWx3 uint8 arrays. Returns, per frame, a list of
        {"box": [x0, y0, x1, y1], "score": detector confidence, "kp": 17x3 array of x, y, score}
        in pixels."""
        h, w = frames[0].shape[:2]
        with torch.inference_mode():
            out = self.det(**self.det_proc(images=frames, return_tensors="pt").to(self.device))
        found = self.det_proc.post_process_object_detection(
            out, target_sizes=torch.tensor([(h, w)] * len(frames)), threshold=MIN_PERSON_SCORE)
        boxes, scores = [], []
        for r in found:
            person = (r["labels"] == 0).cpu().numpy()
            b, s = r["boxes"].cpu().numpy()[person], r["scores"].cpu().numpy()[person]
            tall = (b[:, 3] - b[:, 1]) >= MIN_HEIGHT * h
            b, s = b[tall], s[tall]
            keep = []           # the detector sometimes boxes one person twice: keep the surer box
            for k in np.argsort(-s):
                if all(_iou(b[k], b[j]) < SAME_BOX for j in keep):
                    keep.append(k)
            boxes.append(b[keep])
            scores.append(s[keep])
        people = [[] for _ in frames]
        hits = [i for i, b in enumerate(boxes) if len(b)]
        if hits:
            xywh = [np.concatenate([boxes[i][:, :2], boxes[i][:, 2:] - boxes[i][:, :2]], axis=1) for i in hits]
            with torch.inference_mode():
                out = self.pose(**self.pose_proc([frames[i] for i in hits], boxes=xywh,
                                                 return_tensors="pt").to(self.device))
            for i, poses in zip(hits, self.pose_proc.post_process_pose_estimation(out, boxes=xywh)):
                for b, s, p in zip(boxes[i], scores[i], poses):
                    kp = np.concatenate([p["keypoints"].cpu().numpy(), p["scores"].cpu().numpy()[:, None]], axis=1)
                    people[i].append({"box": b.tolist(), "score": float(s), "kp": kp})
        return people


class Segmenter:
    """Exact outline of each detected person (SAM 2, prompted with the person's box)."""

    def __init__(self, device):
        self.device = device
        self.proc = Sam2Processor.from_pretrained(SEGMENTER)
        self.model = Sam2Model.from_pretrained(SEGMENTER).to(device).eval()

    def __call__(self, frame, people):
        """Adds a boolean "mask" (frame-sized) to every person in the list."""
        if not people:
            return
        inputs = self.proc(images=frame, input_boxes=[[p["box"] for p in people]], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs, multimask_output=False)
        masks = np.asarray(self.proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0])
        for person, mask in zip(people, masks.reshape(len(people), -1, *frame.shape[:2])[:, 0]):
            person["mask"] = mask.astype(bool)


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def _expected(t, i):
    """Where a track's box should be in frame i, carrying on its recent motion."""
    box = np.array(t["box"], float)
    if t["velocity"] is not None:
        box = box + 0.7 * t["velocity"] * (i - t["last"])
    return box.tolist()


def _match(expected, box):
    """How well a detection continues a track whose box was expected at `expected` (0 = not at all).
    Mostly overlap; a box of the same size close by, but hardly overlapping, still counts a
    little (fast motion, or someone walking toward the camera, whose box grows quickly)."""
    overlap = _iou(expected, box)
    if overlap >= MATCH_IOU:
        return overlap
    he, hb = expected[3] - expected[1], box[3] - box[1]
    reach = 0.5 * max(he, hb)
    d = np.hypot((expected[0] + expected[2] - box[0] - box[2]) / 2, (expected[1] + expected[3] - box[1] - box[3]) / 2)
    if he > 0 and 2 / 3 <= hb / he <= 1.5 and d < reach:
        return 0.99 * MATCH_IOU * (1 - d / reach)
    return 0.0


def _confidence(pose):
    """How sure the pose model is that this is a person: its eight surest keypoints, on average
    (a person half hidden behind a desk still scores high; a coat on a hanger does not)."""
    return float(np.sort(pose["kp"][:, 2])[-8:].mean())


def _same_body(a, b):
    """Whether two detections in one frame are the same person: one box mostly inside the other,
    and their surest keypoints coincide."""
    (a0, a1, a2, a3), (b0, b1, b2, b3) = a["box"], b["box"]
    w, h = min(a2, b2) - max(a0, b0), min(a3, b3) - max(a1, b1)
    if w <= 0 or h <= 0 or w * h < SAME_PART * min((a2 - a0) * (a3 - a1), (b2 - b0) * (b3 - b1)):
        return False
    sure = (a["kp"][:, 2] >= 0.5) & (b["kp"][:, 2] >= 0.5)
    if sure.sum() < 3:
        return False
    height = max(a3 - a1, b3 - b1)
    return float(np.median(np.hypot(*(a["kp"][sure, :2] - b["kp"][sure, :2]).T))) < SAME_POSE * height


def _one_per_person(found):
    """One frame's detections, without the extra boxes the detector puts on part of someone it
    has already boxed whole (or round more than them): of the boxes on one person, the one the
    detector and the pose model are surest of together."""
    kept = []
    for p in sorted(found, key=lambda p: -p["score"] * _confidence(p)):
        if not any(_same_body(k, p) for k in kept):
            kept.append(p)
    return kept


def _centre_height(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]), box[3] - box[1]


def _join(a, b):
    """Track b carries on track a after a gap: one track, the frames between missed."""
    gap = b["start"] - a["last"]
    old, new = np.array(a["box"], float), np.array(b["poses"][0]["box"], float)
    for step in range(1, gap):
        a["poses"].append({"box": ((1 - step / gap) * old + step / gap * new).tolist(), "kp": None, "missed": True})
    a["poses"] += b["poses"]
    a.update(last=b["last"], box=b["box"], velocity=b["velocity"], hits=a["hits"] + b["hits"])


def _stitch(tracks):
    """Join tracks that are one person the detector lost for a while, or whose box changed
    shape too fast to follow (getting up, lying down): one ends and, soon after and close by,
    the other starts. Nearest in time first."""
    while True:
        best = None
        for a in tracks:
            (ca, ha) = _centre_height(a["box"])
            for b in tracks:
                gap = b["start"] - a["last"]
                if b is a or not 0 < gap <= STITCH_GAP or (best is not None and gap >= best[0]):
                    continue
                cb, hb = _centre_height(b["poses"][0]["box"])
                if np.hypot(*(ca - cb)) < STITCH_REACH * max(ha, hb):
                    best = (gap, a, b)
        if best is None:
            return tracks
        _join(best[1], best[2])
        tracks = [t for t in tracks if t is not best[2]]


def _same_person(a, b):
    """Whether two tracks running at the same time follow one person whom the detector boxed
    twice (all of a crawling man and again just his front, say): over the frames they share,
    their boxes mostly overlap, or their surest keypoints coincide."""
    overlaps, distances = [], []
    for i in range(max(a["start"], b["start"]), min(a["last"], b["last"]) + 1):
        pa, pb = a["poses"][i - a["start"]], b["poses"][i - b["start"]]
        if pa.get("missed") and pb.get("missed"):
            continue
        overlaps.append(_iou(pa["box"], pb["box"]))
        if pa.get("missed") or pb.get("missed"):
            continue
        sure = (pa["kp"][:, 2] >= 0.5) & (pb["kp"][:, 2] >= 0.5)
        if sure.sum() >= 3:
            height = max(pa["box"][3] - pa["box"][1], pb["box"][3] - pb["box"][1])
            distances.append(np.median(np.hypot(*(pa["kp"][sure, :2] - pb["kp"][sure, :2]).T)) / height)
    return len(overlaps) >= 5 and (np.median(overlaps) >= SAME_TRACK
                                   or (len(distances) >= 5 and np.median(distances) < SAME_POSE))


def _combine(a, b):
    """One track from two that follow the same person: a's detections, and b's where a has none."""
    found = {b["start"] + k: p for k, p in enumerate(b["poses"]) if not p.get("missed")}
    found.update({a["start"] + k: p for k, p in enumerate(a["poses"]) if not p.get("missed")})
    frames = sorted(found)
    out = {"start": frames[0], "last": frames[0], "box": found[frames[0]]["box"], "velocity": None, "hits": 1,
           "poses": [found[frames[0]]]}
    for i in frames[1:]:
        _join(out, {"start": i, "last": i, "box": found[i]["box"], "velocity": None, "hits": 1, "poses": [found[i]]})
    return out


def _plausible(t):
    """Whether a track looks like a person: the detector and the pose model are typically sure."""
    seen = [p for p in t["poses"] if not p.get("missed")]
    return (np.median([p.get("score", 1.0) for p in seen]) >= TRACK_SCORE
            and np.median([_confidence(p) for p in seen]) >= TRACK_KEYPOINTS)


def track(people):
    """Link per-frame detections into tracks.

    Returns a list of {"start": frame_index, "poses": [pose, ...]} with a pose for every frame
    from the first detection to the last. In frames where the person was missed the pose is
    {"box": interpolated box, "kp": None, "missed": True}; otherwise it is the detection itself.
    """
    tracks, open_tracks = [], []
    for i, frame_people in enumerate(_one_per_person(found) for found in people):
        open_tracks = [t for t in open_tracks if i - t["last"] <= MAX_GAP]
        pairs = sorted(((_match(_expected(t, i), p["box"]), ti, pi) for ti, t in enumerate(open_tracks)
                        for pi, p in enumerate(frame_people)), reverse=True)
        used_t, used_p = set(), set()
        for quality, ti, pi in pairs:
            if quality <= 0:
                break
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            t, p = open_tracks[ti], frame_people[pi]
            velocity = (np.array(p["box"], float) - np.array(t["box"], float)) / (i - t["last"])
            _join(t, {"start": i, "last": i, "box": p["box"], "velocity": velocity, "hits": 1, "poses": [p]})
        for pi, p in enumerate(frame_people):
            if pi not in used_p:
                t = {"start": i, "last": i, "box": p["box"], "velocity": None, "hits": 1, "poses": [p]}
                tracks.append(t)
                open_tracks.append(t)
    tracks = _stitch([t for t in tracks if _plausible(t)])
    tracks.sort(key=lambda t: -t["hits"])
    kept = []
    for t in tracks:            # the longest first: a person seen twice is the fuller track, completed by the other
        if t["hits"] < MIN_TRACK:
            continue
        twin = next((k for k, other in enumerate(kept) if _same_person(other, t)), None)
        if twin is None:
            kept.append(t)
        else:
            kept[twin] = _combine(kept[twin], t)
    return [{"start": t["start"], "poses": t["poses"]} for t in sorted(kept, key=lambda t: t["start"])]
