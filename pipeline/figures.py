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
POLE = 0.2            # ... or whose sure keypoints typically lie in an upright line less wide than this share of their
                      # height, in a box the frame's side cuts: something upright and narrow at the side of the
                      # picture (the blurred edge of a building, a pole) that looks like someone standing side on
SIDE = 0.01           # a box that ends this near the frame's left or right edge (share of its width) is cut by it
                      # (and one this near its top, as a share of its height: see _carry_on)
STITCH_GAP = 48       # a track that ends and another that starts at most this many frames later ...
STITCH_REACH = 1.0    # ... within this many body heights of where it ended are one person
SAME_TRACK = 0.5      # two tracks at the same time whose boxes typically overlap this much ...
SAME_POSE = 0.15      # ... or whose keypoints lie this close (in body heights) are one person seen twice
SAME_PART = 0.6       # a box mostly (this share) inside another, with keypoints that close, is a second
                      # box on the same person (all of them, and again just their upper half)
FAINT_SCORE = 0.1     # people found less surely than MIN_PERSON_SCORE, down to this, never start a track, but carry
FAINT_IOU = 0.5       # one on (see _carry_on) where their box overlaps its last this much, with keypoints as sure
                      # as a track's (TRACK_KEYPOINTS): someone in the dark, or half hidden, whom the detector can
                      # barely tell from the floor
HIDDEN = 0.8          # ... and so does a track's box lying this much (share of it) inside someone else's: hidden
                      # behind them
HOLDS = 0.5           # a box holding this much (share) of someone else's box never carries a track on: it is
                      # theirs, boxed loosely with the dark around them (as is one that overlaps theirs by FAINT_IOU)
# Things people carry, hold, throw or ride on, and small animals (the detector's names): each is drawn as a
# simple shape of its own colour that follows it, as the items a template is made to replace; whatever else
# the detector knows (furniture, a parked car) stays part of the set.
PROPS = {"backpack", "umbrella", "handbag", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
         "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
         "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
         "pizza", "donut", "cake", "laptop", "mouse", "remote", "keyboard", "cell phone", "book", "vase", "scissors",
         "teddy bear", "hair drier", "toothbrush", "bird", "cat", "dog"}
MIN_PROP_SCORE = 0.3  # such things detected less surely than this are ignored ...
MIN_PROP_SIZE = 0.02  # ... and so are those less wide or tall than this share of the frame's height ...
MAX_PROP_AREA = 0.3   # ... or that cover more than this share of the frame (a page held up to the camera, a wall) ...
PROP_PERSON = 0.4     # ... or whose box is this much (overlap) a person's: that person, taken for something else
                      # as well (a man crawling for a dog, someone in mid-air for a bird)
PROP_TRACK_SCORE = 0.35  # a track of one whose detections are typically less sure than this is a false detection
SAME_THING = 0.3      # two tracks at the same time whose boxes typically lie this much (share of the smaller) one
                      # inside the other follow one thing boxed twice (a bag, and the bag with the hand on it)
KINDS = {"backpack": "bag", "handbag": "bag", "suitcase": "bag", "skis": "board", "snowboard": "board",
         "skateboard": "board", "surfboard": "board", "bottle": "vessel", "wine glass": "vessel", "cup": "vessel",
         "vase": "vessel", "cell phone": "handset", "remote": "handset", "mouse": "handset"}
SAME_KIND = 0.6       # a track of a thing is one thing only where at least this share of its detections say it is
                      # the same kind of thing (KINDS: the detector calls one bag a handbag or a suitcase), not a
                      # board in one frame, a bird in the next (a part of someone, half seen)
CARRIED = 0.15        # only things held in at least this share of the frames they are seen in are drawn: what people
                      # carry, hold or throw; the rest (planks taken for books, taxis for phones) stays in the set
HOLD = 0.3            # held: a wrist (keypoint seen at least this surely) within this share of the thing's size of
                      # its box (the hand on a bag's handle) ...
CARRIER = 3.0         # ... of someone whose box is at least this many times as big as the thing's (not a part of
                      # someone taken for something they hold)


class PoseEstimator:
    def __init__(self, device):
        self.device = device
        self.det_proc = AutoProcessor.from_pretrained(DETECTOR)
        self.det = RTDetrForObjectDetection.from_pretrained(DETECTOR).to(device).eval()
        self.pose_proc = AutoProcessor.from_pretrained(POSE)
        self.pose = VitPoseForPoseEstimation.from_pretrained(POSE).to(device).eval()

    def __call__(self, frames):
        """frames: list of HxWx3 uint8 arrays. Returns, per frame, a list of the people in it,
        {"box": [x0, y0, x1, y1], "score": detector confidence, "kp": 17x3 array of x, y, score}
        in pixels; per frame a list of the things in PROPS in it, {"box", "score", "label"}; and per
        frame a list of the people found less surely (see FAINT_SCORE), as the people are."""
        h, w = frames[0].shape[:2]
        with torch.inference_mode():
            out = self.det(**self.det_proc(images=frames, return_tensors="pt").to(self.device))
        found = self.det_proc.post_process_object_detection(
            out, target_sizes=torch.tensor([(h, w)] * len(frames)), threshold=min(FAINT_SCORE, MIN_PROP_SCORE))
        boxes, scores, things = [], [], []
        for r in found:
            labels, b, s = r["labels"].cpu().numpy(), r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()
            names = [self.det.config.id2label[int(k)] for k in labels]
            tall = (labels == 0) & ((b[:, 3] - b[:, 1]) >= MIN_HEIGHT * h)
            person = tall & (s >= MIN_PERSON_SCORE)
            keep = _surest(b[person], s[person], np.flatnonzero(person))     # (one person boxed twice: the surer)
            unsure = tall & (s >= FAINT_SCORE) & (s < MIN_PERSON_SCORE) & np.array(
                [all(_iou(box, b[k]) < SAME_BOX for k in keep) for box in b], bool)     # (not someone found surely)
            prop = (np.array([n in PROPS for n in names], bool) & (s >= MIN_PROP_SCORE)
                    & (np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) >= MIN_PROP_SIZE * h)
                    & ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) <= MAX_PROP_AREA * h * w)
                    & np.array([all(_iou(box, b[k]) < PROP_PERSON for k in keep) for box in b], bool))
            things.append([{"box": b[k].tolist(), "score": float(s[k]), "label": names[k]}
                           for k in _surest(b[prop], s[prop], np.flatnonzero(prop))])
            keep += _surest(b[unsure], s[unsure], np.flatnonzero(unsure))
            boxes.append(b[keep])
            scores.append(s[keep])
        people, faint = [[] for _ in frames], [[] for _ in frames]
        hits = [i for i, b in enumerate(boxes) if len(b)]
        if hits:
            xywh = [np.concatenate([boxes[i][:, :2], boxes[i][:, 2:] - boxes[i][:, :2]], axis=1) for i in hits]
            with torch.inference_mode():
                out = self.pose(**self.pose_proc([frames[i] for i in hits], boxes=xywh,
                                                 return_tensors="pt").to(self.device))
            for i, poses in zip(hits, self.pose_proc.post_process_pose_estimation(out, boxes=xywh)):
                for b, s, p in zip(boxes[i], scores[i], poses):
                    kp = np.concatenate([p["keypoints"].cpu().numpy(), p["scores"].cpu().numpy()[:, None]], axis=1)
                    found_as = people if s >= MIN_PERSON_SCORE else faint
                    found_as[i].append({"box": b.tolist(), "score": float(s), "kp": kp})
        return people, things, faint


def _surest(boxes, scores, index):
    """Of boxes that overlap by SAME_BOX or more (one thing boxed twice), the surest: their entries in
    index."""
    keep = []
    for k in np.argsort(-scores):
        if all(_iou(boxes[k], boxes[j]) < SAME_BOX for j in keep):
            keep.append(k)
    return [int(index[k]) for k in keep]


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


def within(a, b):
    """How much (share) of box a lies inside box b."""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return 0.0 if w <= 0 or h <= 0 else w * h / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


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
    """One track from two that follow the same person (or thing): a's detections, and b's where a has none."""
    found = {b["start"] + k: p for k, p in enumerate(b["poses"]) if not p.get("missed")}
    found.update({a["start"] + k: p for k, p in enumerate(a["poses"]) if not p.get("missed")})
    frames = sorted(found)
    out = {"start": frames[0], "last": frames[0], "box": found[frames[0]]["box"], "velocity": None, "hits": 1,
           "poses": [found[frames[0]]]}
    for i in frames[1:]:
        _join(out, {"start": i, "last": i, "box": found[i]["box"], "velocity": None, "hits": 1, "poses": [found[i]]})
    return out


def _spread(pose):
    """How wide the pose's sure body keypoints (the shoulders down) lie for their height; nan where
    fewer than four are sure."""
    kp = pose["kp"][5:]
    sure = kp[:, 2] >= 0.5
    if sure.sum() < 4:
        return np.nan
    return float(np.ptp(kp[sure, 0]) / max(np.ptp(kp[sure, 1]), 1.0))


def _plausible(t, width):
    """Whether a track looks like a person: the detector and the pose model are typically sure, and
    it is not something upright and narrow that the frame's side cuts (see POLE)."""
    seen = [p for p in t["poses"] if not p.get("missed")]
    if (np.median([p.get("score", 1.0) for p in seen]) < TRACK_SCORE
            or np.median([_confidence(p) for p in seen]) < TRACK_KEYPOINTS):
        return False
    spread = [s for s in map(_spread, seen) if np.isfinite(s)]
    cut = [p["box"][0] <= SIDE * width or p["box"][2] >= (1 - SIDE) * width for p in seen]
    return not (spread and np.median(spread) < POLE and np.mean(cut) >= 0.5)


def _carry_on(tracks, people, faint, width, height):
    """The tracks carried on before their first frame and after their last through the people
    found less surely (faint, and those in people no track took; see FAINT_SCORE): a box that
    overlaps the track's nearest by FAINT_IOU, with keypoints as sure as a track's
    (TRACK_KEYPOINTS), that neither of the frame's sides nor its top cuts (SIDE; width and
    height: the frame's, in pixels: who is in such a box, and how much of them, the detector can
    barely tell, a door's edge or a body flying past the lens; the bottom, which cuts the legs of
    anyone near the camera in an ordinary shot, may), and is no one else's (neither holding HOLDS
    of another track's box there nor the same body as its pose, part of it: see _same_body),
    carries it on to that frame. The frames between are bridged as missed (see _join), at most
    MAX_GAP of them in a row with no such box, not counting those where the track's box lies
    inside someone else's (HIDDEN: hidden behind them); the track ends with the last such box, or
    where it meets someone else's track (its box holding HOLDS of theirs). In those frames the
    pose is that box's, keypoints and all, but missed and carried: outlined and measured later,
    though not taken out of the set (see analyse.outline_missed)."""
    n = len(people)
    poses = [{} for _ in range(n)]          # (frame: {track: its pose there})
    taken = [set() for _ in range(n)]       # (the detections tracks have)
    for k, t in enumerate(tracks):
        for j, p in enumerate(t["poses"]):
            poses[t["start"] + j][k] = p
            taken[t["start"] + j].add(id(p))

    def piece(i, found):
        return {"start": i, "last": i, "box": found["box"], "velocity": None, "hits": 0,
                "poses": [{**found, "missed": True}]}
    for k, t in enumerate(tracks):
        first, length = t["start"], len(t["poses"])
        for step in (1, -1):
            i = t["last"] + 1 if step > 0 else t["start"] - 1
            box, misses, found = t["box"] if step > 0 else t["poses"][0]["box"], 0, []
            while 0 <= i < n and misses <= MAX_GAP:
                others = [q for o, q in poses[i].items() if o != k]
                if any(within(q["box"], box) >= HOLDS for q in others):
                    break                   # (someone else's track takes over there)
                near = [p for p in faint[i] + people[i] if id(p) not in taken[i] and _confidence(p) >= TRACK_KEYPOINTS
                        and SIDE * width < p["box"][0] and p["box"][2] < (1 - SIDE) * width and SIDE * height < p["box"][1]
                        and not any(within(q["box"], p["box"]) >= HOLDS
                                    or q.get("kp") is not None and _same_body(p, q) for q in others)]
                best = max(near, key=lambda p: _iou(p["box"], box), default=None)
                if best is not None and _iou(best["box"], box) >= FAINT_IOU:
                    found.append((i, best))
                    taken[i].add(id(best))
                    box, misses = best["box"], 0
                elif not any(within(box, q["box"]) >= HIDDEN for q in others):
                    misses += 1
                i += step
            if not found:
                continue
            if step > 0:
                for i, p in found:
                    _join(t, piece(i, p))
            else:
                found.reverse()
                ext = piece(*found[0])
                for i, p in found[1:]:
                    _join(ext, piece(i, p))
                _join(ext, t)
                t.update(ext)
        for j, p in enumerate(t["poses"]):
            if not first <= t["start"] + j < first + length:
                p["carried"] = True
            poses[t["start"] + j][k] = p
    return tracks


def track(people, width, faint=None, height=None):
    """Link per-frame detections into tracks; width is the frame's, in pixels (see POLE); faint,
    the people found less surely in each frame, carry them on (see _carry_on; height: the
    frame's, in pixels).

    Returns a list of {"start": frame_index, "poses": [pose, ...]} with a pose for every frame
    from the first detection to the last. In frames where the person was missed the pose is
    {"box": interpolated box, "kp": None, "missed": True}; otherwise it is the detection itself.
    Where a track was carried on, its poses there are also marked "carried" (see _carry_on).
    """
    tracks = _stitch([t for t in _link(_one_per_person(found) for found in people) if _plausible(t, width)])
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
    kept.sort(key=lambda t: t["start"])
    if faint is not None:
        kept = _carry_on(kept, people, faint, width, height)
    return [{"start": t["start"], "poses": t["poses"]} for t in kept]


def track_things(things, people):
    """Link the things in PROPS found in each frame (see PoseEstimator) into tracks, as people's
    are (see track), and keep those someone carries (see CARRIED; people: the people found in
    each frame): a list of {"start", "label", "poses"}, its label the one its detections give
    the most weight (the detector can call a bag a handbag in one frame, a suitcase in the
    next)."""
    tracks = _stitch([t for t in _link(things)
                      if np.median([p["score"] for p in t["poses"] if not p.get("missed")]) >= PROP_TRACK_SCORE])
    kept = []
    for t in sorted(tracks, key=lambda t: -t["hits"]):     # (the longest first, completed by its twin)
        twin = next((k for k, other in enumerate(kept) if _same_thing(other, t)), None)
        if twin is None:
            kept.append(t)
        else:
            kept[twin] = _combine(kept[twin], t)
    out = []
    for t in sorted(kept, key=lambda t: t["start"]):
        seen = [(t["start"] + k, p) for k, p in enumerate(t["poses"]) if not p.get("missed")]
        kinds = [KINDS.get(p["label"], p["label"]) for _, p in seen]
        if (t["hits"] >= MIN_TRACK and max(map(kinds.count, kinds)) >= SAME_KIND * len(kinds)
                and sum(held(p["box"], people[i]) for i, p in seen) >= CARRIED * len(seen)):
            weight = {}
            for p in t["poses"]:
                if not p.get("missed"):
                    weight[p["label"]] = weight.get(p["label"], 0.0) + p["score"]
            out.append({"start": t["start"], "label": max(weight, key=weight.get), "poses": t["poses"]})
    return out


def held(box, people):
    """Whether someone in people (one frame's) holds the thing in box (see HOLD, CARRIER)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    for p in people:
        b = p["box"]
        if p.get("kp") is None or (b[2] - b[0]) * (b[3] - b[1]) < CARRIER * w * h:
            continue
        for x, y, score in p["kp"][[9, 10]]:        # (the wrists)
            if score >= HOLD and x0 - HOLD * w <= x <= x1 + HOLD * w and y0 - HOLD * h <= y <= y1 + HOLD * h:
                return True
    return False


def _same_thing(a, b):
    """Whether two tracks of things running at the same time follow one thing boxed twice: over the
    frames they share (their boxes where they were detected, else where they were expected), one
    box typically lies SAME_THING inside the other."""
    inside = []
    for i in range(max(a["start"], b["start"]), min(a["last"], b["last"]) + 1):
        (a0, a1, a2, a3), (b0, b1, b2, b3) = a["poses"][i - a["start"]]["box"], b["poses"][i - b["start"]]["box"]
        w, h = min(a2, b2) - max(a0, b0), min(a3, b3) - max(a1, b1)
        smaller = min((a2 - a0) * (a3 - a1), (b2 - b0) * (b3 - b1))
        inside.append(w * h / smaller if w > 0 and h > 0 and smaller > 0 else 0.0)
    return len(inside) >= 3 and np.median(inside) >= SAME_THING


def _link(found):
    """Per-frame detections (lists of {"box", ...}) linked into tracks by where each track's box
    is expected next (see _match), the frames between two detections missed (see _join)."""
    tracks, open_tracks = [], []
    for i, here in enumerate(found):
        open_tracks = [t for t in open_tracks if i - t["last"] <= MAX_GAP]
        pairs = sorted(((_match(_expected(t, i), p["box"]), ti, pi) for ti, t in enumerate(open_tracks)
                        for pi, p in enumerate(here)), reverse=True)
        used_t, used_p = set(), set()
        for quality, ti, pi in pairs:
            if quality <= 0:
                break
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            t, p = open_tracks[ti], here[pi]
            velocity = (np.array(p["box"], float) - np.array(t["box"], float)) / (i - t["last"])
            _join(t, {"start": i, "last": i, "box": p["box"], "velocity": velocity, "hits": 1, "poses": [p]})
        for pi, p in enumerate(here):
            if pi not in used_p:
                t = {"start": i, "last": i, "box": p["box"], "velocity": None, "hits": 1, "poses": [p]}
                tracks.append(t)
                open_tracks.append(t)
    return tracks
