"""People for the white model: detect persons, estimate their 2D pose, follow them across frames."""
import numpy as np
import torch
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

DETECTOR = "PekingU/rtdetr_r50vd_coco_o365"   # Apache-2.0
POSE = "usyd-community/vitpose-base-simple"   # Apache-2.0, 17 COCO keypoints
MIN_PERSON_SCORE = 0.35
MIN_KEYPOINT = 0.35   # keypoints scored below this are treated as not visible
MIN_HEIGHT = 0.15     # ignore people shorter than this fraction of the frame height
MATCH_IOU = 0.2       # a detection continues a track when their boxes overlap at least this much
MAX_GAP = 12          # frames a person may go undetected before the track ends
MIN_TRACK = 8         # shorter tracks are treated as false detections
SMOOTH = 0.5          # weight of the previous frame's keypoints (reduces jitter)


class PoseEstimator:
    def __init__(self, device):
        self.device = device
        self.det_proc = AutoProcessor.from_pretrained(DETECTOR)
        self.det = RTDetrForObjectDetection.from_pretrained(DETECTOR).to(device).eval()
        self.pose_proc = AutoProcessor.from_pretrained(POSE)
        self.pose = VitPoseForPoseEstimation.from_pretrained(POSE).to(device).eval()

    def __call__(self, frames):
        """frames: list of HxWx3 uint8 arrays. Returns, per frame, a list of
        {"box": [x0, y0, x1, y1], "kp": 17x3 array of x, y, score} in pixels."""
        h, w = frames[0].shape[:2]
        with torch.inference_mode():
            out = self.det(**self.det_proc(images=frames, return_tensors="pt").to(self.device))
        found = self.det_proc.post_process_object_detection(
            out, target_sizes=torch.tensor([(h, w)] * len(frames)), threshold=MIN_PERSON_SCORE)
        boxes = []
        for r in found:
            b = r["boxes"][r["labels"] == 0].cpu().numpy()
            boxes.append(b[(b[:, 3] - b[:, 1]) >= MIN_HEIGHT * h])
        people = [[] for _ in frames]
        hits = [i for i, b in enumerate(boxes) if len(b)]
        if hits:
            xywh = [np.concatenate([boxes[i][:, :2], boxes[i][:, 2:] - boxes[i][:, :2]], axis=1) for i in hits]
            with torch.inference_mode():
                out = self.pose(**self.pose_proc([frames[i] for i in hits], boxes=xywh,
                                                 return_tensors="pt").to(self.device))
            for i, box, poses in zip(hits, boxes, self.pose_proc.post_process_pose_estimation(out, boxes=xywh)):
                for b, p in zip(box, poses):
                    kp = np.concatenate([p["keypoints"].cpu().numpy(), p["scores"].cpu().numpy()[:, None]], axis=1)
                    people[i].append({"box": b.tolist(), "kp": kp})
        return people


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def _between(a, b, t):
    """Pose part-way from a to b; a joint counts as seen only if both ends saw it."""
    kp = (1 - t) * a["kp"] + t * b["kp"]
    kp[:, 2] = np.minimum(a["kp"][:, 2], b["kp"][:, 2])
    box = [(1 - t) * p + t * q for p, q in zip(a["box"], b["box"])]
    return {"box": box, "kp": kp, "z": (1 - t) * a["z"] + t * b["z"]}


def track(people):
    """Link per-frame detections into tracks.

    Returns a list of {"start": frame_index, "poses": [pose, ...]}; a pose is {"box", "kp", "z"}
    with keypoints smoothed over time. Short gaps inside a track are bridged by interpolation.
    """
    tracks, open_tracks = [], []
    for i, frame_people in enumerate(people):
        open_tracks = [t for t in open_tracks if i - t["last"] <= MAX_GAP]
        free = list(frame_people)
        for t in sorted(open_tracks, key=lambda t: -t["last"]):
            best = max(free, key=lambda p: _iou(p["box"], t["box"]), default=None)
            if best is None or _iou(best["box"], t["box"]) < MATCH_IOU:
                continue
            free.remove(best)
            kp = best["kp"].copy()
            kp[:, :2] = SMOOTH * t["kp"][:, :2] + (1 - SMOOTH) * kp[:, :2]
            new, old, gap = {**best, "kp": kp}, t["poses"][-1], i - t["last"]
            for step in range(1, gap):      # bridge the frames where the person was missed
                t["poses"].append(_between(old, new, step / gap))
            t["poses"].append(new)
            t.update(last=i, box=best["box"], kp=kp)
        for p in free:
            t = {"start": i, "last": i, "box": p["box"], "kp": p["kp"], "poses": [dict(p)]}
            tracks.append(t)
            open_tracks.append(t)
    return [{"start": t["start"], "poses": t["poses"]} for t in tracks if len(t["poses"]) >= MIN_TRACK]
