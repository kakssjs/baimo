"""Depth for single frames (Depth Anything 3, monocular) and the clean-up applied to it."""
import sys
import types

import cv2
import numpy as np
import torch

from .config import DA3_SRC

MODEL = "depth-anything/DA3MONO-LARGE"  # Apache-2.0
PROCESS_PIXELS = 420_000   # area the model works at; time per frame grows with it
MAX_RES = 1008             # ... but never more than this on the long side
NEAR = 0.2             # depth (model units) that maps to white in the stored 16-bit frames


class DepthEstimator:
    def __init__(self, device):
        if DA3_SRC not in sys.path:
            sys.path.insert(0, DA3_SRC)
        # The package's export helpers import video and 3D-viewer libraries that are not needed
        # here (and not installed); only inference is used, so that module is replaced by a stub.
        stub = types.ModuleType("depth_anything_3.utils.export")
        stub.export = None
        sys.modules.setdefault("depth_anything_3.utils.export", stub)
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(MODEL).to(device).eval()

    def __call__(self, frame):
        """frame: HxWx3 uint8. Returns relative depth (h, w) float32, larger = farther."""
        with torch.inference_mode():
            h, w = frame.shape[:2]
            res = min(MAX_RES, int((PROCESS_PIXELS * max(w, h) / min(w, h)) ** 0.5))
            p = self.model.inference([frame], process_res=res)
        depth = np.maximum(p.depth[0].astype(np.float32), 1e-3)
        sky = getattr(p, "sky", None)
        if sky is not None and sky[0].any() and not sky[0].all():
            depth[sky[0]] = 2.0 * np.percentile(depth[~sky[0]], 99)   # sky has no depth: push it far back
        return depth


def sharpen_edges(depth, frame):
    """Guided filter: snaps the depth map's soft edges onto the edges of the video frame, and
    smooths the surfaces in between. Works on inverse depth, guided by the frame's brightness."""
    h, w = depth.shape
    guide = cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    guide = guide.astype(np.float32) / 255.0
    src = 1.0 / depth
    k = (13, 13)
    mean_g, mean_s = cv2.blur(guide, k), cv2.blur(src, k)
    var_g = cv2.blur(guide * guide, k) - mean_g * mean_g
    cov = cv2.blur(guide * src, k) - mean_g * mean_s
    a = cov / (var_g + 8e-3)
    b = mean_s - a * mean_g
    out = cv2.blur(a, k) * guide + cv2.blur(b, k)
    return 1.0 / np.clip(out, 1.0 / depth.max(), 1.0 / depth.min())


def lift_out(depth, people):
    """Erase people from a depth map and return each person's own depth.

    people: the frame's detections, each with a full-frame boolean "mask" (removed here).
    Every masked pixel is repainted from the surfaces around it, so the set shows what is
    behind the person and a figure can stand in their place.
    """
    h, w = depth.shape
    gone = np.zeros((h, w), dtype=np.uint8)
    depths = []
    for person in people:
        mask = cv2.resize(person.pop("mask").astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        inside = depth[mask.astype(bool)]
        depths.append(float(np.median(inside)) if inside.size else float(np.median(depth)))
        gone |= mask
    if gone.any():
        # Depth maps blur a little past the outline, so the hole is widened before filling.
        gone = cv2.dilate(gone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        if gone.all():
            return depths
        inv = 1.0 / depth
        lo, hi = float(inv.min()), float(inv.max())
        scaled = ((inv - lo) / max(hi - lo, 1e-9) * 65535).astype(np.uint16)
        filled = cv2.inpaint(scaled, gone, 5, cv2.INPAINT_TELEA).astype(np.float32) / 65535 * (hi - lo) + lo
        smooth = cv2.GaussianBlur(filled, (0, 0), 3)
        depth[gone.astype(bool)] = 1.0 / np.maximum(smooth[gone.astype(bool)], 1e-6)
    return depths


def encode(depth):
    """16-bit image of inverse depth: white = NEAR, black = infinitely far."""
    return (np.clip(NEAR / depth, 0, 1) * 65535).astype(np.uint16)
