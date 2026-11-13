"""Depth for single frames (Depth Anything 3, monocular) and removal of people from it."""
import sys
import types

import cv2
import numpy as np
import torch

from . import figures
from .config import DA3_SRC

MODEL = "depth-anything/DA3MONO-LARGE"  # Apache-2.0
PROCESS_RES = 756      # long side the model works at; depth maps come back at this size


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
            p = self.model.inference([frame], process_res=PROCESS_RES)
        depth = np.maximum(p.depth[0].astype(np.float32), 1e-3)
        sky = getattr(p, "sky", None)
        if sky is not None and sky[0].any() and not sky[0].all():
            depth[sky[0]] = 2.0 * np.percentile(depth[~sky[0]], 99)   # sky has no depth: push it far back
        return depth


def lift_out(depth, person, sx, sy):
    """Return a person's depth and erase them from the depth map, so the set behind them is
    what gets built and a figure can stand in their place.

    Inside the person's box, each row gets a baseline running straight from the surface left of
    the box to the surface right of it; whatever stands out in front of that baseline is pressed
    back onto it.
    """
    h, w = depth.shape
    x0, y0, x1, y1 = person["box"]
    px, py = 0.1 * (x1 - x0), 0.06 * (y1 - y0)
    xa, xb = max(int((x0 - px) * sx), 0), min(int((x1 + px) * sx) + 1, w)
    ya, yb = max(int((y0 - py) * sy), 0), min(int((y1 + py) * sy) + 1, h)
    region = depth[ya:yb, xa:xb]
    if region.size == 0:
        return float(np.median(depth))
    body = [depth[int(np.clip(y * sy, 0, h - 1)), int(np.clip(x * sx, 0, w - 1))]
            for x, y, score in person["kp"][[5, 6, 11, 12]] if score >= figures.MIN_KEYPOINT]
    z = float(np.median(body)) if body else float(np.percentile(region, 25))

    def edge(x):    # the column just outside the box, steadied vertically; None at the frame edge
        if not 0 <= x < w:
            return None
        return cv2.medianBlur(np.ascontiguousarray(depth[ya:yb, x:x + 1]), 5)[:, 0] if yb - ya >= 5 else depth[ya:yb, x]

    left, right = edge(xa - 1), edge(xb)
    if left is None and right is None:      # the box spans the frame: fall back to its far side
        left = right = np.full(yb - ya, np.percentile(region, 90), dtype=np.float32)
    left, right = (right if left is None else left), (left if right is None else right)
    t = np.linspace(0, 1, xb - xa, dtype=np.float32)
    baseline = left[:, None] * (1 - t) + right[:, None] * t
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ya:yb, xa:xb] = region < 0.97 * baseline
    np.copyto(region, baseline, where=region < 0.97 * baseline)
    # Soften the patch and its seam.
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8)).astype(bool)
    depth[mask] = cv2.GaussianBlur(depth, (0, 0), 4)[mask]
    return z
