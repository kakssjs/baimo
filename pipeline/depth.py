"""Depth for single frames (Depth Anything 3, monocular), and where they are too dark for it."""
import sys
import types

import cv2
import numpy as np
import torch

from .config import DA3_SRC

MODEL = "depth-anything/DA3MONO-LARGE"  # Apache-2.0
RES = 1008             # long side the model works at (fine detail for crisp set edges)
DARK = 16              # mean brightness (0..255) below which a frame is too dark for the model to read
UNLIT = 10             # parts of a frame darker than this (0..255, over 2% of its width) and nearly even
UNLIT_EVEN = 4         # (brightness spread below this) show nothing the model can read: it makes up depth
UNLIT_MIN = 0.01       # there (a dark room behind someone, the far end of a tunnel); none smaller than this
                       # share of the frame


def unlit(frame, shape):
    """Where frame (HxWx3 uint8, RGB) is too dark to see anything, at the depth model's working
    size shape ((h, w) bool): large patches nearly black and nearly even (see UNLIT)."""
    h, w = shape
    luma = cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(3, round(0.02 * w) | 1)
    mean = cv2.blur(luma, (k, k))
    spread = np.sqrt(np.maximum(cv2.blur(luma * luma, (k, k)) - mean * mean, 0))
    m = cv2.morphologyEx(((mean < UNLIT) & (spread < UNLIT_EVEN)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= UNLIT_MIN * h * w
    big[0] = False
    return big[cc]


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
        """frame: HxWx3 uint8. Returns depth ((h, w) float32, larger = farther, at the model's
        working size, which covers the whole frame) and where the sky is ((h, w) bool)."""
        with torch.inference_mode():
            p = self.model.inference([frame], process_res=RES)
        depth = np.maximum(p.depth[0].astype(np.float32), 1e-3)
        sky = getattr(p, "sky", None)
        sky = np.zeros(depth.shape, bool) if sky is None else sky[0].astype(bool)
        if sky.mean() > 0.95 and frame.mean() < DARK:   # all "sky" in the dark: the model could not read it
            sky[:] = False
        return depth, sky
