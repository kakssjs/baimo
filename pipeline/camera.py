"""The camera's path through each shot.

Background features are followed from frame to frame (Tracker). Through a shot, they are placed
in 3D, each by the depth of the frames it is seen in, which builds up a map of the background;
the camera's position and direction in each frame are then found from where the features of
that map appear in it (perspective-n-point, path). The shot's space is its first frame's camera
space, with that frame's depth from the depth model as the unit of length.

A pose is (R, T, factor): a point X of the shot's space is at R @ X + T in the frame's camera
space, and the frame's depth from the depth model, divided by factor, is in the shot's units.
Camera space is OpenCV's: x right, y down, z forward.
"""
import cv2
import numpy as np
from scipy import ndimage

FEATURES = 800         # features followed at a time
MIN_POINTS = 12        # fewest features that place the camera in a frame
PIXEL = 2.0            # how far (pixels, at the tracker's size) a feature may land from where the camera puts it
AGREE = 0.1            # a feature's depth in a later frame updates its place where it is within 10% of it
MAX_TURN = 20.0        # degrees per frame: a camera found to turn faster ...
MAX_STEP = 0.25        # ... or to move further than this share of the distance to what it sees is lost there
SMOOTH = 1.5           # frames: the camera's path is smoothed over about this many, which removes jitter
                       # from placing each frame on its own but keeps the shake of a hand-held camera
STRETCH_TURN = 45.0    # a stretch of a shot (one set, see stretches) ends where the camera has turned this
STRETCH_TRAVEL = 0.5   # many degrees, or moved this share of the distance to what it saw, from its start ...
MIN_STRETCH = 24       # ... but lasts at least this many frames (a set that changes more often looks unsteady)


class Tracker:
    """Follows background features from frame to frame. Call once per frame, in order, with
    planes.motion_view(frame) and whether the frame starts a new shot. Afterwards ids[i] and
    pts[i] hold the features seen in frame i (an id stays with one feature for as long as it is
    followed) and their positions in pixels of the motion view, whose size is `size` (h, w)."""

    def __init__(self):
        self.ids, self.pts, self.size = [], [], None
        self._prev, self._next = None, 0

    def __call__(self, view, cut):
        grey, background = view
        self.size = grey.shape
        h, w = grey.shape
        ids, pts = np.zeros(0, np.int64), np.zeros((0, 2), np.float32)
        if self._prev is not None and not cut and len(self._prev[1]):
            g0, ids0, p0 = self._prev
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, grey, p0[:, None], None, winSize=(21, 21), maxLevel=3)
            # Only features that also follow back to where they started (noise in dark frames does not),
            # and that land on background (not on someone walking in front of it).
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(grey, g0, p1, None, winSize=(21, 21), maxLevel=3)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb[:, 0] - p0, axis=1) < 1.0)
            xy = np.round(p1[:, 0]).astype(int)
            ok &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            ok[ok] &= background[xy[ok, 1], xy[ok, 0]] > 0
            ids, pts = ids0[ok], p1[ok, 0]
        if len(ids) < FEATURES * 3 // 4:            # top up, away from the features already followed
            mask = background.copy()
            for x, y in pts:
                cv2.circle(mask, (int(x), int(y)), 7, 0, -1)
            new = cv2.goodFeaturesToTrack(grey, FEATURES - len(ids), 0.01, 7, mask=mask)
            if new is not None:
                ids = np.concatenate([ids, np.arange(self._next, self._next + len(new))])
                pts = np.concatenate([pts, new[:, 0].astype(np.float32)])
                self._next += len(new)
        self.ids.append(ids)
        self.pts.append(pts)
        self._prev = (grey, ids, pts)


def _intrinsics(lens_ratio, h, w):
    """Camera matrix (OpenCV pixel convention: pixel centres on whole numbers) of an h x w image."""
    f = lens_ratio * w
    return np.array([[f, 0.0, w / 2 - 0.5], [0.0, f, h / 2 - 0.5], [0.0, 0.0, 1.0]])


def _lift(pts, size, depth, hole, sky, lens_ratio):
    """Features (pixels of a view of `size`) in the frame's camera space at the depth model's
    depth, and which of them have a depth to trust: none on people, sky or a depth edge."""
    h, w = depth.shape
    gh, gw = size
    smooth = cv2.medianBlur(depth.astype(np.float32), 5)
    near, far = ndimage.minimum_filter(smooth, 5), ndimage.maximum_filter(smooth, 5)
    xd, yd = (pts[:, 0] + 0.5) * w / gw, (pts[:, 1] + 0.5) * h / gh       # depth-map pixels from the corner
    xi, yi = np.clip(xd.astype(int), 0, w - 1), np.clip(yd.astype(int), 0, h - 1)
    z = smooth[yi, xi]
    ok = ~hole[yi, xi] & ~sky[yi, xi] & (far[yi, xi] < 1.1 * near[yi, xi]) & np.isfinite(z) & (z > 0)
    f = lens_ratio * w
    return np.stack([(xd - w / 2) / f * z, (yd - h / 2) / f * z, z], 1), ok


def _angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def path(tracker, s, e, frames, lens_ratio):
    """The camera in frames s..e-1 (one shot): a pose per frame (see the module's docstring),
    and how far away what the camera sees is (median distance to the features placed in 3D, in
    the shot's units). frames(i) -> (depth, hole, sky) of frame i. Where the camera cannot be
    placed it takes the path between its neighbours (or stands still at the ends)."""
    K = _intrinsics(lens_ratio, *tracker.size)
    total, count = {}, {}                     # per feature id: sum of its places in 3D, and how many
    poses, dists, last = [], [], (np.eye(3), np.zeros(3), 1.0, 1.0, s)
    for t in range(s, e):
        ids, pts = tracker.ids[t], tracker.pts[t]
        X, ok = _lift(pts, tracker.size, *frames(t), lens_ratio)
        pose = (np.eye(3), np.zeros(3), 1.0) if t == s else None
        known = np.array([i in total for i in ids], bool)
        if pose is None and known.sum() >= MIN_POINTS:
            obj = np.array([total[i] / count[i] for i in ids[known]])
            img = pts[known].astype(np.float64)
            found, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=100,
                                                        reprojectionError=PIXEL, confidence=0.999,
                                                        flags=cv2.SOLVEPNP_EPNP)
            if found and inl is not None and len(inl) >= MIN_POINTS:
                inl = inl[:, 0]
                rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
                R, T = cv2.Rodrigues(rvec)[0], tvec[:, 0]
                zp = (obj[inl] @ R.T + T)[:, 2]
                sure = ok[known][inl] & (zp > 0)
                R0, T0, f0, d0, t0 = last
                if sure.sum() >= MIN_POINTS:
                    f = float(np.median(X[known][inl][sure, 2] / zp[sure]))
                    gap = t - t0
                    if (_angle(R @ R0.T) <= MAX_TURN * gap and 0.6 < f / f0 < 1.6
                            and np.linalg.norm(R.T @ T - R0.T @ T0) <= MAX_STEP * d0 * gap):
                        pose = (R, T, f)
        R, T, f = pose if pose is not None else last[:3]    # where lost, features are placed as if
        Xw = (X[ok] / f - T) @ R                                # the camera had stood still
        for i, p in zip(ids[ok], Xw):
            if i in total:
                mean = total[i] / count[i]
                if abs((R @ mean + T)[2] - (R @ p + T)[2]) > AGREE * (R @ mean + T)[2]:
                    continue
                total[i], count[i] = total[i] + p, count[i] + 1
            else:
                total[i], count[i] = p, 1
        seen = [total[i] / count[i] for i in ids if i in total]
        d = float(np.median(np.linalg.norm(np.array(seen) @ R.T + T, axis=1))) if seen else last[3]
        if pose is not None:
            last = (*pose, d, t)
        poses.append(pose)
        dists.append(d)
    return _fill(poses), np.array(dists)


def keyframe(poses, a, b, frames, lens_ratio):
    """The frame of a stretch a..b-1 (of poses) whose view takes in most of what all of its
    frames see, which its set is seen from (planes.SetDepth.splat): for a camera that pulls back
    the last, for one that pans the middle. frames(t) -> (depth, hole, sky) of frame t."""
    tests = np.unique(np.linspace(a, b - 1, min(b - a, 12)).round().astype(int))
    points = []                                     # some background of each tested frame, in the shot's space
    for t in tests:
        depth, hole, sky = frames(t)
        h, w = depth.shape
        ys, xs = np.mgrid[8:h:16, 8:w:16]
        z = depth[ys, xs]
        ok = ~hole[ys, xs] & ~sky[ys, xs] & np.isfinite(z) & (z > 0)
        R, T, factor = poses[t]
        f = lens_ratio * w
        z = z[ok] / factor
        X = np.stack([(xs[ok] + 0.5 - w / 2) / f * z, (ys[ok] + 0.5 - h / 2) / f * z, z], 1)
        points.append((X - T) @ R)
    best, choice, mid = -1.0, (a + b) // 2, (a + b - 1) / 2
    for c in range(a, b, max(1, (b - a) // 24)):
        R, T, _ = poses[c]
        inside = []
        for X in points:
            p = X @ R.T + T
            z = np.maximum(p[:, 2], 1e-9)
            u, v = f * p[:, 0] / z / w, f * p[:, 1] / z / h        # from the centre, in frame sizes
            inside.append(np.mean((p[:, 2] > 0) & (np.abs(u) < 0.55) & (np.abs(v) < 0.55)) if len(p) else 0.0)
        score = float(np.mean(inside)) - 0.01 * abs(c - mid) / max(b - a, 1)
        if score > best:
            best, choice = score, c
    return choice


def stretches(poses, dists):
    """Split a shot (poses and dists from path) into stretches [(a, b)] of its frames a..b-1
    that each keep one set: one ends where the camera has turned STRETCH_TURN degrees or moved
    STRETCH_TRAVEL of the distance to what it saw from where it started. A camera that stands
    still or barely moves keeps one for the whole shot."""
    n, out, a = len(poses), [], 0
    while a < n:
        Ra, Ta, _ = poses[a]
        b = a + 1
        while b < n:
            R, T, _ = poses[b]
            far = _angle(R @ Ra.T) > STRETCH_TURN or \
                np.linalg.norm(R.T @ T - Ra.T @ Ta) > STRETCH_TRAVEL * dists[a]
            if far and b - a >= MIN_STRETCH:
                break
            b += 1
        if n - b < MIN_STRETCH // 2:                # no sliver of a stretch at the end of the shot
            b = n
        out.append((a, b))
        a = b
    return out


def relative(pose, key):
    """A pose relative to the camera at a keyframe's pose instead of the shot's space."""
    R, T, f = pose
    Rk, Tk, _ = key
    return R @ Rk.T, T - R @ Rk.T @ Tk, f


def _quat(R):
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _fill(poses):
    """Poses with the missing ones (None) filled in between their neighbours, then smoothed.
    The camera's centre is what is interpolated and smoothed, not T, which turns with it."""
    n = len(poses)
    known = np.array([p is not None for p in poses])
    if not known.any():
        return [(np.eye(3), np.zeros(3), 1.0)] * n
    q = np.array([_quat(p[0]) if p else np.zeros(4) for p in poses])
    idx = np.flatnonzero(known)
    for k, prev in zip(idx[1:], idx[:-1]):          # one side of the double cover, so averages are sound
        if q[k] @ q[prev] < 0:
            q[k] = -q[k]
    c = np.array([-p[0].T @ p[1] if p else np.zeros(3) for p in poses])
    s = np.array([np.log(p[2]) if p else 0.0 for p in poses])
    at = np.arange(n)
    q = np.stack([np.interp(at, idx, q[idx, k]) for k in range(4)], 1)
    c = np.stack([np.interp(at, idx, c[idx, k]) for k in range(3)], 1)
    s = np.interp(at, idx, s[idx])
    if n > 2:
        q = ndimage.gaussian_filter1d(q, SMOOTH, axis=0, mode="nearest")
        c = ndimage.gaussian_filter1d(c, SMOOTH, axis=0, mode="nearest")
        s = ndimage.gaussian_filter1d(s, SMOOTH, mode="nearest")
    out = []
    for k in range(n):
        R = _matrix(q[k])
        out.append((R, -R @ c[k], float(np.exp(s[k]))))
    return out
