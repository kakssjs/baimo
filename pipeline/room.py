"""The set as a box room: floor, back wall, side walls and ceiling fitted to a depth map.

A depth map of film footage is far too soft to show directly, but it does say where the big
surfaces are. Each frame is reduced to six numbers (which way the room is turned, and how far
away each surface is); the numbers are then smoothed over the shot, so the room stays steady
and turns with the camera.

Axes are the camera's: x right, y forward, z up, camera at the origin, assumed level.
Distances are in the depth map's units until `smooth` scales them to metres.
"""
import cv2
import numpy as np

KEYS = ("floor", "ceiling", "back", "left", "right")
MIN_SHARE = 0.03        # a surface needs this share of the frame's points to count as seen
FIT_WIDTH = 160         # the fit runs on a small depth map; fine detail only adds noise
CAMERA_HEIGHT = 1.5     # metres; where the floor goes in shots that never show it
MIN_SEEN = 0.25         # share of a shot's frames that must show a wall or ceiling for it to be built


def fit(depth, focal):
    """One frame -> {"theta", "floor", "ceiling", "back", "left", "right"}; absent surfaces are None.
    focal is in pixels of the full-size frame's width, as a fraction: focal_px / frame_width."""
    h, w = depth.shape
    fw = FIT_WIDTH
    fh = max(8, round(h * fw / w))
    d = cv2.GaussianBlur(cv2.resize(depth, (fw, fh), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    u, v = np.meshgrid(np.arange(fw) + 0.5, np.arange(fh) + 0.5)
    f = focal * fw
    pts = np.stack([(u - fw / 2) / f * d, d, -(v - fh / 2) / f * d], axis=-1)
    normal = np.cross(np.gradient(pts, axis=1), np.gradient(pts, axis=0))
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-9
    normal *= -np.sign((normal * pts).sum(axis=-1, keepdims=True))      # face the camera
    pts, normal = pts.reshape(-1, 3), normal.reshape(-1, 3)
    need = MIN_SHARE * len(pts)

    def level(mask, values):
        return float(np.median(values[mask])) if mask.sum() >= need else None

    out = {"floor": level((normal[:, 2] > 0.8) & (pts[:, 2] < 0), pts[:, 2]),
           "ceiling": level((normal[:, 2] < -0.8) & (pts[:, 2] > 0), pts[:, 2])}

    # Walls: the room's turn is the angle that best lines the wall normals up with two axes.
    wall = np.abs(normal[:, 2]) < 0.35
    yaw = np.arctan2(normal[wall, 0], -normal[wall, 1])     # 0 = a wall facing straight at the camera
    theta = float(np.angle(np.exp(4j * yaw).sum()) / 4) if wall.sum() >= need else 0.0
    c, s = np.cos(theta), np.sin(theta)
    across = pts[wall, 0] * c + pts[wall, 1] * s            # position along the room's own x axis
    along = -pts[wall, 0] * s + pts[wall, 1] * c            # ... and its y axis (away from camera)
    n_across = normal[wall, 0] * c + normal[wall, 1] * s
    n_along = -normal[wall, 0] * s + normal[wall, 1] * c
    side = np.abs(n_across) > np.abs(n_along)
    out.update(theta=theta,
               back=level(~side & (n_along < 0), along),
               left=level(side & (n_across > 0) & (across < 0), across),
               right=level(side & (n_across < 0) & (across > 0), across))
    if out["back"] is None:
        out["back"] = float(np.percentile(pts[:, 1], 90))
    return out


def _steady(values, window):
    """Median then mean filter over time, ignoring missing values; None if nothing was seen."""
    x = np.array([np.nan if v is None else v for v in values], dtype=float)
    if np.isnan(x).all():
        return None
    idx = np.arange(len(x))
    x = np.interp(idx, idx[~np.isnan(x)], x[~np.isnan(x)])
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    med = np.array([np.median(padded[i:i + window]) for i in range(len(x))])
    return np.convolve(np.pad(med, half, mode="edge"), np.ones(window) / window, mode="valid")


def smooth(frames, scale, fps):
    """Per-frame fits of one shot -> steady per-frame lists in metres.

    A wall or ceiling seen in too few frames is dropped for the whole shot, so surfaces never
    pop in and out. The floor always exists; unseen, it sits at standing eye height below the camera.
    """
    n = len(frames)
    window = max(3, int(fps * 0.8) | 1)
    turn = _steady(np.unwrap([4 * f["theta"] for f in frames]) / 4, window)
    out = {"theta": turn.tolist()}
    for key in KEYS:
        seen = sum(f[key] is not None for f in frames) / n
        series = _steady([f[key] for f in frames], window) if seen >= MIN_SEEN else None
        out[key] = None if series is None else (series * scale).tolist()
    if out["floor"] is None:
        out["floor"] = [-CAMERA_HEIGHT] * n
    return out


# How far each surface must stay from the camera, and from any person, in metres.
CLEARANCE = {"floor": (-0.3, -0.02), "ceiling": (0.3, 0.3), "back": (1.0, 0.6), "left": (-0.6, -0.5), "right": (0.6, 0.5)}


def contain(rooms, joints, fps):
    """Push surfaces outward wherever the fit would put a wall through the camera or a person.

    rooms: the smoothed lists from `smooth` (edited in place); joints: per frame, an (n, 3) array
    of every visible joint of every figure. The push is held for about a second either side,
    so a wall moves aside smoothly instead of jumping.
    """
    n = len(rooms["theta"])
    window = max(3, int(fps * 1.2) | 1)
    half = window // 2
    for key, (camera_gap, person_gap) in CLEARANCE.items():
        if rooms[key] is None:
            continue
        sign = 1.0 if camera_gap > 0 else -1.0          # +1: surface lies at larger values (back, right, ceiling)
        need = np.full(n, sign * camera_gap)
        for i, pts in enumerate(joints):
            if len(pts) == 0:
                continue
            c, s_ = np.cos(rooms["theta"][i]), np.sin(rooms["theta"][i])
            value = {"floor": pts[:, 2], "ceiling": pts[:, 2], "back": -pts[:, 0] * s_ + pts[:, 1] * c}.get(
                key, pts[:, 0] * c + pts[:, 1] * s_)
            need[i] = max(need[i], (sign * value).max() + abs(person_gap))
        padded = np.pad(need, half, mode="edge")
        held = np.array([padded[i:i + window].max() for i in range(n)])
        held = np.convolve(np.pad(held, half, mode="edge"), np.ones(window) / window, mode="valid")
        rooms[key] = (sign * np.maximum(sign * np.asarray(rooms[key]), held)).tolist()
