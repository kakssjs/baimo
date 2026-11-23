"""A 3D mannequin for each tracked person: one body with fixed proportions, posed in every frame
so that its joints land on the person's keypoints in the picture.

The body is a rigid torso, a head that turns on it, and two-part arms and legs of fixed length.
Its pose in every frame of a track is found together by gradient descent, which also keeps the
motion smooth, keeps knees and elbows bending the right way and, where the picture shows no
limb (out of frame, hidden), lets it hang down in a natural rest position. Nothing of it reaches
below the floor it is seen on, so a body whose legs the picture misses kneels or crawls there
rather than standing through the floor. The whole body is always there, with the same
proportions from the first frame to the last.

Positions are in the camera's space (x right, y down, z forward) in the units of the depth maps;
analyse.py turns them into metres.
"""
import numpy as np
import torch

HEIGHT = 1.75   # metres; the body below is an adult of this height (track scale 1.0)
# Torso and head points: (towards the person's left, up, forward), from the middle of the hips.
TORSO = {5: (0.19, 0.50, 0.0), 6: (-0.19, 0.50, 0.0), 11: (0.10, 0.0, 0.0), 12: (-0.10, 0.0, 0.0)}
NECK = (0.0, 0.50, 0.0)
HEAD = 0.17                     # head centre above the neck
FACE = {0: (0.0, -0.02, 0.10), 1: (0.035, 0.02, 0.085), 2: (-0.035, 0.02, 0.085),
        3: (0.075, 0.0, 0.0), 4: (-0.075, 0.0, 0.0)}
# (from joint, to joint, length, rest direction: it hangs down, leaning by its other two numbers
# towards the body's left and front; the shins hang along the body, so they lie back along the
# floor, knees down, when it is on all fours)
LIMBS = [(5, 7, 0.29, (0.15, -1.0, 0.0)), (7, 9, 0.26, (0.0, -1.0, 0.3)),
         (6, 8, 0.29, (-0.15, -1.0, 0.0)), (8, 10, 0.26, (0.0, -1.0, 0.3)),
         (11, 13, 0.42, (0.03, -1.0, 0.05)), (13, 15, 0.42, (0.0, -1.0, -0.08)),
         (12, 14, 0.42, (-0.03, -1.0, 0.05)), (14, 16, 0.42, (0.0, -1.0, -0.08))]
MIN_SCORE = 0.3     # keypoints scored lower are ignored
FLOOR_SLACK = 0.03  # the joints may reach this share of the box's height below its floor line
HOLD_GROW, HOLD_SHRINK = 5.0, 50.0  # how firmly the body keeps its size: foreshortening only ever makes a body
                                    # look smaller, so it grows more readily than it shrinks (it folds instead)
CRAWL = 0.6         # a box less tall than this share of the body's height, and wider than tall, holds someone
                    # on all fours seen side on: the body starts out lying along it, head first where the face is
CRAWL_SPAN = 9      # frames over which that is decided (the middle one's neighbours all have a say)
FACING = 0.1        # the face is to one side where it is off the box's middle by this share of its width
BRIDGED = 0.3       # how much the box's edges count where the person is not seen (a box bridged across a gap)
DEPTH_LOOSE, DEPTH_FIRM = 2.0, 2000.0   # how firmly the body keeps to its depth: the hips loosely to the person's where
                                        # that is the depth model's; its lowest part firmly to the set's under their
                                        # feet (about as firmly as the keypoints hold it in the picture), so that it
                                        # stands on the set rather than coming nearer to look as big as its box
STEPS = 500

# What the mannequin is made of: boxes, and a sphere for the head. Sizes (across, along, front to
# back) are for the HEIGHT adult. Each limb box runs from joint to joint and a little past each
# end (the forearm on to cover the hand, the shin to cover the foot).
PART_SIZES = {"Torso": (0.38, 0.60, 0.22), "Head": (0.27, 0.27, 0.27), "Visor": (0.16, 0.055, 0.07)}
TORSO_CENTRE = 0.26     # the torso box's centre above the middle of the hips
LIMB_PARTS = {  # name: (index in LIMBS, thickness, reach before the first joint, reach past the second)
    "Arm_Upper_L": (0, 0.10, 0.03, 0.01), "Arm_Lower_L": (1, 0.09, 0.01, 0.08),
    "Arm_Upper_R": (2, 0.10, 0.03, 0.01), "Arm_Lower_R": (3, 0.09, 0.01, 0.08),
    "Leg_Upper_L": (4, 0.15, 0.03, 0.01), "Leg_Lower_L": (5, 0.13, 0.01, 0.06),
    "Leg_Upper_R": (6, 0.15, 0.03, 0.01), "Leg_Lower_R": (7, 0.13, 0.01, 0.06),
}
for _name, (_k, _thick, _top, _bottom) in LIMB_PARTS.items():
    PART_SIZES[_name] = (_thick, LIMBS[_k][2] + _top + _bottom, _thick)


def _rotation(r6):
    """Rotation matrices (..., 3, 3) from 6-number representations; columns are the body's axes
    (left, up, forward)."""
    a, b = r6[..., :3], r6[..., 3:]
    left = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    up = b - (left * b).sum(-1, keepdim=True) * left
    up = up / up.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.stack((left, up, torch.cross(left, up, dim=-1)), dim=-1)


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _pose(p, R, Rh, u, s):
    """Joint positions (N, 17, 3), head centre (N, 3) and head axes (N, 3, 3) for the parameters."""
    joints = [None] * 17
    off = lambda frame_axes, o: frame_axes @ torch.tensor(o, device=p.device, dtype=p.dtype)   # noqa: E731
    for j, o in TORSO.items():
        joints[j] = p + s * off(R, o)
    neck = p + s * off(R, NECK)
    head_axes = R @ Rh
    head = neck + s * HEAD * head_axes[..., 1]
    for j, o in FACE.items():
        joints[j] = head + s * off(head_axes, o)
    dirs = _unit(u)
    for k, (a, b, length, _) in enumerate(LIMBS):
        joints[b] = joints[a] + s * length * dirs[:, k]
    return torch.stack(joints, dim=1), head, head_axes


def _robust(e, delta):
    return delta * delta * (torch.sqrt(1 + (e / delta) ** 2) - 1)


def _outline(J, head, R, dirs, s):
    """Points on the posed body (N, K, 3) and how far its parts reach around each (K,), which
    together give the body's extent in the picture: the torso box's corners, the head's centre,
    the elbows, wrists, knees and ankles, and the ends of the hands and feet."""
    t = [PART_SIZES["Torso"][0] / 2, PART_SIZES["Torso"][1] / 2, PART_SIZES["Torso"][2] / 2]
    box = torch.tensor([(a * t[0], TORSO_CENTRE + b * t[1], c * t[2]) for a in (-1, 1) for b in (-1, 1)
                        for c in (-1, 1)], device=J.device, dtype=J.dtype)
    hips = (J[:, 11] + J[:, 12]) / 2
    torso = hips[:, None] + s * (R[:, None] @ box[None, :, :, None])[..., 0]
    hand, foot = LIMB_PARTS["Arm_Lower_L"], LIMB_PARTS["Leg_Lower_L"]
    ends = torch.cat([J[:, [9, 10]] + s * hand[3] * dirs[:, [1, 3]], J[:, [15, 16]] + s * foot[3] * dirs[:, [5, 7]]], 1)
    points = torch.cat([torso, head[:, None], J[:, 7:11], J[:, 13:17], ends], dim=1)
    arm, thigh, shin = LIMB_PARTS["Arm_Upper_L"][1] / 2, LIMB_PARTS["Leg_Upper_L"][1] / 2, foot[1] / 2
    # (the ends of the feet are the shin boxes' flat ends, the soles: nothing of the body reaches below them)
    reach = [0.0] * 8 + [PART_SIZES["Head"][1] / 2] + [arm] * 4 + [thigh] * 2 + [shin] * 2 + [hand[1] / 2] * 2 + [0.0] * 2
    return points, torch.tensor(reach, device=J.device, dtype=J.dtype)


def sizes(track, focal, width, height):
    """The body's scale (see fit) as the person's keypoints give it, in every frame they are seen:
    from the torso's length where it is seen, else from the shoulders' width; from the box's
    height where neither ever is. Foreshortening only ever makes these too small."""
    cx, cy = width / 2, height / 2
    found, boxes = [], []
    for pose in track:
        if pose.get("kp") is None:
            continue
        kp, z = np.asarray(pose["kp"], np.float32), pose["z"]
        g = kp[:, 2] >= MIN_SCORE
        at = lambda j: np.array([(kp[j, 0] - cx) / focal * z, (kp[j, 1] - cy) / focal * z, z])     # noqa: E731
        if g[5] and g[6] and g[11] and g[12]:
            found.append(np.linalg.norm((at(5) + at(6)) / 2 - (at(11) + at(12)) / 2) / 0.50)
        elif g[5] and g[6]:
            found.append(np.linalg.norm(at(5) - at(6)) / 0.38)
        boxes.append(max(pose["box"][3] - pose["box"][1], 1.0) / focal * z / HEIGHT)
    return found or boxes


def fit(track, focal, width, height, device, scale=None):
    """Fit the mannequin to one track, of about the given scale (by default what its keypoints
    say: see sizes).

    track: list over the track's frames of {"kp": 17x3 (x, y, score) or None, "box": [x0,y0,x1,y1],
           "z": depth of the person, "grounded": whether that is where the set lies under their
           feet (see analyse.on_set: then the body's lowest part in the picture is that deep, else
           its hips), "jz": 17 depths sampled at the keypoints (nan where unknown),
           "floor": the lowest y in pixels the body may reach, or None (see analyse.floor_line),
           "top", "feet", "left", "right": where the picture shows the body's ends (the box's edges:
           y of its top and lowest point, x of either side; see analyse.box_edges and floor_line),
           or None}.
    The body fills its box from edge to edge where these are known, so it is as big as the person
    looks, and where the box is short for its size, it bends to fit (a crouch, a crawl).
    Returns (scale, joints (N,17,3), head (N,3), head axes (N,3,3), torso axes (N,3,3)) as numpy,
    where scale is the body's size relative to the HEIGHT adult, in depth units per metre.
    """
    n = len(track)
    dt = torch.float32
    cx, cy = width / 2, height / 2
    kp = np.zeros((n, 17, 3), np.float32)
    seen = np.zeros(n, bool)
    for t, pose in enumerate(track):
        if pose.get("kp") is not None:
            kp[t] = pose["kp"]
            seen[t] = True
    z = np.array([pose["z"] for pose in track], np.float32)
    firm = np.array([DEPTH_FIRM if pose.get("grounded") else DEPTH_LOOSE for pose in track], np.float32)
    box_h = np.array([max(pose["box"][3] - pose["box"][1], 1.0) for pose in track], np.float32)
    floor = np.array([1e9 if pose.get("floor") is None else pose["floor"] for pose in track], np.float32)
    floor += FLOOR_SLACK * box_h
    edges = {k: np.array([np.nan if pose.get(k) is None else pose[k] for pose in track], np.float32)
             for k in ("top", "feet", "left", "right")}
    good = seen[:, None] & (kp[:, :, 2] >= MIN_SCORE)

    def lift(t, j, depth=None):
        d = z[t] if depth is None else depth
        return np.array([(kp[t, j, 0] - cx) / focal * d, (kp[t, j, 1] - cy) / focal * d, d], np.float32)

    if scale is None:
        found = sizes(track, focal, width, height) or [z.mean() * 0.4]
        scale = np.percentile(found, 75)        # foreshortening only ever makes things look smaller
    s0 = float(scale)

    # Starting pose: hips (or shoulders) at the person's depth, upright, facing as the shoulders say.
    p0 = np.zeros((n, 3), np.float32)
    left0 = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (n, 1))
    up0 = np.tile(np.array([0.0, -1.0, 0.0], np.float32), (n, 1))
    for t in range(n):
        g = good[t]
        if not seen[t]:
            p0[t] = p0[t - 1] if t else [0.0, 0.0, z[t]]
            left0[t], up0[t] = (left0[t - 1], up0[t - 1]) if t else (left0[t], up0[t])
            continue
        hips = (lift(t, 11) + lift(t, 12)) / 2 if g[11] and g[12] else None
        shoulders = (lift(t, 5) + lift(t, 6)) / 2 if g[5] and g[6] else None
        if hips is not None and shoulders is not None:
            up0[t] = (shoulders - hips) / max(np.linalg.norm(shoulders - hips), 1e-6)
        if hips is not None:
            p0[t] = hips
        elif shoulders is not None:
            p0[t] = shoulders - up0[t] * 0.5 * s0
        else:
            x0, y0, x1, y1 = track[t]["box"]
            p0[t] = [((x0 + x1) / 2 - cx) / focal * z[t], (y0 + 0.45 * (y1 - y0) - cy) / focal * z[t], z[t]]
        if g[5] and g[6]:
            v = lift(t, 5) - lift(t, 6)
            jz = track[t].get("jz")
            if jz is not None and np.isfinite(jz[5]) and np.isfinite(jz[6]):
                v[2] = np.clip(jz[5] - jz[6], -0.3 * s0, 0.3 * s0)
            if np.linalg.norm(v) > 1e-6:
                left0[t] = v / np.linalg.norm(v)
    # Someone on all fours seen side on starts out lying along their box, chest down, head towards
    # their face (+1: to the right, -1: to the left; 0: upright), decided over CRAWL_SPAN frames.
    # Where the face is not seen (or is straight on), it is to the side it was last seen on.
    facing = np.full(n, np.nan, np.float32)
    for t in range(n):
        x0, _, x1, _ = track[t]["box"]
        face = [j for j in range(5) if good[t, j]]
        off = kp[t, face, 0].mean() - (x0 + x1) / 2 if face else 0.0
        if abs(off) > FACING * (x1 - x0):
            facing[t] = np.sign(off)
    known_face = np.flatnonzero(np.isfinite(facing))
    if known_face.size:
        last = np.maximum.accumulate(np.where(np.isfinite(facing), np.arange(n), -1))
        facing = facing[np.where(last >= 0, last, known_face[0])]
    # (Not where a desk or the frame's edge cuts the box off: someone behind a desk, or close up,
    # looks short too.)
    short = np.array([(p["box"][3] - p["box"][1]) / focal * z[t] < CRAWL * HEIGHT * s0
                      and p.get("floor") is not None for t, p in enumerate(track)])
    lie = np.zeros(n, np.float32)
    for t in range(n):
        x0, y0, x1, y1 = track[t]["box"]
        if np.isfinite(facing[t]) and short[t] and x1 - x0 > y1 - y0:
            lie[t] = facing[t]
    half = CRAWL_SPAN // 2
    lie = np.sign(np.median(np.lib.stride_tricks.sliding_window_view(np.pad(lie, half, mode="edge"), CRAWL_SPAN), 1))
    # The crawl goes on, either way, while the box stays that short: turned towards the camera, or
    # partly hidden, a crawler's box is no wider than tall.
    for t in range(1, n):
        if not lie[t] and short[t]:
            lie[t] = lie[t - 1]
    for t in range(n - 2, -1, -1):
        if not lie[t] and short[t]:
            lie[t] = lie[t + 1]
    for t in np.flatnonzero(lie):
        x0, y0, x1, y1 = track[t]["box"]
        hips = x0 + 0.3 * (x1 - x0) if lie[t] > 0 else x1 - 0.3 * (x1 - x0)
        p0[t] = [(hips - cx) / focal * z[t], (y0 + 0.4 * (y1 - y0) - cy) / focal * z[t], z[t]]
        up0[t], left0[t] = (lie[t], 0.0, 0.0), (0.0, 0.0, lie[t])
    r6 = np.concatenate([left0, up0], axis=1)

    dev = device
    T = lambda a: torch.tensor(a, device=dev, dtype=dt)     # noqa: E731
    p = T(p0).requires_grad_()
    r = T(r6).requires_grad_()
    h = T(np.tile([1, 0, 0, 0, 1, 0], (n, 1))).requires_grad_()
    side = T([(d[0], 0.0, d[2]) for *_, d in LIMBS])     # (rest directions: lean, hanging down)
    down = T([0.0, 1.0, 0.0])
    shins = [k for k, (a, *_) in enumerate(LIMBS) if a in (13, 14)]       # (knee to ankle)

    def hanging(R):
        hang = down.expand(n, len(LIMBS), 3).clone()
        hang[:, shins] = -R[:, None, :, 1]
        return _unit((R[:, None] @ side[None, :, :, None])[..., 0] + hang)

    with torch.no_grad():
        u0 = hanging(_rotation(r))
    u = u0.clone().requires_grad_()
    log_s = torch.tensor(np.log(s0), device=dev, dtype=dt, requires_grad=True)

    obs = T(kp[:, :, :2])
    weight = T(np.where(good, kp[:, :, 2], 0.0))
    norm = T(box_h)[:, None]
    zt = T(z)
    firmness = T(firm)
    on_set = torch.tensor([bool(pose.get("grounded")) for pose in track], device=dev)
    lowest = T(floor)
    known = {k: T(np.where(np.isfinite(v), np.where(seen, 1.0, BRIDGED), 0.0)) for k, v in edges.items()}
    edge = {k: T(np.nan_to_num(v)) for k, v in edges.items()}
    tau = 0.01 * norm       # (the body's extent: a soft minimum and maximum, about 1% of the box)
    opt = torch.optim.Adam([p, r, h, u, log_s], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    eye6 = T([1, 0, 0, 0, 1, 0])
    for step in range(STEPS):
        opt.zero_grad()
        s = log_s.exp()
        R = _rotation(r)
        Rh = _rotation(h)
        J, head, head_axes = _pose(p, R, Rh, u, s)
        proj = torch.stack((J[..., 0] / J[..., 2].clamp_min(1e-4) * focal + cx,
                            J[..., 1] / J[..., 2].clamp_min(1e-4) * focal + cy), dim=-1)
        err = (proj - obs).norm(dim=-1) / norm
        loss = (_robust(err, 0.03) * weight).sum() / weight.sum().clamp_min(1.0) * 1e3
        grow = log_s - np.log(s0)                               # (its size)
        loss = loss + (HOLD_GROW + (HOLD_SHRINK - HOLD_GROW) * (grow < 0)) * grow ** 2
        below = torch.relu((proj[..., 1] - lowest[:, None]) / norm)      # (nothing below the floor)
        loss = loss + 1e3 * (below ** 2).sum(-1).mean()
        dirs = _unit(u)
        pts, reach = _outline(J, head, R, dirs, s)          # (the body fills its box)
        z_pts = pts[..., 2].clamp_min(1e-4)
        x, y = pts[..., 0] / z_pts * focal + cx, pts[..., 1] / z_pts * focal + cy
        r_px = reach * s / z_pts * focal
        # Its depth: where the set under their feet is known, the body's lowest part in the picture (a
        # sole where they stand, a knee or a hand where they kneel or crawl) is on it, as deep as the set
        # is there (on a stair, the lower foot a step nearer than the other: the legs take that up);
        # elsewhere the hips keep to the person's depth.
        lowest_part = torch.softmax((y + r_px) / tau, dim=1)
        at = torch.where(on_set, (lowest_part * z_pts).sum(1), p[:, 2])
        loss = loss + (firmness * ((at - zt) / zt) ** 2).mean()
        extent = {"top": -tau[:, 0] * torch.logsumexp(-(y - r_px) / tau, dim=1),
                  "feet": tau[:, 0] * torch.logsumexp((y + r_px) / tau, dim=1),
                  "left": -tau[:, 0] * torch.logsumexp(-(x - r_px) / tau, dim=1),
                  "right": tau[:, 0] * torch.logsumexp((x + r_px) / tau, dim=1)}
        for k, on in known.items():
            miss = _robust((extent[k] - edge[k]) / norm[:, 0], 0.03)
            loss = loss + (miss * on).sum() / on.sum().clamp_min(1.0) * 1e3
        want = hanging(R)                                       # (limbs hang down)
        loss = loss + 0.05 * ((dirs - want) ** 2).sum(-1).mean()
        left = R[..., 0]
        knees = torch.cross(dirs[:, [4, 6]], dirs[:, [5, 7]], dim=-1)
        elbows = torch.cross(dirs[:, [0, 2]], dirs[:, [1, 3]], dim=-1)
        loss = loss + 2.0 * (torch.relu(-(knees * left[:, None]).sum(-1)) ** 2).mean()
        loss = loss + 1.0 * (torch.relu((elbows * left[:, None]).sum(-1)) ** 2).mean()
        loss = loss + 0.05 * ((h - eye6) ** 2).sum(-1).mean()
        loss = loss + 0.02 * ((1 + R[..., 1][:, 1]) ** 2).mean()              # upright when nothing says otherwise
        if n > 2:       # smooth motion: keypoint jitter is about 1% of a body height per frame
            body = J / (s * HEIGHT)
            loss = loss + 1000.0 * ((body[2:] - 2 * body[1:-1] + body[:-2]) ** 2).sum(-1).mean()
            loss = loss + 100.0 * ((R[2:] - 2 * R[1:-1] + R[:-2]) ** 2).sum((-1, -2)).mean()
            loss = loss + 100.0 * ((head_axes[2:] - 2 * head_axes[1:-1] + head_axes[:-2]) ** 2).sum((-1, -2)).mean()
            loss = loss + 5.0 * (((p[1:, 2] - p[:-1, 2]) / zt[1:]) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        s = log_s.exp()
        R = _rotation(r)
        J, head, head_axes = _pose(p, R, _rotation(h), u, s)
    return float(s), J.cpu().numpy(), head.cpu().numpy(), head_axes.cpu().numpy(), R.cpu().numpy()


def parts(scale, joints, head, head_axes, torso_axes):
    """Where every part of a fitted mannequin is (see fit for the arguments): {name: (centres
    (N, 3), axes (N, 3, 3) with columns across, along and front to back)}, in the camera's space.
    The parts' sizes are PART_SIZES times scale."""
    J, R, s = joints, torso_axes, scale
    hips = (J[:, 11] + J[:, 12]) / 2
    out = {"Torso": (hips + s * TORSO_CENTRE * R[..., 1], R),
           "Head": (head, head_axes),
           "Visor": (head + s * (head_axes @ np.array([0.0, 0.01, 0.105])), head_axes)}
    for name, (k, _, top, bottom) in LIMB_PARTS.items():
        a, b, length, _ = LIMBS[k]
        d = J[:, b] - J[:, a]
        d = d / np.linalg.norm(d, axis=-1, keepdims=True).clip(1e-9)
        # The box's front faces the way the body faces (or, for a limb pointing that way, up).
        front = R[..., 2] - (R[..., 2] * d).sum(-1, keepdims=True) * d
        up = R[..., 1] - (R[..., 1] * d).sum(-1, keepdims=True) * d
        front = np.where(np.linalg.norm(front, axis=-1, keepdims=True) < 1e-3, up, front)
        front = front / np.linalg.norm(front, axis=-1, keepdims=True).clip(1e-9)
        out[name] = (J[:, a] + s * (length + bottom - top) / 2 * d, np.stack((np.cross(d, front), d, front), axis=-1))
    return out
