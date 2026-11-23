"""The sets: the depth model's maps from across a stretch of a shot laid onto one picture, seen
from one camera, and turned into flat walls, floors and steps, square to each other where the set
is built square, with straight outlines, and people taken out. What fits no plane (a railing, a
radiator, a lamp, furniture) becomes flat cards in its own outline, the faces of the simple blocks
a set is roughed out with, and small pieces lying on a larger surface become part of it, so the set
is a few large, plain surfaces. What is too dark to see is closed by the surfaces around it, or
left empty where it opens into the dark.

Works in inverse depth, where any plane the camera sees is an affine function of pixel position
(1 / depth = a*x + b*y + c, x and y measured from the image centre), so finding and fitting
planes is plain linear algebra.
"""
import itertools
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

TOLERANCE = 0.02      # a pixel lies on a plane when its depth is within 2% of the plane's ...
ANGLE = 14.0          # ... and its surface turns less than this many degrees away from it
CURVATURE = 0.004     # bend (relative Laplacian of inverse depth) above which a pixel is an edge
MERGE_GAP = 0.04      # two planes that stay within 4% of each other in depth over one's pixels ...
MERGE_ANGLE = 10.0    # ... and turn less than this many degrees apart are one surface (the depth model
                      # bends flat things a little: a printed card, a long wall)
MAX_PLANES = 40
MIN_PLANE = 0.002     # smallest plane, as a fraction of the frame
MIN_STEP = 0.002      # ... and in a second search over what the first left (furniture faces, steps seen from above)
SPREAD = 0.025        # planes take in the pixels next to them within this share of depth (mouldings, ...
SPREAD_STEPS = 32     # ... seams), up to this many pixels away
SQUARE = 20.0         # planes within this many degrees of one of the set's three main directions (square to
SQUARE_SHARE = 0.5    # each other, where at least this share of the main planes' area lines up with them)
                      # are turned onto it: walls, floors and steps come out square, as built
PIECE = 0.01          # pieces of a plane smaller than this fraction of the frame, and its parts thinner than
THIN = 0.004          # this share of the frame width (slivers, spikes), join the planes around them: a set
                      # built by hand has no chips or slivers
CARD_JUMP = 0.05      # what fits no plane becomes flat cards, split where its depth jumps by more than this
CARD_MIN = 0.004      # between neighbouring pixels; pieces smaller than this fraction of the frame join the
                      # surface next to them; the cards are turned square too, as the planes are
ABSORB = 0.1          # pieces smaller than this fraction of a frame that lie on a larger piece next to them,
ABSORB_FIT = 0.12     # within this share of depth at the median of their pixels, become part of it, smallest
SHARD = 0.005         # first, and pieces smaller than this join the one next to them they lie nearest even so:
                      # a set is a few large, plain surfaces, and only what stands out from them its own piece
STRAIGHT = 0.005      # outlines of planes are straightened to within this fraction of the frame width ...
OUTLINE = 2.0         # ... the clutter's this many times as coarsely
CREASE_GAP = 0.03     # planes that meet at an angle (CREASE_ANGLE or more apart) and lie within this share of
CREASE_ANGLE = 10.0   # each other in depth along their seam are cut along the line where they meet (a room's
CREASE_REACH = 0.02   # corner, a box's edge), where it runs within this share of the frame width of the seam
MIN_SEAM = 20         # pixels: shorter seams between planes are left as they are
STAIR_ANGLE = 40.0    # stairs: risers are surfaces facing one of the floor's two main directions to within this
STAIR_SPLIT = 0.01    # many degrees, split where their distance jumps by more than this share into faces of at
MIN_RISER = 0.0003    # least this fraction of a frame; faces that stack like steps (each starting within
STAIR_GAP = 0.35      # this share of a rise of where the one before ends in height, and at most STAIR_RUN
STAIR_RUN = 5.0       # rises further back), MIN_STAIRS or more of them, are a run of stairs where one rise
MIN_STAIRS = 4        # and one run fit them all to within STAIR_EVEN of a rise or run and climb at between
STAIR_EVEN = 0.4      # STAIR_PITCH degrees; it is rebuilt as even steps where the surface lies within
STAIR_PITCH = (10.0, 60.0)  # STAIR_FIT of depth of them
STAIR_FIT = 0.05
BIG_PLANE = 0.03     # planes this large (walls, floors, ceilings) close off the holes people leave
BEHIND = 0.05         # ... and no plane closes it more than this much nearer than the person standing in it
RUN_OFF = 1.5         # no surface lies more than this many times as far back as what is around it, or than
                      # anything seen in the set: a plane carried on towards its horizon runs off for kilometres
HALO = 0.10           # depth maps blur a person into what is next to them: the plane closing a hole takes ...
HALO_REACH = 0.03     # ... back what someone stood over up to this share of a frame's width from it and up
                      # to HALO nearer than the plane, so no rim outlines the spot they stood in
FLOOR = 40.0          # a main plane facing up to within this many degrees (a floor) carries on under the dark
                      # next to it: a floor in shadow is still there
CLOSE = 0.9           # the rest of the dark is closed by the surfaces around it, like the holes people leave,
                      # where they close at least this share of a patch; otherwise, where the depth there
OPENING = 0.75        # lies behind what is around it (inverse depth under this share of it), it is a way
                      # into the dark (a doorway into a dark room) and is left empty
SEEDS, SAMPLE = 384, 16384
MOTION_WIDTH = 480    # frames are matched at this width to follow the camera
CANVAS = 2.5          # a stretch is laid onto one picture at most this many frames wide and high, ...
NEAREST = 0.15        # ... of what is no nearer its camera than this share of the distance to what it sees
SHOT_SAMPLES = 40     # frames whose depth is combined into that picture
SEEN = 0.3            # ... and the share of them a spot must be seen in (not behind anyone) to count as seen
BLEND = 12            # a stretch's set is also built from this many frames around it, half on either side
GROW = 0.15           # a person's outline also takes in the pixels next to it within their own depth range
GROW_REACH = 0.08     # (give or take 15%), up to this share of their size away: the outline model misses
                      # dark clothes against a dark background, which would stay in the set as a ghost
BLEED = 7             # ... and this many pixels more all round, as depth maps bleed past a person's outline
THROUGH = 1.5         # a pixel of a person's mask counts no farther back than this much behind the nearest of
THROUGH_REACH = 48    # them within this many pixels: farther, it shows what is behind them (depth maps blur
                      # their outline into it, and masks take in gaps between arm and body)


def _blur(x, sigma):
    k = int(sigma * 3) * 2 + 1
    g = torch.exp(-((torch.arange(k, device=x.device, dtype=x.dtype) - k // 2) ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    x = F.conv2d(F.pad(x[None, None], (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))[0, 0]


def _derivatives(q):
    p = F.pad(q[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    qu = (p[1:-1, 2:] - p[1:-1, :-2]) / 2
    qv = (p[2:, 1:-1] - p[:-2, 1:-1]) / 2
    lap = p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4 * q
    return qu, qv, lap


def _normal(a, b, c, f):
    """Unit normal, facing the camera, of the plane 1/depth = a*x + b*y + c (camera looks along +z).
    A plane seen almost edge-on (a corridor's side wall) has no clear front, and its normal flips
    from one side to the other on noise: compare normals without their sign (abs of the dot)."""
    n = torch.stack((a * f, b * f, c), dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(n[..., 2:3] > 0, -n, n)


def _fit(x, y, q):
    """Least-squares plane through inverse depths q at pixels (x, y), errors relative to depth."""
    x, y, q = x.double(), y.double(), q.double()
    wt = 1.0 / (q * q)
    A = torch.stack((x, y, torch.ones_like(x)), dim=1)
    Aw = A * wt[:, None]
    return torch.linalg.solve(Aw.T @ A + 1e-12 * torch.eye(3, device=q.device, dtype=q.dtype), Aw.T @ q).float()


def _card(x, y, q):
    """A plane through the bulk of inverse depths q at pixels (x, y): fitted three times, each
    time without the points furthest off it. Where a tilted plane would run far past their depths
    (a thin or ragged piece), one facing the camera at their median instead."""
    flat = torch.stack((torch.zeros_like(q[0]), torch.zeros_like(q[0]), q.median()))
    if float(x.max() - x.min()) < 2 or float(y.max() - y.min()) < 2:
        return flat
    keep = torch.ones_like(q, dtype=torch.bool)
    for _ in range(3):
        abc = _fit(x[keep], y[keep], q[keep])
        off = (q - (abc[0] * x + abc[1] * y + abc[2])).abs() / q
        keep = off <= max(2.0 * float(off.median()), 0.01)
        if keep.sum() < 16:
            break
    pred = abc[0] * x + abc[1] * y + abc[2]
    return flat if pred.min() <= 0.5 * q.min() or pred.max() >= 2.0 * q.max() else abc


def _nearest(lab, where):
    """lab with the pixels where (bool) given the label of the nearest labelled pixel."""
    if where.any() and (lab > 0).any():
        _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(where, lab[iy, ix], lab)
    return lab


def _riser_run(u, v, s, facing, min_px):
    """The longest run of risers stacked like steps: the pixels facing (bool) the way the steps
    run, split where their distance v jumps (more than STAIR_SPLIT between neighbours) into faces
    of at least min_px pixels; a face follows another where it starts within STAIR_GAP of a rise
    (the other's height, u) of where that one ends, is about as high (0.6 to 1.6 times), lies at
    most STAIR_RUN rises further back, and side by side with it (s ranges overlapping by half).
    Returns (faces of the run, nearest first: dicts of v, lo, hi, s0, s1, n, c; component per pixel)."""
    lv = np.log(np.maximum(v, 1e-9))
    jump = np.zeros(facing.shape, bool)
    across, down = np.abs(np.diff(lv, axis=1)) > STAIR_SPLIT, np.abs(np.diff(lv, axis=0)) > STAIR_SPLIT
    jump[:, 1:] |= across
    jump[:, :-1] |= across
    jump[1:] |= down
    jump[:-1] |= down
    n, cc, stats, _ = cv2.connectedComponentsWithStats((facing & ~jump).astype(np.uint8), connectivity=4)
    faces = []
    for c in np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_px):
        if c > 0:
            m = cc == c
            lo, hi = np.percentile(u[m], [5, 95])
            s0, s1 = np.percentile(s[m], [2, 98])
            faces.append(dict(v=float(np.median(v[m])), lo=lo, hi=hi, s0=s0, s1=s1, n=int(m.sum()), c=c))
    faces.sort(key=lambda t: t["v"])

    def follows(a, b):
        rise = a["hi"] - a["lo"]
        return (rise > 0 and 0.6 * rise < b["hi"] - b["lo"] < 1.6 * rise
                and abs(b["lo"] - a["hi"]) <= STAIR_GAP * rise and 0 < b["v"] - a["v"] <= STAIR_RUN * rise
                and min(a["s1"], b["s1"]) - max(a["s0"], b["s0"]) > 0.5 * min(a["s1"] - a["s0"], b["s1"] - b["s0"]))

    length, before = [1] * len(faces), [-1] * len(faces)
    for j in range(len(faces)):
        for i in range(j):
            if length[i] + 1 > length[j] and follows(faces[i], faces[j]):
                length[j], before[j] = length[i] + 1, i
    run, j = [], int(np.argmax(length)) if faces else -1
    while j >= 0:
        run.append(faces[j])
        j = before[j]
    return run[::-1], cc


def _even_steps(steps):
    """One run and one rise fitted to a run of risers (see _riser_run): (distance of the first
    riser, run, height of its bottom, rise), or None where they do not fit every riser to within
    STAIR_EVEN of a run or rise, or climb at a pitch outside STAIR_PITCH: not stairs."""
    K = len(steps)
    k = np.arange(K, dtype=np.float64)
    v = np.array([t["v"] for t in steps])
    lo, hi = np.array([t["lo"] for t in steps]), np.array([t["hi"] for t in steps])
    wt = np.sqrt([t["n"] for t in steps])
    v0, run = np.linalg.lstsq(np.stack([wt, wt * k], 1), wt * v, rcond=None)[0]
    A = np.concatenate([np.stack([np.ones(K), k], 1), np.stack([np.ones(K), k + 1], 1)])
    h0, rise = np.linalg.lstsq(A, np.r_[lo, hi], rcond=None)[0]
    if rise <= 0 or run <= 0 or not STAIR_PITCH[0] <= math.degrees(math.atan2(rise, run)) <= STAIR_PITCH[1]:
        return None
    off = max(np.abs(v - v0 - k * run).max() / run, np.abs(lo - h0 - k * rise).max() / rise,
              np.abs(hi - h0 - (k + 1) * rise).max() / rise)
    return (v0, run, h0, rise) if off <= STAIR_EVEN else None


def _boxes(lab, pad=0):
    """(label, (rows, cols) slices of its bounding box, widened by pad pixels) for every label > 0
    of lab, so that work on one plane looks at its own part of the picture only."""
    h, w = lab.shape
    for k, box in enumerate(ndimage.find_objects(lab), 1):
        if box is not None:
            r, c = box
            yield k, (slice(max(0, r.start - pad), min(h, r.stop + pad)), slice(max(0, c.start - pad), min(w, c.stop + pad)))


def _unit_normals(P, f):
    """Unit normals (n, 3; float64, sign as the plane's coefficients give it) of planes P (n x 3)."""
    n = np.asarray(P, np.float64) * np.array([f, f, 1.0])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _frame(normal, areas):
    """The three directions square to each other (3 x 3, a direction per row) that most of the
    area of planes (unit normals, areas) lines up with to within 15 degrees, or None where less
    than SQUARE_SHARE of it does (a cave, a crowd, a landscape): tried from every two of the
    largest planes, then each direction set to the mean of the normals near it, weighted by area."""
    cos15 = math.cos(math.radians(15))
    best = None
    for i, j in itertools.permutations(np.argsort(-areas)[:12], 2):
        b = normal[j] - (normal[j] @ normal[i]) * normal[i]
        if np.linalg.norm(b) < 0.5:         # (more than 30 degrees off square)
            continue
        R = np.stack([normal[i], b / np.linalg.norm(b), np.cross(normal[i], b / np.linalg.norm(b))])
        score = areas[np.abs(normal @ R.T).max(1) > cos15].sum()
        if best is None or score > best[0]:
            best = (score, R)
    if best is None or best[0] < SQUARE_SHARE * areas.sum():
        return None
    R = best[1]
    for _ in range(2):
        c = normal @ R.T
        axis = np.abs(c).argmax(1)
        M = 1e-9 * R
        for k in np.flatnonzero(np.abs(c).max(1) > cos15):
            M[axis[k]] += areas[k] * np.sign(c[k, axis[k]]) * normal[k]
        u, _, vt = np.linalg.svd(M)
        R = u @ vt                          # (the nearest directions square to each other)
    return R


def _limit(people):
    """The nearest (inverse depth) a hole may be closed where people stood (people: the farthest
    anyone stood at each pixel, 0 where nobody did): no more than BEHIND in front of the farthest
    of them around it (the points frames left there are scattered, and the fill must not switch
    planes from pixel to pixel)."""
    nobody = np.float32(1e30)
    far = cv2.erode(np.where(people > 0, people, nobody).astype(np.float32), np.ones((7, 7), np.uint8))
    return np.where(far < nobody, far * (1 + BEHIND), np.inf).astype(np.float32)


def _sides(lab, src, axis, where=False):
    """Labels of the nearest src pixels before and after every pixel along an axis (0 = none),
    and with where, also their positions along it (-1 and n where there is none)."""
    n = lab.shape[axis]
    pos = np.arange(n).reshape((-1, 1) if axis == 0 else (1, -1))
    before = np.maximum.accumulate(np.where(src, pos, -1), axis=axis)
    after = n - 1 - np.flip(np.maximum.accumulate(np.flip(np.where(src, n - 1 - pos, -1), axis), axis=axis), axis)
    across = np.indices(lab.shape)[1 - axis]

    def label_at(i):
        ii = np.clip(i, 0, n - 1)
        return np.where((i >= 0) & (i < n), lab[ii, across] if axis == 0 else lab[across, ii], 0)

    return (label_at(before), label_at(after)) + ((before, after) if where else ())


def _keep_large(mask, min_px):
    """The parts of a boolean mask that form connected pieces of at least min_px pixels."""
    m = mask.cpu().numpy().astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return torch.from_numpy(big[cc]).to(mask.device)


def _grow(mask, depth):
    """A person's mask (uint8, depth-map size) widened over the pixels next to it whose depth lies
    within the person's own depth range, give or take GROW, up to GROW_REACH of their size away."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return mask
    h, w = mask.shape
    reach = max(2, int(GROW_REACH * max(ys.max() - ys.min(), xs.max() - xs.min())))
    y0, y1 = max(0, ys.min() - reach), min(h, ys.max() + reach + 1)
    x0, x1 = max(0, xs.min() - reach), min(w, xs.max() + reach + 1)
    m, d = mask[y0:y1, x0:x1], depth[y0:y1, x0:x1]
    core = cv2.erode(m, np.ones((9, 9), np.uint8)).astype(bool)      # depth maps blur across outlines
    if core.sum() < 50:
        return mask
    lo, hi = np.percentile(d[core], (5, 95))
    near = ((d >= lo / (1 + GROW)) & (d <= hi * (1 + GROW))).astype(np.uint8)
    grown, step = m.copy(), np.ones((3, 3), np.uint8)
    for _ in range(reach):          # spread only through pixels at the person's depth
        grown |= cv2.dilate(grown, step) & near
    out = mask.copy()
    out[y0:y1, x0:x1] = grown
    return out


def hole_mask(masks, depth):
    """Where people are, at the size of the depth map: the union of their frame-sized masks, each
    widened over the pixels next to it at the person's depth, then a little more because depth maps
    bleed past a person's outline."""
    h, w = depth.shape
    person = np.zeros((h, w), np.uint8)
    for m in masks:
        person |= _grow(cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST), depth)
    return cv2.dilate(person, _disc(BLEED)).astype(bool)


def _body_depth(depth, person):
    """depth (float32) with the pixels of people's masks (person: bool) that show what is behind
    them brought forward to their body (see THROUGH), so that they do not seem to have stood far
    back there."""
    if not person.any():
        return depth
    window = np.ones((2 * THROUGH_REACH + 1, 2 * THROUGH_REACH + 1), np.uint8)
    body = cv2.erode(np.where(person, depth, np.float32(np.inf)), window)   # the nearest of them around
    out = depth.copy()
    out[person] = np.minimum(depth[person], THROUGH * body[person])
    return out


def _disc(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def motion_view(frame, hole, sky):
    """What camera.Tracker needs from a frame: a small grey copy, its contrast evened out (a dark
    frame otherwise yields too few corners to follow), and where the background is."""
    h = round(frame.shape[0] * MOTION_WIDTH / frame.shape[1])
    grey = cv2.cvtColor(cv2.resize(frame, (MOTION_WIDTH, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(grey)
    still = ~hole if sky is None else ~(hole | sky)
    return grey, cv2.resize(still.astype(np.uint8) * 255, (MOTION_WIDTH, h), interpolation=cv2.INTER_NEAREST)


class SetDepth:
    """Builds the sets (splat), one stretch at a time."""

    def __init__(self, device, lens_ratio):
        self.device = device
        self.lens_ratio = lens_ratio        # focal length / image width
        self.gen = torch.Generator(device=device).manual_seed(0)
        self.centre = None                  # principal point in pixels; None = image centre

    def _grid(self, h, w, step=1):
        """Pixel-centre coordinates (y, x), measured from the principal point, of an h x w image or
        of its step-times smaller copy (as avg_pool2d makes it)."""
        cx, cy = self.centre or (w / 2, h / 2)
        dev = self.device
        return torch.meshgrid((torch.arange(h // step, device=dev) + 0.5) - cy / step,
                              (torch.arange(w // step, device=dev) + 0.5) - cx / step, indexing="ij")

    def solve(self, q, hole, sky, f, area, people, dark=None):
        """Planes, hole filling and the final inverse depth for one view (torch tensors on the
        device); area is the size of one video frame in pixels, which plane sizes are measured
        against; people is the inverse depth of the farthest anyone stood at each hole pixel (0
        where not known); dark, if given, where the view is too dark to see. Walls, floors and
        steps come out flat, with straight outlines, as in a set built by hand: the planes found in
        the depth, turned square to each other where the set is built square. What fits none of
        them becomes flat cards (see _cards), and small pieces that lie on a larger surface next to
        them become part of it (see _absorb), so the set is a few large, plain surfaces; a run of
        stairs becomes even steps (see _stairs). What is too dark to see is closed by the surfaces
        around it, as the holes people leave are (a floor carries on under it), and left empty
        only where it is a way into the dark that they cannot close (see _shade, _empty). Returns
        (inverse depth, 0 where sky and -1 where left empty for the dark; plane label per pixel)."""
        width = f / self.lens_ratio             # (of a frame, in pixels)
        unlit = None if dark is None else dark & ~hole & ~sky
        planes, label = self._planes(q, hole, sky, f, area, unlit=unlit)
        planes, label, frame = self._square(q, planes, label, f)
        found = len(planes)         # (the main surfaces, which may close the holes people leave)
        label = self._spread(q, label, planes, ~(hole | sky), SPREAD, SPREAD_STEPS)
        shade, empty, by_people = None, None, hole
        if unlit is not None and unlit.any():       # (the dark is closed like the holes people leave)
            shade, label = self._shade(q, label, planes, f, unlit)
            hole = hole | shade
        # A second search over what is left, for the smaller surfaces the first missed.
        left = (label == 0) & ~hole & ~sky
        more, extra = self._planes(q, hole | ~left, sky, f, area, MIN_STEP)
        label = torch.where(left & (extra > 0), extra + len(planes), label)
        planes = torch.cat([planes, more])
        if frame is not None:       # (the second search's planes square too, none merged)
            planes, label, _ = self._square(q, planes, label, f, frame, found, merge=False)
        label = self._fill_holes(q, hole, sky, planes, label, area, f, people, found)
        if shade is not None:
            empty, label = self._empty(q, label, shade, by_people, width, area)
            sky, hole = sky | empty, hole & ~empty
        planes, label = self._close_rest(q, label, planes, hole, sky, people)
        label = self._take_halo(q, label, planes, hole, sky, people, round(HALO_REACH * width))
        closed = label
        label = self._tidy(label, sky, hole, PIECE * area, max(1, round(THIN * width)))
        label = self._straighten(label, sky, STRAIGHT * width, found, OUTLINE)
        label = self._crease(label, planes, f, width)
        first_card = len(planes)
        planes, label = self._cards(q, label, planes, hole, sky, area)
        if frame is not None:       # (the cards square too)
            planes, label, _ = self._square(q, planes, label, f, frame, first_card)
        label = self._absorb(q, label, planes, hole, people, area)
        label = self._straighten(label, sky, STRAIGHT * width, found, OUTLINE)
        if frame is not None:
            planes, label = self._stairs(q, label, planes, hole, sky, f, frame, area)
        label = self._keep_behind(label, closed, planes, hole, people, area)
        out = self._compose(q, hole, sky, planes, label, people)
        return (out if empty is None else torch.where(empty, torch.full_like(out, -1.0), out)), label

    def splat(self, frames, poses, key, dark=None):
        """One set for a stretch of a shot (see camera.stretches), seen from its keyframe.

        Every pixel of frames from across the stretch is placed in 3D at its depth, where the
        camera was (poses), and laid onto one picture in the keyframe's view; the median of their
        depths is taken (people, and anything else passing through, drop out of it) and its planes
        are found once, so the set holds perfectly still while the camera moves through it.
        frames(t) -> (depth, hole, sky) for t in range(len(poses)); poses[t] = (R, T, factor) of
        frame t relative to the keyframe (see camera.relative); dark(t), if given, -> where frame
        t is too dark to see (see depth.unlit), or None. Returns the picture's inverse depth (0
        where there is no set: sky, or never seen; -1 where it is left empty for the dark) and its
        principal point (cx, cy) in pixels from its top-left corner (same focal length as the
        frames'), or None if too little of it is seen.
        """
        dev = self.device
        n = len(poses)
        depth, hole, sky = frames(key)
        h, w = depth.shape
        f = self.lens_ratio * w
        # Points much nearer the keyframe's camera than what it looks at are seen at too steep an
        # angle from it to place well (the side walls next to a camera that pulls back): left out.
        ok = np.isfinite(depth) & (depth > 0) & ~hole & (True if sky is None else ~sky)
        nearest = NEAREST * float(np.median(depth[ok])) / poses[key][2] if ok.any() else 1e-3
        ys, xs = torch.meshgrid((torch.arange(h, device=dev) + 0.5 - h / 2) / f,
                                (torch.arange(w, device=dev) + 0.5 - w / 2) / f, indexing="ij")
        rays = torch.stack((xs, ys, torch.ones_like(xs)), -1)      # (h, w, 3), camera space

        def place(t, depth, step=1):
            """Pixels of frame t (every step-th) in the keyframe's camera space; directions where
            depth is None."""
            R, T, factor = (torch.tensor(np.asarray(x, np.float32), device=dev) for x in poses[t])
            r = rays[::step, ::step]
            if depth is None:
                return r @ R                        # R^T r, row by row
            return (r * (depth / factor)[..., None] - T) @ R

        def project(p):
            """Keyframe-view pixel coordinates (from the top-left corner) and inverse depth of points."""
            z = p[..., 2]
            ahead = z > nearest
            zs = torch.where(ahead, z, torch.ones_like(z))
            return f * p[..., 0] / zs + w / 2, f * p[..., 1] / zs + h / 2, torch.where(ahead, 1.0 / zs, torch.zeros_like(z))

        picks = np.unique(np.linspace(0, n - 1, min(n, SHOT_SAMPLES)).round().astype(int))
        picks = sorted(picks, key=lambda t: abs(t - key))
        # The picture covers what the frames see, up to CANVAS frame widths and heights.
        reach = (CANVAS - 1) / 2
        lo, hi = np.array([0.0, 0.0]), np.array([w, h], float)
        for t in picks:
            depth, hole, sky = frames(t)
            d = torch.from_numpy(np.where(np.isfinite(depth) & (depth > 0), depth, 0)[::8, ::8].copy()).to(dev)
            u, v, q = project(place(t, d, 8))
            good = (q > 0) & (d > 0)
            if good.sum() > 20:             # (a few stray points do not widen it)
                uv = torch.stack((u[good], v[good]), 1)
                lo = np.minimum(lo, torch.quantile(uv, 0.01, dim=0).cpu().numpy())
                hi = np.maximum(hi, torch.quantile(uv, 0.99, dim=0).cpu().numpy())
        lo = np.floor(np.maximum(lo, [-reach * w, -reach * h]))
        hi = np.ceil(np.minimum(hi, [(1 + reach) * w, (1 + reach) * h]))
        cw, ch = int(hi[0] - lo[0]), int(hi[1] - lo[1])
        centre = (w / 2 - lo[0], h / 2 - lo[1])

        def splat(p, value=None):
            """Canvas index and inverse depth of every point (each covers 2 x 2 canvas pixels, so
            nothing falls through between them where a frame sees from further away)."""
            u, v, q = project(p)
            u, v = u - float(lo[0]) - 0.5, v - float(lo[1]) - 0.5
            j0, i0 = torch.floor(u).long(), torch.floor(v).long()
            idx, val = [], []
            for di in (0, 1):
                for dj in (0, 1):
                    i, j = i0 + di, j0 + dj
                    ok = (q > 0) & (i >= 0) & (i < ch) & (j >= 0) & (j < cw)
                    idx.append((i * cw + j)[ok])
                    val.append((q if value is None else value)[ok])
            return torch.cat(idx), torch.cat(val)

        stack = np.full((len(picks), ch, cw), np.nan, np.float32)
        count = torch.zeros(ch * cw, device=dev, dtype=torch.int32)
        looks = torch.zeros(ch * cw, device=dev, dtype=torch.int32)
        sky_votes = torch.zeros(ch * cw, device=dev, dtype=torch.int32)
        dark_votes = torch.zeros(ch * cw, device=dev, dtype=torch.int32)
        # The farthest anyone stood at each spot: what closes the hole there must stay behind it,
        # or it would hide them in the frames they stood further back (walking towards the camera).
        people = torch.full((ch * cw,), float("inf"), device=dev)
        for k, t in enumerate(picks):
            depth, hole, sky = frames(t)
            sky = np.zeros((h, w), bool) if sky is None else sky
            finite = np.isfinite(depth) & (depth > 0)
            dn = np.where(finite, depth, 1.0).astype(np.float32)
            d = torch.from_numpy(dn).to(dev)
            background = torch.from_numpy(finite & ~hole & ~sky).to(dev)
            # (how far back people stood comes from the people themselves, not the margin around
            # them that hole_mask adds for bleeding depth, which shows what is behind them, and at
            # their outline from the body just inside it)
            proper = cv2.erode(hole.astype(np.uint8), _disc(BLEED)).astype(bool)
            person = torch.from_numpy(finite & proper & ~sky).to(dev)
            margin = torch.from_numpy(finite & hole & ~proper & ~sky).to(dev)
            skyt = torch.from_numpy(sky).to(dev)
            p = place(t, d)
            q = torch.zeros(ch * cw, device=dev)
            idx, val = splat(p[background])
            q.scatter_reduce_(0, idx, val, "amax")      # where the frame saw two things, the nearer
            seen = torch.zeros(ch * cw, device=dev, dtype=torch.bool)
            seen[idx] = True
            body = torch.from_numpy(_body_depth(dn, proper & finite & ~sky)).to(dev)
            pidx, pval = splat(place(t, body)[person])
            people.scatter_reduce_(0, pidx, pval, "amin")
            midx, _ = splat(p[margin])
            sidx, _ = splat(place(t, None)[skyt], torch.ones(int(skyt.sum()), device=dev))
            looked = seen.clone()
            looked[pidx] = True
            looked[midx] = True
            looked[sidx] = True
            sky_here = torch.zeros_like(seen)
            sky_here[sidx] = True
            count += seen
            looks += looked
            sky_votes += sky_here & ~seen
            unlit = None if dark is None else dark(t)
            if unlit is not None and unlit.any():
                didx, _ = splat(p[torch.from_numpy(unlit).to(dev) & background])
                dark_here = torch.zeros_like(seen)
                dark_here[didx] = True
                dark_votes += dark_here
            stack[k] = torch.where(seen, q, torch.full_like(q, float("nan"))).view(ch, cw).cpu().numpy()
        qc = np.empty((ch, cw), np.float32)
        for r0 in range(0, ch, 128):
            part = torch.from_numpy(np.ascontiguousarray(stack[:, r0:r0 + 128])).to(dev)
            qc[r0:r0 + 128] = part.nanmedian(0).values.cpu().numpy()
        del stack
        count, looks, sky_votes, dark_votes = (x.view(ch, cw).cpu().numpy() for x in (count, looks, sky_votes, dark_votes))
        people = people.view(ch, cw).cpu().numpy()
        people[~np.isfinite(people)] = 0
        unseen = looks == 0
        sky_c = ((2 * sky_votes > looks) & ~unseen) | unseen       # never seen: left out, like the sky
        # ... except specks and thin cracks of it inside the set: spots between the points of
        # something near the camera that no frame happened to put a point on, holes to close.
        close = np.ones((7, 7), np.uint8)
        solid = cv2.morphologyEx((~sky_c).astype(np.uint8), cv2.MORPH_CLOSE, close) > 0
        _, sky_cc, sky_stats, _ = cv2.connectedComponentsWithStats((sky_c & ~solid).astype(np.uint8), connectivity=8)
        sky_c = np.isin(sky_cc, np.flatnonzero(sky_stats[:, cv2.CC_STAT_AREA] >= MIN_PLANE * h * w)) & (sky_cc > 0)
        # Background seen in only a few of the frames that looked at it is mostly where someone
        # stayed (crouching, sitting): those few are likely frames where their outline was missed.
        known = np.isfinite(qc) & ~sky_c & (count >= SEEN * looks)
        if known.sum() < 0.05 * h * w:
            return None
        hole_c = ~known & ~sky_c            # always behind someone; in one piece where it is speckled
        hole_c = cv2.morphologyEx(hole_c.astype(np.uint8), cv2.MORPH_CLOSE, close).astype(np.float32)
        hole_c = (cv2.blur(hole_c, (5, 5)) > 0.5) & ~sky_c
        # What no frame put a point on next to where people stood is mostly the floor just in front
        # of the camera behind them, too near to place (see NEAREST): closed like their holes, not
        # left out like the sky (unless it is large, and so out of view).
        _, un_cc, un_stats, _ = cv2.connectedComponentsWithStats((sky_c & unseen).astype(np.uint8), connectivity=8)
        touch = np.unique(un_cc[cv2.dilate(hole_c.astype(np.uint8), close) > 0])
        small = np.flatnonzero(un_stats[:, cv2.CC_STAT_AREA] < 0.1 * h * w)
        behind = np.isin(un_cc, np.intersect1d(touch, small[small > 0]))
        hole_c, sky_c = hole_c | behind, sky_c & ~behind
        _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
        qc = qc[iy, ix]                     # finite everywhere; solve() decides holes and sky
        self.centre = centre
        dark_c = None if dark is None else torch.from_numpy(2 * dark_votes > count).to(dev)    # (most frames
        try:                                                    # that saw it there saw it too dark)
            out_q, _ = self.solve(torch.from_numpy(qc).to(dev), torch.from_numpy(hole_c).to(dev),
                                  torch.from_numpy(sky_c).to(dev), f, h * w, torch.from_numpy(people).to(dev), dark_c)
        finally:
            self.centre = None
        return np.where(sky_c, 0, out_q.cpu().numpy()).astype(np.float32), centre

    def _planes(self, q, hole, sky, f, area, min_plane=None, unlit=None):
        """Find the view's planes, none smaller than min_plane of a frame (MIN_PLANE if None), and
        label every pixel with its plane (0 = none). None is found in unlit pixels (too dark to
        see, where the depth model makes up what it cannot see), but they take one found elsewhere
        where their depth fits it."""
        min_plane = MIN_PLANE if min_plane is None else min_plane
        dev = q.device
        h, w = q.shape
        # Planes are found at half size (fast), then every full-size pixel picks its plane.
        qh = F.avg_pool2d(q[None, None], 2)[0, 0]
        hh, wh = qh.shape
        off = hole | sky if unlit is None else hole | sky | unlit
        excluded = F.max_pool2d(off.float()[None, None], 2)[0, 0] > 0
        ys, xs = self._grid(h, w, 2)
        qs = _blur(qh, 1.0)
        qu, qv, lap = _derivatives(qs)
        smooth = lap.abs() < CURVATURE * 4 * qs        # (Laplacian is 4x larger at half size)
        normal = _normal(qu, qv, qs - qu * xs - qv * ys, f / 2)
        valid = smooth & ~excluded
        free = valid.clone()
        cos_t = math.cos(math.radians(ANGLE))
        min_px = min_plane * area / 4

        def support(abc):
            res = (qh - (abc[0] * xs + abc[1] * ys + abc[2])).abs() / qh
            return free & (res < TOLERANCE) & ((normal * _normal(*abc, f / 2)).sum(-1).abs() > cos_t)

        def settle(abc):
            for _ in range(3):
                inl = support(abc)
                if inl.sum() < 50:
                    return None, None
                abc = _fit(xs[inl], ys[inl], qh[inl])
            keep = _keep_large(support(abc), 0.5 * min_px)
            if keep.sum() < min_px:
                return None, support(abc)
            return _fit(xs[keep], ys[keep], qh[keep]), keep

        planes, supports = [], []
        for _ in range(3 * MAX_PLANES):
            if len(planes) >= MAX_PLANES:
                break
            idx = free.flatten().nonzero()[:, 0]
            if len(idx) < min_px:
                break
            sample = idx[torch.randint(len(idx), (min(SAMPLE, len(idx)),), generator=self.gen, device=dev)]
            seeds = idx[torch.randint(len(idx), (SEEDS,), generator=self.gen, device=dev)]
            a, b = qu.flatten()[seeds], qv.flatten()[seeds]
            c = qs.flatten()[seeds] - a * xs.flatten()[seeds] - b * ys.flatten()[seeds]
            qS, xS, yS = qh.flatten()[sample], xs.flatten()[sample], ys.flatten()[sample]
            res = (qS[None] - (a[:, None] * xS[None] + b[:, None] * yS[None] + c[:, None])).abs() / qS[None]
            agree = (normal.flatten(0, 1)[sample][None] * _normal(a, b, c, f / 2)[:, None]).sum(-1).abs() > cos_t
            score = ((res < TOLERANCE) & agree).sum(1)
            best = int(score.argmax())
            if score[best] < 0.5 * min_px * len(sample) / len(idx):
                break
            abc, keep = settle(torch.stack((a[best], b[best], c[best])))
            if abc is None:
                if keep is None or not keep.any():
                    break
                free &= ~keep               # only scattered support: do not seed here again
                continue
            planes.append(abc)
            supports.append(keep)
            free &= ~keep
        planes = self._merge(planes, supports, xs, ys, qh, f / 2)

        # Every full-size pixel takes its best plane, if any fits well enough.
        ys, xs = self._grid(h, w)
        if not planes:
            return torch.zeros((0, 3), device=dev), torch.zeros((h, w), device=dev, dtype=torch.long)
        P = torch.stack([torch.stack((p[0] / 2, p[1] / 2, p[2])) for p in planes])    # to full-size pixels
        pred = P[:, 0, None, None] * xs + P[:, 1, None, None] * ys + P[:, 2, None, None]
        qf = _blur(q, 1.5)
        qu, qv, lap = _derivatives(qf)
        smooth = lap.abs() < CURVATURE * qf
        normal = _normal(qu, qv, qf - qu * xs - qv * ys, f)
        cos = torch.einsum("hwc,kc->khw", normal, _normal(P[:, 0], P[:, 1], P[:, 2], f)).abs()
        cost = (q[None] - pred).abs() / q[None] / TOLERANCE
        cost = cost + torch.where(smooth[None], (1 - cos).clamp_min(0) / (1 - cos_t), torch.zeros_like(cos))
        cost = torch.where(pred > 0, cost, torch.full_like(cost, 1e9))
        best, label = cost.min(0)
        label = torch.where(best < 2.0, label + 1, torch.zeros_like(label))
        label[hole | sky] = 0
        # Tidy: majority vote in 5x5 windows, then drop specks.
        K = len(planes) + 1
        votes = F.one_hot(label, K).permute(2, 0, 1).float()[None]
        for _ in range(2):
            votes = F.one_hot(F.avg_pool2d(votes, 5, 1, 2).argmax(1)[0], K).permute(2, 0, 1).float()[None]
        label = votes.argmax(1)[0]
        label[hole | sky] = 0
        lab = label.cpu().numpy().astype(np.int32)
        for k in range(1, K):
            n, cc, stats, _ = cv2.connectedComponentsWithStats((lab == k).astype(np.uint8), connectivity=4)
            small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < 0.25 * min_plane * area) + 1
            if len(small):
                lab[np.isin(cc, small)] = 0
        label = torch.from_numpy(lab).to(dev).long()
        # Grow planes over the unlabelled seams beside them (corners the depth model rounded off).
        return P, self._spread(q, label, P, ~(hole | sky), 4 * TOLERANCE, 8)

    def _spread(self, q, label, P, open_px, tolerance, steps):
        """Planes (label) spread over the unlabelled open_px pixels next to them whose depth is
        within tolerance of theirs, one pixel further each step, at most steps times."""
        h, w = q.shape
        ys, xs = self._grid(h, w)
        Pz = torch.cat([torch.zeros((1, 3), device=q.device), P])
        for _ in range(steps):
            pad = F.pad(label[None, None].float(), (1, 1, 1, 1), mode="replicate")[0, 0].long()
            shifts = torch.stack([pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                                  for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
            p = Pz[shifts]
            r = (q - (p[..., 0] * xs + p[..., 1] * ys + p[..., 2])).abs() / q
            r = torch.where(shifts > 0, r, torch.full_like(r, 1e9))
            rmin, arg = r.min(0)
            take = open_px & (label == 0) & (rmin < tolerance)
            if not take.any():
                break
            label = torch.where(take, shifts.gather(0, arg[None])[0], label)
        return label

    @staticmethod
    def _merge(planes, supports, xs, ys, q, f):
        """Join planes that are the same surface found in pieces (a floor the depth model bent
        slightly), so it comes out as one flat surface without seams."""
        if len(planes) < 2:
            return planes
        while True:
            P = torch.stack(planes)
            n = _normal(P[:, 0], P[:, 1], P[:, 2], f)
            pred = P[:, 0, None, None] * xs + P[:, 1, None, None] * ys + P[:, 2, None, None]
            S = torch.stack(supports).float()
            area = S.sum((1, 2)).clamp_min(1)
            best = None
            for i in range(len(planes)):
                # How far plane i is from plane j, over the pixels plane j explains.
                gap = ((pred[i] - pred).abs() / q * S).sum((1, 2)) / area
                ok = (gap < MERGE_GAP) & ((n @ n[i]).abs() > math.cos(math.radians(MERGE_ANGLE)))
                ok[i] = False
                if ok.any():
                    j = int(torch.where(ok, gap, torch.full_like(gap, 9.0)).argmin())
                    if best is None or gap[j] < best[2]:
                        best = (i, j, float(gap[j]))
            if best is None:
                break
            i, j = sorted(best[:2])
            union = supports[i] | supports[j]
            planes[i], supports[i] = _fit(xs[union], ys[union], q[union]), union
            del planes[j], supports[j]
        return planes

    def _square(self, q, P, label, f, frame=None, first=0, merge=True):
        """Planes first.. of P turned onto the nearest of the set's three main directions (frame;
        found from the planes if None, see _frame) where within SQUARE degrees of one, each at the
        distance that fits its pixels best. With merge, planes that then lie on one another
        (facing the same way, within MERGE_GAP in depth) become one: a wall the depth model bent
        into pieces comes out one flat wall. Returns (planes, label, frame); frame is None where
        the set has no main directions, and nothing is turned then."""
        Pn = P.cpu().numpy().astype(np.float64)
        lab = label.cpu().numpy().astype(np.int64)
        areas = np.bincount(lab.ravel(), minlength=len(Pn) + 1)[1:].astype(np.float64)
        if len(Pn) <= first or not areas[first:].any():
            return P, label, frame
        normal = _unit_normals(Pn, f)
        if frame is None:
            frame = _frame(normal, areas)
            if frame is None:
                return P, label, None
        qn = q.cpu().numpy().astype(np.float64)
        ys, xs = (g.cpu().numpy().astype(np.float64) for g in self._grid(*qn.shape))
        boxes = ndimage.find_objects(lab, len(Pn))
        c = normal @ frame.T
        axis = np.abs(c).argmax(1)
        turn = np.flatnonzero((np.abs(c).max(1) > math.cos(math.radians(SQUARE))) & (areas > 0)
                              & (np.arange(len(Pn)) >= first))

        def distance(ks, m):
            # The plane m . (x / f, y / f, 1) = d / z (unit normal m) through the pixels of planes
            # ks: d at the median of theirs.
            ratio = []
            for k in ks:
                box = boxes[k]
                px = lab[box] == k + 1
                ratio.append((m[0] * xs[box][px] / f + m[1] * ys[box][px] / f + m[2]) / qn[box][px])
            return float(np.median(np.concatenate(ratio)))

        d = {}
        for k in turn:
            m = frame[axis[k]] * np.sign(c[k, axis[k]])
            dk = distance([k], m)
            if dk > 0:              # (else seen edge-on, where the turn would put it behind the camera)
                normal[k], d[k] = m, dk
        groups = {k: [k] for k in range(len(Pn))}
        while merge:
            pairs = [(abs(d[i] - d[j]) / max(d[i], d[j]), i, j) for i, j in itertools.combinations(sorted(d), 2)
                     if normal[i] @ normal[j] > 0.99]
            pairs = [p for p in pairs if p[0] < MERGE_GAP]
            if not pairs:
                break
            _, i, j = min(pairs)
            groups[i] += groups.pop(j)
            del d[j]
            d[i] = distance(groups[i], normal[i])
        new, remap = [], np.zeros(len(Pn) + 1, np.int64)
        for k, ks in sorted(groups.items()):
            new.append(np.array([normal[k][0] / f, normal[k][1] / f, normal[k][2]]) / d[k] if k in d else Pn[k])
            remap[np.asarray(ks) + 1] = len(new)
        return (torch.tensor(np.array(new), dtype=P.dtype, device=P.device),
                torch.from_numpy(remap[lab]).to(label.device), frame)

    def _shade(self, q, label, P, f, unlit):
        """The patches of the dark (unlit) on none of the main surfaces (label, planes P), where
        the depth the model made up fits nothing; pieces of the main surfaces wholly in the dark
        (made-up depth that happens to fit a plane) are taken off them. The patches are closed like
        the holes people leave, by the surfaces around them, or else left empty (see _empty). One
        that is mostly floor is floor, though, as a floor in shadow is still there: a floor next to
        a patch carries on under it, out to RUN_OFF times as far back as what is around the patch,
        where that covers most of it. Of the floors around a patch (facing up to within FLOOR
        degrees and on at least 5% of its edge), the lowest is the floor; the others are steps and
        tables. Returns (patches, label)."""
        lab = label.cpu().numpy().astype(np.int64)
        dark, qn = unlit.cpu().numpy(), q.cpu().numpy()
        for k, box in _boxes(lab):
            n, cc = cv2.connectedComponents((lab[box] == k).astype(np.uint8), connectivity=8)
            lit = np.zeros(n, bool)
            lit[cc[(cc > 0) & ~dark[box]]] = True
            lab[box][(cc > 0) & ~lit[cc]] = 0
        Pn = P.cpu().numpy().astype(np.float64)
        n = Pn * np.array([f, f, 1.0])
        floor = np.r_[False, n[:, 1] > math.cos(math.radians(FLOOR)) * np.linalg.norm(n, axis=1)]
        low = np.r_[0.0, 1 / np.linalg.norm(n, axis=1)]      # (how far below the camera: d of n.X = d)
        if floor.any():
            ys, xs = (g.cpu().numpy() for g in self._grid(*qn.shape))
            _, patches = cv2.connectedComponents((dark & (lab == 0)).astype(np.uint8), connectivity=8)
            for c, box in _boxes(patches, pad=2):
                m, lab_b = patches[box] == c, lab[box]
                ring = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~m & (lab_b > 0)
                ks, counts = np.unique(lab_b[ring], return_counts=True)
                ks = ks[floor[ks] & (counts >= 0.05 * counts.sum())]
                edge = qn[box][ring]
                if len(ks) and (edge > 0).any():
                    k = ks[np.argmax(low[ks])]
                    qk = Pn[k - 1, 0] * xs[box] + Pn[k - 1, 1] * ys[box] + Pn[k - 1, 2]
                    under = m & (qk > edge[edge > 0].min() / RUN_OFF)
                    if 2 * under.sum() >= m.sum():
                        lab_b[under] = k
        return torch.from_numpy(dark & (lab == 0)).to(label.device), torch.from_numpy(lab).to(label.device)

    @staticmethod
    def _empty(q, label, shade, hole, width, area):
        """What is left empty of the patches of the dark (shade, see _shade) once the holes are
        closed: each patch that the surfaces around it could not close all but a little of (see
        CLOSE) and that opens into the dark, the depth there lying well behind what is around it
        (OPENING; else it is something in the dark, coats on a rack, which the planes next to it
        close), whole, so that the dark stays one plain opening, not a patchwork of it and walls
        (the rest of a patch that is closed goes to the plane next to it); and the parts of the
        holes people leave (hole) that are nearer to that than to anything seen. It gets as clean
        an outline as the clutter's (see _tidy, _straighten; a frame is width pixels wide and area
        pixels in all), and what it gives up goes to the plane next to it. Returns (empty, label
        without it)."""
        lab = label.cpu().numpy().astype(np.int64)
        dark, hole_n, qn = shade.cpu().numpy(), hole.cpu().numpy(), q.cpu().numpy()
        empty = np.zeros_like(dark)
        _, patches = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
        for c, box in _boxes(patches, pad=4):
            mine = patches[box] == c
            if (mine & (lab[box] > 0)).sum() < CLOSE * mine.sum():
                ring = cv2.dilate(mine.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
                around = qn[box][ring & ~dark[box] & ~hole_n[box]]
                if not around.size or np.median(qn[box][mine]) < OPENING * np.median(around):
                    empty[box] |= mine
        lab = _nearest(np.where(empty, 0, lab), dark & ~empty & (lab == 0))
        if empty.any() and hole_n.any():
            _, (iy, ix) = ndimage.distance_transform_edt(hole_n, return_indices=True)
            empty |= hole_n & empty[iy, ix]
        if empty.any():         # (no slivers or chips, and straight sides, as the clutter has)
            m = cv2.morphologyEx(empty.astype(np.uint8), cv2.MORPH_OPEN, _disc(max(1, round(THIN * width))))
            _, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = stats[:, cv2.CC_STAT_AREA] >= PIECE * area
            keep[0] = False
            outlines, tree = cv2.findContours(keep[cc].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            m[:] = 0
            if outlines:
                eps = STRAIGHT * OUTLINE * width
                cv2.fillPoly(m, [cv2.approxPolyDP(o, eps, True) for o, t in zip(outlines, tree[0]) if t[3] < 0], 1)
                cv2.fillPoly(m, [cv2.approxPolyDP(o, eps, True) for o, t in zip(outlines, tree[0]) if t[3] >= 0], 0)
            lab = _nearest(np.where(m > 0, 0, lab), empty & (m == 0) & ~hole_n)
            empty = m > 0
        return torch.from_numpy(empty).to(label.device), torch.from_numpy(lab).to(label.device)

    def _fill_holes(self, q, hole, sky, P, label, area, f, people, found):
        """Planes for the holes people leave where no frame saw behind them, one hole at a time.
        A plane beside a hole pixel along its row (a step, a wall) carries on across it where it
        passes behind what is seen at the hole's other end, the nearer of the two in view where
        both do (a step behind a banister); else one on both sides of it along its column does.
        Elsewhere the main surfaces around the hole close it: of the
        planes facing one way, the one with most of the hole's edge (the wall, not a pillar in
        front of it; where it may not close a spot, see below, one of the others), and where
        those facing different ways meet, the nearer is in view (a wall above a floor). Main
        surfaces are the main search's big planes and any plane much of the edge lies on (the top
        step of a stair carries on under someone standing on it, the wall behind them above it).
        Nothing goes in front of the people who stood there (see _limit), where it would hide
        their figures, nor much farther back than everything around the hole (a floor carried on
        towards its horizon runs off for kilometres, which also spoils the shadows of everything
        else). Nor does a main surface close a spot where it would stand in front of what is seen
        at the nearest point of the hole's edge: a big wall from elsewhere carried across an
        opening would show as a panel the shape of the hole."""
        if not hole.any() or not len(P):
            return label
        h, w = q.shape
        lab = label.cpu().numpy()
        hole_n, sky_n, qn = (x.cpu().numpy() for x in (hole, sky, q))
        limits = _limit(people.cpu().numpy())
        Pn = P.cpu().numpy().astype(np.float64)
        normal = Pn * np.array([f, f, 1.0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        parallel = np.abs(normal @ normal.T) > math.cos(math.radians(ANGLE))
        ys, xs = (g.cpu().numpy() for g in self._grid(h, w))
        areas = np.bincount(lab.ravel(), minlength=len(P) + 1)
        src = (lab > 0) & ~hole_n
        above, below = _sides(lab, src, 0)
        cols = np.where((above == below) & hole_n, above, 0)
        before, after = _sides(lab, src, 1)                 # the planes on either side along the row ...
        _, _, edge0, edge1 = _sides(lab, ~hole_n, 1, True)  # ... and where the hole ends on either side
        out = lab.copy()
        n, comp, stats, _ = cv2.connectedComponentsWithStats(hole_n.astype(np.uint8), connectivity=8)
        for c in range(1, n):
            x, y, bw, bh = stats[c, :4]
            y0, y1, x0, x1 = max(0, y - 6), min(h, y + bh + 6), max(0, x - 6), min(w, x + bw + 6)
            region = comp[y0:y1, x0:x1] == c
            ring = cv2.dilate(region.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~hole_n[y0:y1, x0:x1]
            lab_w, sky_w = lab[y0:y1, x0:x1], sky_n[y0:y1, x0:x1]
            source = ring & ((lab_w > 0) | sky_w)
            if not source.any():
                continue
            _, (iy, ix) = ndimage.distance_transform_edt(~source, return_indices=True)
            X, Y = xs[y0:y1, x0:x1], ys[y0:y1, x0:x1]
            limit = limits[y0:y1, x0:x1]
            around, count = np.unique(lab_w[ring & (lab_w > 0)], return_counts=True)
            share = dict(zip(around.tolist(), (count / max(count.sum(), 1)).tolist()))
            cand = [k for k in range(1, len(P) + 1)
                    if share.get(k, 0) >= 0.05 or (k <= found and areas[k] >= BIG_PLANE * area)]
            rank = {k: (share.get(k, 0), areas[k]) for k in cand}
            main = np.array([k for k in cand if not any(parallel[k - 1, j - 1] and rank[j] > rank[k]
                                                        for j in cand if j != k)], int)
            hidden = np.array([k for k in cand if k not in main], int)      # (behind or before a main one)
            fill = np.zeros(region.shape, np.int64)
            edge, seen = qn[y0:y1, x0:x1][ring], qn[y0:y1, x0:x1][iy, ix]
            for group in (main, hidden):        # where no main surface may close a spot, one they hid may
                if not len(group):
                    continue
                pm = Pn[group - 1]
                qm = pm[:, 0, None, None] * X + pm[:, 1, None, None] * Y + pm[:, 2, None, None]
                valid = (qm > max(float(edge.min()) / RUN_OFF, 0.0)) & (qm < 1.5 * float(edge.max()))
                valid &= qm[:, iy, ix] <= seen * (1 + BEHIND)                      # not before the edge
                behind = np.where(valid & (qm <= limit), qm, -1.0)
                fill = np.where((fill == 0) & (behind.max(0) > 0), group[behind.argmax(0)], fill)  # the nearest
                # is in view. (Where none may close a spot, it is left to _close_rest.)
            k = cols[y0:y1, x0:x1]
            pk = Pn[np.maximum(k - 1, 0)]
            qk = pk[..., 0] * X + pk[..., 1] * Y + pk[..., 2]
            fill = np.where((k > 0) & (qk > 0) & (qk <= limit), k, fill)
            # Along its row (which wins over the rest): a plane on either side carries on across
            # where it passes behind what is seen at the hole's other end (a step behind a banister,
            # which carried across would stand in front of the step), the nearer in view where both do.
            r = np.arange(y0, y1)[:, None]
            row, nearest = np.zeros(region.shape, np.int64), np.full(region.shape, -np.inf)
            for k, end in ((before, edge1), (after, edge0)):
                k, end = k[y0:y1, x0:x1], end[y0:y1, x0:x1]
                pk = Pn[np.maximum(k - 1, 0)]
                qk = pk[..., 0] * X + pk[..., 1] * Y + pk[..., 2]
                e = np.clip(end, 0, w - 1)
                there = pk[..., 0] * xs[r, e] + pk[..., 1] * ys[r, e] + pk[..., 2]
                behind = (end < 0) | (end >= w) | sky_n[r, e] | (there <= qn[r, e] * (1 + BEHIND))
                ok = (k > 0) & behind & (qk > 0) & (qk <= limit) & (qk > nearest)
                row, nearest = np.where(ok, k, row), np.where(ok, qk, nearest)
            fill = np.where(row > 0, row, fill)
            fill = np.where(sky_w[iy, ix], 0, fill)            # next to the sky: left open
            out[y0:y1, x0:x1][region] = fill[region]
        return torch.from_numpy(out).to(q.device).long()

    def _close_rest(self, q, label, P, hole, sky, people):
        """What is left of the holes (no plane carries on into it) is closed with cards too, one
        per piece, fitted to a surface painted in from around it and kept behind the people who
        stood there (see _limit); crumbs join the plane next to them. Returns (planes, label)."""
        dev = q.device
        h, w = q.shape
        lab = label.cpu().numpy().astype(np.int64)
        sky_n = sky.cpu().numpy()
        gap = (lab == 0) & hole.cpu().numpy() & ~sky_n
        if not gap.any():
            return P, label
        ys, xs = self._grid(h, w)
        _, (iy, ix) = ndimage.distance_transform_edt(gap | sky_n, return_indices=True)
        qn = q.cpu().numpy().astype(np.float32)[iy, ix]
        lo, hi = float(qn.min()), float(qn.max())
        scaled = ((qn - lo) / max(hi - lo, 1e-9) * 65535).astype(np.uint16)
        qn = cv2.inpaint(scaled, gap.astype(np.uint8), 5, cv2.INPAINT_NS).astype(np.float32) / 65535 * (hi - lo) + lo
        limit = _limit(people.cpu().numpy())
        qn = np.minimum(qn, limit)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(gap.astype(np.uint8), connectivity=8)
        qt, cct, lim = torch.from_numpy(qn).to(dev), torch.from_numpy(cc).to(dev), torch.from_numpy(limit).to(dev)
        cards = []
        for c in np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 64):
            if c > 0:
                m = cct == c
                card = _card(xs[m], ys[m], qt[m])
                # Moved back, as a whole, until (nearly) all of it is behind them.
                over = float(torch.quantile((card[0] * xs[m] + card[1] * ys[m] + card[2]) / lim[m], 0.99))
                cards.append(card / max(over, 1.0))
                lab[cc == c] = len(P) + len(cards)
        lab = _nearest(lab, gap & (lab == 0))
        return (torch.cat([P, torch.stack(cards)]) if cards else P), torch.from_numpy(lab).to(dev)

    def _take_halo(self, q, label, P, hole, sky, people, reach):
        """Depth maps blur a person into what is next to them, so the set keeps a rim a little
        nearer than the surface behind, outlining the spot they stood in. The surfaces that close
        a hole and carry on outside it (not the cards made for it) spread back over the pixels
        next to it that someone stood over, where those lie up to HALO nearer than them (or
        within TOLERANCE behind), up to reach pixels out, and never in front of the people who
        stood there (see _limit)."""
        h, w = q.shape
        ys, xs = self._grid(h, w)
        Pz = torch.cat([torch.zeros((1, 3), device=q.device), P])
        limit = torch.from_numpy(_limit(people.cpu().numpy())).to(q.device)
        band = (people > 0) & ~hole & ~sky
        outside = torch.unique(label[~hole & (label > 0)])
        grow = torch.where(hole & torch.isin(label, outside), label, torch.zeros_like(label))
        far = torch.full_like(q, float("inf"))
        for _ in range(reach):
            pad = F.pad(grow[None, None].float(), (1, 1, 1, 1))[0, 0].long()
            shifts = torch.stack([pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                                  for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
            p = Pz[shifts]
            flat = p[..., 0] * xs + p[..., 1] * ys + p[..., 2]
            r = q / flat.clamp_min(1e-9)
            ok = (shifts > 0) & (flat > 0) & (r >= 1 - TOLERANCE) & (r <= 1 + HALO) & (flat <= limit)
            score, arg = torch.where(ok, (r - 1).abs(), far).min(0)
            take = band & (grow == 0) & torch.isfinite(score)
            if not take.any():
                break
            grow = torch.where(take, shifts.gather(0, arg[None])[0], grow)
        return torch.where(band & (grow > 0), grow, label)

    @staticmethod
    def _tidy(label, sky, hole, piece_px, radius):
        """Pieces of a plane smaller than piece_px pixels and its parts thinner than about 2 *
        radius pixels (slivers, spikes) are no plane: a set built by hand has none. They are taken
        as what fits no plane (see _cards), except in the holes (where the model saw no surface),
        which the planes around them close."""
        lab = label.cpu().numpy().astype(np.int64)
        disc = _disc(radius)
        drop = np.zeros(lab.shape, bool)
        for k, box in _boxes(lab, radius + 1):
            mine = lab[box] == k
            m = cv2.morphologyEx(mine.astype(np.uint8), cv2.MORPH_OPEN, disc)
            _, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = stats[:, cv2.CC_STAT_AREA] >= piece_px
            keep[0] = False
            drop[box] |= mine & ~keep[cc]
        lab[drop] = 0
        return torch.from_numpy(_nearest(lab, (lab == 0) & hole.cpu().numpy() & ~sky.cpu().numpy())).to(label.device)

    @staticmethod
    def _straighten(label, sky, eps, first, coarse):
        """Outlines of planes made straight, as on a set built by hand: each plane's outline becomes
        a polygon within eps pixels of it (coarse times that for the planes after the first first,
        the clutter), and the polygons are laid down largest first, so that smaller ones keep their
        shape on top of larger ones (a pixel no polygon covers keeps its plane)."""
        lab = label.cpu().numpy().astype(np.int32)
        sky_n = sky.cpu().numpy()
        out = lab.copy()
        boxes = dict(_boxes(lab, 1))
        ks, areas = np.unique(lab[lab > 0], return_counts=True)
        for k in ks[np.argsort(-areas)]:
            box, e = boxes[k], eps * (coarse if k > first else 1.0)
            outlines, tree = cv2.findContours((lab[box] == k).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            canvas = np.zeros(out[box].shape, np.uint8)
            cv2.fillPoly(canvas, [cv2.approxPolyDP(o, e, True) for o, t in zip(outlines, tree[0]) if t[3] < 0], 1)
            cv2.fillPoly(canvas, [cv2.approxPolyDP(o, e, True) for o, t in zip(outlines, tree[0]) if t[3] >= 0], 0)
            out[box][canvas.astype(bool) & ~sky_n[box]] = k
        return torch.from_numpy(out).to(label.device).long()

    def _crease(self, label, P, f, width):
        """Planes that meet at an angle (a wall and the floor, two walls, the sides of a box) are
        cut along the line where they meet. Where their depths lie within CREASE_GAP of each other
        along their seam, every pixel within CREASE_REACH of a frame's width (width, in pixels) of
        it goes to the nearer of the two there where they fold towards the camera (a room's
        corner), or to the farther where they fold away (the edge of a box), whichever moves fewer
        pixels: the seam becomes the straight line the set was built with rather than the depth
        model's rounded one. A seam the meeting line does not pass near is left as it is."""
        lab = label.cpu().numpy().astype(np.int64)
        Pn = P.cpu().numpy().astype(np.float64)
        if len(Pn) < 2:
            return label
        h, w = lab.shape
        ys, xs = (g.cpu().numpy().astype(np.float64) for g in self._grid(h, w))
        normal = _unit_normals(Pn, f)
        reach = max(2, round(CREASE_REACH * width))
        K = len(Pn) + 1
        pos, keys = [], []
        for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1], lab[1:])):
            m = (a > 0) & (b > 0) & (a != b)
            pos.append(np.argwhere(m))
            keys.append(np.minimum(a[m], b[m]) * K + np.maximum(a[m], b[m]))
        pos, keys = np.concatenate(pos), np.concatenate(keys)
        order = np.argsort(keys, kind="stable")
        pos, keys = pos[order], keys[order]
        seams, start, count = np.unique(keys, return_index=True, return_counts=True)
        for s in np.argsort(-count):
            if count[s] < MIN_SEAM:
                break
            i, j = divmod(int(seams[s]), K)
            if abs(normal[i - 1] @ normal[j - 1]) > math.cos(math.radians(CREASE_ANGLE)):
                continue
            seam = pos[start[s]:start[s] + count[s]]
            (y0, x0), (y1, x1) = np.maximum(seam.min(0) - reach - 2, 0), np.minimum(seam.max(0) + reach + 3, (h, w))
            sub = lab[y0:y1, x0:x1]                 # (a view: what is cut here is cut in lab)
            si, sj = sub == i, sub == j
            if not si.any() or not sj.any():        # (an earlier cut took one side)
                continue
            di, dj = ndimage.distance_transform_edt(~sj), ndimage.distance_transform_edt(~si)
            X, Y = xs[y0:y1, x0:x1], ys[y0:y1, x0:x1]
            qi = Pn[i - 1, 0] * X + Pn[i - 1, 1] * Y + Pn[i - 1, 2]
            qj = Pn[j - 1, 0] * X + Pn[j - 1, 1] * Y + Pn[j - 1, 2]
            along = (si & (di <= 1.5)) | (sj & (dj <= 1.5))
            gap = np.abs(qi - qj) / np.maximum(np.maximum(qi, qj), 1e-9)
            if np.median(gap[along]) > CREASE_GAP:
                continue
            band = ((si & (di <= reach)) | (sj & (dj <= reach))) & (qi > 0) & (qj > 0)
            front = (qi > qj)[band]
            if front.all() or not front.any():
                continue
            now = sub[band]
            inward, outward = np.where(front, i, j), np.where(front, j, i)
            moved_in, moved_out = (inward != now).sum(), (outward != now).sum()
            if min(moved_in, moved_out) <= 0.5 * band.sum():
                sub[band] = inward if moved_in <= moved_out else outward
        return torch.from_numpy(lab).to(label.device)

    def _cards(self, q, label, P, hole, sky, area):
        """What fits no plane (a railing, a lamp, a radiator, coats on a rack) as flat cards, the
        faces of the simple blocks a set is roughed out with, rather than the model's own surface
        there, which is its lumpy guess at a coat, a baluster or a dark corner. It is split where
        its depth jumps (more than CARD_JUMP between neighbouring pixels); each piece of at least
        CARD_MIN of a frame (area, in pixels) becomes one card through the bulk of its depths (see
        _card), and smaller pieces, and the jumps between pieces, join the surface next to them.
        Returns (planes, label)."""
        lab = label.cpu().numpy().astype(np.int64)
        rest = (lab == 0) & ~sky.cpu().numpy() & ~hole.cpu().numpy()
        if not rest.any():
            return P, label
        lq = np.log(np.maximum(q.cpu().numpy().astype(np.float64), 1e-9))
        jump = np.zeros(lab.shape, bool)
        across, down = np.abs(np.diff(lq, axis=1)) > CARD_JUMP, np.abs(np.diff(lq, axis=0)) > CARD_JUMP
        jump[:, 1:] |= across
        jump[:, :-1] |= across
        jump[1:] |= down
        jump[:-1] |= down
        h, w = lab.shape
        ys, xs = self._grid(h, w)
        _, cc, stats, _ = cv2.connectedComponentsWithStats((rest & ~jump).astype(np.uint8), connectivity=4)
        cct = torch.from_numpy(cc).to(q.device)
        cards = []
        for c in np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= CARD_MIN * area):
            if c > 0:
                m = cct == c
                cards.append(_card(xs[m], ys[m], q[m]))
                lab[cc == c] = len(P) + len(cards)
        lab = _nearest(lab, rest & (lab == 0))
        return (torch.cat([P, torch.stack(cards)]) if cards else P), torch.from_numpy(lab).to(label.device)

    def _absorb(self, q, label, P, hole, people, area):
        """Pieces (connected parts of one plane or card) smaller than ABSORB of a frame (area, in
        pixels) that lie on a larger piece next to them, within ABSORB_FIT of depth at the median
        of their pixels, become part of it, smallest first; pieces smaller than SHARD join the
        piece next to them they lie nearest even so. So a wall the depth model saw lumpy, in
        patches, comes out one wall, no chips are left between the surfaces, and only what stands
        out from them stays a piece of its own. In the holes people leave, a piece's depth is its
        plane's, the model's elsewhere, and it never joins a plane that would lie in front of
        where they stood there (see _limit)."""
        lab = label.cpu().numpy().astype(np.int64)
        if (ABSORB <= 0 and SHARD <= 0) or not (lab > 0).any():
            return label
        Pn = P.cpu().numpy().astype(np.float64)
        qn = q.cpu().numpy().astype(np.float64)
        h, w = lab.shape
        ys, xs = (g.cpu().numpy().astype(np.float64).ravel() for g in self._grid(h, w))
        piece, owner = np.zeros(lab.shape, np.int64), [0]
        for k, box in _boxes(lab):
            n, cc = cv2.connectedComponents((lab[box] == k).astype(np.uint8), connectivity=8)
            piece[box][cc > 0] = cc[cc > 0] + len(owner) - 1
            owner += [k] * (n - 1)
        flat_ids = piece.ravel()
        order = np.argsort(flat_ids, kind="stable")
        starts = np.searchsorted(flat_ids[order], np.arange(len(owner) + 1))
        pixels = [[order[starts[i]:starts[i + 1]]] for i in range(len(owner))]
        size = np.diff(starts)
        p = np.concatenate([np.zeros((1, 3)), Pn])[lab].reshape(-1, 3)
        flat = p[:, 0] * xs + p[:, 1] * ys + p[:, 2]
        in_hole = hole.cpu().numpy().ravel()
        ref = np.where(in_hole | (qn.ravel() <= 0), flat, qn.ravel())
        limit = _limit(people.cpu().numpy()).ravel()
        near = float(qn[qn > 0].min()) / RUN_OFF if (qn > 0).any() else 0.0
        adj = [set() for _ in owner]
        for a, b in ((piece[:, :-1], piece[:, 1:]), (piece[:-1], piece[1:])):
            m = (a > 0) & (b > 0) & (a != b)
            for i, j in np.unique(np.stack([a[m], b[m]], 1), axis=0):
                adj[i].add(j)
                adj[j].add(i)
        root = np.arange(len(owner))
        for i in np.argsort(size[1:], kind="stable") + 1:
            if size[i] >= max(ABSORB, SHARD) * area:
                continue
            px = np.concatenate(pixels[i])
            r = ref[px]
            best, best_off = 0, math.inf
            for j in adj[i]:
                if size[j] < size[i]:
                    continue
                a, b, c = Pn[owner[j] - 1]
                pred = a * xs[px] + b * ys[px] + c
                if pred.min() <= near or ((pred > limit[px]) & in_hole[px]).any():  # (it would run off
                    continue        # behind everything there, see RUN_OFF, or stand in front of someone)
                off = float(np.median(np.abs(pred - r) / r))
                if off < best_off:
                    best, best_off = j, off
            if not best or (best_off > ABSORB_FIT and size[i] >= SHARD * area):
                continue
            root[i] = best
            pixels[best] += pixels[i]
            size[best] += size[i]
            for m in adj[i]:
                adj[m].discard(i)
                if m != best:
                    adj[m].add(best)
                    adj[best].add(m)
            adj[i] = set()
        for i in range(len(owner)):         # (each piece to the piece it became part of, in the end)
            j = i
            while root[j] != j:
                j = root[j]
            root[i] = j
        final = np.asarray(owner)[root]
        return torch.from_numpy(np.where(piece > 0, final[piece], lab)).to(label.device)

    def _stairs(self, q, label, P, hole, sky, f, frame, area):
        """A run of stairs rebuilt as even steps, as a set builder makes them: the depth model sees
        the steps, but they are small and its rounded corners tilt them, so the plane search breaks
        them up unevenly. Risers are found facing one of the floor's two main directions (frame:
        the set's main directions; see _riser_run); the longest run of at least MIN_STAIRS gets one
        rise and one run fitted to it, and its risers and treads are laid where the surface lies
        within STAIR_FIT of depth of them, or people hid it, next to the risers found: sideways as
        far as the surface lies on them, and down to the floor and up to the landing one step at a
        time while it bears them out (lies on at least half of what the step would show).
        Returns (planes, label)."""
        h, w = q.shape
        ys, xs = self._grid(h, w)
        up = int(np.abs(frame[:, 1]).argmax())          # (the main direction nearest the picture's up-down)
        su = -np.sign(frame[up, 1])                     # (heights count up; the picture's y runs down)
        qs = _blur(q.clamp_min(1e-6), 1.0)
        qu, qv, _ = _derivatives(qs)
        Fr = torch.tensor(frame, dtype=q.dtype, device=q.device)
        facing = (_normal(qu, qv, qs - qu * xs - qv * ys, f) @ Fr.T).abs().cpu().numpy()
        ray = torch.stack((xs / f, ys / f, torch.ones_like(xs)), dim=-1).cpu().numpy().astype(np.float64) @ frame.T
        qn = q.cpu().numpy().astype(np.float64)
        z = 1.0 / np.maximum(qn, 1e-9)
        hole_n, sky_n = hole.cpu().numpy(), sky.cpu().numpy()
        seen = ~hole_n & ~sky_n & (qn > 0)
        best = None
        for run in (a for a in range(3) if a != up):
            sv = np.sign(frame[run, 2]) or 1.0          # (distances count away from the camera)
            u, v, s = su * ray[..., up] * z, sv * ray[..., run] * z, ray[..., 3 - up - run] * z
            steps, cc = _riser_run(u, v, s, seen & (facing[..., run] > math.cos(math.radians(STAIR_ANGLE))) & (v > 0),
                                   MIN_RISER * area)
            even = _even_steps(steps) if len(steps) >= MIN_STAIRS else None
            if even is not None and (best is None or len(steps) > len(best[0])):
                best = steps, cc, run, sv, even
        if best is None:
            return P, label
        steps, cc, run, sv, (v0, run_d, h0, rise) = best
        side = 3 - up - run
        K = len(steps)
        Du, Dv, Ds = su * ray[..., up], sv * ray[..., run], ray[..., side]
        Dv_, Du_ = np.where(np.abs(Dv) > 1e-12, Dv, np.nan), np.where(np.abs(Du) > 1e-12, Du, np.nan)
        found = np.isin(cc, [t["c"] for t in steps])

        def cast(ks, s0, s1):
            # Each pixel's ray to the nearest riser or tread of steps ks (riser i at v0 + i run_d
            # from bottom h0 + i rise up one rise; its tread on top, one run deep): (depth, step,
            # tread or riser), inf and -1 where it meets none.
            near = np.full((h, w), np.inf)
            idx, tread = np.full((h, w), -1), np.zeros((h, w), bool)
            for i in ks:
                vk, bottom = v0 + i * run_d, h0 + i * rise
                for is_tread, zk in ((False, vk / Dv_), (True, (bottom + rise) / Du_)):
                    uk, sk, vk_ = Du * zk, Ds * zk, Dv * zk
                    inside = (vk <= vk_) & (vk_ <= vk + run_d) if is_tread else (bottom <= uk) & (uk <= bottom + rise)
                    hit = (zk > 0) & inside & (s0 <= sk) & (sk <= s1) & (zk < near)
                    near, idx, tread = np.where(hit, zk, near), np.where(hit, i, idx), np.where(hit, is_tread, tread)
            return near, idx, tread

        def joined(mask):
            # the parts of mask (8-connected) that take in some of the risers found
            n, parts = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            keep = np.zeros(n, bool)
            keep[np.unique(parts[found & mask])] = True
            keep[0] = False
            return keep[parts]

        def lies(near):
            return seen & np.isfinite(near) & (np.abs(qn * near - 1.0) < STAIR_FIT)

        near, _, _ = cast(range(K), -np.inf, np.inf)
        on = joined(lies(near) | (hole_n & np.isfinite(near))) & lies(near)
        if on.sum() < 50:
            return P, label
        s0, s1 = np.percentile(ray[..., side][on] * near[on], [1, 99])
        lo, hi = 0, K
        for down in (True, False):
            while True:
                extra = lo - 1 if down else hi
                near, idx, _ = cast(range(min(lo, extra), max(hi, extra + 1)), s0, s1)
                shown = (idx == extra) & seen
                if shown.sum() < MIN_RISER * area or lies(near)[shown].mean() < 0.5:
                    break
                lo, hi = min(lo, extra), max(hi, extra + 1)
        near, idx, tread = cast(range(lo, hi), s0, s1)
        on = joined(np.isfinite(near) & ~sky_n & (lies(near) | hole_n))
        lab = label.cpu().numpy().astype(np.int64)
        new = []
        for i in range(lo, hi):
            for is_tread, m, d in ((False, sv * frame[run], v0 + i * run_d), (True, su * frame[up], h0 + (i + 1) * rise)):
                new.append(np.array([m[0] / f, m[1] / f, m[2]]) / d)
                lab[on & (idx == i) & (tread == is_tread)] = len(P) + len(new)
        return (torch.cat([P, torch.tensor(np.array(new), dtype=P.dtype, device=P.device)]),
                torch.from_numpy(lab).to(label.device))

    def _keep_behind(self, label, closed, P, hole, people, area):
        """Hole pixels that tidying or straightening put on a plane in front of the people who
        stood there (see _limit) keep the plane the holes were closed with (closed), in patches of
        at least SHARD of a frame (area, in pixels): smaller ones would be specks in the surface,
        and _compose keeps them behind the people anyway."""
        h, w = label.shape
        ys, xs = self._grid(h, w)
        p = torch.cat([torch.zeros((1, 3), device=P.device), P])[label]
        flat = p[..., 0] * xs + p[..., 1] * ys + p[..., 2]
        limit = torch.from_numpy(_limit(people.cpu().numpy())).to(P.device)
        back = (hole & (flat > limit)).cpu().numpy()
        if SHARD > 0 and back.any():
            _, cc, stats, _ = cv2.connectedComponentsWithStats(back.astype(np.uint8), connectivity=8)
            back = (stats[:, cv2.CC_STAT_AREA] >= SHARD * area)[cc] & back
        return torch.where(torch.from_numpy(back).to(label.device), closed, label)

    def _compose(self, q, hole, sky, P, label, people):
        """Inverse depth of the set: its planes and cards, the model's own (smoothed) surface
        where neither may lie (see RUN_OFF), and a painted-in surface where a hole could not be closed;
        in the holes, never in front of the people who stood there (see _limit)."""
        h, w = q.shape
        dev = q.device
        qn = q.cpu().numpy().astype(np.float32)
        gap = (hole & (label == 0)).cpu().numpy()
        if gap.any():
            lo, hi = float(qn.min()), float(qn.max())
            scaled = ((qn - lo) / max(hi - lo, 1e-9) * 65535).astype(np.uint16)
            filled = cv2.inpaint(scaled, gap.astype(np.uint8), 5, cv2.INPAINT_TELEA)
            qn = filled.astype(np.float32) / 65535 * (hi - lo) + lo
        qn = cv2.GaussianBlur(cv2.medianBlur(qn, 5), (0, 0), 1.2)   # calms the model's fine ripples
        out = torch.from_numpy(qn).to(dev)
        if len(P):
            ys, xs = self._grid(h, w)
            p = torch.cat([torch.zeros((1, 3), device=dev), P])[label]
            flat = p[..., 0] * xs + p[..., 1] * ys + p[..., 2]
            # (a plane that tidying or straightening carried onto a pixel can run off there far
            # behind everything seen: such a pixel keeps the model's surface; see RUN_OFF)
            out = torch.where((label > 0) & (flat > float(q.min()) / RUN_OFF), flat, out)
        limit = torch.from_numpy(_limit(people.cpu().numpy())).to(dev)
        out = torch.where(hole, torch.minimum(out, limit), out)
        return torch.where(sky, torch.zeros_like(out), out.clamp_min(1e-6))
