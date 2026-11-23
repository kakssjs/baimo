# Baimo workflow guide

How the video → white-model website works, how to run it, how to change it, and what to build next.
For a short description of what the site does, see [README.md](README.md).

---

## 1. The workflow at a glance

```
 You                     Website                         Processing (one video at a time)
 ───                     ───────                         ────────────────────────────────
 choose a video  ──▶  POST /api/jobs  ──▶ queue ──▶  1 ingest    ffmpeg   → source.mp4
                                                     2 analyse   AI       → sets/*.npy, scene.json
 watch the stages ◀── GET /api/jobs  ◀── job.json ◀─ 3 build     Blender  → scene.blend
                                                     4 render    Blender  → white.mp4
 play / download ◀──  GET …/files/…  ◀─────────────  5 compose   ffmpeg   → final.mp4
```

Every video becomes one **job** with its own folder `jobs/<id>/`. The page polls the job's
`job.json` once a second and shows the current stage.

| Stage (page label) | What happens | Code | Output |
|---|---|---|---|
| `queued` (Waiting) | Upload saved, waiting for the worker | `server/app.py` | `input.<ext>`, `job.json` |
| `ingest` (Preparing video) | Check limits; convert to constant frame rate, even size, H.264 + AAC | `media.ingest` | `source.mp4` |
| `analyse` (Reading depth and people) | See section 2 | `pipeline/analyse.py`, `depth.py`, `camera.py`, `planes.py`, `figures.py`, `mannequin.py` | `sets/NNN.npy`, `scene.json` |
| `build` (Building 3D scene) | Blender builds the sets, the figures, the moving camera and the light | `pipeline/blender/build_scene.py` | `scene.blend` |
| `render` (Rendering in Blender) | Blender renders every frame (the .blend, which holds the whole scene, is zipped later, on its first download) | `pipeline/blender/render_shot.py` | `white.mp4` (`scene.zip`) |
| `compose` (Combining) | Original (scaled to the render size) stacked over the white model, original audio kept | `media.compose` | `final.mp4` |
| `done` / `failed` | Result shown, or the error message | `pipeline/job.py` | — |

---

## 2. How a frame becomes a white model

**Ingest.** Before anything else the picture area is found (`media.find_picture`): black bars
are cut off, and in a video that is already a comparison (an original above, below or beside a
render) only the filmed half is kept. Shot cuts are found by checking whether each frame still
correlates with the one before it; nothing is carried across a cut.

**Analyse (Python, GPU)** runs in three passes (`pipeline/analyse.py`).

*1. Look at every frame.*

- **People**: RT-DETR finds each person, ViTPose finds their 17 body keypoints, and SAM 2
  traces each person's exact outline. The outline is the hole taken out of the set, widened
  over the pixels next to it that lie at the person's own depth (the outline model misses hair,
  hands and loose clothes), and then by 7 pixels all round, as a depth map blurs a person into
  what is next to them.
- **Depth**: Depth Anything 3 (monocular, large) estimates depth and the sky at 1008 pixels on
  the long side. Where a frame is too dark to show anything (patches nearly black and nearly
  even over 2% of its width, together at least 1% of the frame), that is marked too: the model
  makes depth up there.
- **Background features**: up to 800 corners on the background (people and sky masked out)
  are followed from frame to frame (`camera.Tracker`). The frames' contrast is evened out first
  (a dark frame otherwise has too few corners to follow), and only features that also track
  back to where they started are kept.

*Between passes: one track per person* (`pipeline/figures.py`).

- Detections are linked frame to frame by box overlap (or a box of the same size close by, for
  fast motion), and a person missed for up to 24 frames is bridged.
- Two tracks are joined when one ends and the other starts soon after (within 48 frames) and
  close by. This covers a detector that lost someone for a while, or a box that changed shape
  too fast (getting up, lying down).
- A person the detector boxed twice becomes one track, also when the second box covers only
  part of them (their upper half, say).
- Tracks are dropped as false detections when they have fewer than 8 detections, or when the
  detector or the pose model was typically unsure about them (coats on a rack, statues,
  posters).
- Where a kept person was missed, SAM 2 outlines them again inside the bridged box, so they are
  still taken out of the set there.

*2. The camera and the sets* (`pipeline/camera.py`, `pipeline/planes.py`).

- **Camera path.** Through each shot, the followed features are placed in 3D at their depth,
  which builds up a map of the background. Each frame's camera is placed and turned so that the
  map's points land where the frame sees them (perspective-n-point with RANSAC), and the depth
  model's scale in that frame is matched to the map's. A move that is not plausible (a turn of
  more than 20°, or a step of more than a quarter of the distance to what it sees, in one frame)
  is not taken; such frames, and frames with too few features, are filled in from their
  neighbours, and the path is smoothed over about 1.5 frames (the shake of a hand-held camera
  stays).
- **Stretches.** A shot is split where the camera has turned more than 30°, or moved more than
  0.3 times the distance to what it sees, since the stretch began; a camera that stands still or
  pans gently keeps one stretch. Each stretch gets a keyframe: the frame whose view takes in
  most of what the others see (the last frame of a pull-back).
- **Sets.** For each stretch, every pixel of up to 40 of its frames (and of the 6 frames on
  either side) is placed in 3D at its depth and laid onto one picture in the keyframe's view.
  The median of the depths at each spot is taken, so people and anything else passing through
  drop out. A spot counts as seen when it is uncovered in at least 30% of the frames that looked
  at it, and as too dark when it was too dark in most of the frames that saw it. What is very
  near the keyframe's camera (less than 15% of the distance to what it looks at) is left out, as
  it is seen too steeply from there to place well.
- **Planes.** The work is done in inverse depth (1 / depth), where every flat surface is a linear
  function of pixel position. Walls, floors, steps and furniture faces are found as planes: a
  pixel joins a plane when it is within 2% of it in depth and faces the same way (planes are
  only looked for in what is lit). Where the set is built square (at least half of the main
  planes' area lines up with three directions at right angles), every plane within 20° of one of
  them is turned onto it, so walls, floors and steps come out square, as built. Planes that are
  then really one surface are merged, and each takes in the pixels next to it that are within
  2.5% of it (mouldings, seams). A second search over what is left finds the smaller surfaces
  the first missed.
- **Cards.** Everything else (railings, lamps, radiators, coats on a rack) becomes flat cards,
  the faces of the simple blocks a set is roughed out with. It is split where its depth jumps
  by more than 5% from one pixel to the next, and each piece of at least 0.4% of the frame
  becomes one flat card through the bulk of its depths (facing the camera where a tilted one
  would run far past them), turned square like the planes where the set is built square;
  smaller pieces join the surface next to them. (The depth model's own surface there, kept in
  relief before, was a lumpy guess at coats, balusters and dark corners and made the set look
  cluttered.) A plane's pieces smaller than 1% of the frame, and its slivers and spikes thinner
  than 0.4% of the frame's width, are no plane: they become cards too, except in the holes,
  where the planes around them close them.
- **Few, plain surfaces.** Then every piece (a connected part of one plane or card) smaller
  than 10% of the frame that lies on a larger piece next to it, within 12% of depth at the
  median of its pixels, becomes part of it, smallest first, and pieces under 0.5% of the frame
  join the piece next to them they lie nearest even so. So a wall the depth model saw lumpy,
  in patches (most of all in dark footage), comes out one wall, no chips are left between the
  surfaces, and only what stands out from them (a staircase's side, a counter, a door frame)
  stays a piece of its own; small things close to a wall (a wall lamp, a picture, a door
  panel) become part of it. In the holes people leave, a piece never joins a plane that would
  stand in front of where they stood. Each outline is then straightened into a polygon (the
  second search's and the cards' twice as coarsely as the main planes'), and where two planes
  meet at a fold (10° or more apart, within 3% of each other in depth along their seam), the
  seam is moved onto the line where they really meet. So walls, floors and steps are flat,
  corners sharp and edges straight and crisp, as on a set built by hand.
- **Stairs.** The depth model sees a staircase's steps, but they are small and its rounded
  corners tilt them, so the plane search breaks them up unevenly. So where the set is built
  square, what faces one of the floor's two main directions (to within 40°) is split into faces
  where its distance jumps by more than 1% between neighbouring pixels; faces of at least 0.03%
  of the frame that stack like steps (each starting within 35% of a rise of where the one before
  ends in height, about as high, at most five rises further back, and side by side with it) are
  a run of stairs. The longest run gets one rise and one run fitted to it. It is a staircase when
  it has four or more risers, the fit is within 40% of a rise or run at every one of them, and it
  climbs at between 10° and 60°. Its risers and treads then take every pixel whose surface lies
  within 5% of depth of them, or that people hid, next to the risers found: sideways as far as
  the surface lies on them, and down to the floor and up to the landing one step at a time,
  while at least half of what the next step would show lies on it. So the steps come out even,
  with straight edges.
- **The dark.** Where the set was too dark to see, its depth is not used. A patch of it that is
  mostly floor is floor: the floor next to it carries on under it (a floor in shadow is still
  there). The rest is closed like the holes people leave (next bullet), by the walls, floors and
  ceilings around it, as the reference films do. A patch that they cannot close nearly all of
  (90%), and that lies behind what is around it (at most 75% of its inverse depth), is a way
  into the dark, such as a doorway into an unlit room. It is left empty, in one piece with a
  straight outline, and so is the part of a hole that lies within it. A dark patch that does
  not lie behind its surroundings is something in the dark (coats on a rack), which the planes
  next to it close.
- **Holes.** Where a spot was behind people in most frames, it is closed from around it. A
  plane beside it along its row (a step, a wall) carries on across where it passes behind what
  is seen at the hole's other end; where the planes on both sides do, the nearer one in view
  does (a step seen past a banister: carried on across, the banister's plane would stand in
  front of the step). Else a plane on both sides of it along its column carries on across.
  Elsewhere the main surfaces around the hole close it: of the planes facing one way,
  the one with most of the hole's edge (the wall, not a pillar in front of it), and where those
  facing different ways meet, the nearer one shows, which puts the corner in the right place.
  A main surface may not close a spot where it would stand in front of what is seen at the
  nearest point of the hole's edge (a big wall from elsewhere carried across an opening shows
  as a flat panel the shape of the hole); there one of the planes facing the same way that it
  won over may. Nothing may close a hole more than 5% nearer than the farthest anyone stood
  there (someone walking towards the camera stood further back earlier, and would be hidden).
  How far back people stood comes from the people themselves, not from the margin around them
  (which shows what is behind them), and a pixel of their mask more than half again as far
  back as the nearest of them within 48 pixels counts no farther back than that: there the
  mask shows what is behind them (their blurred outline, a gap between arm and body), and the
  wall behind them could not close the hole there. Nor may a main surface close a hole more
  than half again as far back as anything around it: a floor carried on past the foot of a
  wall runs off towards its horizon, kilometres away, and that spoils the shadows on
  everything else. What is still open becomes a card fitted to a surface painted in from
  around it, moved back behind them. The surface that closes a hole also takes back the thin
  rim next to it where the depth map blurred the person into it (up to 10% nearer, up to 3% of
  the frame's width out), so no outline of the person stays in the set. A spot next to people
  that no frame could place (mostly the floor just in front of the camera) is closed the same
  way, not left empty.
- **Hand-over.** Each set is shown for its own stretch and stands perfectly still while the
  camera moves through it; where one stretch ends, the next set takes over. Each set is also
  built from the 6 frames on either side of its stretch, so neighbours match where they meet,
  and over its last 8 frames a set takes on the next one's shape wherever both show the same
  surface, so the switch itself changes next to nothing.
- **Storage.** Each set is `sets/NNN.npy`: inverse depth in the shot's units on its keyframe's
  picture, 0 where there is nothing (sky, or never seen), and -1 where it is left empty for the
  dark.

*3. Figures* (`pipeline/mannequin.py`).

- Each track gets one mannequin of fixed proportions: an adult 1.75 m tall, scaled to the
  person. Everyone in a shot is about the same size, as all their keypoints together say (the
  torso's length, else the shoulders' width): one person's alone can mislead, as someone
  crawling side on looks narrow. The size is taken high (the 75th percentile), since
  foreshortening only ever makes a body look smaller, and in the fit a body grows more
  readily than it shrinks: where its box is short for its size, it folds instead. It grows by
  a quarter at most, about as far as foreshortening explains: frames that are not that person
  (a false detection joined onto their track) can otherwise call for a figure twice the size.
- **On the floor.** The band just below a person's box (1% to 5% of the frame's height down)
  shows what they stand on. Where it is the floor (no nearer than 80% of their depth), their
  feet are at the box's bottom and they stand on the set there: on the set as the frame's camera
  sees it through that band, carried on up to the box's bottom (a floor exactly, a stair along
  its slope). The depth model puts people a little in front of what they stand on, and the
  floor just below someone on a stair can lie well in front of the set's, so they would float.
  It is not carried across an edge (a jump of over 12% in depth from one line of the band to
  the next): of the pieces either side, the nearest is what they stand on, such as the riser
  just below their step, while what lies further back is the floor beyond the step's edge or
  the hole they left in the set, closed with what was behind them. Nor is the set what they
  stand on where it lies more than twice as deep as they are (or half as deep): that is a floor
  beyond what hides their feet, such as the floor far down a dark corridor, and standing on it
  would make them giants far back in the set. How deep the set is under their feet is the median
  over the nine frames around (where it was found), as one frame's can be well off, such as below
  someone at the frame's edge. Where the floor is not seen,
  they stand as much deeper than their outline as it says elsewhere in the track. The top of a
  desk or counter just below someone standing behind it can pass for the floor for a moment, so
  where most of the second around it (25 frames) says a desk hides the floor, it does. Where
  someone else's box covers the band, that person hides the floor, even where the band's depth
  would pass for it, and their body reaches no further down than that person's box; where
  their box reaches the frame's bottom edge, or a desk hides what is below it, their legs may be
  anywhere down there.
- **Who is in front** of whom is what the depth model says: someone 10% nearer is in front
  (where it has no depth for one of them, someone whose box reaches 2% further down the
  picture). The boxes cannot tell: the box of someone behind can take in part of whoever is in
  front, and then both reach as far down.
- It is fitted in 3D to every frame of the track at once, by gradient descent:
  - joints land on the keypoints in the picture. Keypoints inside the box of someone in front
    of them are left out, as the pose model, given a box that takes in part of that person, can
    put joints on them;
  - where they stand on the set, the body's lowest part in the picture (a sole, or a knee or a
    hand where they kneel or crawl) is as deep as the set there, held about as firmly as the
    keypoints hold the joints, so the figure stands on the set rather than coming nearer to look
    as big as its box; on a stair the lower foot is a step nearer than the other, and the legs
    take that up. The soles are the shin boxes' flat ends, and nothing of the body reaches below
    them. Where the floor is not seen, the hips keep loosely to the person's depth;
  - the body fills the person's box from edge to edge where the picture shows them, so it is as
    big as the person looks, and a short box makes it crouch or crawl. An edge is not used where
    the frame's edge cuts it, or another person's box overlaps it or ends within 2% of it (the
    one may hide the other there, or the box take them in). Where the box only bridges a gap in
    the detections, its edges count for less;
  - someone on all fours seen side on (a box less than 60% of their height tall and wider than
    tall, over most of nine frames) starts out lying along it, chest down, head towards the side
    their face is on (or was last seen on), so the fit finds the crawl rather than a kneel; the
    crawl goes on, before and after, while the box stays that short (turned towards the camera,
    or partly hidden, a crawler's box is no wider than tall). A box that a desk or the frame's
    edge cuts off does not count as short: someone behind a desk, or close up, looks short too;
  - the motion is smooth;
  - knees and elbows bend the right way;
  - limbs the picture does not show (out of frame, hidden) hang in a rest pose, so the whole
    body is always there: arms and thighs hang down, shins hang along the body, so on all
    fours they lie back along the floor, knees down.
- Torso and head turn with the person.
- Each shot gets a **scale to metres**: the median person in it is 1.75 m tall, or, with nobody
  in the shot, the typical surface is 4 m away.

**Build (Blender).** One scene in metres, z up. Each shot has its own space, that of its first
frame's camera (looking along +Y), scaled to metres:

- **Sets**: a mesh per set, standing where its keyframe's camera stood.
  - Every pixel of its picture is a vertex out along its line of sight, so the set lines up
    with the footage from the keyframe; a flat 8 × 8-pixel square (walls, floors, steps) is
    merged into one face.
  - Where something near stands in front of something far, the surface is not stretched
    across the gap (that would show as a sheet once the camera moves): it breaks there, and
    what is behind carries on a few pixels under the edge. Not, though, at the top of
    something level with the camera (the treads of a stair and its landing, the top of a
    radiator), which the camera sees edge-on: where the near side is below the far side, the
    sight line between them runs level or a little down (at most 1 in 10), and the far side is
    at most 40% further back, the surface carries on from the front edge back to what stands
    behind it, as that top, and whoever stands there stands on it.
  - Folds sharper than 40° stay as hard creases; everything else is smooth-shaded clay.
  - What was left empty for the dark is a dark navy backdrop, unlit and casting no shadow, as in
    the reference films: one flat surface twice as far back as the farthest of the set, reaching
    3% of the set's width in behind its edges, so no light gap opens beside it as the camera
    moves.
  - Each is shown only during its stretch; a shape key takes it onto the next set's surfaces
    over its last 8 frames.
- **Figures**: for each track, a sphere head, a box torso, two-part box arms and legs, and a
  small teal visor that shows which way the face points.
  - All figures are one colour.
  - Every part keeps one size throughout.
  - Parts are keyframed on every frame and shown only while the person is in the shot.
- **Camera**: placed and turned in every frame along the recovered path.
- **Light**: a soft sky, a sun from over the left shoulder of each shot's first camera (it
  stands still while the camera moves), ambient occlusion and soft shadows, so figures cast
  shadows on the set.

**Render.** EEVEE, at the source size enlarged so the long side is at least 1280 pixels.

---

## 3. First-time setup

Needed on the machine (all already in place here):

| Tool | Where | Check |
|---|---|---|
| Blender 5.x | path in `config.toml` → `[tools].blender` | `& "E:\blander\blender.exe" --version` |
| ffmpeg + ffprobe | on PATH (`winget install Gyan.FFmpeg`) | `ffprobe -version` |
| uv | on PATH | `uv --version` |
| NVIDIA GPU + driver | — | `nvidia-smi` |
| `whitebox.py` helper | folder in `config.toml` → `[paths].whitebox_scripts` | file exists |
| Depth Anything 3 source | folder in `config.toml` → `[paths].depth_anything_3` | see below |

Depth Anything 3 is not on PyPI in a form this project can install, so it is used from a clone
(only its inference code is loaded; its own heavy extras are not needed):

```powershell
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git _vendor/Depth-Anything-3
```

Install the Python packages (FastAPI, PyTorch with CUDA, transformers, OpenCV — about 3 GB the
first time):

```powershell
uv sync
```

The AI models (about 2.2 GB together) download from Hugging Face automatically the first time a
video is processed. Without a GPU the analysis still runs, but far too slowly to be useful.

---

## 4. Daily use

Start the site:

```powershell
uv run uvicorn server.app:app --port 8000
```

Open <http://localhost:8000>, then:

1. **Choose a video** (or drop one on the box). Limits are shown under the box.
2. **Watch the stages** light up. Timing is in section 8.
3. **Play the result** — original on top, white model below.
4. **Download** the comparison video, the white model alone, or the Blender scene.
5. Older videos stay in **Your videos**; select one to see it again, or delete it.

Plain footage and finished comparison videos both work. In a video that is already a comparison
(like the sample files in this folder, with a white model and black bars baked in), only the
filmed half is used; the page notes the crop under the result.

**Without the browser** — process one file from the command line, optionally with a crop of your
own (it replaces the automatic one):

```powershell
uv run python -m pipeline my_clip.mp4
```

```powershell
uv run python -m pipeline bad19affd9d603bbb1ae2e3fe3f4a12b.mp4 --crop "592:240:0:636"
```

`--crop` is `width:height:x:y` in source pixels. The result lands in `jobs/<id>/final.mp4`.

**Editing the white model in Blender** — download the scene (the zip is built on the first
request and can be around a hundred megabytes), unzip it and open `scene.blend`, which holds
everything. The sets are ordinary meshes in `WB/Set` (`Set_001`, … shown only during their
shot); figures are in the `WB/Figures` collections, one per person, with ordinary keyframes on
every part; `WB_Camera` follows the recovered path; `WB_Sun` and the `WB_Clay` material control
the look.

---

## 5. Settings and tuning

| To change | Edit |
|---|---|
| Blender / ffmpeg / library locations | `config.toml` → `[tools]`, `[paths]` |
| Upload limits (size, length, resolution) | `config.toml` → `[limits]` |
| Depth model and its working resolution (quality vs. speed) | `MODEL`, `RES` in `pipeline/depth.py` |
| What counts as one flat surface (looser = fewer, larger planes; stricter = more detail kept) | `TOLERANCE`, `ANGLE`, `CURVATURE`, `MERGE_GAP`, `MERGE_ANGLE`, `MIN_PLANE` in `pipeline/planes.py` |
| When planes are turned square to each other | `SQUARE`, `SQUARE_SHARE` in `pipeline/planes.py` |
| How much around a person's outline is taken out of the set | `GROW`, `GROW_REACH`, `BLEED` in `pipeline/planes.py` |
| Which pixels of a person's mask show what is behind them | `THROUGH`, `THROUGH_REACH` in `pipeline/planes.py` |
| How the rest of the set is cleaned up (the second plane search, no chips or slivers, straight outlines, sharp corners) | `MIN_STEP`, `SPREAD`, `SPREAD_STEPS`, `PIECE`, `THIN`, `CARD_JUMP`, `CARD_MIN`, `STRAIGHT`, `OUTLINE`, `CREASE_GAP`, `CREASE_ANGLE`, `CREASE_REACH`, `MIN_SEAM` in `pipeline/planes.py` |
| How plain the set is (larger = fewer, larger surfaces; 0 = every piece found kept) | `ABSORB`, `ABSORB_FIT`, `SHARD` in `pipeline/planes.py` |
| Which runs of stairs are rebuilt as even steps | `STAIR_ANGLE`, `STAIR_SPLIT`, `MIN_RISER`, `STAIR_GAP`, `STAIR_RUN`, `MIN_STAIRS`, `STAIR_EVEN`, `STAIR_PITCH`, `STAIR_FIT` in `pipeline/planes.py` |
| What counts as too dark to see, and what becomes of it (closed by the surfaces around, or a dark void) | `UNLIT`, `UNLIT_EVEN`, `UNLIT_MIN` in `pipeline/depth.py`; `FLOOR`, `CLOSE`, `OPENING` in `pipeline/planes.py`; `VOID_COLOUR`, `VOID_MARGIN`, `VOID_FAR` in `pipeline/blender/build_scene.py` |
| How the holes people leave are closed | `BIG_PLANE`, `BEHIND`, `RUN_OFF`, `HALO`, `HALO_REACH` in `pipeline/planes.py` |
| How the camera's path is found and smoothed | `FEATURES`, `MIN_POINTS`, `PIXEL`, `MAX_TURN`, `MAX_STEP`, `SMOOTH` in `pipeline/camera.py` |
| How a shot is split into sets, and how each is built | `STRETCH_TURN`, `STRETCH_TRAVEL`, `MIN_STRETCH` in `pipeline/camera.py`; `CANVAS`, `NEAREST`, `SHOT_SAMPLES`, `SEEN`, `BLEND` in `pipeline/planes.py` |
| Assumed lens (changes how deep the scene looks) | `LENS` in `pipeline/analyse.py` |
| Which detections count (confidence, minimum size) | `MIN_PERSON_SCORE`, `MIN_HEIGHT` in `pipeline/figures.py` |
| Which tracks count, and when two tracks are one person | `MIN_TRACK`, `TRACK_SCORE`, `TRACK_KEYPOINTS`, `MAX_GAP`, `STITCH_GAP`, `STITCH_REACH`, `SAME_TRACK`, `SAME_POSE`, `SAME_PART` in `pipeline/figures.py` |
| Which keypoints the figures follow | `MIN_SCORE` in `pipeline/mannequin.py` |
| Where figures stand, and how firmly they keep their size and depth | `OPEN_EDGE`, `BELOW`, `OCCLUDER`, `BRINK`, `GROUND_SPAN`, `GROUND_STEADY`, `FLOOR_SPAN`, `TOUCH`, `NEARER` in `pipeline/analyse.py`; `FLOOR_SLACK`, `HOLD_GROW`, `HOLD_SHRINK`, `GROW_LIMIT`, `BRIDGED`, `DEPTH_LOOSE`, `DEPTH_FIRM` in `pipeline/mannequin.py` |
| When a figure starts out on all fours, and which way it faces | `CRAWL`, `CRAWL_SPAN`, `FACING` in `pipeline/mannequin.py` |
| Figure proportions and part sizes | `HEIGHT`, `TORSO`, `LIMBS`, `PART_SIZES`, `TORSO_CENTRE`, `LIMB_PARTS` in `pipeline/mannequin.py` |
| Figure, visor and set colours; bevel; which set squares become one face | `FIGURE_COLOUR`, `VISOR_COLOUR`, `SET_COLOUR`, `BEVEL`, `BLOCK`, `FLAT`, `CREASE`, `EDGE_ON`, `BACK`, `LEDGE`, `LEDGE_GAP` in `pipeline/blender/build_scene.py` |
| How one set hands over to the next | `HAND_OVER`, `MATCH` in `pipeline/blender/build_scene.py` |
| Light and render samples | the light and render sections at the end of `build_scene.py` |
| Smallest output width | `output_size` in `pipeline/media.py` |
| Stage names on the page | `STAGES` in `web/index.html` |

Restart the server after changing Python files or `config.toml`. The page only needs a refresh.

---

## 6. API

| Endpoint | Purpose |
|---|---|
| `GET /api/limits` | Upload limits |
| `POST /api/jobs` | Upload a video (form field `file`) → job state |
| `GET /api/jobs` | All jobs, newest first |
| `GET /api/jobs/{id}` | One job: `stage`, `error`, `source`, `final` |
| `GET /api/jobs/{id}/files/{name}` | `final.mp4`, `white.mp4`, `source.mp4`, `scene.zip` |
| `DELETE /api/jobs/{id}` | Delete a finished job (409 while it is running) |

Interactive reference: <http://localhost:8000/docs>.

---

## 7. When something goes wrong

| Symptom | Likely cause and fix |
|---|---|
| Job fails: "could not be read as a video" | Not a video, or a damaged file |
| Job fails with a size / length / resolution message | Over the limits in `config.toml`; trim the clip or raise the limit |
| Job fails: "blender not found" / "ffmpeg not found" | Fix the path in `config.toml`; after installing ffmpeg, open a new terminal so PATH updates |
| Fails at *Reading depth and people* with "No module named depth_anything_3" | The clone in section 3 is missing or `[paths].depth_anything_3` points elsewhere |
| Fails at that stage the first time, other message | No internet for the one-time model downloads, or out of GPU memory (close other GPU apps) |
| A person has no figure | Too small, too dark, seen in under 8 frames, or the detector and pose model were unsure of them (a person half hidden behind a counter, say); lower `MIN_HEIGHT`, `MIN_PERSON_SCORE`, `MIN_TRACK`, `TRACK_SCORE` or `TRACK_KEYPOINTS` in `figures.py` (lower values also let in coats, statues and posters) |
| One person becomes two figures, one after the other | They were lost for more than 48 frames or moved far while lost, or there was a cut; raise `STITCH_GAP` or `STITCH_REACH` |
| An arm or leg points the wrong way | The pose model was unsure of those joints; raise `MIN_SCORE` in `mannequin.py` so they hang in the rest pose instead |
| A flat patch or blob remains where a person stood | The spot was behind people in most of the frames that looked at it (someone who stays in one place, or covers much of the frame), and the planes around it could not close it |
| Rows of faint dashes or thin stripes on walls and figures | Part of a set reaches very far away (kilometres), which spreads the sun's shadows too thin. Hole fills are kept near what is around them for this reason; if it shows, look in the `.blend` for set vertices far from the camera |
| The page will not open | The server is not running; start it (section 4) |
| An old video shows a different look | It was processed by an earlier version; upload it again |
| Set looks cut-out or blocky in places | What lies on no flat surface becomes flat cards in its own outline (`CARD_JUMP`, `CARD_MIN` in `pipeline/planes.py`). Dark, blurry or very low-resolution footage carries little depth detail, so fewer planes are found and more of the set is cards |
| A dark doorway or corner is a dark, empty void in the white model | It was too dark to see in most frames, and the surfaces around it could not close it (section 2, *The dark*). Lower `CLOSE` in `planes.py` to close more of the dark with the surfaces around it; lowering `UNLIT` in `depth.py` makes less count as too dark, but the depth the model makes up there then shows as shards |
| `uv` fails with "os error 17" inside the Claude desktop app | Set `$env:UV_CACHE_DIR = "C:\Users\kakssjs\.cache\uv"` first (not needed in a normal terminal) |
| Server was closed mid-job | Start it again; unfinished jobs restart automatically from the first stage |

The full error text of a failed job is in `jobs/<id>/job.json`.

---

## 8. What was verified

Checked on 2026-11-23 with an RTX 5060 laptop GPU (8 GB).

**Through the website.** The reference video `7182b76184e0351fc372b1f7f7d5c203.mp4` was sent to
the website's API as the page sends it. Its film part (a hall with a staircase: 592×424, 30 fps,
787 frames, 26 s, with sound) went through every stage to `final.mp4` (1280×1832, two 1280×916
halves, 787 frames, with sound), `white.mp4` and `scene.blend`, and the page listed it as done
with its three downloads.

| Stage | Time |
|---|---|
| Preparing | 11 s |
| Analysis (depth, people, camera path, sets, figures) | 821 s |
| Building the Blender scene | 10 s |
| Rendering | 176 s |
| Combining | 11 s |
| **Total** | **1029 s (17 min), about 1.3 s per frame** |

The stage times were read every 10 s. At that rate a 60-second clip at 30 fps takes about 40
minutes. `scene.zip` downloaded (36 MB: the `.blend` alone, which holds the whole scene), and the
`.blend` in it opened in Blender with its 7 sets, the figures, the camera and all 787 frames, and
no missing files.

**Four test clips**, run with the same code from the command line. Each clip reused its first
pass (depth, people and camera path) and was checked at four to six sampled frames against the
previous version. Each set was also checked for how far it reaches, and for jumps where one set
hands over to the next.

| Clip | Source |
|---|---|
| Hall and staircase (the reference video) | 592×424, 30 fps, 787 frames |
| Studio lot (well lit, handheld) | 592×240, 30 fps, 327 frames |
| City rooftops (fast aerial camera, a stunt performer, sky) | 592×444, 30 fps, 669 frames |
| Corridor (very dark, 5 shots) | 1280×528, 24 fps, 472 frames |

- **Figures stand on the set.** The height of each figure's lowest shin corner above the set
  straight below it was measured on every frame where the set under the person's feet is found:
  - The man coming down the hall's stairs is a median 2 cm above the step below. The 10th to
    90th percentile runs from 6 cm into it to 10 cm above it; before this version he was 27 cm
    above it.
  - Blender agrees at seven frames on the stairs: at six of them a foot is within 7 cm of the
    step below. The foot stepping down can dip into the step.
  - Everyone in the hall is a median 7 cm off the set, down from 18 cm.
  - In the studio lot it is 2 cm, down from 3 cm.
  - In the corridor the floor under people is found in only one frame.
  - In the rooftop clip the numbers mean little: the camera turns fast and the performer is
    mostly in mid-air.
- **No figure outgrows its shot.** In the rooftop clip a blurred building edge, taken for a
  person, was joined onto the leaping performer's track. Filling its tall box at the depth of
  the street below made his figure 2.3 times the size the shot's people share: a giant torso and
  head over the water tower for about two seconds. With the size limit it is 1.25 times, about
  his size in the picture, and it spreads its arms as he does. Three other figures were over
  the limit, and all three still fill their person's outline:
  - a far walker in the studio lot, 1.6 times before and 1.24 now;
  - the man close to the camera in the corridor, 1.29 times before and 1.17 now;
  - one more in the studio lot, 1.26 times before and 1.22 now.

  In the hall no figure changes size, and its frames render as before (seven sampled frames
  differ by half a grey level on average), so the website run above still stands.
- **Nothing else changed in the picture.** Apart from the leap, the sampled frames of the
  rooftop, studio and corridor clips match the previous version.
- **The website's result matches the command line's.** At eight frames, the man on the stairs
  walks on the steps there too.
- **Sets:** no part of any set is more than 37 m from its camera. The largest jump where one set
  hands over to the next is at the hall's frame 200→201: a visible shift of the set.
- **Earlier checks that still apply:** page layout at phone and desktop width, failed-upload
  message, delete, restart recovery, status codes for unknown jobs and files.

Not verified:

- the whole videos frame by frame (four to eight sampled frames each, plus the stairs);
- the native file picker and real drag-and-drop;
- the over-size upload rejection;
- clips near the 60 s limit;
- opening the `.blend` by hand in Blender's window (it was opened without the window);
- `uv run` from a normal terminal (the environment's Python was called directly).

---

## 9. Known limits and next steps

- **The set is what the camera saw.** Flat surfaces (walls, floors, steps, furniture faces)
  become clean planes with sharp edges; irregular things (coats, plants, lamps) become flat
  cards in their own outline, not their real shape, and small things close to a wall or floor
  (wall lamps, pictures, door panels) become part of it: the set is kept plain on purpose.
  Stairs are made even only where their risers face the camera (looking up a straight flight).
  What no frame saw (the backs of things, the room behind the camera) is missing, so it cannot
  be walked around freely. Dark or motion-blurred footage yields fewer, rougher planes. What was
  too dark to see is the surfaces around it carried on, not what is really there; where it
  opens into the dark it is a flat dark backdrop with no depth.
- **Behind people the set is reconstructed.** Usually other frames saw the spot; otherwise the
  walls and floors around the person carry on behind them. A few small specks of another
  surface can still show in that wall or floor next to a figure (mostly slivers left where
  outlines are straightened), and where the surfaces around a person meet behind them, a thin
  edge can show between them. When a person stays in one place or fills much of the frame, a
  flat patch can show behind the figure.
- **Four alternatives were built and dropped:** fitting real walls and boxes from a multi-view
  reconstruction (nonsense geometry on a handheld shot), a plain box room (clean, but it threw
  away the set detail that the reference videos keep), the depth model's raw surface, lightly
  smoothed (detailed, but soft and blurry-looking, and it flickered), and planes with the depth
  model's surface kept in relief for everything else (recognisable, but lumpy and cluttered
  next to the clean planes; the reference videos rough such things out as simple blocks).
- **The camera path is an estimate.** It is found from features on the background, so a shot
  with little background to follow (a close-up, darkness, a whip pan) gets a rougher path, and
  over a long move its scale drifts. Each set is right for its own stretch, but where one
  stretch hands over to the next, the set can change slightly.
- **People can be missed.** The detector is unsure of people half hidden behind furniture, in
  near darkness or mostly out of frame, and tracks it is unsure of are dropped as false
  detections. Two people who overlap in the picture (one crawling behind another) can be boxed
  as one, and the one behind then has no figure there. Identity is kept across gaps of up to 48
  frames, not across cuts.
- **Crawls and hidden people are partly guesses.** A crawl is found from the box's shape
  (short, and at first wider than tall) and the side the face is on, so someone going down onto
  all fours is shown kneeling until their box says otherwise, and someone crawling straight at
  the camera can stay kneeling. For someone mostly hidden behind another person, the depth
  model's depth can be the two blurred together, so their figure can stand too near.
- **Someone in mid-air is taken to stand on what is below them.** A leap above a water tower is
  placed as deep as the tower.
- **Fast, blurred footage can hold false detections.** In the rooftop test clip, the blurred
  edge of a building is taken for a person for 18 frames (0.6 s) and gets a figure at the
  frame's edge. The performer, who reappears next to it, carries on as that figure, so his
  figure starts at the frame's edge and jumps across to him. His size no longer follows it (see
  section 8).
- **Scale and lens are estimates**, so distances in the `.blend` are approximate.
- One job at a time; no progress percentage inside a stage; no accounts, so anyone who can open
  the site sees and can delete every job; Blender/ffmpeg error text includes server paths.

Next steps, in order of value: catching half-hidden and dark people (and telling apart two
who overlap) without letting in coats and posters; simpler, boxier shapes for what lies on no
plane; then, for public hosting,
accounts, old-job cleanup, a cancel button and cleaned-up error text.
