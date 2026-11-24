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
| `analyse` (Reading depth and people) | See section 2 | `pipeline/analyse.py`, `depth.py`, `camera.py`, `planes.py`, `figures.py`, `mannequin.py` | `sets/NNN.npy`, `sets/NNN_edges.npz`, `sets/NNN_tone.png`, `scene.json` |
| `build` (Building 3D scene) | Blender builds the sets in the footage's greys, the figures, a coloured shape for each thing someone carries, the moving camera and the light | `pipeline/blender/build_scene.py` | `scene.blend` |
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
- **Things people carry**: the same detector finds what people carry, hold, throw or ride on
  (bags, cases, umbrellas, balls, bottles, cups, phones, boards, small animals; `PROPS` in
  `figures.py`), the items a template is made to replace. It is ignored when the detector is
  less than 30% sure of it, when it is less than 2% of the frame's height wide or tall, when it
  covers more than 30% of the frame (a page held up to the camera, a wall), or when its box is
  mostly a person's (40% overlap: that person, taken for a dog or a bird as well). SAM 2
  outlines it and its depth is measured inside the outline, as a person's is. Where someone
  holds it (see below), it is taken out of the set with the people, as it is drawn where it
  goes; what nobody holds stays part of the set.
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
  posters). So are tracks whose keypoints typically lie in an upright line, in a box that the
  frame's side cuts: something upright and narrow at the side of the picture (the blurred edge
  of a building, a pole), which looks like someone standing side on. Over a track, the
  keypoints of someone walking side on spread about a third as wide as they are tall; a track
  is dropped under a fifth.
- Where a kept person was missed, SAM 2 outlines them again inside the bridged box, so they are
  still taken out of the set there.

*Between passes: one track per thing carried* (`figures.track_things`).

- Things are linked into tracks as people are. A track is dropped as a false detection when it
  has fewer than 8 detections, when the detector was typically less than 35% sure of it, or when
  fewer than 60% of its detections say it is one kind of thing (a bag the detector calls a
  handbag in one frame and a suitcase in the next is one kind; a board in one frame and a bird
  in the next is a part of someone, half seen).
- Two tracks at the same time whose boxes typically lie at least 30% one inside the other follow
  one thing boxed twice (the bag, and the bag with the hand on it), and become one.
- Only what someone carries is kept: a thing held in at least 15% of the frames it is seen in.
  Held means that a wrist the pose model is at least 30% sure of lies on its box (to within 30%
  of the box's size), and that the person's box is at least three times as big as the thing's
  (not a part of someone taken for something they hold). The rest (planks taken for books, a
  taxi for a phone, a bag left on the floor) stays in the set.
- Each track takes the name its detections give the most weight (`suitcase`, say).
- A thing found again after it was lost (out of the picture, or in the dark) or after a cut is
  the one of its kind last seen, unless that one is still in view (`analyse.follow_things`). So
  a briefcase carried down the stairs, out of the picture for seconds and on into the next shot
  stays one item, drawn as one shape in one colour.

*2. The camera and the sets* (`pipeline/camera.py`, `pipeline/planes.py`).

- **Camera path.** Through each shot, the followed features are placed in 3D at their depth,
  which builds up a map of the background. Each frame's camera is placed and turned so that the
  map's points land where the frame sees them (perspective-n-point with RANSAC), and the depth
  model's scale in that frame is matched to the map's. A move that is not plausible (a turn of
  more than 20°, or a step of more than a quarter of the distance to what it sees, in one frame)
  is not taken; such frames, and frames with too few features, are filled in from their
  neighbours, and the path is smoothed over about 1.5 frames (the shake of a hand-held camera
  stays).
- **Stretches.** A shot is split where the camera has turned more than 45°, or moved more than
  half the distance to what it sees, since the stretch began; a camera that stands still or
  pans gently keeps one stretch. A stretch lasts at least 24 frames, even in a fast move: a set
  that changes more often makes the walls look unsteady. A last stretch of under 12 frames joins
  the one before it. Each stretch gets a keyframe: the frame whose view takes in most of what
  the others see (the last frame of a pull-back).
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
  median of its pixels, becomes part of it, smallest first, and pieces under 1% of the frame
  join the piece next to them they lie nearest even so. So a wall the depth model saw lumpy,
  in patches (most of all in dark footage), comes out one wall, no chips are left between the
  surfaces, and only what stands out from them (a staircase's side, a counter, a door frame)
  stays a piece of its own; small things close to a wall (a wall lamp, a picture, a door
  panel) become part of it. In the holes people leave, a piece never joins a plane that would
  stand in front of where they stood. Each outline is then rounded off with a disc of 2% of the
  frame's width (notches and teeth the depth model leaves filled, thin bits sticking out cut
  off) and straightened into a polygon within 1% of the frame's width of it (the second
  search's and the cards' twice as coarsely as the main planes'); neither changes a piece by
  more than a quarter of its thickness, so a rail or a lamp keeps its shape. Where the set is
  built square, a side of a plane turned square that runs within 15° of one of the set's lines
  (upright, or level along a wall or floor) is laid along it, as the picture shows that line
  there, its corners moving at most three times as far as the straightening may. Where two
  planes meet at a fold (10° or more apart, within 3% of each other in depth along their seam),
  the seam is moved onto the line where they really meet. So walls, floors and steps are flat,
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
- **Flat everywhere.** Last, whatever is not on its plane (a plane that would run off there, a
  hole pushed back behind the people who stood in it, a surface painted in) is made flat too,
  so the set is flat surfaces only, with no ripples, dents or specks to show as the camera
  moves. Such pixels of one plane at most 2% of the frame's width apart are one patch with the
  plane's pixels between them, and the patch becomes the plane that fits it best, moved back
  behind those people (facing the camera, where that plane would run off). Where pixels pushed
  back behind people are less than half of their patch, they are specks of someone's outline
  that the depth model blurred back, and the patch stays on its plane.
- **Greys.** The set is painted as a set builder paints one, in 6 even greys, light and dark
  where the footage is (`planes.set_tones`). The frames' grey levels are laid onto the set's
  picture with their depths and the median is taken at each spot, so people passing through
  drop out; a spot behind people throughout takes the median grey of the rest of its surface.
  The levels are spread from the 2nd to the 98th percentile over all the sets of the shot
  together, so a wall keeps its grey when one set hands over to the next. Each surface (as
  Build follows them round) gets the median grey seen on it, so walls, floors and steps are one
  even grey each, with no blotches where the light fell unevenly. A part of a surface that
  stands out from the rest of it, lighter or darker by 18% of that spread once an even fall of
  light across the surface is taken out, gets the median grey seen on that part: a window's
  panes, a door's panels, a radiator or a lamp on a wall. Its outline is rounded off first with
  a disc 0.3% of the picture's width in radius (so it is one whole shape, not a fringe of
  specks), and it must then cover at least 0.02% of the picture. Each grey is then rounded to
  the nearest of the 6, from 0.12 (dark grey) to 0.7 (light grey) albedo. A spot that was
  behind people throughout takes the grey of the nearest spot of its own surface that was seen,
  so no outline of them shows.
- **Hand-over.** Each set is shown for its own stretch and stands perfectly still while the
  camera moves through it; where one stretch ends, the next set takes over in a single frame.
  Each set is also built from the 6 frames on either side of its stretch, so neighbours match
  where they meet. They are not blended into each other: two sets agree only in places, and
  bending one towards the other corner by corner twisted its flat faces out of their planes on
  the way, which showed as torn slabs for a third of a second.
- **Storage.** Each set is `sets/NNN.npy`: inverse depth in the shot's units on its keyframe's
  picture, 0 where there is nothing (sky, or never seen), and -1 where it is left empty for the
  dark. Beside it, `sets/NNN_edges.npz` holds where the pixels along its outlines go in the
  mesh, on the straight lines the outlines were drawn as, and which surface each pixel is on
  (see Build), and `sets/NNN_tone.png` its greys, on the same picture.

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
  they stand as much deeper than their outline as it says elsewhere in the track. Where the
  picture shows them, the set is behind them (their outline was cut out of it and closed with
  what was behind them): it is found along the lines of sight through the keypoints the pose
  model is at least 50% sure of (the median of where they meet the set), and taken from the
  frames either side where the detector missed them. A floor more than 10% further back than
  that is not what they stand on but one far below (someone in mid-air over a roof's edge), and
  where they stand on no floor they stand at least 5% of its depth in front of it, so their
  figure is never hidden inside a wall. The top of a
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
    one may hide the other there, or the box take them in), or where the person's own keypoints
    lie more than 3% of the box's size past it (a box can stop short of a blurred arm flung
    out, and would fold the arm in). Where the box only bridges a gap in the detections, its
    edges count for less;
  - someone on all fours seen side on (a box less than 60% of their height tall and wider than
    tall, over most of nine frames) starts out lying along it, chest down, head towards the side
    their face is on (or was last seen on), so the fit finds the crawl rather than a kneel; the
    crawl goes on, before and after, while the box stays that short (turned towards the camera,
    or partly hidden, a crawler's box is no wider than tall). A box that a desk or the frame's
    edge cuts off does not count as short: someone behind a desk, or close up, looks short too;
  - the motion is smooth;
  - knees and elbows bend the right way;
  - a limb the picture does not show (out of frame, hidden, too blurred to place) keeps the pose,
    relative to the body, that it is seen in up to 24 frames before or after (an arm still
    pointing, a leg still mid-stride), blended between the two where it is seen on both sides;
    further from any sight of it, it hangs in a rest pose, so the whole body is always there:
    arms and thighs hang down, shins hang along the body, so on all fours they lie back along
    the floor, knees down.
- Torso and head turn with the person.
- Each shot gets a **scale to metres**: the median person in it is 1.75 m tall, or, with nobody
  in the shot, the typical surface is 4 m away.

*Last: the things carried* (`analyse.place_props`). Each kept thing is placed in 3D in every frame of
its track: at the middle of its box, as deep as measured inside its outline (between
detections, as deep as on either side), and as big as its box typically spans at that depth,
across and up. A detection 1.5 times nearer or further than the median of the 5 around it has
boxed something else with the thing (the floor beyond a bag) and is treated as a miss: its box
and depth are bridged from either side, so the thing does not fly off to the wall behind for
the frames in between. Its path is the median of where it was over 5 frames, as its depth is measured
afresh in every frame; so it moves as the thing moves and hangs still where it is held still.
It stays upright, turned as the camera is in the middle of its track.

**Build (Blender).** One scene in metres, z up. Each shot has its own space, that of its first
frame's camera (looking along +Y), scaled to metres:

- **Sets**: a mesh per set, standing where its keyframe's camera stood.
  - Every pixel of its picture is a vertex out along its line of sight, so the set lines up
    with the footage from the keyframe; a flat 8 × 8-pixel square (walls, floors, steps) is
    merged into one face.
  - The pixels along each surface's outline are not left as a staircase of whole pixels, which
    shows as one where the camera sees the surface from another side or from nearer than the
    set was built from (a wall seen edge-on from the keyframe, stretched out later). Each
    surface (neighbouring pixels whose slope carries on, as on one plane) is followed round as
    a polygon within a pixel of its outline, and the outline's pixels go onto it, on along their
    own plane and out to the line where the next surface's pixels meet them, so outlines stay
    straight lines from everywhere and surfaces still meet. A pixel more than a pixel from the
    polygon (a sharp point) or on two stretches of outline (a surface one pixel thin) stays
    where it is, and an 8 × 8 square that a pixel moves in is not merged. A pixel square with
    three corners on one surface and the fourth on another (a step where the outline runs at a
    slant to the pixels) is cut along the diagonal that leaves the fourth out, so the surface's
    edge runs from one outline pixel to the next rather than dipping in at every step, which
    showed as a row of teeth from close up. This is written with the set, with the surface each
    pixel is on (`sets/NNN_edges.npz`).
  - Where something near stands in front of something far, the surface is not stretched
    across the gap (that would show as a sheet once the camera moves): it breaks there, and
    what is behind carries on a few pixels under the edge. Not, though, at the top of
    something level with the camera (the treads of a stair and its landing, the top of a
    radiator), which the camera sees edge-on: where the near side is below the far side, the
    sight line between them runs level or a little down (at most 1 in 10), and the far side is
    at most 40% further back, the surface carries on from the front edge back to what stands
    behind it, as that top, and whoever stands there stands on it.
  - Every face is shaded flat, as the plane it lies on, so walls and floors look like the plain
    boards of a set built by hand, not like moulded clay.
  - It is painted with its greys (`sets/NNN_tone.png`, mapped onto it pixel for pixel and kept
    inside the `.blend`), so each wall, floor and step is one even grey, darker or lighter as in
    the footage, with its windows, doors and lamps picked out.
  - What was left empty for the dark is a dark navy backdrop, unlit and casting no shadow, as in
    the reference films: one flat surface twice as far back as the farthest of the set, reaching
    3% of the set's width in behind its edges, so no light gap opens beside it as the camera
    moves.
  - Each is shown only during its stretch, and the next one replaces it in a single frame.
- **Figures**: for each track, a sphere head, a box torso, two-part box arms and legs, and a
  small teal visor that shows which way the face points.
  - All figures are one light clay grey (0.48 albedo), lighter than most of the set but not
    white.
  - Every part keeps one size throughout.
  - Parts are keyframed on every frame and shown only while the person is in the shot.
- **Props**: for each item carried, a simple shape in a colour of its own, so the items to
  replace stand out from the grey model: a ball for a ball or fruit, an upright cylinder for a
  bottle, cup, vase or bowl, and for everything else (bags, cases, phones, boards) a box half as
  deep as the smaller of its width and height.
  - The colours go in turn, item by item: brown, orange, blue, red, yellow, green, purple, pink.
  - Each item is one object named after what it is (`Prop_01_suitcase`), in `WB/Props`, also
    when it was lost and found again: keyframed on every frame it is seen in and shown only
    then, in each stretch it is seen in as big as it looks there and turned as there.
- **Camera**: placed and turned in every frame along the recovered path.
- **Light**: a soft sky, a sun from over the left shoulder of each shot's first camera (it
  stands still while the camera moves), and soft shadows, so figures cast shadows on the set.
  The set casts none on itself, and ambient occlusion reaches 0.4 m, so walls and floors stay
  plain, each one its own grey, lit by the way it faces, without blotches of shadow. The render
  is exposed 1.2 stops down (`EXPOSURE`), so the clay shows in mid greys, as a white model is
  shown, not glaring white.

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
shot), each painted by its own material (`WB_Set_001`, …); figures are in the `WB/Figures`
collections, one per person, with ordinary keyframes on every part; the things people carry are
in `WB/Props` (`Prop_01_suitcase`, …), each with its own material, to swap for the real item;
`WB_Camera` follows the recovered path; `WB_Sun` and the `WB_Figure` material control the look.

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
| How the rest of the set is cleaned up (the second plane search, no chips or slivers, rounded and straight outlines laid along the set's lines, sharp corners, flat everywhere) | `MIN_STEP`, `SPREAD`, `SPREAD_STEPS`, `PIECE`, `THIN`, `CARD_JUMP`, `CARD_MIN`, `SMOOTH`, `STRAIGHT`, `OUTLINE`, `SNAP`, `ALONG_REACH`, `CREASE_GAP`, `CREASE_ANGLE`, `CREASE_REACH`, `MIN_SEAM`, `SPECKS`, `EDGE_FIT` in `pipeline/planes.py` |
| How plain the set is (larger = fewer, larger surfaces; 0 = every piece found kept) | `ABSORB`, `ABSORB_FIT`, `SHARD` in `pipeline/planes.py` |
| Which runs of stairs are rebuilt as even steps | `STAIR_ANGLE`, `STAIR_SPLIT`, `MIN_RISER`, `STAIR_GAP`, `STAIR_RUN`, `MIN_STAIRS`, `STAIR_EVEN`, `STAIR_PITCH`, `STAIR_FIT` in `pipeline/planes.py` |
| What counts as too dark to see, and what becomes of it (closed by the surfaces around, or a dark void) | `UNLIT`, `UNLIT_EVEN`, `UNLIT_MIN` in `pipeline/depth.py`; `FLOOR`, `CLOSE`, `OPENING` in `pipeline/planes.py`; `VOID_COLOUR`, `VOID_MARGIN`, `VOID_FAR` in `pipeline/blender/build_scene.py` |
| How the holes people leave are closed | `BIG_PLANE`, `BEHIND`, `RUN_OFF`, `HALO`, `HALO_REACH` in `pipeline/planes.py` |
| How the camera's path is found and smoothed | `FEATURES`, `MIN_POINTS`, `PIXEL`, `MAX_TURN`, `MAX_STEP`, `SMOOTH` in `pipeline/camera.py` |
| How a shot is split into sets, and how each is built | `STRETCH_TURN`, `STRETCH_TRAVEL`, `MIN_STRETCH` in `pipeline/camera.py`; `CANVAS`, `NEAREST`, `SHOT_SAMPLES`, `SEEN`, `BLEND` in `pipeline/planes.py` |
| Assumed lens (changes how deep the scene looks) | `LENS` in `pipeline/analyse.py` |
| Which detections count (confidence, minimum size) | `MIN_PERSON_SCORE`, `MIN_HEIGHT` in `pipeline/figures.py` |
| Which tracks count, and when two tracks are one person | `MIN_TRACK`, `TRACK_SCORE`, `TRACK_KEYPOINTS`, `POLE`, `SIDE`, `MAX_GAP`, `STITCH_GAP`, `STITCH_REACH`, `SAME_TRACK`, `SAME_POSE`, `SAME_PART` in `pipeline/figures.py` |
| Which keypoints the figures follow, and how long an unseen limb keeps the pose it was seen in | `MIN_SCORE`, `REMEMBER` in `pipeline/mannequin.py` |
| Where figures stand, and how firmly they keep their size and depth | `OPEN_EDGE`, `BELOW`, `OCCLUDER`, `BRINK`, `GROUND_SPAN`, `GROUND_STEADY`, `FLOOR_SPAN`, `TOUCH`, `NEARER`, `PAST`, `SURE`, `FRONT`, `BEYOND` in `pipeline/analyse.py`; `FLOOR_SLACK`, `HOLD_GROW`, `HOLD_SHRINK`, `GROW_LIMIT`, `BRIDGED`, `DEPTH_LOOSE`, `DEPTH_FIRM` in `pipeline/mannequin.py` |
| When a figure starts out on all fours, and which way it faces | `CRAWL`, `CRAWL_SPAN`, `FACING` in `pipeline/mannequin.py` |
| Figure proportions and part sizes | `HEIGHT`, `TORSO`, `LIMBS`, `PART_SIZES`, `TORSO_CENTRE`, `LIMB_PARTS` in `pipeline/mannequin.py` |
| How many greys the set is painted in, how dark and light, and which parts of a surface get a grey of their own | `TONES`, `TONE_RANGE`, `TONE_CLIP`, `TONE_EDGE`, `TONE_SPECK`, `TONE_ROUND` in `pipeline/planes.py` |
| Which things count as carried (what, how sure, how big, held by whom), and when two tracks are one thing | `PROPS`, `MIN_PROP_SCORE`, `MIN_PROP_SIZE`, `MAX_PROP_AREA`, `PROP_PERSON`, `PROP_TRACK_SCORE`, `SAME_THING`, `KINDS`, `SAME_KIND`, `CARRIED`, `HOLD`, `CARRIER` in `pipeline/figures.py`; how steady their path is, and which depth readings are taken as misses: `PROP_STEADY`, `PROP_JUMP` in `pipeline/analyse.py` |
| Figure, visor, prop and set colours; how bright the render is; prop shapes; bevel; which set squares become one face | `FIGURE_COLOUR`, `VISOR_COLOUR`, `PROP_COLOURS`, `EXPOSURE`, `ROUND`, `UPRIGHT`, `PROP_DEPTH`, `SET_COLOUR`, `BEVEL`, `BLOCK`, `FLAT`, `EDGE_ON`, `BACK`, `LEDGE`, `LEDGE_GAP` in `pipeline/blender/build_scene.py` |
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
| A person has no figure | Too small, too dark, seen in under 8 frames, or the detector and pose model were unsure of them (a person half hidden behind a counter, say); lower `MIN_HEIGHT`, `MIN_PERSON_SCORE`, `MIN_TRACK`, `TRACK_SCORE` or `TRACK_KEYPOINTS` in `figures.py` (lower values also let in coats, statues and posters). Someone who stands still, side on, at the very side of the picture can be taken for a pole; lower `POLE` (0 lets in the edges of buildings) |
| One person becomes two figures, one after the other | They were lost for more than 48 frames or moved far while lost, or there was a cut; raise `STITCH_GAP` or `STITCH_REACH` |
| Something carried has no coloured shape, or only part of the time | The detector missed it for long stretches (a bag in shadow, mostly behind a leg), or no wrist was seen on it in 15% of the frames it was seen in; lower `MIN_PROP_SCORE` or `CARRIED` in `figures.py` (lower values also let in false detections: planks taken for books, a taxi for a phone). Only the kinds in `PROPS` are looked for |
| Something that is not carried has a coloured shape | A false detection that someone's hand passed over; raise `CARRIED` or `PROP_TRACK_SCORE` in `figures.py` |
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

Checked on 2026-11-24 with an RTX 5060 laptop GPU (8 GB).

**Through the website.** The reference video `7182b76184e0351fc372b1f7f7d5c203.mp4` was sent to
the website's API as the page sends it. Its film part (a hall with a staircase: 592×424, 30 fps,
787 frames, 26 s, with sound) went through every stage to `final.mp4` (1280×1832, two 1280×916
halves, 787 frames, with sound), `white.mp4` and `scene.blend`, and the page listed it as done
with its three downloads.

| Stage | Time |
|---|---|
| Preparing, analysis (depth, people, things carried, camera path, sets, figures) and building the Blender scene | 597 s |
| Rendering and combining | 165 s |
| **Total** | **762 s (13 min), about 1 s per frame** |

The stage was read every 10 s, too seldom to time the short stages on their own. At that rate a
60-second clip at 30 fps takes about 30 minutes. `scene.zip` downloaded (26 MB: the `.blend`
alone, which holds the whole scene), and the `.blend` in it opened in Blender with its 5 sets, 7
figures, the briefcase (`Prop_01_handbag`), the camera and all 787 frames, and no missing files.
Its picture matches the command-line run's below in the figures, their poses and the briefcase;
the sets differ in small details of their panels and greys, as the website analyses the video
afresh (the command-line run reused its first pass, and found 6 figures to the website's 7).

**Against the reference white model** (2026-11-24). Under its film, the reference video holds
the white model made by hand that this project copies. Six frames of the hall (60, 205, 280,
400, 600 and 750) were laid side by side: the film, their model and ours.

- **Brightness.** The median grey of the picture is now 96 (theirs 102, before this version
  150), and of the main figure's torso 130 (theirs 122, before 210), in levels 0–255.
- **The briefcase is one item.** It is one brown object, `Prop_01_handbag`, shown in the three
  stretches it is seen in (frames 48–81, 301–347 and 500–786), as big in each as it looks there.
  Before, it was three objects, in brown, orange and blue.
- **A wrong reading no longer throws it.** At frame 314 the detector boxed the bag together with
  the floor beyond it, 4.3 m away against 1.0–1.2 m in the frames around. The frames bridged on
  either side followed it, so the bag flew back and shrank to a speck for about 20 frames. That
  reading is now taken as a miss, and the bag stays in the man's hand.
- **One spread of greys per shot.** The three sets of the hall's first shot spread their grey
  levels from 5–116, 7–88 and 4–108, so the middle one painted the same walls a level lighter;
  they now share one (5–111). Where one set hands over to the next (frames 231→232 and
  314→315), the mean change over the picture fell from 28.4 and 19.2 grey levels to 15.3 and
  10.4. The change between ordinary frames fell from 1.3 to 0.8 (the render is darker), so most
  of what is left is the two sets' shapes, not their greys.
- **Plainer walls.** The set textures have far fewer specks: what stands out on a surface is a
  whole, rounded shape (a window's panes, a lamp, a door's panels), and the walls around it are
  one grey.

**Four test clips**, run with the same code from the command line: the hall reused its first
pass (depth, people and camera path), the other three ran it afresh. Each was checked at four
to six sampled frames, for how far each set reaches, and for jumps where one set hands over to
the next.

| Clip | Source | Figures | Items carried | Sets per shot | Farthest set |
|---|---|---|---|---|---|
| Hall and staircase (the reference video) | 592×424, 30 fps, 787 frames | 6 | 1 (a briefcase) | 3, 1, 1 | 14.5 m |
| Studio lot (well lit, handheld) | 592×240, 30 fps, 327 frames | 9 | 0 | 4, 1 | 48.9 m |
| City rooftops (fast aerial camera, a stunt performer, sky) | 592×444, 30 fps, 669 frames | 6 | 0 | 7, 4, 5, 2 | 45.6 m |
| Corridor (very dark, 5 shots) | 1280×528, 24 fps, 472 frames | 7 | 0 | 7, 4, 1, 1, 1 | 34.6 m |

- The studio, rooftop and corridor stills show the same figures in the same places as before,
  on sets in plain greys that follow the footage's light and dark; the dark corridor gives a
  darker model. In the rooftop clip open sky is still built as set in places, as before.
- No set reaches more than 50 m from its camera.

*Checked on 2026-11-23, before the changes to the set's greys, the props and the render
above:*

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
  differ by half a grey level on average), so that day's website run still stood.
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
  stretch gives way to the next (at least 24 frames apart), the set changes in a single frame,
  and surfaces the two sets placed differently jump a little.
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
- **Fast, blurred footage can still hold false detections.** Something upright and narrow at
  the side of the picture (the blurred edge of a building in the rooftop clip) is dropped, since
  its keypoints lie in a line; a false detection whose keypoints spread as a person's do still
  gets a figure. The same rule can drop someone who stands still, side on and cut by the
  frame's side, for most of the time they are seen; lower `POLE` in `figures.py` if it does.
- **Only held things become props, and only while the detector sees them.** A bag in shadow
  or mostly behind a leg can be lost for seconds, and is not drawn then; a thing nobody holds
  stays grey in the set. Items are numbered, and coloured, in the order they first appear, not
  by kind, and a thing found again is taken to be the last one of its kind seen, so two bags,
  one lost from view as the other appears, are drawn as one item. A depth reading that is wrong
  in more than two of five frames in a row is not caught, and the thing is placed at that depth.
- **Scale and lens are estimates**, so distances in the `.blend` are approximate.
- One job at a time; no progress percentage inside a stage; no accounts, so anyone who can open
  the site sees and can delete every job; Blender/ffmpeg error text includes server paths.

Next steps, in order of value: catching half-hidden and dark people (and telling apart two
who overlap) without letting in coats and posters; simpler, boxier shapes for what lies on no
plane; then, for public hosting,
accounts, old-job cleanup, a cancel button and cleaned-up error text.
