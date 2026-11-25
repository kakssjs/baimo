# Baimo — video to white-model film

A local website that turns a video into a reusable 3D white-model (白模) template of its shots,
with the same framing, camera moves and timing, and a comparison film:
**the original on top, the Blender white model of the same footage below.**

```
┌──────────────────────────┐
│   original video         │
├──────────────────────────┤
│   white model (Blender)  │   same framing, same timing, original sound
└──────────────────────────┘
```

## What it can do

- **Upload a video in the browser** and watch it move through the stages: preparing, reading
  depth and people, building the 3D scene, rendering in Blender, combining.
- **Use comparison videos as they are.** Black bars are cut off, and in a video that already
  shows an original next to a render, only the filmed half is used.
- **Turn the set into a crisp clay model in the footage's own light and dark.** A large depth
  model reads every frame; walls, floors, steps and furniture faces are rebuilt from it as flat
  planes with straight, sharp outlines and clean corners, turned square to each other where the
  set is built square, so the set looks built by hand; everything else (rails, lamps, radiators,
  coats) becomes flat cards in its own outline, like the blocks a set is roughed out with. Small
  pieces that lie on a larger surface become part of it, so the set is a few large, plain
  surfaces that render cleanly, and only what stands out from them (a staircase, a counter, a
  door frame) is a piece of its own. A run of stairs is rebuilt as even steps, one rise and one
  run, as a set builder makes them. The treads of a stair and its landing, which a camera level
  with them sees edge-on, are still there to stand on. Outlines stay straight lines from every
  side the camera sees them, not staircases of pixels. Each surface is painted one even grey,
  the grey the footage shows it in (six greys in all, from dark to light), and each part of it
  that stands out from the rest (a window's panes, a door's panels, a lamp on a wall) its own
  grey, so the set keeps its windows, doors, lamps, rails and darker and lighter walls, with no
  blotches where the light fell unevenly; a part that is nearly convex (a pane, or a lamp without
  the glow around it) is drawn as a clean round or straight-sided shape. The set's straight
  edges (window frames, door and panel outlines, rails, floor tiles) are drawn on it as crisp
  lines one grey darker, closed at the corners where they meet, so they keep their outlines even
  where the footage is too dim for the greys to tell them apart; where one surface stands in
  front of another (a door in its frame, a pillar before a wall) the line is drawn on the nearer,
  so it moves with it. The greys are the same in every set of a shot, so a wall keeps its grey
  as the camera moves on. Every surface is shaded flat, like the boards of a hand-built set:
  Blender renders the figures' soft shadows on it, but the set casts none on itself.
- **Build the set through the dark.** Where the footage is too dark to see, the walls, floors and
  ceilings around it carry on through it, as on a real set. Only where it opens into darkness
  that nothing around can close (a doorway into an unlit room) is it left as a plain dark void.
- **Move the camera through a still set, like a real one.** The camera's path through each shot
  (where it stands and where it looks, frame by frame) is recovered from the background. The set
  is built from the depth of many frames at once and stands still while the camera moves, turns
  or pulls back through it: nothing in it flickers or changes shape. A long camera move is
  covered by a few such sets, one after the other, each kept for at least 24 frames (a second
  at 24 fps), so the walls do not keep shifting during a fast move.
- **Replace people with sphere-and-box figures in light clay.** Each person is detected,
  followed through the shot and cut out of the set along their exact outline. The space behind
  them is closed with the walls and floor around them, or with what other frames saw there, so
  no outline of them stays in the set.
- **Keep figures true to the people.** Each person becomes one 3D mannequin of fixed
  proportions: sphere head, box torso, box arms and legs. It stands on the set where they
  stand (on the floor, or on a stair), at their size, fills their outline from edge to edge (so
  someone kneeling folds down rather than shrinks, and someone on all fours crawls, knees down),
  and copies their pose and the way they face, frame by frame, with the whole body always
  there: an arm or leg the picture loses for a moment (out of frame, hidden, blurred) keeps the
  pose it is seen in just before or after, rather than dropping to the side. A teal visor shows
  where the face points, and the figure casts a shadow on the set. Someone the detector loses
  for a moment, or boxes twice, stays one figure.
- **Colour what people carry, as the items to replace.** A bag, bottle, cup, phone, ball,
  board or book (any of the everyday things the detector knows) that someone holds becomes a
  box, cylinder or ball of its own colour (the first brown, then orange, blue, red, yellow,
  green, purple, pink), as big as it is seen, following its path frame by frame: where it is
  carried, how long it hangs still, how near it comes. Found again after it went out of the
  picture, or after a cut, it is the same item in the same colour. It is cut out of the set
  where it is held. Things nobody holds stay part of the set.
- **Keep the two halves in step.** The white model lines up with the footage pixel for pixel
  and has the same frame rate and frame count.
- **Render at a watchable size.** Small uploads are enlarged so each half is at least 1280
  pixels on its long side.
- **Play and download the results:** the stacked comparison video (in step with the original,
  with its sound), the white-model MP4 on its own, and the editable Blender scene, with every
  set, figure, prop (named after what it is, such as `Prop_01_handbag`) and the camera.
- **Keep a history** of processed videos, with delete.
- **Recover from a restart:** videos that were waiting or processing when the server stopped are
  run again when it starts.
- **Run without the browser** from the command line, including cropping the input.

## What it cannot do yet

- The set is built from what the camera saw, not from separate objects: what no frame saw (the
  backs of things, the room behind the camera) is missing, so it cannot be walked around freely
  in Blender.
- Irregular things (coats, plants, a lamp) become flat cards in their own outline, not their
  real shape, and dark, blurry or very low-resolution footage yields fewer, rougher planes.
  Stairs are made even only where their risers face the camera (looking up a straight flight);
  seen from above, or winding, they keep the planes found. Only the straight edges the footage
  shows clearly are drawn as lines: detail too dim to tell apart (a door's panels in a dark hall)
  is left out, and a lamp on a piece of set turned away from the light is shaded darker than it
  is. What is built through the dark is the surfaces around it carried on, not what is really
  there, and a dark void is a flat backdrop with no depth.
- Where a long camera move passes from one set to the next (at least 24 frames apart), the set
  changes in a single frame: each set is cut into panels on its own, so some panels jump a
  little.
- Where a person stood, the set behind them is a reconstruction. A few small specks or a thin
  edge can show in the wall or floor next to a figure, and a flat patch can show behind
  someone who stays in one place or fills much of the frame.
- People the detector is unsure of (half hidden behind a counter, very dark, mostly out of
  frame, or overlapping someone else so that it boxes them together) can get no figure.
  Someone mostly hidden behind another person can be placed too near, and someone going down
  onto all fours can be shown kneeling for a second or so before their figure crawls. Someone
  in mid-air (a leap over a water tower) is placed as deep as what is below them. In fast,
  blurred footage something that is not a person can still get a figure for a moment; the edge
  of a building or a pole at the side of the picture no longer does, but someone standing still,
  side on, at the very side of the picture can be taken for a pole and get no figure. A person
  who comes back after a cut, or after more than 48 frames out of sight, becomes a new figure.
- Only what someone holds becomes a coloured prop, and only while the detector sees it (it is
  unsure of a bag in dark footage, and can lose it for seconds, when it is not shown). A thing
  that nobody holds stays grey in the set, and the colours go by the order in which the items
  first appear, not by kind. Two things of one kind, one lost from view as the other appears,
  can be taken for one item.
- One video is processed at a time. Limits: 200 MB, 60 seconds, 1920×1080 (portrait allowed).
- It is built for one machine and one user: no accounts, so anyone who can open the site sees
  every video.

## Quick start

Requirements: Windows, Blender 5.x, ffmpeg, uv, an NVIDIA GPU. Paths are set in `config.toml`.

One-time setup (a clone of the depth model's code, then the Python packages, about 3 GB):

```powershell
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git _vendor/Depth-Anything-3
```

```powershell
uv sync
```

Start the site:

```powershell
uv run uvicorn server.app:app --port 8000
```

Open <http://localhost:8000>, choose a video, and wait for the result. The first run also
downloads the AI models (about 2.2 GB). Timing is in [WORKFLOW.md](WORKFLOW.md), section 8.

Command line alternative:

```powershell
uv run python -m pipeline my_clip.mp4
```

## Project layout

```
baimo/
├─ README.md          this file
├─ WORKFLOW.md        full guide: stages, setup, use, tuning, troubleshooting, limits
├─ config.toml        tool paths and upload limits
├─ pyproject.toml     Python dependencies (uv)
├─ server/app.py      website backend: API, job queue, serves the page
├─ web/index.html     the page
├─ pipeline/
│  ├─ job.py          runs the stages for one video, writes job.json
│  ├─ media.py        ffmpeg steps (finding the picture area), Blender launchers
│  ├─ analyse.py      shot cuts, the three analysis passes; writes the sets and scene.json
│  ├─ depth.py        the depth model
│  ├─ camera.py       the camera's path through each shot, from features on the background
│  ├─ planes.py       the sets: depth from many frames laid together, flat surfaces, people taken out
│  ├─ figures.py      person detection, pose estimation, outlines and tracking
│  ├─ mannequin.py    the 3D mannequin fitted to each tracked person
│  └─ blender/        scripts that run inside Blender (build_scene.py, render_shot.py)
├─ jobs/              one folder per processed video (not in git)
├─ WB_SmokeTest/      earlier Blender white-model experiments
└─ _vendor/           cloned third-party code (not in git)
```

## Licences of the AI models

Depth Anything 3 Mono-Large, RT-DETR, ViTPose and SAM 2.1 are all released under Apache-2.0.
