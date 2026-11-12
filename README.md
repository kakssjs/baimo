# Baimo — video to white-model film

A local website that turns a video into a comparison film:
**the original on top, a Blender white-model (白模) version of the same footage below.**

```
┌──────────────────────────┐
│   original video         │
├──────────────────────────┤
│   white model (Blender)  │   same size, same timing, original sound
└──────────────────────────┘
```

## What it can do

- **Upload a video in the browser** and watch it move through the stages: preparing, estimating
  depth, building the 3D relief, rendering in Blender, combining.
- **Convert every frame to a white model automatically.** An AI depth model (Depth Anything V2)
  measures how far each part of the picture is; Blender turns that into a clay-white 3D relief of
  the set, lights it, and renders it. No manual modelling.
- **Replace people with sphere-and-box figures.** Each person is detected, followed through the
  shot, and rebuilt as a simple mannequin (sphere head, box torso, box arms and legs) that copies
  their pose frame by frame. Each tracked person gets a colour.
- **Keep the two halves in step.** The white model has the same resolution, frame rate and frame
  count as the footage and lines up with it pixel for pixel.
- **Play and download the results:** the stacked comparison video, the white model on its own,
  and the editable Blender scene (`.blend` plus its depth frames).
- **Keep a history** of processed videos, with delete.
- **Recover from a restart:** videos that were waiting or processing when the server stopped are
  run again when it starts.
- **Run without the browser** from the command line, including cropping the input.

## What it cannot do yet

- The **set is still a depth relief** (one sculpted surface seen from the front), not separate
  walls, floors and props like the two sample videos in this folder. That is the next step.
- Figures are flat-on to the camera: they follow the 2D pose and depth of each person, not a true
  3D skeleton. Arms or legs the pose model cannot see are left out.
- A person who is lost and found again (after a cut, or when hidden) comes back as a new figure
  with a different colour. Small or very dark people can be missed.
- Traces of the outline of a person can remain in the relief behind their figure.
- Dark, blurry or extreme close-up shots give a soft, flat relief, and the surface can shimmer.
- One video is processed at a time. Limits: 200 MB, 60 seconds, 1920×1080 (portrait allowed).
- It is built for one machine and one user: no accounts, so anyone who can open the site sees
  every video.

## Quick start

Requirements: Windows, Blender 5.x, ffmpeg, uv, an NVIDIA GPU. Paths are set in `config.toml`.

Install the Python packages (first time only, about 3 GB):

```powershell
uv sync
```

Start the site:

```powershell
uv run uvicorn server.app:app --port 8000
```

Open <http://localhost:8000>, choose a video, and wait for the result. An 11-second clip takes
about 2 minutes on an RTX 5060 laptop GPU. The first run also downloads the AI models (~600 MB).

Command line alternative:

```powershell
uv run python -m pipeline my_clip.mp4
```

## Project layout

```
baimo/
├─ README.md          this file
├─ WORKFLOW.md        full guide: stages, setup, use, tuning, troubleshooting, next steps
├─ config.toml        tool paths and upload limits
├─ pyproject.toml     Python dependencies (uv)
├─ server/app.py      website backend: API, job queue, serves the page
├─ web/index.html     the page
├─ pipeline/
│  ├─ job.py          runs the stages for one video, writes job.json
│  ├─ media.py        ffmpeg steps and Blender launchers
│  ├─ depth.py        AI depth estimation, writes depth frames and figures.json
│  ├─ figures.py      person detection, pose estimation and tracking
│  └─ blender/        scripts that run inside Blender (build_relief.py, render_shot.py)
├─ jobs/              one folder per processed video (not in git)
├─ WB_SmokeTest/      earlier Blender white-model experiments
└─ _vendor/           reference repositories (not in git)
```

## More

See [WORKFLOW.md](WORKFLOW.md) for how each stage works, how to tune the look, the API,
troubleshooting, what was tested, and the plan for real set geometry.
