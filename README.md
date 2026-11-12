# Baimo — video to white-model film

A local website that turns a video into a comparison film:
**the original on top, a Blender white-model (白模) version of the same footage below.**

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
- **Turn the set into a white clay model.** A large depth model reads every frame at high
  resolution; Blender shapes a perspective surface from it, with edges snapped to the real
  edges in the picture, and renders it with soft shadows and ambient occlusion.
- **Replace people with sphere-and-box figures.** Each person is detected, followed through the
  shot, erased from the set, and rebuilt as a 3D mannequin — sphere head, box torso, box arms
  and legs — that copies their pose frame by frame and casts a shadow on the set. A teal visor
  shows which way the face points. Each tracked person gets a colour.
- **Keep the two halves in step.** The white model lines up with the footage pixel for pixel
  and has the same frame rate and frame count.
- **Render at a watchable size.** Small uploads are enlarged so each half is at least 1280
  pixels on its long side.
- **Play and download the results:** the stacked comparison video, the white model on its own,
  and the editable Blender scene.
- **Keep a history** of processed videos, with delete.
- **Recover from a restart:** videos that were waiting or processing when the server stopped are
  run again when it starts.
- **Run without the browser** from the command line, including cropping the input.

## What it cannot do yet

- The set is one sculpted surface seen from the camera, not separate walls, floors and props,
  and there is no recovered camera path. It cannot be walked around in Blender.
- Figures follow poses but always face the camera; arms or legs the pose model cannot see are
  left out.
- A person who is lost and found again (after a cut, or when hidden) comes back as a new figure
  with a different colour. Small or very dark people can be missed.
- Dark, blurry or very low-resolution footage gives a softer set, and a soft lump can remain
  where a person filled most of the frame.
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
downloads the AI models (about 2 GB). Timing is in [WORKFLOW.md](WORKFLOW.md), section 8.

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
│  ├─ media.py        ffmpeg steps and Blender launchers
│  ├─ analyse.py      shot cuts, per-frame depth, people; writes depth frames and scene.json
│  ├─ depth.py        depth model, person removal, edge sharpening
│  ├─ figures.py      person detection, pose estimation and tracking
│  └─ blender/        scripts that run inside Blender (build_scene.py, render_shot.py)
├─ jobs/              one folder per processed video (not in git)
├─ WB_SmokeTest/      earlier Blender white-model experiments
└─ _vendor/           cloned third-party code (not in git)
```

## Licences of the AI models

Depth Anything 3 Mono-Large, RT-DETR and ViTPose are all released under Apache-2.0.
