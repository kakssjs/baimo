# Baimo workflow guide

How the video → white-model website works, how to run it, how to change it, and what to build next.
For a short description of what the site does, see [README.md](README.md).

---

## 1. The workflow at a glance

```
 You                     Website                         Processing (one video at a time)
 ───                     ───────                         ────────────────────────────────
 choose a video  ──▶  POST /api/jobs  ──▶ queue ──▶  1 ingest    ffmpeg   → source.mp4
                                                     2 analyse   AI       → depth/*.png, figures.json
 watch the stages ◀── GET /api/jobs  ◀── job.json ◀─ 3 build     Blender  → shot01.blend
                                                     4 render    Blender  → white.mp4
 play / download ◀──  GET …/files/…  ◀─────────────  5 compose   ffmpeg   → final.mp4
```

Every video becomes one **job** with its own folder `jobs/<id>/`. The page polls the job's
`job.json` once a second and shows the current stage.

| Stage (page label) | What happens | Code | Output |
|---|---|---|---|
| `queued` (Waiting) | Upload saved, waiting for the worker | `server/app.py` | `input.<ext>`, `job.json` |
| `ingest` (Preparing video) | Check limits; convert to constant frame rate, even size, H.264 + AAC | `media.ingest` | `source.mp4` |
| `analyse` (Finding depth and people) | Depth Anything V2 estimates a depth map for every frame. RT-DETR finds people, ViTPose finds their 17 body keypoints, and overlapping boxes are linked into tracks. Each person is pressed out of the depth map | `pipeline/depth.py`, `pipeline/figures.py` | `depth/NNNN.png` (16-bit, white = near), `figures.json` |
| `build` (Building 3D scene) | Blender builds a clay-white grid displaced by the depth frames, a keyframed sphere-and-box figure per tracked person, an orthographic camera and a sun light | `pipeline/blender/build_relief.py` | `shot01.blend` |
| `render` (Rendering in Blender) | Blender renders the scene at the source size, frame rate and frame count; scene is zipped | `pipeline/blender/render_shot.py` | `white.mp4`, `scene.zip` |
| `compose` (Combining) | Original stacked over the white model, original audio kept | `media.compose` | `final.mp4` |
| `done` / `failed` | Result shown, or the error message | `pipeline/job.py` | — |

Why the white model lines up with the footage: the relief is a flat grid the shape of the frame,
pushed toward an orthographic camera. Each point stays on its own pixel, so the bottom half matches
the top half frame for frame. Figures live in the same frame space: each body part is a cube
stretched between two keypoints (the head is a sphere), placed at the depth the relief had for
that person, so they also line up and cast shadows on the set.

---

## 2. First-time setup

Needed on the machine (all already installed here):

| Tool | Where | Check |
|---|---|---|
| Blender 5.x | path in `config.toml` → `[tools].blender` | `& "E:\blander\blender.exe" --version` |
| ffmpeg + ffprobe | on PATH (`winget install Gyan.FFmpeg`) | `ffprobe -version` |
| uv | on PATH | `uv --version` |
| NVIDIA GPU + driver | — | `nvidia-smi` |
| `whitebox.py` helper | folder in `config.toml` → `[paths].whitebox_scripts` | file exists |

Install the Python packages (FastAPI, PyTorch with CUDA, transformers — about 3 GB the first time):

```powershell
uv sync
```

The three AI models (about 600 MB together) download from Hugging Face automatically the first
time a video is processed. Without a GPU the depth stage still runs on the CPU, but much slower.

---

## 3. Daily use

Start the site:

```powershell
uv run uvicorn server.app:app --port 8000
```

Open <http://localhost:8000>, then:

1. **Choose a video** (or drop one on the box). Limits are shown under the box.
2. **Watch the stages** light up. A 11-second 592×240 clip takes about 2 minutes on the RTX 5060.
3. **Play the result** — original on top, white model below.
4. **Download** the comparison video, the white model alone, or the Blender scene.
5. Older videos stay in **Your videos**; select one to see it again, or delete it.

Upload plain footage. A video that is already a comparison (like the two sample files in this
folder, which have a white model and black bars baked in) must be cropped to the film part first.

**Without the browser** — process one file from the command line, optionally cropping it:

```powershell
uv run python -m pipeline my_clip.mp4
```

```powershell
uv run python -m pipeline bad19affd9d603bbb1ae2e3fe3f4a12b.mp4 --crop "592:240:0:636"
```

`--crop` is `width:height:x:y` in source pixels. The result lands in `jobs/<id>/final.mp4`.

**Editing the white model in Blender** — unzip `scene.zip`, open `shot01.blend` (the `depth`
folder must stay next to it). The object `Relief` has a *Displace* modifier called `Depth`;
its strength, the `WB_Sun` light and the `WB_Clay` material control the look. Figures are in
the `WB/Figures` collections, one per person, with ordinary keyframes on every part.

---

## 4. Settings and tuning

| To change | Edit |
|---|---|
| Blender / ffmpeg locations | `config.toml` → `[tools]` |
| Upload limits (size, length, resolution) | `config.toml` → `[limits]` |
| How far the relief sticks out | `STRENGTH` in `pipeline/blender/build_relief.py` |
| Mesh detail | `COLS` in `build_relief.py` |
| Light direction, brightness, ambient | the `WB_Sun` and world lines in `build_relief.py` |
| Flicker vs. responsiveness | `SMOOTH` in `pipeline/depth.py` (0 = none, higher = steadier but more trailing) |
| How fast depth contrast adapts after a cut | `RANGE_WINDOW` in `pipeline/depth.py` |
| Depth model | `MODEL` in `pipeline/depth.py` (the Small model is Apache-2.0; Base and Large are non-commercial) |
| Which people count (confidence, minimum size, shortest track) | `MIN_PERSON_SCORE`, `MIN_HEIGHT`, `MIN_TRACK` in `pipeline/figures.py` |
| When a limb is drawn | `MIN_KEYPOINT` in `pipeline/figures.py` |
| Figure jitter | `SMOOTH` in `pipeline/figures.py` |
| Figure colours, limb thickness, head size | `FIGURE_COLOURS` and `placements()` in `build_relief.py` |
| Stage names on the page | `STAGES` in `web/index.html` |

Restart the server after changing Python files or `config.toml`. The page (`web/index.html`) only
needs a browser refresh.

---

## 5. API

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

## 6. When something goes wrong

| Symptom | Likely cause and fix |
|---|---|
| Job fails: "could not be read as a video" | Not a video, or a damaged file |
| Job fails with a size / length / resolution message | Over the limits in `config.toml`; trim the clip or raise the limit |
| Job fails: "blender not found" / "ffmpeg not found" | Fix the path in `config.toml`; after installing ffmpeg, open a new terminal so PATH updates |
| Fails at *Estimating depth* the first time | No internet for the one-time model download, or out of GPU memory (close other GPU apps) |
| White model is flat or soft | Dark, blurry or close-up shots carry little depth; raise `STRENGTH`, or accept it for that shot |
| A person has no figure | Too small, too dark, or seen for under 8 frames; lower `MIN_HEIGHT`, `MIN_PERSON_SCORE` or `MIN_TRACK` |
| A figure changes colour mid-video | The person was lost and found again, so a new track started; expected after cuts |
| A figure is missing arms or legs | The pose model was unsure of those joints; lower `MIN_KEYPOINT` (may add wrong limbs) |
| White model shimmers | Depth is estimated frame by frame; raise `SMOOTH` |
| Depth contrast "breathes" for a second after a cut | Expected; lower `RANGE_WINDOW` to shorten it |
| `uv` fails with "os error 17" inside the Claude desktop app | Set `$env:UV_CACHE_DIR = "C:\Users\kakssjs\.cache\uv"` first (not needed in a normal terminal) |
| Server was closed mid-job | Start it again; unfinished jobs restart automatically from the first stage |

The full error text of a failed job is in `jobs/<id>/job.json`.

---

## 7. What was verified (2026-11-12)

- Command line and website both processed the film part of `bad19affd9d603bbb1ae2e3fe3f4a12b.mp4`
  (592×240, 30 fps, 327 frames) to a 592×480 `final.mp4` with audio, in about 100 seconds.
- White model frames line up with the footage; walls and props are recognisable in well-lit
  shots and soft in the dark close-ups.
- Nine person tracks were found in that clip. Figures sit on the people, follow them, and show
  full bodies for the background people and head-and-torso for close-ups. Seen in six sampled
  frames, not reviewed frame by frame.
- Page: stage indicator, playable result, three downloads, failed-upload message, delete, no
  layout overflow at phone or desktop width.
- API: status codes for unknown jobs and files, range requests, delete refused while running.
- Server stopped during *Estimating depth* and restarted: the job ran again and finished.

Not verified: the native file picker and real drag-and-drop (uploads were injected by script),
the over-size upload rejection, videos near the 60 s / 1080p limits, and CPU-only processing.

---

## 8. What is not built, and the next steps

People are now sphere-and-box figures, as in the two sample videos. The **set is still a depth
relief**: one continuous surface seen from a fixed camera, not simple walls, floors and props.

Next steps, in the order that closes that gap:

1. **Better figures.** Keep one figure (and colour) per person across cuts and gaps; estimate a
   3D pose so figures can turn instead of staying flat-on; remove the leftover outline in the
   relief with a proper person mask.
2. **Simple set geometry.** Replace the background relief with fitted planes and boxes (floor,
   walls, large props) built with `whitebox.py`, one scene per shot. Needs real shot-cut detection
   and a camera estimate per shot.
3. **Blockout mode on the page.** A mode selector (relief / blockout), and re-rendering after an
   edited `.blend` is uploaded.
4. **For public hosting.** Accounts or private job links, cleaned-up error text, automatic deletion
   of old jobs, cancel button, per-stage progress percentage, terms for uploaded footage.

Other known limits: one job at a time; no progress percentage inside a stage; anyone who can open
the site sees and can delete every job; error text for Blender/ffmpeg failures includes server paths.
