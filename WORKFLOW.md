# Baimo workflow guide

How the video → white-model website works, how to run it, how to change it, and what to build next.
For a short description of what the site does, see [README.md](README.md).

---

## 1. The workflow at a glance

```
 You                     Website                         Processing (one video at a time)
 ───                     ───────                         ────────────────────────────────
 choose a video  ──▶  POST /api/jobs  ──▶ queue ──▶  1 ingest    ffmpeg   → source.mp4
                                                     2 analyse   AI       → scene.json
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
| `analyse` (Reading the room and people) | See section 2 | `pipeline/analyse.py`, `depth.py`, `room.py`, `figures.py` | `scene.json` |
| `build` (Building 3D scene) | Blender builds the box room, the figures, the camera and the light | `pipeline/blender/build_scene.py` | `scene.blend` |
| `render` (Rendering in Blender) | Blender renders every frame | `pipeline/blender/render_shot.py` | `white.mp4` |
| `compose` (Combining) | Original (scaled to the render size) stacked over the white model, original audio kept | `media.compose` | `final.mp4` |
| `done` / `failed` | Result shown, or the error message | `pipeline/job.py` | — |

---

## 2. How a frame becomes a white model

The white model is deliberately a **box style**: a plain room and sphere-and-box figures. It shows
where things are and how people move, not surface detail. An earlier version rendered the depth
map itself as a sculpted surface; on dark or blurred footage that turned into unreadable blobs,
so it was removed.

**Analyse (Python, GPU).** For every frame:

1. **Shot cuts** are found first, by checking whether each frame still correlates with the one
   before it. Cuts reset the smoothing and the people tracking.
2. **People**: RT-DETR finds each person, ViTPose finds their 17 body keypoints. Overlapping
   boxes in nearby frames are linked into tracks, and frames where a person was missed (up to
   12 in a row) are bridged by interpolation.
3. **Depth**: Depth Anything 3 (monocular, large) estimates how far away everything is.
   Each person's depth is noted, then they are erased from the depth map.
4. **Room fit**: the depth map is reduced to six numbers: which way the room is turned, and the
   distance to the floor, ceiling, back wall, left wall and right wall. A surface only counts if
   enough of the frame faces the right way.

After the last frame, per shot:

- A **scale to metres** is set from the average torso length of the people in the shot (or a
  default distance when nobody is fully visible).
- The room numbers are **smoothed over time**, so the room is steady and turns with camera pans.
  A wall or ceiling seen in fewer than a quarter of the shot's frames is left out for the whole
  shot; the floor always exists.
- Walls are **pushed outward** wherever they would cut through the camera or a person.
- Every tracked person's keypoints become **3D joints in metres**.

**Build (Blender).** One scene in metres, z up, camera fixed at the origin looking along +Y:

- **Room**: five whitebox boxes (`Room_Floor`, `Room_Back`, `Room_Left`, `Room_Right`,
  `Room_Ceiling`) parented to an empty called `Room`. The empty carries the turn; each box is
  keyframed to its distance.
- **Figures**: for each tracked person, a sphere head, a box torso, box arms and legs, and a
  small teal visor that shows which way the face points. A person seen from behind has no visor.
- **Light**: a soft sky, a sun from the camera's left, ambient occlusion and soft shadows.

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

The AI models (about 2 GB together) download from Hugging Face automatically the first time a
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

The site only works while that command is running. Videos processed before a pipeline change keep
their old look; upload them again to get the current one.

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

**Editing the white model in Blender** — open the downloaded `.blend`; it is self-contained.
Everything is an ordinary keyframed object: the room boxes under the `Room` empty, figures in
the `WB/Figures` collections (one per person), `WB_Camera`, `WB_Sun`.

---

## 5. Settings and tuning

| To change | Edit |
|---|---|
| Blender / ffmpeg / library locations | `config.toml` → `[tools]`, `[paths]` |
| Upload limits (size, length, resolution) | `config.toml` → `[limits]` |
| Depth model and its working resolution | `MODEL`, `PROCESS_RES` in `pipeline/depth.py` |
| How much of the frame a wall, floor or ceiling needs to count | `MIN_SHARE` in `pipeline/room.py` |
| How often a wall must be seen in a shot to be built | `MIN_SEEN` in `pipeline/room.py` |
| How steady the room is (longer = steadier, slower to follow pans) | the `0.8` second window in `room.smooth` |
| Gap kept between walls and people or the camera | `CLEARANCE` in `pipeline/room.py` |
| Floor height when the floor is never seen | `CAMERA_HEIGHT` in `pipeline/room.py` |
| Assumed lens | `LENS` in `pipeline/analyse.py` |
| Which people count (confidence, minimum size, shortest track, longest gap bridged) | `MIN_PERSON_SCORE`, `MIN_HEIGHT`, `MIN_TRACK`, `MAX_GAP` in `pipeline/figures.py` |
| When a limb is drawn | `MIN_KEYPOINT` in `pipeline/figures.py` |
| Figure colours, proportions, visor | `FIGURE_COLOURS` and `placements()` in `pipeline/blender/build_scene.py` |
| Wall and floor greys, light, render samples | the room and light sections of `build_scene.py` |
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
| `GET /api/jobs/{id}/files/{name}` | `final.mp4`, `white.mp4`, `source.mp4`, `scene.blend` |
| `DELETE /api/jobs/{id}` | Delete a finished job (409 while it is running) |

Interactive reference: <http://localhost:8000/docs>.

---

## 7. When something goes wrong

| Symptom | Likely cause and fix |
|---|---|
| The page will not open | The server is not running; start it (section 4) |
| Job fails: "could not be read as a video" | Not a video, or a damaged file |
| Job fails with a size / length / resolution message | Over the limits in `config.toml`; trim the clip or raise the limit |
| Job fails: "blender not found" / "ffmpeg not found" | Fix the path in `config.toml`; after installing ffmpeg, open a new terminal so PATH updates |
| Fails at *Reading the room and people* with "No module named depth_anything_3" | The clone in section 3 is missing or `[paths].depth_anything_3` points elsewhere |
| Fails at that stage the first time, other message | No internet for the one-time model downloads, or out of GPU memory (close other GPU apps) |
| A person has no figure | Too small, too dark, or seen for under 8 frames; lower `MIN_HEIGHT`, `MIN_PERSON_SCORE` or `MIN_TRACK` |
| A figure changes colour mid-video | The person was lost for more than 12 frames, or there was a cut, so a new track started |
| A figure is missing arms or legs | The pose model was unsure of those joints; lower `MIN_KEYPOINT` (may add wrong limbs) |
| The room has no side walls, or only a back wall | Too little of the frame showed them; lower `MIN_SHARE` or `MIN_SEEN` |
| The room drifts or tilts oddly | Dark or blurred depth; lengthen the smoothing window in `room.smooth` |
| An old video still shows a lumpy grey surface | It was processed by an earlier version; upload it again |
| `uv` fails with "os error 17" inside the Claude desktop app | Set `$env:UV_CACHE_DIR = "C:\Users\kakssjs\.cache\uv"` first (not needed in a normal terminal) |
| Server was closed mid-job | Start it again; unfinished jobs restart automatically from the first stage |

The full error text of a failed job is in `jobs/<id>/job.json`.

---

## 8. What was verified

Checked on 2026-11-13 with an RTX 5060 laptop GPU (8 GB), by uploading the film halves of the
two sample videos through the website.

| Clip | Source | Result | Total time | Of which analysis |
|---|---|---|---|---|
| Studio lot (well lit, handheld) | 592×240, 30 fps, 327 frames | 1280×1036, 327 frames, audio | 131 s | 68 s |
| Corridor (very dark, 5 shots) | 1280×528, 24 fps, 472 frames | 1280×1056, 472 frames, audio | 178 s | 87 s |

So expect roughly 0.2 s per frame for analysis plus 0.18 s per frame for rendering: a 60-second
clip at 30 fps takes about 11 minutes. The `.blend` download is under 1 MB.

- **Studio lot:** every sampled frame shows a clean room with all visible people as figures in
  the right places and poses; walls no longer cut through anyone.
- **Corridor:** the long corridor shots read as a corridor with a figure walking down it; in the
  two darkest, most blurred sampled frames nobody was detected, so the room is empty there.
- **Website:** both uploads went through every stage; `final.mp4`, `white.mp4` and
  `scene.blend` were produced and the `.blend` downloaded.
- **Earlier checks that still apply:** page layout at phone and desktop width, failed-upload
  message, delete, restart recovery, status codes for unknown jobs and files.

Not verified: the whole videos frame by frame (six sampled frames each), how steady the room is
during playback, opening the `.blend` by hand in Blender, the native file picker and real
drag-and-drop, the over-size upload rejection, clips near the 60 s limit, and `uv run` from a
normal terminal (the environment's Python was called directly).

---

## 9. Known limits and next steps

- **The room is always a plain box.** There are no doorways, windows, props or furniture, and
  an outdoor scene still gets a back wall. The room is an estimate of the big surfaces, so its
  walls only roughly match the real ones.
- **No camera path.** The camera stays at the origin; the room turns and shifts around it.
- **Figures face the camera.** Joints sit at one depth per person, so figures follow poses but
  do not turn in depth.
- **Identity is not kept** across cuts or long gaps: the person returns as a new figure.
- **Scale and lens are estimates**, so distances in the `.blend` are approximate.
- One job at a time; no progress percentage inside a stage; no accounts, so anyone who can open
  the site sees and can delete every job; Blender/ffmpeg error text includes server paths.

Next steps, in order of value: prop boxes for large foreground objects; openings (doors,
windows) in walls; one figure per person across gaps; a recovered camera path; then, for public
hosting, accounts, old-job cleanup, a cancel button and cleaned-up error text.
