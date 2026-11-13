# Baimo workflow guide

How the video → white-model website works, how to run it, how to change it, and what to build next.
For a short description of what the site does, see [README.md](README.md).

---

## 1. The workflow at a glance

```
 You                     Website                         Processing (one video at a time)
 ───                     ───────                         ────────────────────────────────
 choose a video  ──▶  POST /api/jobs  ──▶ queue ──▶  1 ingest    ffmpeg   → source.mp4
                                                     2 analyse   AI       → depth/*.png, scene.json
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
| `analyse` (Reading depth and people) | See section 2 | `pipeline/analyse.py`, `depth.py`, `figures.py` | `depth/NNNNNN.png`, `scene.json` |
| `build` (Building 3D scene) | Blender builds the set surface, the figures, the camera and the light | `pipeline/blender/build_scene.py` | `scene.blend` |
| `render` (Rendering in Blender) | Blender renders every frame; the scene is zipped with its depth frames | `pipeline/blender/render_shot.py` | `white.mp4`, `scene.zip` |
| `compose` (Combining) | Original (scaled to the render size) stacked over the white model, original audio kept | `media.compose` | `final.mp4` |
| `done` / `failed` | Result shown, or the error message | `pipeline/job.py` | — |

---

## 2. How a frame becomes a white model

**Analyse (Python, GPU).** For every frame:

1. **Shot cuts** are found first, by checking whether each frame still correlates with the one
   before it. Cuts reset the smoothing and the people tracking.
2. **People**: RT-DETR finds each person, ViTPose finds their 17 body keypoints, and SAM 2
   traces each person's exact outline. Overlapping boxes in nearby frames are linked into
   tracks, and frames where a person was missed (up to 12 in a row) are bridged.
3. **Depth**: Depth Anything 3 (monocular, large) estimates depth at 1008 pixels wide. It is
   blended lightly with the previous frame to reduce shimmer.
4. **People are erased from the depth map**: every pixel inside a person's outline (widened a
   little) is repainted from the surfaces around it, so the set shows what is behind them. Their
   depth is noted first, so their figure can be placed where they stood.
5. **Edges are sharpened** with a guided filter, which snaps the depth map's soft edges onto the
   real edges in the video frame.
6. The result is stored as a 16-bit image (white = near).

After the last frame, each shot gets a **scale to metres** (from the average torso length of the
people in it, or a default distance when nobody is fully visible), and every tracked person's
keypoints are turned into **3D joints in metres**.

**Build (Blender).** One scene in metres, z up, with the camera fixed at the origin looking along +Y:

- **Set**: a 640-column grid that fills the view one unit in front of the camera. A Geometry
  Nodes modifier pushes every vertex out along its line of sight to the depth in that frame's
  image, so the set is a true perspective surface that lines up with the footage pixel for
  pixel. Sharp folds stay as hard creases; everything else is smooth-shaded clay.
- **Figures**: for each tracked person, a sphere head, a box torso, box arms and legs, and a
  small teal visor that shows which way the face points. Parts are real 3D objects stretched
  between the 3D joints and keyframed on every frame; a person seen from behind has no visor.
- **Light**: a soft sky, a sun from over the camera's left shoulder, ambient occlusion and soft
  shadows, so figures cast shadows on the set.

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

**Editing the white model in Blender** — download the scene (the zip is built on the first
request and can be several hundred megabytes), unzip it and open `scene.blend` (the `depth`
folder must stay next to it). The `Set` object carries the `Depth` Geometry Nodes modifier;
figures are in the `WB/Figures` collections, one per person, with ordinary keyframes on every
part; `WB_Sun` and the `WB_Clay` material control the look.

---

## 5. Settings and tuning

| To change | Edit |
|---|---|
| Blender / ffmpeg / library locations | `config.toml` → `[tools]`, `[paths]` |
| Upload limits (size, length, resolution) | `config.toml` → `[limits]` |
| Depth model and its working resolution (quality vs. speed) | `MODEL`, `PROCESS_PIXELS`, `MAX_RES` in `pipeline/depth.py` |
| Shimmer vs. trailing on moving edges | `SMOOTH` in `pipeline/analyse.py` |
| How strongly edges snap to the picture (lower = sharper, more surface texture) | the `8e-3` term in `sharpen_edges`, `pipeline/depth.py` |
| How far past a person's outline the set is repainted | the `(15, 15)` widening in `lift_out`, `pipeline/depth.py` |
| Assumed lens (changes how deep the scene looks) | `LENS` in `pipeline/analyse.py` |
| Which people count (confidence, minimum size, shortest track, longest gap bridged) | `MIN_PERSON_SCORE`, `MIN_HEIGHT`, `MIN_TRACK`, `MAX_GAP` in `pipeline/figures.py` |
| When a limb is drawn | `MIN_KEYPOINT` in `pipeline/figures.py` |
| Figure colours, proportions, visor | `FIGURE_COLOURS` and `placements()` in `pipeline/blender/build_scene.py` |
| Light, clay colour, render samples | the light and render sections at the end of `build_scene.py` |
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
| A person has no figure | Too small, too dark, or seen for under 8 frames; lower `MIN_HEIGHT`, `MIN_PERSON_SCORE` or `MIN_TRACK` |
| A figure changes colour mid-video | The person was lost for more than 12 frames, or there was a cut, so a new track started |
| A figure is missing arms or legs | The pose model was unsure of those joints; lower `MIN_KEYPOINT` (may add wrong limbs) |
| A soft patch remains where a person stood | The person covered a large part of the frame, so there was little surface around them to repaint from |
| The page will not open | The server is not running; start it (section 4) |
| An old video shows a different look | It was processed by an earlier version; upload it again |
| Set looks soft or lumpy | Dark, blurry or very low-resolution footage carries little depth detail |
| `uv` fails with "os error 17" inside the Claude desktop app | Set `$env:UV_CACHE_DIR = "C:\Users\kakssjs\.cache\uv"` first (not needed in a normal terminal) |
| Server was closed mid-job | Start it again; unfinished jobs restart automatically from the first stage |

The full error text of a failed job is in `jobs/<id>/job.json`.

---

## 8. What was verified

Checked on 2026-11-13 with an RTX 5060 laptop GPU (8 GB), using the film part of the reference
video `7182b76184e0351fc372b1f7f7d5c203.mp4` (a well-lit hall with a staircase), uploaded
through the website.

| Clip | Source | Result | Total time |
|---|---|---|---|
| Hall and staircase | 592×438, 30 fps, 787 frames | 1280×1896 (two 1280×948 halves), 787 frames, audio | 1156 s |

That run spent 855 s on analysis and 282 s on rendering. Afterwards the depth model's working
area was capped (`PROCESS_PIXELS`), which should roughly halve the analysis time for 4:3 and
16:9 clips; that has not been re-timed.

- **Set:** stairs, banisters, door panels, wall lamps and the counter are recognisable in all
  sampled frames. Surfaces are a soft relief, not clean planes, and the floor near the camera is
  lumpy.
- **People:** every person in the sampled frames was removed from the set along their outline
  and replaced by a figure in the right place and pose, including the two people on the floor.
- **Website:** the upload went through every stage; `final.mp4` and `white.mp4` were produced.

Not verified: the whole video frame by frame (ten sampled frames), the dark corridor and studio
samples with this version, the on-demand `scene.zip` download after it was changed, opening the
`.blend` by hand, the native file picker and real drag-and-drop, the over-size upload rejection,
clips near the 60 s limit, and `uv run` from a normal terminal.

---|---|---|---|---|
| Studio lot (well lit, handheld) | 592×240, 30 fps, 327 frames | 1280×1036, 327 frames, audio | 211 s (command line) | about 150 s |
| Corridor (very dark, 5 shots) | 1280×528, 24 fps, 472 frames | 1280×1056, 472 frames, audio | 296 s (website) | 169 s |

So expect roughly 0.4 s per frame for analysis plus 0.25 s per frame for rendering: a 60-second
clip at 30 fps takes about 20 minutes.

- **Look, studio lot:** the set shows the wall, door panels, timber frame and lumber pile as a
  clean clay surface; all visible people became figures with correct poses and visors; four
  consecutive frames were compared and were stable.
- **Look, corridor:** figures are placed correctly in most shots; the set is recognisable where
  there is light and soft where the picture is nearly black; in the darkest opening frames the
  person was not detected.
- **Website:** the corridor clip went through every stage in the browser's API to a finished
  result, and `scene.zip` downloaded (176 MB for that clip: the depth frames are large).
- **Earlier checks that still apply:** page layout at phone and desktop width, failed-upload
  message, delete, restart recovery, status codes for unknown jobs and files.

Not verified: the whole videos frame by frame (six to ten sampled frames each), the native file
picker and real drag-and-drop, the over-size upload rejection, clips near the 60 s limit, and
`uv run` from a normal terminal (the environment's Python was called directly).

---

## 9. Known limits and next steps

- **The set is a sculpted surface, not hand-clean geometry.** It follows the footage closely
  and shows stairs, doors, railings and furniture, but as a relief seen from the camera: edges
  are softer than a hand-built model's, and it cannot be walked around. Clear, well-lit footage
  gives a detailed set; dark or motion-blurred footage gives a soft, lumpy one.
- **Two alternatives were built and dropped:** fitting real walls and boxes from a multi-view
  reconstruction (nonsense geometry on a handheld shot), and a plain box room (clean, but it
  threw away the set detail that the reference videos keep).
- **No camera path.** The camera stays at the origin and the set changes around it. The light
  therefore moves with the camera.
- **Figures face the camera.** Joints sit at one depth per person, so figures follow poses but
  do not turn in depth.
- **Identity is not kept** across cuts or after someone is hidden: they return as a new figure.
- **Scale and lens are estimates**, so distances in the `.blend` are approximate.
- One job at a time; no progress percentage inside a stage; no accounts, so anyone who can open
  the site sees and can delete every job; Blender/ffmpeg error text includes server paths.

Next steps, in order of value: sharper, flatter set surfaces (plane-aware smoothing); keep one
figure per person across gaps; a recovered camera path with world-fixed light; then, for
public hosting, accounts, old-job cleanup, a cancel button and cleaned-up error text.
