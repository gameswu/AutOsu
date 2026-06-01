# AutOsu

Vision-based osu! standard mode AI player. Uses screen capture, object detection,
approach-circle timing, and a learned attention-based trajectory model to play
beatmaps autonomously.

## Architecture

```
Screen Capture   →  YOLOv8n Detect  →  Approach (ring box)  →  Controller       →  SendInput
  (DXcam ~2ms)       (6 classes)        (ring size → ratio)    (trajectory model)   (1000Hz)
```

Detection is learned (YOLO); the player is **vision-only** with an
**attention-based trajectory model** for cursor motion and **rule-based key
presses**:

```
targets, keys, phase = reference(scene)                       # target list + keys (rule-based)
v = trajectory_model(cursor, velocity, phase, targets)        # learned (cross-attention)
v = arrival_safeguard(v, cursor, primary, tth)                # approach phase only
cursor(t) = cursor(t-1) + v · dt
```

A rule-based *reference* layer determines the key state (taps and holds) and
builds the list of visible targets from detections + approach ratios. An
**attention-based trajectory model** (`TrajectoryModel`) produces the cursor
velocity: the cursor state queries a variable-length set of targets via
cross-attention, letting the model learn which target to prioritise and how to
move. Output is raw `(vx, vy)` — no tanh, no speed cap; the velocity range is
learned from human replay data. An `arrival_safeguard` (not part of the network)
ensures the cursor makes sufficient progress toward the primary target during
approach. Without trained weights the cursor holds position (model inactive).

## Project Structure

```
AutOsu/
├── configs/
│   └── default.yaml            # All configuration (paths, thresholds, etc.)
├── lib/
│   ├── osr2mp4/                # Extracted osr2mp4-core rendering library (patched)
│   └── recordclass.py          # Mutable named-tuple shim for osr2mp4
├── scripts/
│   ├── collect_data.py         # Automated beatmap + replay collection from osu.ppy.sh
│   ├── prepare_data.py         # Extract .osz, verify replay↔beatmap matching
│   ├── generate_dataset.py     # .osu + .osr → training data (via osr2mp4 renderer)
│   ├── preview.py              # Interactive rendering debug / video export
│   ├── train_detector.py       # YOLOv8n training
│   ├── demo_inference.py       # Offline "fake inference" — render + run full stack → annotated MP4
│   ├── debug_action.py         # Controller (live) + approach-ratio (approach-geo) diagnostics
│   ├── debug_overlay.py        # Live detection overlay on captured frames
│   └── run.py                  # Runtime entry point (play / observe)
├── src/
│   ├── data/
│   │   ├── osu_parser.py       # .osu beatmap parser → Beatmap dataclass
│   │   ├── replay_parser.py    # .osr replay parser + .osz extraction + MD5 matching
│   │   ├── skin_loader.py      # Skin textures + cursor/trail + tinting
│   │   ├── slider_path.py      # Bezier/Perfect/Linear curve computation
│   │   └── renderer.py         # osr2mp4 rendering wrapper → pixel-perfect frames
│   ├── vision/
│   │   ├── detector.py         # YOLO inference wrapper
│   │   ├── approach_from_boxes.py  # approach_ratio from detected approach-ring boxes (primary)
│   │   └── approach_geometry.py    # geometric (CV) approach_ratio from approach ring (fallback)
│   ├── control/
│   │   ├── tracker.py          # online timing tracker (approach-ratio → time-to-hit)
│   │   ├── planner.py          # scene building + target selection
│   │   ├── reference.py        # target list + key state (taps/holds, rule-based)
│   │   ├── motion_net.py       # TrajectoryModel (attention-based) + TrajectoryPolicy + arrival_safeguard
│   │   └── controller.py       # reference + trajectory model → cursor velocity + keys
│   └── runtime/
│       ├── pipeline.py         # Main loop: capture→detect→approach→controller→inject
│       ├── capture.py          # DXcam / mss screen capture
│       ├── window.py           # Win32 osu! window detection + coord mapping
│       └── injector.py         # Win32 SendInput + MockInjector
├── pyproject.toml
└── .gitignore
```

## Requirements

- **OS**: Windows 10/11 (native, not WSL)
- **GPU**: NVIDIA RTX 3060+ (CUDA 12.1, 6 GB+ VRAM)
- **Python**: 3.11+
- **osu!**: Stable client, borderless or fullscreen
- **Skin**: Single skin for consistency (default: WhiteCat - Selyu v2.3)

## Setup

```bash
# Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and install
git clone <repo-url>
cd AutOsu
uv sync
```

## Configuration

All settings live in `configs/default.yaml`. CLI arguments override config values
when provided. There is no `.env` file.

```yaml
device: "cuda:0"

data:
  raw_data_dir: "raw_data"      # contains beatmaps/ and replays/
  skin_dir: "C:/Users/ASUS/AppData/Local/osu!/Skins/WhiteCat - Selyu v2.3"
  output_dir: "dataset"
  fps: 30
  max_beatmaps: 100
```

## Raw Data

Training data comes from two sources, both downloadable from the osu! website:

| File | What | Where to get |
|------|------|-------------|
| `.osz` | Beatmap archive (contains `.osu` + background + audio) | [Beatmap listing](https://osu.ppy.sh/beatmapsets) — download button |
| `.osr` | Human replay file | User profile top plays or beatmap leaderboard — replay download icon |

### Automated collection

`scripts/collect_data.py` automates the entire process — searching for beatmaps,
downloading `.osz` files, filtering NoMod scores, and downloading replay `.osr` files.

**Prerequisites:** Add your osu! API credentials and session cookie to `configs/default.yaml`:

```yaml
osu_api:
  client_id: 12345            # From https://osu.ppy.sh/home/account/edit#oauth
  client_secret: "..."
  osu_session: "..."           # Browser cookie value (F12 → Application → Cookies)
```

**Usage:**

```bash
# Collect 50 beatmapsets (4-6 star, ranked, sorted by play count):
python scripts/collect_data.py --count 50

# Narrow star range, more replays per beatmap:
python scripts/collect_data.py --count 100 --star-min 4.0 --star-max 5.5 --max-replays 5

# Download only replays (skip .osz if you already have beatmaps):
python scripts/collect_data.py --count 100 --replays-only

# Dry run (show what would be downloaded):
python scripts/collect_data.py --count 10 --dry-run
```

The script uses dual authentication:
- **OAuth Bearer** for API calls (beatmapset search)
- **Web session cookie** + XHR headers for downloads (.osz, .osr) and NoMod score filtering

Score filtering uses the web endpoint `GET /beatmaps/{id}/scores?mods[]=NM` which
returns only NoMod scores, then filters for `has_replay=True` before attempting download.

### Manual collection

Alternatively, place files manually:

```
raw_data/
├── beatmaps/          ← drop .osz files here
│   ├── 123456 Artist - Title.osz
│   └── 789012 Another Song.osz
└── replays/           ← drop .osr files here
    ├── replay1.osr
    └── replay2.osr
```

**No manual pairing required.** The pipeline automatically:

1. Extracts `.osz` (they are zip archives) to get `.osu` files + assets
2. Computes the MD5 hash of every `.osu` file
3. Reads the `beatmap_hash` field inside each `.osr`
4. Matches replays to beatmaps by hash
5. Skips non-std maps and modded replays

## Usage

The complete workflow from raw files to a running AI player:

### Step 0 — Prepare & verify data

```bash
python scripts/prepare_data.py --data raw_data --verbose
```

Extracts `.osz` files, builds the MD5 index, and prints a match report
showing which replays belong to which beatmaps. Run this once after
placing new files to confirm everything is linked correctly.

### Step 1 — Generate training dataset

```bash
python scripts/generate_dataset.py ^
    --data raw_data ^
    --skin "C:\Users\ASUS\AppData\Local\osu!\Skins\WhiteCat - Selyu v2.3" ^
    --output dataset ^
    --max-beatmaps 100
```

Outputs:

| Directory | Contents |
|-----------|----------|
| `dataset/images/` + `dataset/labels/` | YOLO detection images + labels (cursor rendered from replay) |
| `dataset/data.yaml` | YOLO dataset descriptor (6 classes) |

The generator emits two phases per slider — a **slider_head** (class 1) with
its approach-circle during the approach phase, and a **slider_ball** (class 2)
at the follow-circle during the slide phase — plus the **approach_circle**
(class 4) for circles/heads. It class-balances the train split by oversampling
rare-class frames (`--no-balance` / `--balance-max-factor N` to tune).

### Step 1.5 — Preview & debug rendering

Interactive OpenCV window for verifying that osr2mp4-rendered frames match real osu! visuals.
The preview tool auto-extracts `.osz` and matches replays by MD5 — no need to
manually locate `.osu` files:

```bash
# From raw_data directory (interactive beatmap/replay selection):
python scripts/preview.py --data raw_data --skin path/to/skin

# From a single .osz (auto-finds matching replays in raw_data/replays/):
python scripts/preview.py --osz raw_data/beatmaps/123456.osz --skin path/to/skin ^
    --replays raw_data/replays

# Export as video (no GUI):
python scripts/preview.py --data raw_data --skin path/to/skin --export debug.mp4
```

Controls during playback:

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| Right arrow | Step frame (when paused) |
| `+` / `-` | Speed 2x / 0.5x |
| `S` | Save current frame as PNG |
| `Q` / Esc | Quit |

> **Note:** Rendering is forward-only (osr2mp4 is stateful). Backward stepping
> and jump-to-time are not supported.

### Step 2 — Train detection model

```bash
python scripts/train_detector.py --epochs 100 --batch 16
```

Output: `runs/detect/train/weights/best.pt` (auto-exports ONNX).

### Step 3 — Approach ratio (no training)

Approach ratio is computed at runtime from vision only. The **primary** path
(`src/vision/approach_from_boxes.py`) pairs each detected approach-circle box
(class 5) with its hitcircle / slider-head and converts the ring size to a
ratio with exact geometry (ring shrinks from 4.0× → 1.0× the hitcircle radius).
A geometric CV estimator (`src/vision/approach_geometry.py`) remains as a
no-detection fallback. No model, no GPU, no crops.

Validate the geometric estimator against ground-truth timing ratios on
re-rendered replays:

```bash
python scripts/debug_action.py approach-geo \
    --data raw_data --skin "<skin dir>" --frames 1000
```

### Step 4 — Controller (no training needed for keys)

The key-press logic (tap timing, hold/release) is rule-based and needs no
training. Cursor motion comes from a **learned trajectory model** — see Step 4.5.
Inspect the controller live (capture only, no input injected):

```bash
python scripts/debug_action.py live
```

#### Step 4.5 — Train the trajectory model

The trajectory model learns human cursor motion from real replays. Navigation
targets are reconstructed from beatmap ground truth (geometry only, no
rendering) and the model regresses **raw human cursor velocity** given the
cursor state and visible target set. The runtime itself stays vision-only.

```bash
# 1) build the trajectory dataset (cursor features + targets + velocities)
python scripts/build_motion_dataset.py -c configs/default.yaml \
    --data raw_data --output runs/motion/dataset.npz

# 2) train the attention-based trajectory model
python scripts/train_motion.py -c configs/default.yaml \
    --dataset runs/motion/dataset.npz --output runs/motion/trajectory.pt
```

Then set `motion_net_path: runs/motion/trajectory.pt` in your config. Without
trained weights the cursor holds position (model inactive).

### Step 5 — Run the AI

```bash
# Observe mode (detection overlay, no input injection):
python scripts/run.py observe

# Play mode (full AI control):
python scripts/run.py play

# Custom config:
python scripts/run.py play --config configs/my_config.yaml

# Force CPU (no GPU):
python scripts/run.py observe --device cpu
```

### Step 5.5 — Offline inference demo (watch the model play, no osu! needed)

`scripts/demo_inference.py` lets you *see the AI play a real beatmap without
launching osu!*. Instead of capturing the live screen, it re-renders a beatmap +
replay with the same osr2mp4 renderer used for training, then runs the **actual
runtime stack** on every rendered frame and writes an annotated MP4:

```
rendered frame  →  YOLO detect  →  approach-ring ratio
                →  controller (trajectory model + keys)  →  cursor + (z, x)
```

```bash
uv run python scripts/demo_inference.py ^
    --data raw_data ^
    --skin "C:\Users\ASUS\AppData\Local\osu!\Skins\WhiteCat - Selyu v2.3" ^
    --output demo.mp4 --frames 1200 --fps 15 --scale 2
```

The overlay shows:

| Element | Meaning |
|---------|---------|
| Yellow boxes + `0.42` | Detections + approach ratio (actionable objects) |
| Ring / ball boxes | Detected approach circles and slider balls |
| **Red** dot + trail | Controller-driven cursor |
| **Green** crosshair | Human (replay) cursor — ground-truth reference |
| Z / X lamps + HUD | Controller key state, target, frame/time |

Useful options: `--index N` (which matched pair), `--fps` (should match training
fps), `--scale` (output upscale), `--conf` (detector confidence). Requires the
trained detector (`runs/detect/train/weights/best.pt`); optionally the
trajectory model (`runs/motion/trajectory.pt`).

### Optional — TensorRT export

```bash
trtexec --onnx=runs\detect\train\weights\best.onnx ^
        --saveEngine=best.engine --fp16
```

## Models

| Component | Architecture | Parameters | Input | Output |
|-----------|-------------|-----------|-------|--------|
| Detector | YOLOv8-nano | 3.2M | 640x384 BGR | 6-class bounding boxes |
| Approach Estimator | Ring-box geometry (no model) | 0 | detections | approach_ratio [0, 1] |
| Reference | Rule-based phase FSM | 0 | detections + ratios | target list + (key_z, key_x) + phase |
| Trajectory Model | Cross-attention + MLP | ~50K | cursor(8) + targets(N×8) | raw (vx, vy) unbounded |
| Arrival Safeguard | `v_toward ≥ d/tth` constraint | 0 | velocity + primary target | adjusted velocity (approach phase only) |

Detection classes: `hitcircle`(0), `slider_head`(1), `slider_ball`(2),
`spinner`(3), `approach_circle`(4), `slider_body`(5). Sliders are labeled in
two phases: a `slider_head` (+`approach_circle`) while approaching, then a
`slider_ball` at the follow-circle while sliding. `slider_body` indicates a
slider is still active (used when the slider_ball is momentarily lost).

## Coordinate System

osu! uses a 512x384 playfield. The renderer uses osr2mp4-core's coordinate
mapping (from `Utils/Resolution.py`):

```
scale          = render_height / 768
playfieldscale = (595.5 / 768) * scale * 2
pf_height      = render_height * 595.5 / 768
moveright      = round((render_width - pf_height * 4/3) / 2.1)
movedown       = round((render_height - pf_height) / 1.9)

render_x = int(osu_x * playfieldscale) + moveright
render_y = int(osu_y * playfieldscale) + movedown
```

This formula is consistent across the renderer, YOLO labels, and runtime transforms.

## Constraints

- NoMod only (no HR/DT/HD/EZ/FL)
- Single skin (model trained on specific skin textures)
- Static backgrounds only
- osu! standard mode only (mode 0)
- Academic project — not for ranked play

## License

MIT. Academic use only. Not intended for ranked osu! play.
