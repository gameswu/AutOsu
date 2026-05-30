# AutOsu

Vision-based osu! standard mode AI player. Uses screen capture, object detection,
approach-circle timing, and a deterministic vision-only controller to play
beatmaps autonomously.

## Architecture

```
Screen Capture   →  YOLOv8n Detect  →  Approach (ring box)  →  Controller       →  SendInput
  (DXcam ~2ms)       (7 classes)        (ring size → ratio)    (target + path)     (1000Hz)
```

Detection is learned (YOLO); everything downstream is **deterministic and
vision-only** — no behavioral cloning. Timing comes from the detected
approach-circle box size, and cursor movement / key timing are produced by a
hand-written controller (min-jerk motion to the most imminent object, reactive
slider-ball following, circular spinner sweep).

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
│   │   ├── motion.py           # human-like min-jerk motion + smoothing
│   │   └── controller.py       # deterministic state machine → cursor target + keys
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
| `dataset/data.yaml` | YOLO dataset descriptor (7 classes) |

The generator also emits the approach-circle (class 5) and slider-ball (class 6)
labels and class-balances the train split by oversampling rare-class frames
(`--no-balance` / `--balance-max-factor N` to tune).

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

### Step 4 — Controller (no training)

There is no action model to train. The deterministic controller in
`src/control/` consumes detections + approach ratios every frame and emits a
cursor target and key state. Inspect it live (capture only, no input injected):

```bash
python scripts/debug_action.py live
```

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
                →  deterministic controller  →  cursor target + (z, x)
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
trained detector (`runs/detect/train/weights/best.pt`); the controller needs no
weights.

### Optional — TensorRT export

```bash
trtexec --onnx=runs\detect\train\weights\best.onnx ^
        --saveEngine=best.engine --fp16
```

## Models

| Component | Architecture | Parameters | Input | Output |
|-----------|-------------|-----------|-------|--------|
| Detector | YOLOv8-nano | 3.2M | 640x384 BGR | 7-class bounding boxes |
| Approach Estimator | Ring-box geometry (no model) | 0 | detections | approach_ratio [0, 1] |
| Controller | Deterministic state machine | 0 | detections + ratios | cursor target + (key_z, key_x) |

Detection classes: `hitcircle`(0), `slider_head`(1), `slider_body`(2),
`slider_end`(3), `spinner`(4), `approach_circle`(5), `slider_ball`(6).

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
