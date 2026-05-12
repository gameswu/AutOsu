# AutOsu Development Roadmap

Vision-based osu! std AI player — fully learned control (no heuristic trajectories).

---

## Phase 0 — Environment & Infrastructure [done]

- [x] Project skeleton with `uv` dependency management
- [x] YAML-based configuration (`configs/default.yaml`)
- [x] GPU environment verified (CUDA 12.1, PyTorch 2.3.1, RTX 3060)

---

## Phase 1 — Data Foundation [done]

### 1a — Beatmap Parser (`src/data/osu_parser.py`)
- [x] Parse all .osu sections (General, Metadata, Difficulty, TimingPoints, Colours, HitObjects)
- [x] HitCircle / Slider / Spinner extraction with full timing
- [x] Timing point handling (BPM, inherited SV)
- [x] Combo colour assignment
- [x] AR/CS/OD → preempt / circle radius conversion

### 1b — Skin Loader (`src/data/skin_loader.py`)
- [x] Parse skin.ini, load std textures as BGRA numpy arrays
- [x] Pre-tint hitcircle by combo colour
- [x] Number font sprites (default-0~9)
- [x] Cursor + cursortrail + cursormiddle texture loading
- [x] @2x / SD resolution handling

### 1c — Replay Parser (`src/data/replay_parser.py`)
- [x] Parse .osr files (osrparse)
- [x] Extract frame-level cursor positions + key states
- [x] Interpolation for arbitrary timestamp queries
- [x] `.osz` extraction (auto-unzip beatmap archives)
- [x] MD5-based beatmap indexing (`build_beatmap_index()`)
- [x] Replay scanning with beatmap_hash extraction (`scan_replays()`)
- [x] `find_replay_pairs()` — automatic .osr->.osu matching by MD5 hash
- [x] Verified on real data: 63/64 replays matched across 652 .osu files

### 1d — Window Detection (`src/runtime/window.py`)
- [x] Win32 API: dynamic osu! window finding
- [x] Client rect, DPI-aware coordinate mapping
- [x] osu!px ↔ screen ↔ model input transforms (no hardcoded values)

---

## Phase 2 — Synthetic Data Pipeline [done]

### 2a — Frame Renderer (`src/data/renderer.py`)
- [x] osr2mp4-core extracted into `lib/osr2mp4/` (stripped audio/ffmpeg/C extension)
- [x] C extension bridged: `lib/osr2mp4/ImageProcess/Curves/curves.py` → pure-Python via `src/data/slider_path.py`
- [x] `recordclass` shim at `lib/recordclass.py` (mutable named-tuples with index access)
- [x] Pillow 10+ compatibility (`textsize` → `textbbox`)
- [x] `Osr2mp4Renderer` wrapper: constructs Settings, parses beatmap+replay, runs `checkmain`, loads skin via `PreparedFrames`, renders frame-by-frame via `FrameObjects` component managers
- [x] Draw order matches `Draw.py:106-143` exactly (background → hitresult → followpoints → hitobjects → hitresult → cursor)
- [x] Tested: 9000 frames, 0 errors, ~200 fps on test data

### 2b — Dataset Generator (`scripts/generate_dataset.py`)
- [x] Unified pipeline: .osu + .osr → YOLO images + approach crops + action sequences
- [x] Uses osr2mp4 renderer for pixel-perfect frame generation
- [x] Cursor + trail rendered from replay data (so YOLO learns to ignore it)
- [x] YOLO labels generated from osr2mp4 beatmap hit objects
- [x] Stacked note deduplication
- [x] Configurable FPS, max_beatmaps (post-filter), train/val split
- [x] Outputs:
  - `dataset/images/` + `dataset/labels/` (YOLO format)
  - `dataset/crops/` (64x64 approach estimator training)
  - `dataset/sequences/` (.npz state/action pairs)
  - `dataset/data.yaml`

### 2c — Data Preparation (`scripts/prepare_data.py`)
- [x] Automatic `.osz` extraction to `beatmaps/` subdirectories
- [x] MD5 index build + replay matching report
- [x] `--verbose` mode with per-replay match details
- [x] Mod filtering (skip non-NoMod replays)
- [x] Mode filtering (skip non-std replays)

### 2d — Preview / Debug Viewer (`scripts/preview.py`)
- [x] Interactive OpenCV playback with osr2mp4 renderer
- [x] Playback speed control (0.5x / 2x)
- [x] Frame stepping (forward only — osr2mp4 is stateful)
- [x] Save frame as PNG
- [x] Video export mode (`--export output.mp4`)

---

## Phase 3 — Vision Models [done]

### 3a — Object Detector (`src/vision/detector.py`)
- [x] YOLOv8-nano, 5 classes, 640x384 input
- [x] Training script (`scripts/train_detector.py`)
- [x] ONNX export built into training pipeline
- [ ] TensorRT FP16 export + benchmark

### 3b — Approach Estimator (`src/vision/approach_estimator.py`)
- [x] 63K-param CNN (64x64 → approach_ratio)
- [x] Training script (`scripts/train_approach.py`)
- [x] Inference wrapper with batched prediction

---

## Phase 4 — Action Model [done]

### 4a — State/Action Definition (`src/action/state.py`)
- [x] GameStateVector: 16 objects × 8 features + cursor(4) + time_delta(1) = 133 dims
- [x] ActionVector: (dx, dy, key_z, key_x) normalised
- [x] Consistent format between offline training and online inference

### 4b — GRU Policy Network (`src/action/model.py`)
- [x] 858K-param GRU (2 layers, hidden=256)
- [x] Stateful inference wrapper (maintains hidden state across frames)
- [x] Training script with MSE + BCE loss (`scripts/train_action.py`)

---

## Phase 5 — Runtime Pipeline [done]

### 5a — Screen Capture (`src/runtime/capture.py`)
- [x] DXcam (DXGI Desktop Duplication) with mss fallback
- [x] Continuous capture mode for DXcam

### 5b — Input Injection (`src/runtime/injector.py`)
- [x] Win32 SendInput (absolute mouse + keyboard)
- [x] 1000Hz dedicated thread
- [x] MockInjector for observe mode

### 5c — Game Pipeline (`src/runtime/pipeline.py`)
- [x] Full loop: capture → detect → approach → state → action → inject
- [x] Configurable FPS target
- [x] Graceful start/stop

### 5d — Entry Point (`scripts/run.py`)
- [x] `play` mode (full injection)
- [x] `observe` mode (detection only)
- [x] Config loading from YAML + CLI overrides

---

## Phase 6 — Training & Evaluation [in progress]

### 6a — Automated Data Collection (`scripts/collect_data.py`) [done]
- [x] `OsuClient` class with dual auth (OAuth API + osu_session cookie)
- [x] Beatmapset search via API (star range, mode, status, sort, pagination)
- [x] NoMod score filtering via web endpoint (`/beatmaps/{id}/scores?mods[]=NM`)
- [x] XHR-style requests with XSRF token for web endpoints
- [x] `.osz` download via web session cookie
- [x] `.osr` download via web session cookie, filtered by `has_replay=True`
- [x] Session validity check (`--check-session`)
- [x] Deduplication (skips already-downloaded files)
- [x] Dry-run mode, replays-only mode, configurable star range / replay count
- [x] End-to-end tested: searches, filters NM, downloads .osz + .osr

### 6b — Training Pipeline [next]
- [ ] Collect raw data: 100+ `.osz` beatmaps + `.osr` replays across 2-5 star difficulty range
- [ ] Run `prepare_data.py` to verify matching coverage
- [ ] Generate full dataset with cursor rendering (~30-50K images)
- [ ] Train detector to mAP50 > 0.9
- [ ] Train approach estimator to MAE < 0.05
- [ ] Train action model: cursor MAE < 10px, key accuracy > 85%
- [ ] End-to-end test on easy maps (2-3 star)
- [ ] Record and review AI gameplay for failure analysis

---

## Phase 7 — Optimization [future]

- [ ] TensorRT FP16/INT8 for detector (target < 3ms)
- [ ] Profile and reduce pipeline latency
- [ ] Adaptive confidence thresholds based on map density
- [ ] Error recovery (handle missed detections gracefully)
- [ ] Test on progressively harder maps (4-5 star)

---

## Dropped / Superseded

- ~~Bezier trajectory generation~~ → replaced by learned GRU action model
- ~~Rule-based controller~~ → fully learned from replays
- ~~.env configuration~~ → unified in `configs/default.yaml`
- ~~C++ pybind11 input module~~ → ctypes SendInput is sufficient at 1000Hz
- ~~Manual .osu/.osr pairing by filename~~ → replaced by automatic MD5 hash matching
- ~~Custom Python renderer~~ → replaced by osr2mp4-core rendering wrapper for pixel-perfect output
