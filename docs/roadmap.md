# AutOsu Development Roadmap

Vision-based osu! std AI player — learned detection + deterministic vision-only control.

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
- [x] Unified pipeline: .osu + .osr → YOLO images + labels (5 classes)
- [x] Uses osr2mp4 renderer for pixel-perfect frame generation
- [x] Cursor + trail rendered from replay data (so YOLO learns to ignore it)
- [x] YOLO labels generated from osr2mp4 beatmap hit objects, incl. approach
      circle (class 4, ring scale 4→1) and slider ball (class 2, at the
      follow-circle during the slide phase)
- [x] Stacked note deduplication
- [x] Class-balanced train split (oversample rare-class frames) + histogram
- [x] Configurable FPS, max_beatmaps (post-filter), train/val split
- [x] Outputs:
  - `dataset/images/` + `dataset/labels/` (YOLO format)
  - `dataset/data.yaml` (7 classes)

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

### 3b — Approach Estimator
- [x] **Primary** (`src/vision/approach_from_boxes.py`): pair each detected
      approach-circle box with its hitcircle/slider-head; `ratio = (4 - r_ring/R_disc)/3`,
      optional temporal global-slope fit; no ring ⇒ ratio 1.0
- [x] **Fallback** (`src/vision/approach_geometry.py`): geometric CV, polar-unwrap
      around each object, read approach-ring radius; temporal linear-fit
      (overall ratio MAE 0.18 → 0.13)
- [x] Validation mode: `debug_action.py approach-geo` (vs ground-truth timing ratio)
- [x] Replaced the old 63K-param CNN (poor calibration on small 64×64 crops)

---

## Phase 4 — Controller [done]

### 4a — Online Timing Tracker (`src/control/tracker.py`)
- [x] Per-object approach-ratio → time-to-hit estimate, EMA-smoothed preempt
- [x] Pure vision, no beatmap parsing

### 4b — Deterministic Controller (`src/control/{planner,motion,controller}.py`)
- [x] Scene building + most-imminent target selection (`planner.py`)
- [x] Human-like min-jerk motion + damped follow + tapered jitter (`motion.py`)
- [x] State machine: approach/tap → slide (reactive ball follow) → spin sweep,
      Z/X alternation (`controller.py`)
- [x] Zero learned parameters; replaced the behavioral-cloning GRU
- [x] Optional baked `MotionProfile` (jitter / overshoot / follow_alpha /
      tap_lead_ms) extracted offline from real replays by
      `scripts/analyze_motion.py`; loaded via `motion_profile_path` (runtime
      stays vision-only — no replays/beatmaps read at play time)

### 4c — Learned Motion Net (CPRP) [implemented; weights pending]

A learned, human-like cursor layer that **cannot violate hard constraints**
(must pass through hit circles / follow the ball / stay on the spin circle),
unlike a raw behavioral-cloning policy. **Constraint-Projected Residual
Policy (CPRP)**.

```
cursor(t) = reference(t) + gate(phase) · residual(t)
```

- **reference(t)** — `src/control/reference.py` (`ReferenceController`).
  Recomputed every frame from the *current detections*: min-jerk interpolation
  toward the most-imminent target (approach/tap), the detected follow-ball
  (slide), or the circular sweep (spin). The old monolithic deterministic
  controller logic now lives here; it guarantees the hard constraints and the
  key/tap timing on its own.
- **residual(t)** — `src/control/motion_net.py` (`ResidualPolicy` + a small
  `tanh`-bounded **MLP**) over *reference-relative* features (phase one-hot,
  time-to-hit, target vector in cursor frame, recent velocity; `FEATURE_DIM=10`,
  shared by runtime and the dataset builder). Output is capped at
  `max_residual_osu`; `gate(phase)` shrinks it to ~0 at the tap/contact instants
  (scales with `1 − approach_ratio` while approaching) so accuracy is never
  sacrificed for style. Keys come straight from the reference — the residual
  only nudges the cursor.
- **composition** — `src/control/controller.py` (`Controller`) keeps the same
  public API (`update`/`reset`/`ControlOutput`). When the residual is active the
  reference's hand-made jitter/overshoot are disabled (the net supplies human
  deviation instead).
- **training** — fully offline / supervised, run on your server:
  `scripts/build_motion_dataset.py` reconstructs the reference frame-by-frame
  from beatmap ground truth (geometry only, no rendering) under teacher forcing
  and records `human − reference`; `scripts/train_motion.py` regresses the
  bounded residual. No RL, no env rollouts.
- **runtime** — vision-only; consumes only detections + phase. With **no
  weights** the `ResidualPolicy` is inactive and the controller emits the pure
  deterministic reference (min-jerk fallback).
- **config** — `motion_net_path` + `max_residual_osu` (entry points
  `autosu-build-motion-dataset`, `autosu-train-motion`).
- [ ] (you) build dataset + train weights on the Linux server, then set
  `motion_net_path`

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
- [x] Full loop: capture → detect → approach → controller → inject
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
- [ ] Generate full dataset with cursor rendering (~30-50K images, 5 classes, balanced)
- [ ] Train detector to mAP50 > 0.9 (incl. approach_circle / slider_ball)
- [ ] End-to-end test on easy maps (2-3 star)
- [ ] Record and review AI gameplay for failure analysis

### 6c — Offline Inference Demo (`scripts/demo_inference.py`) [done]
- [x] Re-renders a beatmap+replay and runs the full runtime stack per frame
      (YOLO → ring-box approach → deterministic controller) without launching osu!
- [x] Annotated MP4 output: detection boxes + approach ratios, controller cursor
      (red, with trail) vs human/replay cursor (green), Z/X key lamps + HUD
- [x] Reuses the shared `open_replay_frames()` render factory (one render frame
      = one inference step; render fps should match training fps)

---

## Phase 7 — Optimization [future]

- [ ] TensorRT FP16/INT8 for detector (target < 3ms)
- [ ] Profile and reduce pipeline latency
- [ ] Adaptive confidence thresholds based on map density
- [ ] Error recovery (handle missed detections gracefully)
- [ ] Test on progressively harder maps (4-5 star)

---

## Dropped / Superseded

- ~~Bezier trajectory generation~~ → replaced by deterministic min-jerk controller
- ~~GRU behavioral-cloning action model~~ → replaced by deterministic vision-only
  controller (BC drifted, suffered covariate shift, and was slider-blind)
- ~~`.npz` state/action sequences~~ → no longer generated (no action model)
- ~~.env configuration~~ → unified in `configs/default.yaml`
- ~~C++ pybind11 input module~~ → ctypes SendInput is sufficient at 1000Hz
- ~~Manual .osu/.osr pairing by filename~~ → replaced by automatic MD5 hash matching
- ~~Custom Python renderer~~ → replaced by osr2mp4-core rendering wrapper for pixel-perfect output
- ~~7-class detection (`slider_body`, `slider_end`)~~ → collapsed to **5 classes**
  (`hitcircle`, `slider_head`, `slider_ball`, `spinner`, `approach_circle`).
  `slider_body` was a noisy huge AABB and `slider_end` was unused by the
  controller. Sliders are now labeled in two phases: head+approach-circle while
  approaching, ball at the follow-circle while sliding.
- ~~slider_ball never emitted (labeling bug)~~ → the visibility filter dropped a
  whole active slider once its head circle faded (`is_fadeout` read from the
  head), so the slide phase produced no labels. Fixed by keeping mid-slide
  sliders visible and splitting the label phase at the object's hit time.
