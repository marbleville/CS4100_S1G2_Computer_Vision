## Module B Plan: Vision Front-End (`preprocessor/`) Without OpenCV

### Summary

Build a Python-only preprocessing package that ingests webcam frames (primary) or directory images (test mode), localizes the hand using classical CV methods implemented from scratch, and outputs:

- current hand features for static gesture models
- motion trajectory + aggregate x/y deltas for dynamic gesture models

This plan targets first-check-in readiness with deterministic tests and inspectable visual outputs before deep performance tuning.

### Public Interfaces (v1)

- `init_preprocessor(config: PreprocessorConfig) -> Preprocessor`
- `Preprocessor.get_current_hand() -> HandFrameResult`
- `Preprocessor.get_current_motion() -> MotionWindowResult`

Proposed types:

- `PreprocessorConfig`
- `buffer_size: int`
- `async_process: bool`
- `input_mode: Literal["webcam","image_dir"]`
- `camera_device: int`
- `image_dir: Optional[str]`
- `frame_size: tuple[int, int]`
- `fps_hint: int`
- `threshold_profile: str`
- `HandFrameResult`
- `timestamp_ms: int`
- `bbox_xyxy_norm: tuple[float, float, float, float]`
- `centroid_xy_norm: tuple[float, float]`
- `contour_points_norm: list[tuple[float, float]]`
- `quality_score: float`
- `MotionWindowResult`
- `window_size: int`
- `trajectory_xy_norm: list[tuple[float, float]]`
- `delta_x_px: float`
- `delta_y_px: float`
- `path_length_px: float`
- `motion_confidence: float`

### Strategy Options for Hand Localization (Pros/Cons)

1. Classical segmentation (recommended for v1)

- Pros: explainable, no model training needed, aligns with “no external landmark model.”
- Cons: sensitive to lighting/background, needs calibration/tuning.

2. Early learned detector

- Pros: can generalize better across environments once trained.
- Cons: requires labeled detection data and longer setup; higher risk for first milestone.

### Implementation Plan

<s>

1. Package skeleton and contracts

- Create `preprocessor/` package structure (`config`, `io`, `pipeline`, `features`, `buffers`, `types`, `tests`).
- Convert `preprocessor/contract.md` into strict API + invariants doc.
- Add typed dataclasses for all request/response objects.

2. Frame ingestion layer

- Implement webcam source via `imageio`/`ffmpeg` wrapper.
- Implement directory-frame source with deterministic ordering.
- Add common frame iterator interface and timestamp normalization.

3. From-scratch preprocessing pipeline

- RGB->grayscale and optional color-space transforms using `numpy`.
- Noise reduction (manual Gaussian/box filter kernels via `numpy` ops).
- Adaptive/percentile thresholding.
- Connected components extraction.
- Largest plausible hand blob selection with heuristics (area, aspect ratio, position continuity).

4. Feature extraction

- Compute normalized bbox and centroid.
- Extract contour/edge points (boundary tracing on binary mask).
- Compute frame quality score (mask stability + area constraints).
  </s>

5. Motion buffering and temporal features

- Circular buffer of recent `HandFrameResult`.
- `get_current_motion()` returns trajectory + aggregate deltas.
- Add smoothing (EMA/median over centroid path) and missing-frame handling.

6. Async vs on-demand execution mode

- `async_process=True`: background frame consumer updates latest state.
- `async_process=False`: process on request from newest available frame.
- Add lock-safe state handoff and stale-frame detection.

7. Diagnostics and artifacts

- Debug visualizer output (mask, bbox, centroid overlay) saved to `artifacts/preprocessor_debug/`.
- Small replay script for dataset clip/image-dir validation.

8. Integration hooks

- Stable output schema for Module C (static) and Module D (motion).
- Explicit error codes/status when hand not found or confidence too low.

### Test Cases and Scenarios

- Unit tests:
- coordinate normalization stays in `[0,1]`
- centroid/bbox correctness on synthetic masks
- motion delta/path length math on known trajectories
- buffer eviction and ordering correctness
- thresholding/component selection deterministic on fixture frames
- Integration tests:
- webcam adapter initialization and graceful fallback on camera unavailable
- image directory pipeline processes full sample set deterministically
- async and sync modes produce consistent outputs on same frame sequence
- Robustness tests:
- no-hand frames return structured “not found” result
- partial occlusion and background clutter fixtures
- sudden lighting change fixtures with bounded failure behavior

### Milestones (High-Level)

1. Week 1: contracts/types + frame IO adapters + synthetic tests.
2. Week 2: segmentation + component selection + hand feature outputs.
3. Week 3: motion buffer + smoothing + API stabilization.
4. Week 4: deterministic test suite + visual diagnostics + check-in demo assets.

### Done Criteria (v1)

- All API contracts implemented and documented.
- Deterministic preprocessing tests passing on fixed fixtures.
- Visual debug outputs generated for representative static/dynamic samples.
- `get_current_hand()` and `get_current_motion()` usable by downstream modules without format changes.

### Assumptions and Defaults

- Python-only implementation, no OpenCV.
- Allowed core dependencies: `numpy`, `imageio[ffmpeg]`, `Pillow`, `pytest`.
- Primary runtime is live webcam; image-directory mode is mandatory for reproducible tests.
- Motion API standard is trajectory + aggregate deltas.
- First milestone prioritizes correctness/reproducibility over strict FPS optimization.
