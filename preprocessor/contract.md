# Preprocessor Contract (v1)

## Purpose

`preprocessor` is Module B's vision front-end contract. It accepts raw frames from a webcam
or image directory source and returns typed, model-ready hand and motion features.

This document freezes Phase 1 public interfaces, result semantics, and invariants.

## Public Python API

```python
init_preprocessor(config: PreprocessorConfig) -> Preprocessor
Preprocessor.get_current_hand() -> HandFrameResult
Preprocessor.get_current_motion() -> MotionWindowResult
```

## Public Types

### `PreprocessorConfig`

- `buffer_size: int`
- `async_process: bool`
- `input_mode: Literal["webcam", "image_dir"]`
- `camera_device: int = 0`
- `image_dir: str | None = None`
- `video_path: str | None = None`
- `frame_size: tuple[int, int] = (640, 480)`
- `fps_hint: int = 30`
- `threshold_profile: str = "default"`

### `ResultStatus`

Required enum values:

- `ok`
- `no_hand`
- `low_confidence`
- `insufficient_history`
- `error`

### `HandFrameResult`

- `status: ResultStatus` (required)
- `timestamp_ms: int`
- `bbox_xyxy_norm: tuple[float, float, float, float] | None`
- `centroid_xy_norm: tuple[float, float] | None`
- `contour_points_norm: list[tuple[float, float]]`
- `quality_score: float | None`
- `error_message: str | None`

### `MotionWindowResult`

- `status: ResultStatus` (required)
- `timestamp_ms: int`
- `window_size: int`
- `trajectory_xy_norm: list[tuple[float, float]]`
- `delta_x_px: float`
- `delta_y_px: float`
- `path_length_px: float`
- `motion_confidence: float | None`
- `error_message: str | None`

## Result Semantics

- `status` is mandatory for every result object.
- No hand detected is represented by:
  - `HandFrameResult(status=ResultStatus.NO_HAND, bbox_xyxy_norm=None, centroid_xy_norm=None, ...)`
- Insufficient temporal history is represented by:
  - `MotionWindowResult(status=ResultStatus.INSUFFICIENT_HISTORY, ...)`
- Errors are represented as structured result objects with `status=ResultStatus.ERROR`
  and an optional `error_message`.

## Invariants (Documented, Not Runtime-Enforced in Phase 1)

- When `status == ResultStatus.OK`, normalized coordinates are in `[0.0, 1.0]`.
- For motion output: `window_size == len(trajectory_xy_norm)`.
- `buffer_size >= 1`.
- If `input_mode == "image_dir"`, then `image_dir` is required.

## Frame Ingestion Layer (Phase 2)

### Activation Rule

- If `video_path` is set, local video ingestion is used.
- If `video_path` is not set, source selection falls back to not-yet-implemented
  modes (webcam/image directory) and currently raises `NotImplementedError`.

### Common Source Interface

```python
FrameSource.open() -> None
FrameSource.read() -> FramePacket | None
FrameSource.close() -> None
```

- `read()` returns a `FramePacket` while frames are available.
- `read()` returns `None` at end-of-stream (EOS).

### `FramePacket`

- `frame_index: int`
- `timestamp_ms: int`
- `frame_rgb: np.ndarray`
- `source_id: str`

Timestamp policy:
- `timestamp_ms` is always derived from frame index and source fps:
  `int((frame_index / fps) * 1000)`.

Frame normalization guarantees:
- `frame_rgb` is RGB, shape `(frame_size[1], frame_size[0], 3)`, dtype `uint8`.
- Source frames are resized to `PreprocessorConfig.frame_size` in IO layer.

## Preprocessing Pipeline (Phase 3)

### Public API

```python
PreprocessingPipeline(config: PreprocessorConfig)
PreprocessingPipeline.process(packet: FramePacket) -> PipelineFrameResult
PreprocessingPipeline.reset() -> None
```

### `PipelineFrameResult`

- `timestamp_ms: int`
- `frame_index: int`
- `mask: np.ndarray` (binary mask)
- `selected_label: int | None`
- `selected_bbox_xyxy_px: tuple[int, int, int, int] | None`
- `selected_centroid_xy_px: tuple[float, float] | None`
- `selected_area_px: int | None`
- `candidate_count: int`
- `quality_score: float`
- `debug: dict[str, float | int | str]`

### Behavior

- Input frame validation:
  - expects shape `(H, W, 3)`.
  - converts to `uint8` if needed.
- Color transforms:
  - RGB->grayscale.
  - RGB->HSV.
  - RGB->YCbCr.
- Optional running-average background model contributes foreground score.
- Fused confidence map is denoised with gaussian blur.
- Thresholding:
  - global percentile threshold.
  - optional local tile correction blended with global threshold.
- Mask cleanup:
  - binary open.
  - binary close.
- Connected components extract per-blob geometry and quality stats.
- Candidate selection applies hard constraints and continuity-aware scoring.
- The selected candidate and diagnostics are returned as `PipelineFrameResult`.

### Continuity and State

- Pipeline maintains prior selected centroid/bbox/area for continuity scoring.
- `reset()` clears continuity and background model state.

### Handoff to Phase 4

- `pipeline_result_to_hand_result(...)` converts `PipelineFrameResult` into
  `HandFrameResult`.
- Phase 4 feature extraction will expand contour extraction and richer output
  fields while preserving this seam.
