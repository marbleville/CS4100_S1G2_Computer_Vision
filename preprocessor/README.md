# Preprocessor Contract

## Purpose

`preprocessor` is Module B's vision front-end contract. It accepts raw frames from a webcam
or local video source and returns normalized candidate hand crops for downstream gesture models.

## How to use

```python
from preprocessor import init_preprocessor
from preprocessor.config.types import PreprocessorConfig

config = PreprocessorConfig(
    input_mode="local_video",
    video_path="path/to/video_file",
)

preprocessor = init_preprocessor(config)

frame_result = preprocessor.get_current_hand_candidates()
for candidate in frame_result.candidates:
    print(candidate.bbox_xyxy_px, candidate.frame_rgb.shape)

# Stream candidates one at a time.
next_candidate = preprocessor.next()
if next_candidate is not None:
    print(next_candidate.source_frame_index, next_candidate.candidate_index)

```

Use `get_current_hand_candidates()` and `next()` as separate access patterns. The batch API
returns the current source frame's candidate list, while `next()` drains the internal FIFO
candidate queue.

## Public Types

### `PreprocessorConfig`

- `input_mode: Literal["webcam", "local_video"]`
- `video_path: str | None = None`
- `frame_size: tuple[int, int] = (640, 480)`
- `threshold_profile: str = "default"`
- `candidate_frame_size_px: int = 224`
- `candidate_buffer_size: int = 32`

### `ResultStatus`

Required enum values:

- `ok`
- `no_hand`
- `error`

### `HandCandidateFrame`

- `frame_rgb: np.ndarray`
- `timestamp_ms: int`
- `source_frame_index: int`
- `source_id: str`
- `candidate_index: int`
- `bbox_xyxy_px: tuple[int, int, int, int]`

### `HandFrameResult`

- `status: ResultStatus`
- `timestamp_ms: int`
- `candidates: list[HandCandidateFrame]`
- `error_message: str | None = None`

## Runtime Behavior

- Candidate crops are sorted largest-first within each source frame.
- Each crop is square padded with black pixels as needed, then resized to
  `candidate_frame_size_px x candidate_frame_size_px`.
- The internal candidate buffer is FIFO and overwrites the oldest items when full.
- `Preprocessor.next()` returns `None` only after the source is exhausted and the queue is empty.
