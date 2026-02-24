# Preprocessor Contract (v1)

## Purpose

`preprocessor` is Module B's vision front-end contract. It accepts raw frames from a webcam
or directory source and returns a set of plausible regions for a model to analyze for gestures.

## How to use

```python
from preprocessor import init_preprocessor
from preprocessor.config.types import PreprocessorConfig

# only local video supported currently
config = PreprocessorConfig(
    input_mode='local_video',
    video_path='path/to/video_file',
)

preprocessor = init_preprocessor(config)

result = preprocessor.get_current_hand_candidates()

# process result further...

```

## Public Types

### `PreprocessorConfig`

- `input_mode: Literal["webcam", "local_video"]`
- `video_path: str | None = None`
- `frame_size: tuple[int, int] = (640, 480)`
- `threshold_profile: str = "default"`

### `ResultStatus`

Required enum values:

- `ok`
- `no_hand`
- `error`

### `HandFrameResult`

- `status: ResultStatus`
- `timestamp_ms: int`
- `candidates_bbox_px: list[tuple[int, int, int, int]]`
- `error_message: str | None = None`
