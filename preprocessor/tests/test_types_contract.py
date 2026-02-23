from pathlib import Path

import preprocessor
from preprocessor.config.types import PreprocessorConfig
from preprocessor.types import HandFrameResult, MotionWindowResult, ResultStatus


def test_preprocessor_config_instantiation() -> None:
    config = PreprocessorConfig(
        buffer_size=8,
        async_process=True,
        input_mode="webcam",
    )
    assert config.buffer_size == 8
    assert config.input_mode == "webcam"
    assert config.frame_size == (640, 480)


def test_result_status_enum_coverage() -> None:
    expected = {
        "ok",
        "no_hand",
        "low_confidence",
        "insufficient_history",
        "error",
    }
    actual = {status.value for status in ResultStatus}
    assert actual == expected


def test_hand_result_supports_no_hand_shape() -> None:
    result = HandFrameResult(
        status=ResultStatus.NO_HAND,
        timestamp_ms=123456,
        bbox_xyxy_norm=None,
        centroid_xy_norm=None,
        contour_points_norm=[],
        quality_score=None,
    )
    assert result.status == ResultStatus.NO_HAND
    assert result.bbox_xyxy_norm is None
    assert result.centroid_xy_norm is None


def test_motion_result_supports_insufficient_history_shape() -> None:
    result = MotionWindowResult(
        status=ResultStatus.INSUFFICIENT_HISTORY,
        timestamp_ms=123456,
        window_size=0,
        trajectory_xy_norm=[],
        delta_x_px=0.0,
        delta_y_px=0.0,
        path_length_px=0.0,
        motion_confidence=None,
    )
    assert result.status == ResultStatus.INSUFFICIENT_HISTORY
    assert result.window_size == len(result.trajectory_xy_norm)


def test_contract_doc_and_public_exports_stay_aligned() -> None:
    contract_path = Path(__file__).resolve().parents[1] / "contract.md"
    contract_text = contract_path.read_text(encoding="utf-8")

    documented_names = [
        "PreprocessorConfig",
        "HandFrameResult",
        "MotionWindowResult",
        "ResultStatus",
        "init_preprocessor",
        "Preprocessor",
    ]

    for name in documented_names:
        assert name in contract_text
        assert hasattr(preprocessor, name)
