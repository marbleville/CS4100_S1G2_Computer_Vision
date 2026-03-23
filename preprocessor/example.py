from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from preprocessor import init_preprocessor
from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.factory import build_frame_source
from preprocessor.pipeline.processor import PreprocessingPipeline, pipeline_result_to_hand_result
from preprocessor.types import HandFrameResult
from preprocessor.visualization import render_pipeline_result

SAMPLE_FILES = ["nothing.mov", "open_hand.mov", "thumbs_up.mov", "two_hands.mov"]
MAX_STREAM_CANDIDATES = 3


def _save_candidate_crops(result: HandFrameResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in result.candidates:
        output_path = output_dir / (
            f"candidate_{candidate.source_frame_index:06d}_{candidate.candidate_index:02d}.png"
        )
        Image.fromarray(candidate.frame_rgb, mode="RGB").save(output_path)


def _config_for(file_name: str) -> PreprocessorConfig:
    config = PreprocessorConfig(
        input_mode="local_video",
        video_path=f"./data/test/{file_name}",
    )
    return config


def demo_batch_api(file_name: str) -> None:
    config = _config_for(file_name)
    source = build_frame_source(config)
    pipeline = PreprocessingPipeline(config)
    packet = source.read()
    if packet is None:
        return

    output_dir = Path(f"artifacts/preprocessor_vis/{file_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    result = pipeline.process(packet)
    hand_result = pipeline_result_to_hand_result(result)
    end_time = time.perf_counter()

    output_path = output_dir / f"frame_{packet.frame_index:06d}.png"
    render_pipeline_result(packet, result, output_path=output_path)
    _save_candidate_crops(hand_result, output_dir / "batch_candidates")

    elapsed_time = end_time - start_time
    print(f"[batch] {file_name}: {len(hand_result.candidates)} candidates in {elapsed_time:.4f}s")


def demo_stream_api(file_name: str) -> None:
    preprocessor = init_preprocessor(_config_for(file_name))
    output_dir = Path(f"artifacts/preprocessor_vis/{file_name}/stream_candidates")
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    candidate = preprocessor.next()
    while candidate is not None and count < MAX_STREAM_CANDIDATES:
        output_path = output_dir / (
            f"candidate_{candidate.source_frame_index:06d}_{candidate.candidate_index:02d}.png"
        )
        Image.fromarray(candidate.frame_rgb, mode="RGB").save(output_path)
        print(
            "[stream] "
            f"{file_name}: frame={candidate.source_frame_index} "
            f"candidate={candidate.candidate_index} "
            f"shape={candidate.frame_rgb.shape}"
        )
        count += 1
        candidate = preprocessor.next()


for sample_file in SAMPLE_FILES:
    demo_batch_api(sample_file)
    demo_stream_api(sample_file)
