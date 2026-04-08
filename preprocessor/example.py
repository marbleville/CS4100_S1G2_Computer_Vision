from __future__ import annotations

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessor import init_preprocessor
from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.factory import build_frame_source
from preprocessor.io.types import FramePacket
from preprocessor.pipeline.processor import (
    PreprocessingPipeline,
    pipeline_result_to_hand_result,
)
from preprocessor.types import HandFrameResult
from preprocessor.visualization import render_pipeline_result

VIDEO_SAMPLE_FILES = ["nothing.mov", "open_hand.mov", "thumbs_up.mov", "two_hands.mov"]
IMAGE_SAMPLE_DIR = Path("./data/test/sample_frames")
IMAGE_GLOB = "*.jpg"
MAX_STREAM_CANDIDATES = 3
DEFAULT_WEBCAM_BATCH_FRAMES = 10
OUTPUT_ROOT = Path("artifacts/preprocessor_vis")


def _save_candidate_crops(result: HandFrameResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in result.candidates:
        output_path = output_dir / (
            f"candidate_{candidate.source_frame_index:06d}_{candidate.candidate_index:02d}.png"
        )
        Image.fromarray(candidate.frame_rgb, mode="RGB").save(output_path)


def _video_config_for(file_name: str) -> PreprocessorConfig:
    return PreprocessorConfig(
        input_mode="local_video",
        video_path=f"./data/test/{file_name}",
    )


def _webcam_config() -> PreprocessorConfig:
    return PreprocessorConfig(input_mode="webcam")


def demo_video_batch_api(file_name: str) -> None:
    config = _video_config_for(file_name)
    source = build_frame_source(config)
    pipeline = PreprocessingPipeline(config)
    packet = source.read()
    if packet is None:
        return

    output_dir = OUTPUT_ROOT / file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    result = pipeline.process(packet)
    hand_result = pipeline_result_to_hand_result(result)
    end_time = time.perf_counter()

    output_path = output_dir / f"frame_{packet.frame_index:06d}.png"
    render_pipeline_result(packet, result, output_path=output_path)
    _save_candidate_crops(hand_result, output_dir / "batch_candidates")

    elapsed_time = end_time - start_time
    print(f"[video batch] {file_name}: {len(hand_result.candidates)} candidates in {elapsed_time:.4f}s")


def demo_video_stream_api(file_name: str) -> None:
    preprocessor = init_preprocessor(_video_config_for(file_name))
    output_dir = OUTPUT_ROOT / file_name / "stream_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    candidate = preprocessor.next()
    while candidate is not None and count < MAX_STREAM_CANDIDATES:
        output_path = output_dir / (
            f"candidate_{candidate.source_frame_index:06d}_{candidate.candidate_index:02d}.png"
        )
        Image.fromarray(candidate.frame_rgb, mode="RGB").save(output_path)
        print(
            "[video stream] "
            f"{file_name}: frame={candidate.source_frame_index} "
            f"candidate={candidate.candidate_index} "
            f"shape={candidate.frame_rgb.shape}"
        )
        count += 1
        candidate = preprocessor.next()


def demo_webcam_stream_api(max_candidates: int = MAX_STREAM_CANDIDATES) -> None:
    preprocessor = init_preprocessor(_webcam_config())

    count = 0
    candidate = preprocessor.next()
    while candidate is not None and count < max_candidates:
        print(
            "[webcam stream] "
            f"frame={candidate.source_frame_index} "
            f"candidate={candidate.candidate_index} "
            f"shape={candidate.frame_rgb.shape}"
        )
        count += 1
        candidate = preprocessor.next()


def demo_webcam_batch_api(frame_count: int = DEFAULT_WEBCAM_BATCH_FRAMES) -> None:
    config = _webcam_config()
    source = build_frame_source(config)
    pipeline = PreprocessingPipeline(config)
    output_dir = OUTPUT_ROOT / "webcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_frames = 0
    total_candidates = 0

    try:
        while processed_frames < frame_count:
            packet = source.read()
            if packet is None:
                break

            start_time = time.perf_counter()
            result = pipeline.process(packet)
            hand_result = pipeline_result_to_hand_result(result)
            end_time = time.perf_counter()

            output_path = output_dir / f"frame_{packet.frame_index:06d}.png"
            render_pipeline_result(packet, result, output_path=output_path)
            _save_candidate_crops(hand_result, output_dir / "batch_candidates")

            elapsed_time = end_time - start_time
            total_candidates += len(hand_result.candidates)
            processed_frames += 1
            print(
                "[webcam batch] "
                f"frame={packet.frame_index} "
                f"candidates={len(hand_result.candidates)} "
                f"time={elapsed_time:.4f}s"
            )
    finally:
        source.close()

    print(
        "[webcam batch] "
        f"processed_frames={processed_frames} "
        f"saved_candidates={total_candidates} "
        f"output_dir={output_dir}"
    )


def demo_image_batch_api(image_path: Path, frame_index: int) -> None:
    frame_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    packet = FramePacket(
        frame_index=frame_index,
        timestamp_ms=frame_index,
        frame_rgb=frame_rgb,
        source_id=image_path.name,
    )

    pipeline = PreprocessingPipeline(_webcam_config())
    output_dir = OUTPUT_ROOT / "sample_frames" / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    result = pipeline.process(packet)
    hand_result = pipeline_result_to_hand_result(result)
    end_time = time.perf_counter()

    render_pipeline_result(packet, result, output_path=output_dir / "frame_000000.png")
    _save_candidate_crops(hand_result, output_dir / "batch_candidates")

    elapsed_time = end_time - start_time
    print(
        "[image batch] "
        f"{image_path.name}: candidates={len(hand_result.candidates)} "
        f"light_mode={result.debug.get('active_light_mode')} "
        f"luma={float(result.debug.get('frame_median_luma', 0.0)):.4f} "
        f"time={elapsed_time:.4f}s"
    )


def run_video_demos() -> None:
    for sample_file in VIDEO_SAMPLE_FILES:
        demo_video_batch_api(sample_file)
        demo_video_stream_api(sample_file)


def run_image_demos(image_dir: Path) -> None:
    image_paths = sorted(image_dir.glob(IMAGE_GLOB))
    for frame_index, image_path in enumerate(image_paths):
        demo_image_batch_api(image_path, frame_index=frame_index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run preprocessor demos on bundled video or image fixtures.",
    )
    parser.add_argument(
        "--mode",
        choices=("videos", "images", "webcam", "all"),
        default="all",
        help="Which demos to run. `webcam` uses the onboard camera.",
    )
    parser.add_argument(
        "--image-dir",
        default=str(IMAGE_SAMPLE_DIR),
        help="Directory containing extracted sample-frame images.",
    )
    parser.add_argument(
        "--webcam-frames",
        type=int,
        default=DEFAULT_WEBCAM_BATCH_FRAMES,
        help="Number of webcam frames to process and save in batch mode.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode in {"videos", "all"}:
        run_video_demos()
    if args.mode in {"images", "all"}:
        run_image_demos(Path(args.image_dir))
    if args.mode == "webcam":
        demo_webcam_batch_api(frame_count=args.webcam_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
