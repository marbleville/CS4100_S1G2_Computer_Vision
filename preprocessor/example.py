from pathlib import Path
import time

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.factory import build_frame_source
from preprocessor.pipeline.processor import PreprocessingPipeline
from preprocessor.visualization import render_pipeline_result

config = PreprocessorConfig(
    buffer_size=8,
    async_process=False,
    input_mode="webcam",
    video_path="./data/test/test.mov",
)

source = build_frame_source(config)
pipeline = PreprocessingPipeline(config)
output_dir = Path("artifacts/preprocessor_vis")
output_dir.mkdir(parents=True, exist_ok=True)

packet = source.read()

while packet:
    start_time = time.perf_counter()

    result = pipeline.process(packet)

    end_time = time.perf_counter()

    output_path = output_dir / f"frame_{packet.frame_index:06d}.png"
    render_pipeline_result(packet, result, output_path=output_path)

    elapsed_time = end_time - start_time

    print(f"Elapsed time (sec): {elapsed_time}")
    packet = None  # source.read()
