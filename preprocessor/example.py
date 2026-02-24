from pathlib import Path
import time

from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.factory import build_frame_source
from preprocessor.pipeline.processor import PreprocessingPipeline, pipeline_result_to_hand_result
from preprocessor.visualization import render_pipeline_result

for file_name in ['nothing.mov', 'open_hand.mov', 'thumbs_up.mov', 'two_hands.mov']:
    config = PreprocessorConfig(
        input_mode='local_video',
        video_path=f'./data/test/{file_name}',
    )

    source = build_frame_source(config)
    pipeline = PreprocessingPipeline(config)
    output_dir = Path(f'artifacts/preprocessor_vis/{file_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    packet = source.read()

    numFrames = 0
    while packet and numFrames < 1:
        start_time = time.perf_counter()

        result = pipeline.process(packet)
        hand_result = pipeline_result_to_hand_result(result)

        end_time = time.perf_counter()

        output_path = output_dir / f"frame_{packet.frame_index:06d}.png"
        render_pipeline_result(packet, result, output_path=output_path)

        elapsed_time = end_time - start_time
        print(f"Elapsed time (sec): {elapsed_time}")
        print(result)

        packet = source.read()
        numFrames += 1
