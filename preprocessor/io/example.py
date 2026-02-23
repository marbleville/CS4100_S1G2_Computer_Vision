from preprocessor.config.types import PreprocessorConfig
from preprocessor.io.factory import build_frame_source
from preprocessor.io.frame_packet_writer import DiskFramePacketWriter

config = PreprocessorConfig(
    buffer_size=8,
    async_process=False,
    input_mode="webcam",
    video_path="./data/test/test.avi",
)

source = build_frame_source(config)

writer = DiskFramePacketWriter()

packet = source.read()

if packet:
    writer.write_frame_packet(packet, './data/test/output', 'testFrame')
