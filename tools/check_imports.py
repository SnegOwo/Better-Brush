import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.hardware import HardwareDetector, SAM2Profiler, ContentProfile
    print("OK: hardware imports")
    from src.models.checkpoint_manager import CheckpointManager
    from src.models.sam2_manager import SAM2Manager
    print("OK: models imports")
    from src.processing.video_reader import VideoReader
    from src.processing.frame_buffer import FrameBuffer
    from src.processing.inference_worker import InferenceWorker, PropagationWorker
    print("OK: processing imports")
    from src.export.exporter import Exporter
    print("OK: export imports")
    from src.ui.video_canvas import VideoCanvas
    from src.ui.timeline import Timeline
    from src.ui.main_window import MainWindow
    print("OK: ui imports")
    print("\nAll imports successful.")
except Exception as e:
    print(f"IMPORT FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
