import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from main_prefect_dag import dubbing_flow

if __name__ == "__main__":
    dubbing_flow.serve(
        name="video-dubbing-deployment",
        limit=4,
    )