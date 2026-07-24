from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from main_prefect_dag import dubbing_flow

if __name__ == "__main__":
    (
        dubbing_flow
        .from_source(
            source=str(Path(__file__).parent.parent),
            entrypoint="main_prefect_dag.py:dubbing_flow",
        )
        .deploy(
            name="video-dubbing-deployment",
            work_pool_name="video-dubbing-pool",
        )
    )