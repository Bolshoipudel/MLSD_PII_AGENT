from pathlib import Path

HIVETRACE_REVISION = "cd6a18ace16daf23e79247ccf1e2b245d4054654"
REDMADROBOT_REVISION = "f77ea831274daf980cc45c61a93c226be9d978d6"

SOURCES = {
    "hivetrace/domain.parquet": {
        "revision": HIVETRACE_REVISION,
        "url": f"https://huggingface.co/datasets/hivetrace/pii-bench/resolve/{HIVETRACE_REVISION}/data/domain-00000-of-00001.parquet",
        "sha256": "7ef3574273c0fe3a987981e38e32cd2764031a82766ca5b235bfccd1f27058eb",
    },
    "hivetrace/entity.parquet": {
        "revision": HIVETRACE_REVISION,
        "url": f"https://huggingface.co/datasets/hivetrace/pii-bench/resolve/{HIVETRACE_REVISION}/data/entity-00000-of-00001.parquet",
        "sha256": "6e0c77d566c2f04e7213917b26f1038336e28fc006551fac7a7f44f7627d5f96",
    },
    "redmadrobot/test.csv": {
        "revision": REDMADROBOT_REVISION,
        "url": f"https://huggingface.co/datasets/redmadrobot-rnd/pii_benchmark/resolve/{REDMADROBOT_REVISION}/test.csv",
        "sha256": "6bf544a380a3ee5bec94b946124bea3afaecce49e734679ad0f0c0e7c12977bb",
    },
}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "data/raw/pii_benchmarks"


def selected_sources(scope: str) -> dict:
    if scope not in {"development", "evaluation", "all"}:
        raise ValueError(f"Unknown source scope: {scope}")
    return {
        name: metadata
        for name, metadata in SOURCES.items()
        if scope == "all"
        or (
            name.startswith("redmadrobot/")
            if scope == "development"
            else name.startswith("hivetrace/")
        )
    }
