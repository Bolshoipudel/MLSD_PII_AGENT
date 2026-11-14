from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from .sources import DEFAULT_SOURCE_DIR, selected_sources


def download_sources(
    output_dir: Path = DEFAULT_SOURCE_DIR, timeout: int = 60, scope: str = "development"
) -> list[Path]:
    output_dir = Path(output_dir)
    completed = []
    for relative, source in selected_sources(scope).items():
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and _sha256(destination) == source["sha256"]:
            completed.append(destination)
            continue
        request = Request(source["url"], headers={"User-Agent": "benchmark-merge/1.0"})
        temporary = None
        try:
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
                temporary = Path(temp.name)
                with urlopen(request, timeout=timeout) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        digest.update(chunk)
                        temp.write(chunk)
            actual = digest.hexdigest()
            if actual != source["sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch for {relative}: expected {source['sha256']}, got {actual}"
                )
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        completed.append(destination)
    return completed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned benchmark sources.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--scope", choices=["development", "evaluation", "all"], default="development"
    )
    args = parser.parse_args()
    paths = download_sources(args.output_dir, args.timeout, args.scope)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
