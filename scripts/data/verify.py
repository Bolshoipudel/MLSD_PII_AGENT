import argparse
import hashlib
import json

from .sources import DEFAULT_SOURCE_DIR, ROOT


def verify(scope="development"):
    metadata = ROOT / "eda"
    manifest = json.loads((metadata / f"{scope}_manifest.json").read_text(encoding="utf-8"))
    files = []
    for name, item in manifest["sources"].items():
        files.append((DEFAULT_SOURCE_DIR / name, item["sha256"]))
    for name, item in manifest["outputs"].items():
        files.append((ROOT / "data/processed" / scope / name, item["sha256"]))
    files.extend((metadata / name, digest) for name, digest in manifest["metadata_sha256"].items())
    files.extend(
        (ROOT / "data/splits" / name, digest)
        for name, digest in manifest["split_index_sha256"].items()
    )
    files.append((ROOT / "uv.lock", manifest["uv_lock_sha256"]))
    for path, expected in files:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {path.relative_to(ROOT)}")
    for name, expected in manifest["code_sha256"].items():
        path = ROOT / "scripts/data" / name
        if hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest() != expected:
            raise ValueError(f"Preparation code changed: {path.relative_to(ROOT)}")
    print(f"{scope}: verified {len(files)} files and {len(manifest['code_sha256'])} code hashes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["development", "evaluation"], default="development")
    verify(parser.parse_args().scope)


if __name__ == "__main__":
    main()
