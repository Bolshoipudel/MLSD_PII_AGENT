import argparse
import hashlib
import json
import platform
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from .normalize import normalize_sources
from .sources import DEFAULT_SOURCE_DIR, ROOT, selected_sources
from .split import ALGORITHM, split_records, validate_splits


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def jsonl_bytes(rows) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def index_bytes(index) -> bytes:
    lines = ["{"]
    for key in ("scope", "seed", "algorithm"):
        lines.append(f'  "{key}": {json.dumps(index[key])},')
    lines.append('  "parts": {')
    parts = list(index["parts"].items())
    for part_index, (name, rows) in enumerate(parts):
        lines.append(f'    "{name}": [')
        for row_index, row in enumerate(rows):
            suffix = "," if row_index < len(rows) - 1 else ""
            lines.append("      " + json.dumps(row, sort_keys=True, separators=(",", ":")) + suffix)
        lines.append("    ]" + ("," if part_index < len(parts) - 1 else ""))
    return ("\n".join([*lines, "  }", "}"]) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summarize(rows: list[dict]) -> dict:
    import numpy as np

    lengths = [len(row["text"]) for row in rows]
    edges = [0, 25, 50, 100, 200, 400, 800, 1600, 3200]
    counts, _ = np.histogram(lengths, bins=edges)
    labels = Counter(e["type"] for row in rows for e in row["entities"])
    positive_labels = Counter(
        label for row in rows for label in {e["type"] for e in row["entities"]}
    )
    domains = Counter(row.get("domain") or "unspecified" for row in rows)
    scopes = sorted({label for row in rows for label in row["label_scope"]})
    return {
        "records": len(rows),
        "spans": sum(labels.values()),
        "positive_records": sum(bool(row["entities"]) for row in rows),
        "negative_records": sum(not row["entities"] for row in rows),
        "negative_records_with_digits": sum(
            not row["entities"] and bool(re.search(r"\d", row["text"])) for row in rows
        ),
        "records_with_multiple_entities": sum(len(row["entities"]) > 1 for row in rows),
        "max_entities_per_record": max(len(row["entities"]) for row in rows),
        "label_spans": dict(sorted(labels.items())),
        "label_positive_records": dict(sorted(positive_labels.items())),
        "label_scope": scopes,
        "domains": dict(sorted(domains.items())),
        "length_chars": {
            "min": min(lengths),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "max": max(lengths),
            "histogram_edges": edges,
            "histogram_counts": counts.tolist(),
        },
    }


def write_payloads(directory: Path, payloads: dict[str, bytes], force: bool) -> None:
    conflicts = [
        name
        for name, data in payloads.items()
        if (directory / name).exists() and (directory / name).read_bytes() != data
    ]
    if conflicts and not force:
        raise FileExistsError(f"Different generated files in {directory}; use --force: {conflicts}")
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        (directory / name).write_bytes(data)


def prepare(
    source_dir=DEFAULT_SOURCE_DIR,
    output_dir=None,
    metadata_dir=None,
    split_dir=None,
    scope="development",
    seed=42,
    force=False,
) -> dict:
    if scope not in {"development", "evaluation"}:
        raise ValueError("Preparation requires development or evaluation; merging is disabled")
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir or ROOT / "data/processed" / scope).resolve()
    metadata_dir = Path(metadata_dir or ROOT / "eda").resolve()
    split_dir = Path(split_dir or ROOT / "data/splits").resolve()
    for destination in (output_dir, metadata_dir, split_dir):
        if destination == source_dir or source_dir in destination.parents:
            raise ValueError("Outputs must be separate from the immutable source directory")
    rows, audit = normalize_sources(source_dir, scope=scope)
    if scope == "development":
        parts = split_records(rows, seed=seed)
        checks = validate_splits(parts)
        algorithm = ALGORITHM
    else:
        parts = {
            name: [
                dict(row, split="evaluation_" + name)
                for row in rows
                if row["original_split"] == name
            ]
            for name in ("domain", "entity")
        }
        checks = {
            "records": len(rows),
            "entities": audit["spans"],
            "sizes": {name: len(part) for name, part in parts.items()},
            "original_evaluation_slices_preserved": True,
        }
        algorithm = "original_pii_bench_slices_v1"
    payloads = {name + ".jsonl": jsonl_bytes(part) for name, part in parts.items()}
    summary = {
        "scope": scope,
        "all": summarize(rows),
        "parts": {name: summarize(part) for name, part in parts.items()},
        "checks": checks,
    }
    indices = {
        scope + ".json": index_bytes(
            {
                "scope": scope,
                "seed": seed if scope == "development" else None,
                "algorithm": algorithm,
                "parts": {
                    name: [
                        dict(
                            id=row["id"],
                            text_sha256=digest(row["text"].encode()),
                            **({"group_id": row["group_id"]} if "group_id" in row else {}),
                        )
                        for row in sorted(part, key=lambda r: r["id"])
                    ]
                    for name, part in parts.items()
                },
            }
        ),
    }
    metadata = {
        scope + "_summary.json": json_bytes(summary),
        scope + "_corrections.json": json_bytes(audit),
    }
    code_files = sorted(Path(__file__).parent.glob("*.py"))
    manifest = {
        "schema_version": 2,
        "scope": scope,
        "algorithm": algorithm,
        "seed": seed if scope == "development" else None,
        "target_ratio": [0.7, 0.3] if scope == "development" else None,
        "checks": checks,
        "sources": selected_sources(scope),
        "code_hash_policy": "SHA-256 of UTF-8 Python source with LF newlines",
        "code_sha256": {p.name: digest(p.read_text(encoding="utf-8").encode()) for p in code_files},
        "environment": {
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("numpy", "pyarrow")},
        },
        "uv_lock_sha256": digest((ROOT / "uv.lock").read_bytes()),
        "outputs": {
            name: {"sha256": digest(data), "bytes": len(data)}
            for name, data in sorted(payloads.items())
        },
        "metadata_sha256": {name: digest(data) for name, data in sorted(metadata.items())},
        "split_index_sha256": {name: digest(data) for name, data in indices.items()},
    }
    for directory, files in (
        (output_dir, payloads),
        (split_dir, indices),
        (metadata_dir, metadata | {scope + "_manifest.json": json_bytes(manifest)}),
    ):
        if not force:
            for name, data in files.items():
                if (directory / name).exists() and (directory / name).read_bytes() != data:
                    raise FileExistsError(f"Use --force or a new directory: {directory / name}")
    write_payloads(output_dir, payloads, force)
    write_payloads(split_dir, indices, force)
    write_payloads(metadata_dir, metadata | {scope + "_manifest.json": json_bytes(manifest)}, force)
    print(json.dumps({"scope": scope, **checks}, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["development", "evaluation"], default="development")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
