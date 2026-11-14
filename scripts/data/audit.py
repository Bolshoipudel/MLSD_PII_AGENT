import json
from collections import defaultdict

from .normalize import normalize_sources
from .prepare import json_bytes
from .sources import DEFAULT_SOURCE_DIR, ROOT
from .split import group_records, masked_template, normalized_text


def audit_overlap(rows):
    result = {}
    for kind, key in (
        ("normalized_text", lambda r: normalized_text(r["text"])),
        ("masked_template", masked_template),
    ):
        signatures = defaultdict(set)
        for row in rows:
            signatures[key(row)].add(row["source"])
        result[kind + "_cross_source_groups"] = sum(
            len(sources) > 1 for sources in signatures.values()
        )
    groups, grouping = group_records(rows)
    sources = defaultdict(set)
    for row in rows:
        sources[groups[row["id"]]].add(row["source"])
    result["near_template_cross_source_groups"] = sum(len(items) > 1 for items in sources.values())
    result["method"] = grouping["algorithm"]
    result["semantic_independence_verified"] = False
    result["interpretation"] = (
        "Source overlap audit only. Zero heuristic matches does not prove semantic independence. "
        "No evaluation examples or predictions are exposed to optimization methods."
    )
    return result


def main():
    rows, _ = normalize_sources(DEFAULT_SOURCE_DIR, scope="all")
    result = audit_overlap(rows)
    (ROOT / "eda/overlap_audit.json").write_bytes(json_bytes(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
