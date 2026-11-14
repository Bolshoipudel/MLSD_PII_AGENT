import hashlib
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict

SPLIT_NAMES = ("opt_train", "golden")
ALGORITHM = "grouped_masked_char5_jaccard_0.85_balanced_v1"
SIMILARITY_THRESHOLD = 0.85


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def masked_template(record: dict) -> str:
    text = record["text"]
    for entity in sorted(record["entities"], key=lambda e: e["start"], reverse=True):
        text = text[: entity["start"]] + "<" + entity["type"] + ">" + text[entity["end"] :]
    return normalized_text(text)


def group_records(records: list[dict]) -> tuple[dict[str, str], dict]:
    rows = sorted(records, key=lambda row: row["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Record IDs must be unique")
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    exact, templates, inverted = {}, {}, defaultdict(list)
    gram_sets = []
    edges = Counter()
    for index, row in enumerate(rows):
        for signature, seen, kind in [
            (normalized_text(row["text"]), exact, "normalized_text"),
            (masked_template(row), templates, "masked_template"),
        ]:
            if signature in seen:
                union(index, seen[signature])
                edges[kind] += 1
            else:
                seen[signature] = index
        template = masked_template(row)
        grams = {template[i : i + 5] for i in range(len(template) - 4)}
        gram_sets.append(grams)
        if len(template) < 30:
            continue
        overlaps = Counter(prior for gram in sorted(grams) for prior in inverted[gram])
        for prior, intersection in sorted(overlaps.items()):
            if find(index) == find(prior):
                continue
            similarity = intersection / (len(grams) + len(gram_sets[prior]) - intersection)
            if similarity >= SIMILARITY_THRESHOLD:
                union(index, prior)
                edges["near_template"] += 1
        for gram in sorted(grams):
            inverted[gram].append(index)
    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(index)].append(row["id"])
    mapping = {}
    for ids in components.values():
        group_id = "g_" + hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:20]
        mapping.update(dict.fromkeys(ids, group_id))
    sizes = [len(ids) for ids in components.values()]
    return mapping, {
        "algorithm": ALGORITHM,
        "groups": len(sizes),
        "multi_record_groups": sum(size > 1 for size in sizes),
        "records_in_multi_record_groups": sum(size for size in sizes if size > 1),
        "largest_group": max(sizes, default=0),
        "matched_edges": dict(sorted(edges.items())),
        "semantic_independence_verified": False,
    }


def split_records(records: list[dict], seed: int = 42, train_ratio: float = 0.7) -> dict:
    if not math.isfinite(train_ratio) or not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")
    if any(row["source"] != "redmadrobot" for row in records):
        raise ValueError("Only redmadrobot development data may enter opt_train/golden")
    groups, _ = group_records(records)
    members = defaultdict(list)
    for row in sorted(records, key=lambda row: row["id"]):
        members[groups[row["id"]]].append(row)

    def features(rows):
        result = Counter({"records": len(rows)})
        for row in rows:
            result["positive" if row["entities"] else "negative"] += 1
            result.update("label:" + label for label in {e["type"] for e in row["entities"]})
        return result

    totals = features(records)
    targets = {
        "opt_train": {key: count * train_ratio for key, count in totals.items()},
        "golden": {key: count * (1 - train_ratio) for key, count in totals.items()},
    }
    rng = random.Random(seed)
    group_order = sorted(members)
    rng.shuffle(group_order)
    ranks = {group: index for index, group in enumerate(group_order)}
    group_features = {group: features(rows) for group, rows in members.items()}
    group_order.sort(
        key=lambda group: (
            min(
                (totals[key] for key in group_features[group] if key.startswith("label:")),
                default=len(records) + 1,
            ),
            -len(members[group]),
            ranks[group],
        )
    )
    counts = {name: Counter() for name in SPLIT_NAMES}
    group_split = {}
    for group in group_order:
        feature_counts = group_features[group]

        def delta(name, feature_counts=feature_counts):
            return sum(
                (4 if key == "records" else 1)
                * (
                    (counts[name][key] + value - targets[name][key]) ** 2
                    - (counts[name][key] - targets[name][key]) ** 2
                )
                / max(totals[key], 1) ** 2
                for key, value in feature_counts.items()
            )

        selected = min(SPLIT_NAMES, key=lambda name: (delta(name), SPLIT_NAMES.index(name)))
        counts[selected].update(feature_counts)
        group_split[group] = selected
    for _ in range(20):
        moved = False
        for group in group_order:
            origin = group_split[group]
            destination = next(name for name in SPLIT_NAMES if name != origin)
            change = 0.0
            for key, value in group_features[group].items():
                weight = 4 if key == "records" else 1
                for name, sign in ((origin, -1), (destination, 1)):
                    before = counts[name][key] - targets[name][key]
                    change += (
                        weight
                        * ((before + sign * value) ** 2 - before**2)
                        / max(totals[key], 1) ** 2
                    )
            if change < -1e-12:
                counts[origin].subtract(group_features[group])
                counts[destination].update(group_features[group])
                group_split[group] = destination
                moved = True
        if not moved:
            break
    assigned = {name: [] for name in SPLIT_NAMES}
    for group, name in group_split.items():
        assigned[name].extend(dict(row, split=name, group_id=group) for row in members[group])
    return {name: sorted(rows, key=lambda row: row["id"]) for name, rows in assigned.items()}


def validate_splits(parts: dict) -> dict:
    if set(parts) != set(SPLIT_NAMES) or any(not parts[name] for name in SPLIT_NAMES):
        raise ValueError("Both nonempty opt_train and golden are required")
    rows = [row for name in SPLIT_NAMES for row in parts[name]]
    groups, audit = group_records(rows)
    seen_ids, seen_groups = set(), {}
    spans = 0
    for name in SPLIT_NAMES:
        for row in parts[name]:
            if row["source"] != "redmadrobot":
                raise ValueError("Evaluation data cannot enter development splits")
            if row["id"] in seen_ids:
                raise ValueError(f"Repeated record ID: {row['id']}")
            seen_ids.add(row["id"])
            if row.get("split") != name or row.get("group_id") != groups[row["id"]]:
                raise ValueError(f"Invalid split/group metadata: {row['id']}")
            group = groups[row["id"]]
            if group in seen_groups and seen_groups[group] != name:
                raise ValueError(f"Cross-split text/template group: {row['id']}")
            seen_groups[group] = name
            if row["has_entities"] != bool(row["entities"]):
                raise ValueError("Inconsistent has_entities")
            previous_end = 0
            for entity in sorted(row["entities"], key=lambda e: e["start"]):
                start, end = entity["start"], entity["end"]
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(row["text"])
                    or start < previous_end
                    or row["text"][start:end] != entity["text"]
                    or entity["type"] not in row["label_scope"]
                ):
                    raise ValueError(f"Invalid gold annotation: {row['id']}")
                previous_end = end
                spans += 1
    return {
        "records": len(rows),
        "entities": spans,
        "sizes": {name: len(parts[name]) for name in SPLIT_NAMES},
        "id_overlap": 0,
        "normalized_text_overlap": 0,
        "template_group_overlap": 0,
        "grouping": audit,
    }
