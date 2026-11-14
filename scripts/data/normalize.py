from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from .sources import HIVETRACE_REVISION, REDMADROBOT_REVISION, SOURCES, selected_sources

ALIASES = {
    "PHONE_NUMBER": "PHONE",
    "BANK_CARD_NUMBER": "CREDIT_CARD",
    "PASSPORT_NUMBER": "PASSPORT",
}
HIVE_LABELS = sorted(
    [
        "NAME",
        "ADDRESS",
        "PHONE_NUMBER",
        "EMAIL",
        "BANK_CARD_NUMBER",
        "CVC",
        "INN",
        "KPP",
        "OGRN",
        "OGRNIP",
        "SNILS",
        "PASSPORT_NUMBER",
        "TOKEN",
    ]
)
RED_LABELS = sorted(
    [
        "FIRST_NAME",
        "LAST_NAME",
        "MIDDLE_NAME",
        "COUNTRY",
        "REGION",
        "DISTRICT",
        "CITY",
        "STREET",
        "HOUSE",
        "EMAIL",
        "PHONE",
        "URL",
        "IP_ADDRESS",
        "PASSPORT",
        "INN",
        "SNILS",
        "OMS",
        "CREDIT_CARD",
        "DRIVER_LICENSE",
        "MILITARY_ID",
        "BIRTH_CERTIFICATE",
    ]
)
EXPECTED_REVISIONS = {
    "hivetrace": HIVETRACE_REVISION,
    "redmadrobot": REDMADROBOT_REVISION,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _base(source: str, split: str, original_id: str, text: str, labels: list[str]) -> dict:
    _require(isinstance(text, str) and bool(text.strip()), f"{source}/{original_id}: empty text")
    return {
        "id": f"{source}/{split}/{original_id}",
        "text": text,
        "source": source,
        "source_revision": EXPECTED_REVISIONS[source],
        "original_split": split,
        "original_id": original_id,
        "domain": None,
        "label_scope": sorted({ALIASES.get(label, label) for label in labels}),
        "native_label_scope": labels.copy(),
    }


def _entity(text: str, start: int, end: int, native_type: str) -> dict:
    _require(type(start) is int and type(end) is int, "Offsets must be integers")
    _require(0 <= start < end <= len(text), f"Invalid span {start}:{end}")
    return {
        "start": start,
        "end": end,
        "type": ALIASES.get(native_type, native_type),
        "native_type": native_type,
        "text": text[start:end],
    }


def _red_entities(
    text: str, tokens: list[str], tags: list[str], row_id: str, corrections: list[dict]
) -> list[dict]:
    _require(isinstance(tokens, list) and isinstance(tags, list), f"{row_id}: expected arrays")
    _require(len(tokens) == len(tags), f"{row_id}: token/tag length mismatch")
    _require(all(isinstance(t, str) for t in tokens + tags), f"{row_id}: non-string token/tag")
    positions, clean_tags = [], []
    cursor = 0
    for index, (token, tag) in enumerate(zip(tokens, tags, strict=True)):
        if token == "":
            _require(
                row_id == "row_1431" and index == 6 and tag == "O",
                f"{row_id}: unexpected empty token at {index}",
            )
            corrections.append(
                {
                    "source": "redmadrobot",
                    "original_id": row_id,
                    "kind": "omit_empty_O_token_in_derived_alignment",
                    "token_index": index,
                    "original_token": token,
                    "tag": tag,
                }
            )
            continue
        start = cursor
        while start < len(text) and text[start].isspace():
            start += 1
        end = start + len(token)
        if text[start:end] != token:
            _require(
                row_id in {"row_0", "row_1628"} and token == "WWW" and text[start:end] == "www",
                f"{row_id}: token {index} does not align at {start}",
            )
            corrections.append(
                {
                    "source": "redmadrobot",
                    "original_id": row_id,
                    "kind": "case_insensitive_alignment",
                    "token_index": index,
                    "original_token": token,
                    "surface": text[start:end],
                    "start": start,
                    "end": end,
                }
            )
        _require(end <= len(text), f"{row_id}: token exceeds text bounds")
        positions.append((start, end))
        clean_tags.append(tag)
        cursor = end
    _require(not text[cursor:].strip(), f"{row_id}: unaligned trailing text")
    entities = []
    active = None
    for (start, end), tag in zip(positions, clean_tags, strict=True):
        if tag == "O":
            if active:
                entities.append(_entity(text, *active))
                active = None
            continue
        _require(re.fullmatch(r"[BI]-.+", tag) is not None, f"{row_id}: invalid BIO {tag}")
        prefix, label = tag.split("-", 1)
        _require(label in RED_LABELS, f"{row_id}: unexpected type {label}")
        if prefix == "B":
            if active:
                entities.append(_entity(text, *active))
            active = (start, end, label)
        else:
            _require(
                active is not None and active[2] == label, f"{row_id}: invalid BIO transition {tag}"
            )
            active = (active[0], end, label)
    if active:
        entities.append(_entity(text, *active))
    return entities


def normalize_sources(source_dir: Path, scope: str = "development") -> tuple[list[dict], dict]:
    source_dir = Path(source_dir)
    selected = selected_sources(scope)
    records, corrections, source_files = [], [], []

    def verified(relative: str) -> Path:
        path = source_dir / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = SOURCES[relative]["sha256"]
        _require(digest == expected, f"Source hash mismatch: {relative}")
        source_files.append({"file": relative, "sha256": digest, "bytes": path.stat().st_size})
        return path

    for split in ("domain", "entity"):
        if f"hivetrace/{split}.parquet" not in selected:
            continue
        path = verified(f"hivetrace/{split}.parquet")
        for raw in pq.read_table(path).to_pylist():
            record = _base("hivetrace", split, raw["id"], raw["text"], HIVE_LABELS)
            record["domain"] = raw["domain"]
            entities = []
            for entity in raw["entities"]:
                _require(entity["type"] in HIVE_LABELS, f"{raw['id']}: unknown type")
                item = _entity(raw["text"], entity["start"], entity["end"], entity["type"])
                _require(item["text"] == entity["text"], f"{raw['id']}: surface mismatch")
                entities.append(item)
            record["entities"] = entities
            record["native_annotation"] = {"entities": raw["entities"]}
            record["has_entities"] = bool(entities)
            records.append(record)

    if "redmadrobot/test.csv" in selected:
        path = verified("redmadrobot/test.csv")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
    else:
        raw_rows = []
    for index, raw in enumerate(raw_rows):
        native_id = f"row_{index}"
        tokens, tags = json.loads(raw["tokens"]), json.loads(raw["ner_tags"])
        record = _base("redmadrobot", "test", native_id, raw["text"], RED_LABELS)
        record["entities"] = _red_entities(raw["text"], tokens, tags, native_id, corrections)
        record["native_annotation"] = {"tokens": tokens, "ner_tags": tags}
        record["has_entities"] = bool(record["entities"])
        records.append(record)

    _require(len(records) == len({r["id"] for r in records}), "Duplicate stable IDs")
    for record in records:
        entities = sorted(record["entities"], key=lambda e: (e["start"], e["end"]))
        _require(
            all(a["end"] <= b["start"] for a, b in zip(entities, entities[1:], strict=False)),
            f"{record['id']}: overlapping spans",
        )
        _require(
            all(e["type"] in record["label_scope"] for e in entities),
            f"{record['id']}: entity outside label scope",
        )
    row_counts = Counter(r["source"] for r in records)
    span_counts = Counter()
    for record in records:
        span_counts[record["source"]] += len(record["entities"])
    expected_rows = {"hivetrace": 1810, "redmadrobot": 2841}
    expected_spans = {"hivetrace": 1667, "redmadrobot": 5614}
    expected_sources = {name.split("/")[0] for name in selected}
    _require(
        dict(row_counts) == {s: expected_rows[s] for s in expected_sources}, "Unexpected row counts"
    )
    _require(
        dict(span_counts) == {s: expected_spans[s] for s in expected_sources},
        "Unexpected span counts",
    )
    _require(
        len(corrections) == (3 if "redmadrobot" in expected_sources else 0),
        "Unexpected alignment corrections",
    )
    audit = {
        "schema_version": "pii_sources_v2",
        "scope": scope,
        "revisions": {s: EXPECTED_REVISIONS[s] for s in sorted(expected_sources)},
        "rows": len(records),
        "rows_by_source": dict(row_counts),
        "spans": sum(span_counts.values()),
        "spans_by_source": dict(span_counts),
        "canonical_types": sorted({e["type"] for r in records for e in r["entities"]}),
        "corrections": corrections,
        "source_files": source_files,
        "annotation_scope_policy": "Labels outside label_scope are unknown, not negative gold.",
        "text_policy": "Original text unchanged; end offsets exclusive; Unicode code points.",
        "name_address_policy": "Preserve native granularity; no span aggregation.",
    }
    return records, audit
