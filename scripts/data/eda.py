import csv
import json
from pathlib import Path

from .sources import ROOT

TARGET_COVERAGE = {
    "NAME": ("FIRST_NAME,LAST_NAME,MIDDLE_NAME", "different_granularity"),
    "ADDRESS": ("COUNTRY,REGION,DISTRICT,CITY,STREET,HOUSE", "different_granularity"),
    "PHONE": ("PHONE", "compatible"),
    "EMAIL": ("EMAIL", "compatible"),
    "CREDIT_CARD": ("CREDIT_CARD", "compatible"),
    "PASSPORT": ("PASSPORT", "different_granularity"),
    "INN": ("INN", "compatible"),
    "SNILS": ("SNILS", "compatible"),
    "CVC": ("", "absent"),
    "KPP": ("", "absent"),
    "OGRN": ("", "absent"),
    "OGRNIP": ("", "absent"),
    "TOKEN": ("", "absent"),
}


def load_development(root=ROOT):
    root = Path(root)
    return {
        name: [
            json.loads(line)
            for line in (root / f"data/processed/development/{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for name in ("opt_train", "golden")
    }


def load_summaries(root=ROOT):
    return {
        scope: json.loads((Path(root) / f"eda/{scope}_summary.json").read_text(encoding="utf-8"))
        for scope in ("development", "evaluation")
    }


def build_tables(summaries):
    composition, labels, lengths, domains = [], [], [], []
    for scope, summary in summaries.items():
        for part, stats in {"all": summary["all"], **summary["parts"]}.items():
            composition.append(
                {
                    "scope": scope,
                    "part": part,
                    **{
                        key: stats[key]
                        for key in ("records", "spans", "positive_records", "negative_records")
                    },
                }
            )
            lengths.append(
                {
                    "scope": scope,
                    "part": part,
                    **{key: stats["length_chars"][key] for key in ("min", "median", "p95", "max")},
                }
            )
            for label in stats["label_scope"]:
                labels.append(
                    {
                        "scope": scope,
                        "part": part,
                        "label": label,
                        "spans": stats["label_spans"].get(label, 0),
                        "positive_records": stats["label_positive_records"].get(label, 0),
                    }
                )
            for domain, count in stats["domains"].items():
                domains.append({"scope": scope, "part": part, "domain": domain, "records": count})
    coverage = []
    for target, (native, status) in TARGET_COVERAGE.items():
        row = {"target": target, "development_native_types": native, "status": status}
        for part, minimum in (("opt_train", 200), ("golden", 100)):
            available = (
                summaries["development"]["parts"][part]["label_spans"].get(target, 0)
                if status == "compatible"
                else 0
            )
            row[part + "_compatible_spans"] = available
            row[part + "_minimum_target"] = minimum
            row[part + "_additional_spans"] = max(0, minimum - available)
        coverage.append(row)
    return {
        "composition": composition,
        "label_distribution": labels,
        "lengths": lengths,
        "domains": domains,
        "target_coverage": coverage,
    }


def make_figures(summaries, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figures = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=140)
        plt.close(fig)
        figures.append(name)

    dev, evaluation = summaries["development"], summaries["evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, (scope, summary) in enumerate(summaries.items()):
        stats = summary["all"]
        axes[0].bar(index, stats["positive_records"], color="#355c83")
        axes[0].bar(
            index, stats["negative_records"], bottom=stats["positive_records"], color="#69a58a"
        )
        axes[0].text(index, stats["records"] + 25, str(stats["records"]), ha="center")
        length = stats["length_chars"]
        edges = length["histogram_edges"]
        fractions = np.array(length["histogram_counts"]) / stats["records"]
        axes[1].stairs(fractions, edges, label=scope, linewidth=2)
    axes[0].set_xticks([0, 1], ["Development\nredmadrobot", "Final evaluation\nPII-Bench"])
    axes[0].set_ylabel("Records (blue: with annotated PII; green: without)")
    axes[0].set_ylim(0, 3200)
    axes[1].set_xscale("symlog", linthresh=25)
    axes[1].set_xlabel("Text length, characters")
    axes[1].set_ylabel("Fraction of records per bin")
    axes[1].legend()
    save(fig, "sources_and_lengths.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    labels = sorted(dev["all"]["label_spans"])
    y = np.arange(len(labels))
    for i, name in enumerate(("opt_train", "golden")):
        counts = [dev["parts"][name]["label_spans"].get(label, 0) for label in labels]
        ax.barh(y + (i - 0.5) * 0.38, counts, height=0.38, label=name)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Native entity spans; names and address components are not merged")
    ax.legend()
    save(fig, "development_labels.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    for name in ("opt_train", "golden"):
        part = dev["parts"][name]
        rates = [part["label_positive_records"].get(label, 0) / part["records"] for label in labels]
        ax.plot(x, rates, "o-", label=name)
    ax.set_xticks(x, labels, rotation=65, ha="right")
    ax.set_ylabel("Fraction of records with each type")
    ax.legend()
    save(fig, "split_label_rates.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, part in evaluation["parts"].items():
        labels_eval = sorted(evaluation["all"]["label_spans"])
        ax.plot(
            labels_eval,
            [part["label_spans"].get(label, 0) for label in labels_eval],
            "o-",
            label=name,
        )
    ax.set_xticks(range(len(labels_eval)), labels_eval, rotation=45, ha="right")
    ax.set_ylabel("Gold spans (dataset audit only; no model scores)")
    ax.legend()
    save(fig, "evaluation_composition.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    status_values = {"absent": 0, "different_granularity": 1, "compatible": 2}
    target_labels = list(TARGET_COVERAGE)
    matrix = np.array([[status_values[TARGET_COVERAGE[label][1]]] for label in target_labels])
    ax.imshow(
        matrix,
        cmap=matplotlib.colors.ListedColormap(["#cd6a69", "#e4bd67", "#69a58a"]),
        vmin=0,
        vmax=2,
        aspect="auto",
    )
    ax.set_xticks([0], ["Development support for final target schema"])
    ax.set_yticks(range(len(target_labels)), target_labels)
    for i, label in enumerate(target_labels):
        ax.text(0, i, TARGET_COVERAGE[label][1].replace("_", " "), ha="center", va="center")
    save(fig, "target_coverage.png")
    return figures


def main():
    summaries = load_summaries()
    directory = ROOT / "eda"
    for name, rows in build_tables(summaries).items():
        with (directory / (name + ".csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print("Figures:", ", ".join(make_figures(summaries, directory / "figures")))


if __name__ == "__main__":
    main()
