"""
Recommended command
-------------------
python src/analyze_final_rank_near_miss_annotations.py \
  --input results/analysis/final_rank_near_misses/final_rank_near_miss_review.tsv \
  --output_dir results/analysis/final_rank_near_misses/annotation_summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


FINE_LABELS = [
    "equivalent_in_top10",
    "equivalent_variant_in_top10",
    "near_equivalent_in_top10",
    "plausible_alternative_in_top10",
    "same_domain_alternative_in_top10",
    "generic_related_candidate_in_top10",
    "weak_related_candidate_in_top10",
    "no_clear_semantic_support_in_top10",
]

COARSE_MAP = {
    "equivalent_in_top10": "strong_semantic_support",
    "equivalent_variant_in_top10": "strong_semantic_support",
    "near_equivalent_in_top10": "strong_semantic_support",
    "plausible_alternative_in_top10": "strong_semantic_support",
    "same_domain_alternative_in_top10": "partial_semantic_support",
    "generic_related_candidate_in_top10": "partial_semantic_support",
    "weak_related_candidate_in_top10": "partial_semantic_support",
    "no_clear_semantic_support_in_top10": "no_clear_semantic_support",
}

COARSE_LABELS = [
    "strong_semantic_support",
    "partial_semantic_support",
    "no_clear_semantic_support",
]

REQUIRED_COLUMNS = {
    "model_key",
    "id",
    "language",
    "semantic_group",
    "final_gold_rank",
    "rank_bin",
    "final_prediction",
    "final_top10",
    "history_category",
    "manual_label",
    "manual_note",
}

DIMENSIONS = [
    ("model_key", "annotation_by_model.tsv"),
    ("language", "annotation_by_language.tsv"),
    ("semantic_group", "annotation_by_semantic_group.tsv"),
    ("rank_bin", "annotation_by_rank_bin.tsv"),
    ("history_category", "annotation_by_history.tsv"),
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize manual semantic-support annotations for "
            "final-rank 11--30 near-miss cases."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Annotated final_rank_near_miss_review.tsv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/analysis/final_rank_near_misses/annotation_summary"),
    )
    parser.add_argument(
        "--allow_unlabeled",
        action="store_true",
        help=(
            "Allow blank manual_label/manual_note rows. "
            "Blank rows are excluded from annotation statistics."
        ),
    )
    parser.add_argument(
        "--allow_unknown_labels",
        action="store_true",
        help=(
            "Allow labels outside the predefined annotation schema. "
            "They will receive coarse_label='unmapped'."
        ),
    )
    parser.add_argument(
        "--examples_per_label",
        type=int,
        default=3,
        help="Maximum examples exported per fine-grained label.",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(
            "Ignoring unrecognized runtime arguments: " + " ".join(unknown),
            file=sys.stderr,
        )
    return args


def load_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input TSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def write_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def pct(count: int, total: int) -> Optional[float]:
    return None if total == 0 else count / total


def normalise_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def validate_and_prepare(
    rows: list[dict[str, str]],
    *,
    allow_unlabeled: bool,
    allow_unknown_labels: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    missing_columns = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    annotated: list[dict[str, str]] = []
    unlabeled: list[dict[str, str]] = []
    unknown_labels: Counter[str] = Counter()

    seen_keys: set[tuple[str, str, str]] = set()

    for line_no, row in enumerate(rows, start=2):
        key = (
            normalise_text(row.get("model_key")),
            normalise_text(row.get("id")),
            normalise_text(row.get("language")),
        )
        if key in seen_keys:
            raise ValueError(f"Duplicate model/id/language key at row {line_no}: {key}")
        seen_keys.add(key)

        label = normalise_text(row.get("manual_label"))
        note = normalise_text(row.get("manual_note"))

        if not label:
            unlabeled.append(row)
            continue

        if not note:
            if not allow_unlabeled:
                raise ValueError(
                    f"Row {line_no} has manual_label={label!r} but blank manual_note."
                )

        if label not in COARSE_MAP:
            unknown_labels[label] += 1
            if not allow_unknown_labels:
                continue

        prepared = dict(row)
        prepared["manual_label"] = label
        prepared["manual_note"] = note
        prepared["coarse_label"] = COARSE_MAP.get(label, "unmapped")
        annotated.append(prepared)

    if unlabeled and not allow_unlabeled:
        preview = [
            (
                normalise_text(row.get("model_key")),
                normalise_text(row.get("id")),
                normalise_text(row.get("language")),
            )
            for row in unlabeled[:10]
        ]
        raise ValueError(
            f"{len(unlabeled)} rows have blank manual_label. "
            f"Preview: {preview}. Use --allow_unlabeled only for drafts."
        )

    if unknown_labels and not allow_unknown_labels:
        raise ValueError(
            "Unknown manual labels found: "
            + ", ".join(
                f"{label} (n={count})"
                for label, count in sorted(unknown_labels.items())
            )
            + ". Fix the labels or rerun with --allow_unknown_labels."
        )

    if not annotated:
        raise ValueError("No annotated rows available for analysis.")

    return annotated, unlabeled


def count_table(
    rows: Sequence[dict[str, str]],
    label_field: str,
    label_order: Sequence[str],
) -> list[dict[str, Any]]:
    counts = Counter(row[label_field] for row in rows)
    total = len(rows)
    output: list[dict[str, Any]] = []

    seen = set()
    for label in label_order:
        count = counts.get(label, 0)
        output.append(
            {
                "label": label,
                "count": count,
                "total_annotated": total,
                "proportion": pct(count, total),
            }
        )
        seen.add(label)

    for label in sorted(set(counts) - seen):
        count = counts[label]
        output.append(
            {
                "label": label,
                "count": count,
                "total_annotated": total,
                "proportion": pct(count, total),
            }
        )

    return output


def dimension_table(
    rows: Sequence[dict[str, str]],
    dimension: str,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[normalise_text(row.get(dimension))].append(row)

    output: list[dict[str, Any]] = []
    for value, bucket in sorted(buckets.items()):
        fine_counts = Counter(row["manual_label"] for row in bucket)
        coarse_counts = Counter(row["coarse_label"] for row in bucket)

        base = {
            dimension: value,
            "n_annotated": len(bucket),
        }

        for label in FINE_LABELS:
            base[f"{label}_count"] = fine_counts.get(label, 0)
            base[f"{label}_prop"] = pct(
                fine_counts.get(label, 0),
                len(bucket),
            )

        extra_fine = sorted(set(fine_counts) - set(FINE_LABELS))
        for label in extra_fine:
            base[f"{label}_count"] = fine_counts[label]
            base[f"{label}_prop"] = pct(fine_counts[label], len(bucket))

        for label in COARSE_LABELS:
            base[f"{label}_count"] = coarse_counts.get(label, 0)
            base[f"{label}_prop"] = pct(
                coarse_counts.get(label, 0),
                len(bucket),
            )

        if "unmapped" in coarse_counts:
            base["unmapped_count"] = coarse_counts["unmapped"]
            base["unmapped_prop"] = pct(
                coarse_counts["unmapped"],
                len(bucket),
            )

        output.append(base)

    return output


def make_examples(
    rows: Sequence[dict[str, str]],
    per_label: int,
) -> list[dict[str, Any]]:
    if per_label <= 0:
        return []

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row["manual_label"]].append(row)

    # Deterministic preference:
    # 1) closer final rank to Top-10,
    # 2) lower best rank across layers if available,
    # 3) stable model/id/language ordering.
    def int_or_large(value: Any) -> int:
        try:
            return int(float(str(value)))
        except Exception:
            return 10**9

    output: list[dict[str, Any]] = []
    for label in FINE_LABELS + sorted(set(buckets) - set(FINE_LABELS)):
        candidates = buckets.get(label, [])
        candidates = sorted(
            candidates,
            key=lambda row: (
                int_or_large(row.get("final_gold_rank")),
                int_or_large(row.get("best_rank_across_layers")),
                normalise_text(row.get("model_key")),
                normalise_text(row.get("language")),
                normalise_text(row.get("id")),
            ),
        )

        for row in candidates[:per_label]:
            output.append(
                {
                    "manual_label": row["manual_label"],
                    "coarse_label": row["coarse_label"],
                    "model_key": row.get("model_key", ""),
                    "id": row.get("id", ""),
                    "language": row.get("language", ""),
                    "semantic_group": row.get("semantic_group", ""),
                    "subject": row.get("subject", ""),
                    "prefix": row.get("prefix", ""),
                    "gold_target": row.get("gold_target", ""),
                    "full_sentence": row.get("full_sentence", ""),
                    "final_gold_rank": row.get("final_gold_rank", ""),
                    "rank_bin": row.get("rank_bin", ""),
                    "final_prediction": row.get("final_prediction", ""),
                    "final_top10": row.get("final_top10", ""),
                    "history_category": row.get("history_category", ""),
                    "best_rank_across_layers": row.get("best_rank_across_layers", ""),
                    "manual_note": row.get("manual_note", ""),
                }
            )

    return output


def nested_counts(
    rows: Sequence[dict[str, str]],
    field: str,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[normalise_text(row.get(field))].append(row)

    result: dict[str, dict[str, Any]] = {}
    for value, bucket in sorted(buckets.items()):
        fine = Counter(row["manual_label"] for row in bucket)
        coarse = Counter(row["coarse_label"] for row in bucket)
        result[value] = {
            "n": len(bucket),
            "fine_counts": dict(fine),
            "coarse_counts": dict(coarse),
        }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    rows = load_tsv(args.input)
    annotated, unlabeled = validate_and_prepare(
        rows,
        allow_unlabeled=args.allow_unlabeled,
        allow_unknown_labels=args.allow_unknown_labels,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fine_counts = count_table(
        annotated,
        "manual_label",
        FINE_LABELS,
    )
    coarse_counts = count_table(
        annotated,
        "coarse_label",
        COARSE_LABELS,
    )

    write_tsv(
        output_dir / "annotation_label_counts.tsv",
        fine_counts,
    )
    write_tsv(
        output_dir / "annotation_coarse_counts.tsv",
        coarse_counts,
    )

    for dimension, filename in DIMENSIONS:
        write_tsv(
            output_dir / filename,
            dimension_table(annotated, dimension),
        )

    examples = make_examples(
        annotated,
        per_label=args.examples_per_label,
    )
    write_tsv(
        output_dir / "annotation_examples.tsv",
        examples,
    )

    fine_counter = Counter(row["manual_label"] for row in annotated)
    coarse_counter = Counter(row["coarse_label"] for row in annotated)

    report = {
        "analysis": "final_rank_near_miss_manual_annotations",
        "input": str(args.input),
        "sample_design_note": (
            "The reviewed cases form a stratified representative sample "
            "(model x language x semantic group x rank bin), so percentages "
            "are descriptive of the reviewed sample and should not be treated "
            "as simple-random-sample estimates for all rank-11--30 cases."
        ),
        "interpretation_note": (
            "Manual labels describe semantic support in the final Top-10. "
            "They are not correctness labels and do not replace exact-token "
            "Target@10 or full-vocabulary rank."
        ),
        "n_input_rows": len(rows),
        "n_annotated_rows": len(annotated),
        "n_unlabeled_rows": len(unlabeled),
        "fine_label_schema": FINE_LABELS,
        "coarse_mapping": COARSE_MAP,
        "headline": {
            "fine_counts": dict(fine_counter),
            "fine_proportions": {
                label: fine_counter[label] / len(annotated) for label in fine_counter
            },
            "coarse_counts": dict(coarse_counter),
            "coarse_proportions": {
                label: coarse_counter[label] / len(annotated)
                for label in coarse_counter
            },
        },
        "by_model": nested_counts(annotated, "model_key"),
        "by_language": nested_counts(annotated, "language"),
        "by_semantic_group": nested_counts(
            annotated,
            "semantic_group",
        ),
        "by_rank_bin": nested_counts(annotated, "rank_bin"),
        "by_history_category": nested_counts(
            annotated,
            "history_category",
        ),
        "validation": {
            "required_columns_present": True,
            "unique_model_id_language_keys": True,
            "manual_labels_nonblank_for_analyzed_rows": True,
            "manual_notes_nonblank_for_analyzed_rows": True,
            "labels_match_schema_unless_overridden": not args.allow_unknown_labels,
        },
        "outputs": {
            "fine_counts": "annotation_label_counts.tsv",
            "coarse_counts": "annotation_coarse_counts.tsv",
            "by_model": "annotation_by_model.tsv",
            "by_language": "annotation_by_language.tsv",
            "by_semantic_group": "annotation_by_semantic_group.tsv",
            "by_rank_bin": "annotation_by_rank_bin.tsv",
            "by_history": "annotation_by_history.tsv",
            "examples": "annotation_examples.tsv",
            "summary": "annotation_summary.json",
        },
    }

    write_json(
        output_dir / "annotation_summary.json",
        report,
    )

    print("\n========================================")
    print("Near-miss annotation summary")
    print("========================================")
    print(f"Input rows:       {len(rows)}")
    print(f"Annotated rows:   {len(annotated)}")
    print(f"Unlabeled rows:   {len(unlabeled)}")

    print("\nFine-grained labels:")
    for label in FINE_LABELS:
        count = fine_counter.get(label, 0)
        print(f"  {label:40s} {count:3d}  ({100 * count / len(annotated):5.1f}%)")

    print("\nCoarse semantic support:")
    for label in COARSE_LABELS:
        count = coarse_counter.get(label, 0)
        print(f"  {label:30s} {count:3d}  ({100 * count / len(annotated):5.1f}%)")

    if unlabeled:
        print(
            "\nWARNING: unlabeled rows were excluded because "
            "--allow_unlabeled was used."
        )

    print(f"\nOutputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
