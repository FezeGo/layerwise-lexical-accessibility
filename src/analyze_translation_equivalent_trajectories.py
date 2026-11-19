"""
python src/analyze_translation_equivalent_trajectories.py \
    --base_input_dir data/logit_lens/translation_equivalent \
    --output_dir results/analysis/translation_equivalent \
    --models qwen3_0_6b qwen3_4b qwen3_14b \
    --min_consecutive 3
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


DEFAULT_BASE_INPUT_DIR = Path("data/logit_lens/translation_equivalent")
DEFAULT_OUTPUT_DIR = Path("data/analysis/full/translation_equivalent")
DEFAULT_MODELS = ["qwen3_0_6b", "qwen3_4b", "qwen3_14b"]
DEFAULT_INPUT_FILENAME = "translation_equivalent_by_layer.jsonl"
DEFAULT_MIN_CONSECUTIVE = 3
K_VALUES = (1, 5, 10)

REQUIRED_FIELDS = {
    "id",
    "model_key",
    "language",
    "layer",
    "num_layers",
    "semantic_group",
    "zh_gold",
    "zh_gold_token_id",
    "zh_gold_rank",
    "zh_gold_logit",
    "zh_gold_in_top1",
    "zh_gold_in_top5",
    "zh_gold_in_top10",
    "en_equivalent",
    "en_equivalent_token_id",
    "en_equivalent_rank",
    "en_equivalent_logit",
    "en_equivalent_in_top1",
    "en_equivalent_in_top5",
    "en_equivalent_in_top10",
    "rank_difference_en_minus_zh",
    "final_prediction_token_id",
    "final_prediction_text",
    "final_prediction_is_zh_gold",
    "final_prediction_is_en_equivalent",
    "topk_token_ids",
}

TRAJECTORY_METADATA_FIELDS = (
    "model_key",
    "model_name",
    "id",
    "semantic_group",
    "subgroup",
    "relation",
    "subject",
    "prefix",
    "full_sentence",
    "num_layers",
    "zh_gold",
    "zh_gold_token_id",
    "zh_gold_token_surface",
    "en_equivalent",
    "en_equivalent_token_id",
    "en_equivalent_token_surface",
    "en_equivalent_decoded_token",
)

PATTERN_ORDER = (
    "english_before_chinese",
    "chinese_before_english",
    "same_onset",
    "english_only_sustained",
    "chinese_only_sustained",
    "neither_sustained",
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            row["_source_file"] = str(path)
            row["_source_line"] = line_number
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def ordered_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def clean_for_table(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_tsv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = ordered_fieldnames(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: clean_for_table(row.get(key, "")) for key in fieldnames}
            )


def write_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def normalise_id(value: Any) -> str:
    return str(value)


def proportion(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def mean_or_none(values: Sequence[float | int]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def median_or_none(values: Sequence[float | int]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def safe_fraction(layer: Optional[int], num_layers: int) -> Optional[float]:
    if layer is None or num_layers <= 0:
        return None
    return layer / num_layers


# ---------------------------------------------------------------------------
# Validation and grouping
# ---------------------------------------------------------------------------


def validate_row_schema(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        location = f"{row.get('_source_file')}:{row.get('_source_line')}"
        missing = REQUIRED_FIELDS.difference(row)
        if missing:
            raise ValueError(
                f"Missing required fields at {location}: {sorted(missing)}"
            )

        if row["language"] != "zh":
            raise ValueError(
                f"EQ1 extraction must contain Chinese-prompt rows only; "
                f"got language={row['language']!r} at {location}."
            )

        if not isinstance(row["topk_token_ids"], list):
            raise TypeError(f"topk_token_ids must be a list at {location}")
        if len(row["topk_token_ids"]) < 10:
            raise ValueError(f"Need at least Top-10 token IDs at {location}")

        zh_rank = int(row["zh_gold_rank"])
        en_rank = int(row["en_equivalent_rank"])
        stored_delta = int(row["rank_difference_en_minus_zh"])
        if stored_delta != en_rank - zh_rank:
            raise ValueError(
                f"Rank-difference mismatch at {location}: stored={stored_delta}, "
                f"expected={en_rank - zh_rank}."
            )

        top_ids = [int(x) for x in row["topk_token_ids"]]
        zh_id = int(row["zh_gold_token_id"])
        en_id = int(row["en_equivalent_token_id"])

        expected_membership = {
            "zh_gold_in_top1": zh_id == top_ids[0],
            "zh_gold_in_top5": zh_id in top_ids[:5],
            "zh_gold_in_top10": zh_id in top_ids[:10],
            "en_equivalent_in_top1": en_id == top_ids[0],
            "en_equivalent_in_top5": en_id in top_ids[:5],
            "en_equivalent_in_top10": en_id in top_ids[:10],
        }
        for field, expected in expected_membership.items():
            if bool(row[field]) != expected:
                raise ValueError(
                    f"Top-k membership mismatch for {field} at {location}: "
                    f"stored={row[field]!r}, expected={expected}."
                )


def group_trajectories(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["model_key"]), normalise_id(row["id"]))
        grouped[key].append(row)

    for key, trajectory in grouped.items():
        trajectory.sort(key=lambda r: int(r["layer"]))
        layers = [int(r["layer"]) for r in trajectory]
        if len(layers) != len(set(layers)):
            duplicates = [layer for layer, n in Counter(layers).items() if n > 1]
            raise ValueError(f"Duplicate layers for {key}: {duplicates}")

        num_layers_values = {int(r["num_layers"]) for r in trajectory}
        if len(num_layers_values) != 1:
            raise ValueError(f"Inconsistent num_layers for {key}: {num_layers_values}")
        num_layers = next(iter(num_layers_values))
        expected_layers = list(range(0, num_layers + 1))
        if layers != expected_layers:
            raise ValueError(
                f"Non-contiguous/incomplete trajectory for {key}: "
                f"expected 0..{num_layers}, got {layers[:5]}...{layers[-5:]}"
            )

        # Token identity and semantic metadata should not change across layers.
        constant_fields = (
            "semantic_group",
            "zh_gold",
            "zh_gold_token_id",
            "en_equivalent",
            "en_equivalent_token_id",
            "final_prediction_token_id",
            "final_prediction_text",
        )
        for field in constant_fields:
            values = {str(r.get(field, "")) for r in trajectory}
            if len(values) != 1:
                raise ValueError(
                    f"Inconsistent {field} across layers for {key}: {values}"
                )

        final = trajectory[-1]
        top1_id = int(final["topk_token_ids"][0])
        if int(final["final_prediction_token_id"]) != top1_id:
            raise ValueError(f"Final-prediction Top-1 mismatch for {key}")
        if bool(final["final_prediction_is_zh_gold"]) != (
            int(final["zh_gold_token_id"]) == top1_id
        ):
            raise ValueError(f"Final Chinese-gold flag mismatch for {key}")
        if bool(final["final_prediction_is_en_equivalent"]) != (
            int(final["en_equivalent_token_id"]) == top1_id
        ):
            raise ValueError(f"Final English-equivalent flag mismatch for {key}")

    return dict(grouped)


# ---------------------------------------------------------------------------
# Onset helpers
# ---------------------------------------------------------------------------


def first_true_layer(rows: Sequence[dict[str, Any]], field: str) -> Optional[int]:
    for row in rows:
        if bool(row[field]):
            return int(row["layer"])
    return None


def first_sustained_layer(
    rows: Sequence[dict[str, Any]],
    field: str,
    min_consecutive: int,
) -> Optional[int]:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1")

    run_start: Optional[int] = None
    run_length = 0
    previous_layer: Optional[int] = None

    for row in rows:
        layer = int(row["layer"])
        if bool(row[field]):
            if previous_layer is not None and layer == previous_layer + 1:
                run_length += 1
            else:
                run_start = layer
                run_length = 1
            if run_length >= min_consecutive:
                return run_start
            previous_layer = layer
        else:
            run_start = None
            run_length = 0
            previous_layer = None

    return None


def max_consecutive_run(rows: Sequence[dict[str, Any]], field: str) -> int:
    best = 0
    current = 0
    previous_layer: Optional[int] = None
    for row in rows:
        layer = int(row["layer"])
        if bool(row[field]):
            if previous_layer is not None and layer == previous_layer + 1:
                current += 1
            else:
                current = 1
            best = max(best, current)
            previous_layer = layer
        else:
            current = 0
            previous_layer = None
    return best


def classify_sustained_pattern(
    en_onset: Optional[int],
    zh_onset: Optional[int],
) -> str:
    if en_onset is not None and zh_onset is not None:
        if en_onset < zh_onset:
            return "english_before_chinese"
        if zh_onset < en_onset:
            return "chinese_before_english"
        return "same_onset"
    if en_onset is not None:
        return "english_only_sustained"
    if zh_onset is not None:
        return "chinese_only_sustained"
    return "neither_sustained"


# ---------------------------------------------------------------------------
# Trajectory-level analysis
# ---------------------------------------------------------------------------


def build_trajectory_summary(
    rows: Sequence[dict[str, Any]],
    min_consecutive: int,
) -> dict[str, Any]:
    first = rows[0]
    final = rows[-1]
    num_layers = int(first["num_layers"])

    summary: dict[str, Any] = {
        field: first.get(field, "") for field in TRAJECTORY_METADATA_FIELDS
    }
    summary.update(
        {
            "min_consecutive": min_consecutive,
            "zh_final_prediction_text": final["final_prediction_text"],
            "zh_final_prediction_token_id": int(final["final_prediction_token_id"]),
            "zh_final_prediction_is_zh_gold": bool(
                final["final_prediction_is_zh_gold"]
            ),
            "zh_final_prediction_is_en_equivalent": bool(
                final["final_prediction_is_en_equivalent"]
            ),
            "zh_gold_final_rank": int(final["zh_gold_rank"]),
            "en_equivalent_final_rank": int(final["en_equivalent_rank"]),
            "final_rank_difference_en_minus_zh": int(
                final["rank_difference_en_minus_zh"]
            ),
            "zh_gold_best_rank": min(int(r["zh_gold_rank"]) for r in rows),
            "en_equivalent_best_rank": min(int(r["en_equivalent_rank"]) for r in rows),
            "best_rank_difference_en_minus_zh": (
                min(int(r["en_equivalent_rank"]) for r in rows)
                - min(int(r["zh_gold_rank"]) for r in rows)
            ),
            "layers_en_rank_better_count": sum(
                int(r["en_equivalent_rank"]) < int(r["zh_gold_rank"]) for r in rows
            ),
            "layers_zh_rank_better_count": sum(
                int(r["zh_gold_rank"]) < int(r["en_equivalent_rank"]) for r in rows
            ),
            "layers_equal_rank_count": sum(
                int(r["zh_gold_rank"]) == int(r["en_equivalent_rank"]) for r in rows
            ),
        }
    )

    summary["en_rank_better_layer_rate"] = proportion(
        summary["layers_en_rank_better_count"], len(rows)
    )
    summary["zh_rank_better_layer_rate"] = proportion(
        summary["layers_zh_rank_better_count"], len(rows)
    )

    for k in K_VALUES:
        en_field = f"en_equivalent_in_top{k}"
        zh_field = f"zh_gold_in_top{k}"

        en_first = first_true_layer(rows, en_field)
        zh_first = first_true_layer(rows, zh_field)
        en_sustained = first_sustained_layer(rows, en_field, min_consecutive)
        zh_sustained = first_sustained_layer(rows, zh_field, min_consecutive)

        summary.update(
            {
                f"en_equivalent_ever_top{k}": any(bool(r[en_field]) for r in rows),
                f"zh_gold_ever_top{k}": any(bool(r[zh_field]) for r in rows),
                f"en_equivalent_final_top{k}": bool(final[en_field]),
                f"zh_gold_final_top{k}": bool(final[zh_field]),
                f"first_en_equivalent_top{k}_layer": en_first,
                f"first_zh_gold_top{k}_layer": zh_first,
                f"first_en_equivalent_top{k}_fraction": safe_fraction(
                    en_first, num_layers
                ),
                f"first_zh_gold_top{k}_fraction": safe_fraction(zh_first, num_layers),
                f"first_en_equivalent_top{k}_sustained_layer": en_sustained,
                f"first_zh_gold_top{k}_sustained_layer": zh_sustained,
                f"first_en_equivalent_top{k}_sustained_fraction": safe_fraction(
                    en_sustained, num_layers
                ),
                f"first_zh_gold_top{k}_sustained_fraction": safe_fraction(
                    zh_sustained, num_layers
                ),
                f"en_equivalent_max_consecutive_top{k}": max_consecutive_run(
                    rows, en_field
                ),
                f"zh_gold_max_consecutive_top{k}": max_consecutive_run(rows, zh_field),
                f"translation_sustained_pattern_top{k}": classify_sustained_pattern(
                    en_sustained, zh_sustained
                ),
                # Positive => English sustained onset is earlier.
                f"sustained_onset_delta_zh_minus_en_top{k}": (
                    zh_sustained - en_sustained
                    if en_sustained is not None and zh_sustained is not None
                    else None
                ),
            }
        )

    return summary


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def grouping_specs() -> list[tuple[str, Callable[[dict[str, Any]], str]]]:
    return [
        ("overall", lambda row: "all"),
        ("semantic_group", lambda row: str(row.get("semantic_group", ""))),
    ]


def aggregate_pattern_counts(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for grouping_name, group_fn in grouping_specs():
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["model_key"]), group_fn(row))].append(row)

        for (model_key, group_value), group_rows in sorted(grouped.items()):
            n = len(group_rows)
            for k in K_VALUES:
                pattern_field = f"translation_sustained_pattern_top{k}"
                counts = Counter(str(r[pattern_field]) for r in group_rows)

                for pattern in PATTERN_ORDER:
                    count = counts.get(pattern, 0)
                    output.append(
                        {
                            "grouping": grouping_name,
                            "group": group_value,
                            "model_key": model_key,
                            "k": k,
                            "row_type": "pattern",
                            "pattern": pattern,
                            "count": count,
                            "n": n,
                            "rate": proportion(count, n),
                        }
                    )

                en_field = f"first_en_equivalent_top{k}_sustained_layer"
                zh_field = f"first_zh_gold_top{k}_sustained_layer"
                en_count = sum(r[en_field] is not None for r in group_rows)
                zh_count = sum(r[zh_field] is not None for r in group_rows)
                both_count = sum(
                    r[en_field] is not None and r[zh_field] is not None
                    for r in group_rows
                )
                either_count = sum(
                    r[en_field] is not None or r[zh_field] is not None
                    for r in group_rows
                )
                output.append(
                    {
                        "grouping": grouping_name,
                        "group": group_value,
                        "model_key": model_key,
                        "k": k,
                        "row_type": "coverage",
                        "pattern": "__coverage_summary__",
                        "n": n,
                        "english_sustained_count": en_count,
                        "english_sustained_rate": proportion(en_count, n),
                        "chinese_sustained_count": zh_count,
                        "chinese_sustained_rate": proportion(zh_count, n),
                        "both_sustained_count": both_count,
                        "both_sustained_rate": proportion(both_count, n),
                        "either_sustained_count": either_count,
                        "either_sustained_rate": proportion(either_count, n),
                    }
                )

    return output


def aggregate_onset_summary(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for grouping_name, group_fn in grouping_specs():
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["model_key"]), group_fn(row))].append(row)

        for (model_key, group_value), group_rows in sorted(grouped.items()):
            n = len(group_rows)
            for k in K_VALUES:
                en_layer_field = f"first_en_equivalent_top{k}_sustained_layer"
                zh_layer_field = f"first_zh_gold_top{k}_sustained_layer"
                en_frac_field = f"first_en_equivalent_top{k}_sustained_fraction"
                zh_frac_field = f"first_zh_gold_top{k}_sustained_fraction"

                en_layers = [
                    int(r[en_layer_field])
                    for r in group_rows
                    if r[en_layer_field] is not None
                ]
                zh_layers = [
                    int(r[zh_layer_field])
                    for r in group_rows
                    if r[zh_layer_field] is not None
                ]
                en_fracs = [
                    float(r[en_frac_field])
                    for r in group_rows
                    if r[en_frac_field] is not None
                ]
                zh_fracs = [
                    float(r[zh_frac_field])
                    for r in group_rows
                    if r[zh_frac_field] is not None
                ]

                both = [
                    r
                    for r in group_rows
                    if r[en_layer_field] is not None and r[zh_layer_field] is not None
                ]
                deltas = [
                    int(r[f"sustained_onset_delta_zh_minus_en_top{k}"]) for r in both
                ]
                delta_fracs = [
                    float(r[zh_frac_field]) - float(r[en_frac_field]) for r in both
                ]

                output.append(
                    {
                        "grouping": grouping_name,
                        "group": group_value,
                        "model_key": model_key,
                        "k": k,
                        "n": n,
                        "english_sustained_n": len(en_layers),
                        "english_sustained_rate": proportion(len(en_layers), n),
                        "english_mean_onset_layer": mean_or_none(en_layers),
                        "english_median_onset_layer": median_or_none(en_layers),
                        "english_mean_onset_fraction": mean_or_none(en_fracs),
                        "english_median_onset_fraction": median_or_none(en_fracs),
                        "chinese_sustained_n": len(zh_layers),
                        "chinese_sustained_rate": proportion(len(zh_layers), n),
                        "chinese_mean_onset_layer": mean_or_none(zh_layers),
                        "chinese_median_onset_layer": median_or_none(zh_layers),
                        "chinese_mean_onset_fraction": mean_or_none(zh_fracs),
                        "chinese_median_onset_fraction": median_or_none(zh_fracs),
                        "both_sustained_n": len(both),
                        "mean_delta_zh_minus_en_layer_among_both": mean_or_none(deltas),
                        "median_delta_zh_minus_en_layer_among_both": median_or_none(
                            deltas
                        ),
                        "mean_delta_zh_minus_en_fraction_among_both": mean_or_none(
                            delta_fracs
                        ),
                        "median_delta_zh_minus_en_fraction_among_both": median_or_none(
                            delta_fracs
                        ),
                    }
                )

    return output


def aggregate_layerwise_summary(
    layer_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    layer_grouping_specs: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("overall", lambda row: "all"),
        ("semantic_group", lambda row: str(row.get("semantic_group", ""))),
    ]

    for grouping_name, group_fn in layer_grouping_specs:
        grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in layer_rows:
            key = (str(row["model_key"]), group_fn(row), int(row["layer"]))
            grouped[key].append(row)

        for (model_key, group_value, layer), rows in sorted(grouped.items()):
            n = len(rows)
            first = rows[0]
            zh_ranks = [int(r["zh_gold_rank"]) for r in rows]
            en_ranks = [int(r["en_equivalent_rank"]) for r in rows]
            rank_deltas = [int(r["rank_difference_en_minus_zh"]) for r in rows]
            en_better = sum(delta < 0 for delta in rank_deltas)
            zh_better = sum(delta > 0 for delta in rank_deltas)
            equal = sum(delta == 0 for delta in rank_deltas)

            output.append(
                {
                    "grouping": grouping_name,
                    "group": group_value,
                    "model_key": model_key,
                    "layer": layer,
                    "num_layers": int(first["num_layers"]),
                    "layer_fraction": float(
                        first.get("layer_fraction", layer / int(first["num_layers"]))
                    ),
                    "n": n,
                    "zh_gold_at_1": proportion(
                        sum(bool(r["zh_gold_in_top1"]) for r in rows), n
                    ),
                    "zh_gold_at_5": proportion(
                        sum(bool(r["zh_gold_in_top5"]) for r in rows), n
                    ),
                    "zh_gold_at_10": proportion(
                        sum(bool(r["zh_gold_in_top10"]) for r in rows), n
                    ),
                    "en_equivalent_at_1": proportion(
                        sum(bool(r["en_equivalent_in_top1"]) for r in rows), n
                    ),
                    "en_equivalent_at_5": proportion(
                        sum(bool(r["en_equivalent_in_top5"]) for r in rows), n
                    ),
                    "en_equivalent_at_10": proportion(
                        sum(bool(r["en_equivalent_in_top10"]) for r in rows), n
                    ),
                    "mean_zh_gold_rank": mean_or_none(zh_ranks),
                    "median_zh_gold_rank": median_or_none(zh_ranks),
                    "mean_en_equivalent_rank": mean_or_none(en_ranks),
                    "median_en_equivalent_rank": median_or_none(en_ranks),
                    "mean_rank_difference_en_minus_zh": mean_or_none(rank_deltas),
                    "median_rank_difference_en_minus_zh": median_or_none(rank_deltas),
                    "english_better_rank_count": en_better,
                    "english_better_rank_rate": proportion(en_better, n),
                    "chinese_better_rank_count": zh_better,
                    "chinese_better_rank_rate": proportion(zh_better, n),
                    "equal_rank_count": equal,
                    "equal_rank_rate": proportion(equal, n),
                }
            )

    return output


def aggregate_final_layer_summary(
    trajectory_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for grouping_name, group_fn in grouping_specs():
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in trajectory_rows:
            grouped[(str(row["model_key"]), group_fn(row))].append(row)

        for (model_key, group_value), rows in sorted(grouped.items()):
            n = len(rows)
            zh_ranks = [int(r["zh_gold_final_rank"]) for r in rows]
            en_ranks = [int(r["en_equivalent_final_rank"]) for r in rows]
            deltas = [int(r["final_rank_difference_en_minus_zh"]) for r in rows]
            en_better = sum(delta < 0 for delta in deltas)
            zh_better = sum(delta > 0 for delta in deltas)
            equal = sum(delta == 0 for delta in deltas)

            out: dict[str, Any] = {
                "grouping": grouping_name,
                "group": group_value,
                "model_key": model_key,
                "n": n,
                "mean_zh_gold_final_rank": mean_or_none(zh_ranks),
                "median_zh_gold_final_rank": median_or_none(zh_ranks),
                "mean_en_equivalent_final_rank": mean_or_none(en_ranks),
                "median_en_equivalent_final_rank": median_or_none(en_ranks),
                "mean_final_rank_difference_en_minus_zh": mean_or_none(deltas),
                "median_final_rank_difference_en_minus_zh": median_or_none(deltas),
                "english_better_final_rank_count": en_better,
                "english_better_final_rank_rate": proportion(en_better, n),
                "chinese_better_final_rank_count": zh_better,
                "chinese_better_final_rank_rate": proportion(zh_better, n),
                "equal_final_rank_count": equal,
                "equal_final_rank_rate": proportion(equal, n),
                "final_prediction_is_zh_gold_count": sum(
                    bool(r["zh_final_prediction_is_zh_gold"]) for r in rows
                ),
                "final_prediction_is_zh_gold_rate": proportion(
                    sum(bool(r["zh_final_prediction_is_zh_gold"]) for r in rows), n
                ),
                "final_prediction_is_en_equivalent_count": sum(
                    bool(r["zh_final_prediction_is_en_equivalent"]) for r in rows
                ),
                "final_prediction_is_en_equivalent_rate": proportion(
                    sum(bool(r["zh_final_prediction_is_en_equivalent"]) for r in rows),
                    n,
                ),
            }
            for k in K_VALUES:
                out[f"zh_gold_final_top{k}_count"] = sum(
                    bool(r[f"zh_gold_final_top{k}"]) for r in rows
                )
                out[f"zh_gold_final_top{k}_rate"] = proportion(
                    out[f"zh_gold_final_top{k}_count"], n
                )
                out[f"en_equivalent_final_top{k}_count"] = sum(
                    bool(r[f"en_equivalent_final_top{k}"]) for r in rows
                )
                out[f"en_equivalent_final_top{k}_rate"] = proportion(
                    out[f"en_equivalent_final_top{k}_count"], n
                )
            output.append(out)

    return output


# ---------------------------------------------------------------------------
# Input resolution and CLI
# ---------------------------------------------------------------------------


def resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [Path(path) for path in args.input]

    paths = [
        Path(args.base_input_dir) / model / args.input_filename for model in args.models
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following EQ1 extraction files are missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )
    return paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze direct Chinese-gold vs English translation-equivalent "
            "Logit Lens measurements under Chinese prompts."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        help="Explicit translation_equivalent_by_layer.jsonl files.",
    )
    parser.add_argument(
        "--base_input_dir",
        type=Path,
        default=DEFAULT_BASE_INPUT_DIR,
        help=f"Base extraction directory (default: {DEFAULT_BASE_INPUT_DIR}).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model directory keys used with --base_input_dir.",
    )
    parser.add_argument(
        "--input_filename",
        default=DEFAULT_INPUT_FILENAME,
        help=f"Input filename per model (default: {DEFAULT_INPUT_FILENAME}).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Analysis output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--min_consecutive",
        type=int,
        default=DEFAULT_MIN_CONSECUTIVE,
        help=(
            "Consecutive Top-k memberships required for strict sustained onset "
            f"(default: {DEFAULT_MIN_CONSECUTIVE})."
        ),
    )

    # Colab/Jupyter kernels inject arguments such as `-f kernel.json`.
    # parse_known_args keeps this script runnable both as a .py file and inside
    # a notebook cell without changing the analysis logic.
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(
            "Ignoring unrecognized runtime arguments: " + " ".join(unknown),
            file=sys.stderr,
        )
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.min_consecutive < 1:
        raise ValueError("--min_consecutive must be at least 1")

    input_paths = resolve_input_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_layer_rows: list[dict[str, Any]] = []
    for path in input_paths:
        print(f"Loading {path} ...", file=sys.stderr)
        all_layer_rows.extend(load_jsonl(path))

    validate_row_schema(all_layer_rows)
    trajectories = group_trajectories(all_layer_rows)

    trajectory_rows = [
        build_trajectory_summary(rows, args.min_consecutive)
        for _, rows in sorted(trajectories.items())
    ]
    trajectory_rows.sort(key=lambda r: (str(r["model_key"]), normalise_id(r["id"])))

    # Derived tables.
    pattern_counts = aggregate_pattern_counts(trajectory_rows)
    onset_summary = aggregate_onset_summary(trajectory_rows)
    layerwise_summary = aggregate_layerwise_summary(all_layer_rows)
    final_layer_summary = aggregate_final_layer_summary(trajectory_rows)

    english_before_top10 = [
        r
        for r in trajectory_rows
        if r["translation_sustained_pattern_top10"] == "english_before_chinese"
    ]

    # Keep a compact layerwise table containing the direct measurements used by
    # the analysis, rather than duplicating every diagnostic field from raw JSONL.
    compact_layerwise_rows: list[dict[str, Any]] = []
    for row in all_layer_rows:
        compact_layerwise_rows.append(
            {
                "model_key": row["model_key"],
                "model_name": row.get("model_name", ""),
                "id": normalise_id(row["id"]),
                "semantic_group": row.get("semantic_group", ""),
                "subgroup": row.get("subgroup", ""),
                "relation": row.get("relation", ""),
                "layer": int(row["layer"]),
                "layer_name": row.get("layer_name", ""),
                "num_layers": int(row["num_layers"]),
                "layer_fraction": float(
                    row.get(
                        "layer_fraction", int(row["layer"]) / int(row["num_layers"])
                    )
                ),
                "zh_gold": row["zh_gold"],
                "zh_gold_token_id": int(row["zh_gold_token_id"]),
                "en_equivalent": row["en_equivalent"],
                "en_equivalent_token_id": int(row["en_equivalent_token_id"]),
                "zh_gold_rank": int(row["zh_gold_rank"]),
                "en_equivalent_rank": int(row["en_equivalent_rank"]),
                "rank_difference_en_minus_zh": int(row["rank_difference_en_minus_zh"]),
                "zh_gold_logit": float(row["zh_gold_logit"]),
                "en_equivalent_logit": float(row["en_equivalent_logit"]),
                "logit_difference_en_minus_zh": row.get("logit_difference_en_minus_zh"),
                "zh_gold_in_top1": bool(row["zh_gold_in_top1"]),
                "zh_gold_in_top5": bool(row["zh_gold_in_top5"]),
                "zh_gold_in_top10": bool(row["zh_gold_in_top10"]),
                "en_equivalent_in_top1": bool(row["en_equivalent_in_top1"]),
                "en_equivalent_in_top5": bool(row["en_equivalent_in_top5"]),
                "en_equivalent_in_top10": bool(row["en_equivalent_in_top10"]),
                "final_prediction_text": row["final_prediction_text"],
                "final_prediction_is_zh_gold": bool(row["final_prediction_is_zh_gold"]),
                "final_prediction_is_en_equivalent": bool(
                    row["final_prediction_is_en_equivalent"]
                ),
            }
        )

    compact_layerwise_rows.sort(
        key=lambda r: (str(r["model_key"]), normalise_id(r["id"]), int(r["layer"]))
    )

    # Write canonical EQ1 analysis outputs.
    write_tsv(
        compact_layerwise_rows,
        output_dir / "translation_equivalent_layerwise.tsv",
    )
    write_tsv(
        trajectory_rows,
        output_dir / "translation_equivalent_trajectories.tsv",
    )
    write_jsonl(
        trajectory_rows,
        output_dir / "translation_equivalent_trajectories.jsonl",
    )
    write_tsv(
        pattern_counts,
        output_dir / "translation_equivalent_pattern_counts.tsv",
    )
    write_tsv(
        onset_summary,
        output_dir / "translation_equivalent_onset_summary.tsv",
    )
    write_tsv(
        layerwise_summary,
        output_dir / "translation_equivalent_layerwise_summary.tsv",
    )
    write_tsv(
        final_layer_summary,
        output_dir / "translation_equivalent_final_layer_summary.tsv",
    )
    write_tsv(
        english_before_top10,
        output_dir / "english_before_chinese_gold.tsv",
    )

    model_counts = Counter(str(r["model_key"]) for r in trajectory_rows)
    expected_per_model = {model: model_counts.get(model, 0) for model in args.models}
    top10_patterns_by_model = {
        model: dict(
            Counter(
                str(r["translation_sustained_pattern_top10"])
                for r in trajectory_rows
                if str(r["model_key"]) == model
            )
        )
        for model in sorted(model_counts)
    }

    report = {
        "analysis": "translation_equivalent_accessibility_under_chinese_prompt",
        "input_files": [str(path) for path in input_paths],
        "raw_layer_rows": len(all_layer_rows),
        "trajectory_units_model_id": len(trajectory_rows),
        "model_example_counts": dict(model_counts),
        "requested_model_example_counts": expected_per_model,
        "min_consecutive": args.min_consecutive,
        "measurement_source": (
            "Direct token-specific full-vocabulary measurements produced by "
            "logit_lens_translation_equivalent.py; no decoded-string matching."
        ),
        "rank_definition": (
            "1 + number of vocabulary logits strictly greater than the tracked token logit"
        ),
        "rank_difference_definition": "English-equivalent rank minus Chinese-gold rank; negative means English ranks better.",
        "topk_definition": "Actual torch.topk membership from the EQ1 extraction.",
        "sustained_onset_definition": (
            f"Earliest layer beginning at least {args.min_consecutive} consecutive "
            "layers of Top-k membership."
        ),
        "top10_pattern_counts_by_model": top10_patterns_by_model,
        "english_before_chinese_top10_count": len(english_before_top10),
        "validation": {
            "all_rows_chinese_prompt": True,
            "rank_difference_consistency_checked": True,
            "actual_topk_membership_consistency_checked": True,
            "trajectory_contiguity_checked": True,
            "final_layer_prediction_consistency_checked": True,
        },
        "outputs": {
            "layerwise": "translation_equivalent_layerwise.tsv",
            "trajectories_tsv": "translation_equivalent_trajectories.tsv",
            "trajectories_jsonl": "translation_equivalent_trajectories.jsonl",
            "pattern_counts": "translation_equivalent_pattern_counts.tsv",
            "onset_summary": "translation_equivalent_onset_summary.tsv",
            "layerwise_summary": "translation_equivalent_layerwise_summary.tsv",
            "final_layer_summary": "translation_equivalent_final_layer_summary.tsv",
            "english_before_chinese_top10": "english_before_chinese_gold.tsv",
        },
    }
    write_json(report, output_dir / "translation_equivalent_report.json")

    print(f"Wrote EQ1 analysis outputs to: {output_dir}")
    print(f"Trajectory units: {len(trajectory_rows)}")
    print(f"Model counts: {dict(model_counts)}")
    print(
        "English-equivalent sustained before Chinese gold at Top-10: "
        f"{len(english_before_top10)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
