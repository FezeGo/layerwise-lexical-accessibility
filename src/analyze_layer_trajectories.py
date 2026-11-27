from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean as statistics_mean
from statistics import median
from typing import Any, Callable, Iterable, Optional, Sequence


DEFAULT_INPUT_DIR = Path("data/logit_lens/full")
DEFAULT_OUTPUT_DIR = Path("results/analysis/trajectory")
MODEL_KEYS = ("qwen3_0_6b", "qwen3_4b", "qwen3_14b")
INPUT_FILENAME = "logit_lens_by_layer.jsonl"
DEFAULT_MIN_CONSECUTIVE = 3
DEFAULT_EXAMPLES_PER_PATTERN = 3
DEFAULT_SAMPLING_SEED = 42

OUTPUT_FILENAMES = (
    "layer_trajectory_examples.tsv",
    "layer_trajectory_examples.jsonl",
    "layer_trajectory_summary.tsv",
    "trajectory_pattern_counts.tsv",
    "representative_trajectory_examples.tsv",
)
COMBINED_OUTPUT_FILENAMES = tuple(f"all_models_{name}" for name in OUTPUT_FILENAMES)

K_VALUES = (1, 5, 10)

FINAL_EXACT_GOLD = "final_exact_gold"
FINAL_NON_GOLD = "final_non_gold"

OPTIONAL_METADATA_FIELDS = (
    "source",
    "subject",
    "prefix",
    "full_sentence",
    "paired_language",
    "paired_subject",
    "paired_target",
    "paired_prefix",
    "paired_full_sentence",
    "en_subject",
    "en_target",
    "en_prefix",
    "en_full_sentence",
    "zh_subject",
    "zh_target",
    "zh_prefix",
    "zh_full_sentence",
)


# I/O helpers


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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

            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def ordered_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    fields: list[str] = []

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    return fields


def write_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = ordered_fieldnames(rows)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# Validation and conversion helpers


def as_bool(value: Any) -> bool:
    """
    Convert a stored Boolean-like value to bool.

    JSONL output should normally contain true/false, but this helper
    also supports 0/1 and common string representations.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes", "y"}:
            return True

        if normalized in {"false", "0", "no", "n", ""}:
            return False

    raise ValueError(f"Cannot interpret Boolean value: {value!r}")


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(converted):
        return None

    return converted


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [value for value in values if value is not None and math.isfinite(value)]

    if not cleaned:
        return None

    return float(statistics_mean(cleaned))


def safe_median(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [value for value in values if value is not None and math.isfinite(value)]

    if not cleaned:
        return None

    return float(median(cleaned))


def rate(rows: Sequence[dict[str, Any]], key: str) -> Optional[float]:
    values = [as_bool(row[key]) for row in rows if key in row and row[key] is not None]

    if not values:
        return None

    return sum(values) / len(values)


def validate_required_fields(
    rows: Sequence[dict[str, Any]],
    input_path: Path,
) -> None:
    required = {
        "id",
        "model_key",
        "language",
        "semantic_group",
        "subgroup",
        "relation",
        "layer",
        "target",
        "target_token_id",
        "final_prediction_token_id",
        "final_prediction_is_target",
    }

    for k in K_VALUES:
        required.add(f"target_in_top{k}")
        required.add(f"faithful_final_at_{k}")

    missing = sorted(field for field in required if field not in rows[0])

    if missing:
        raise ValueError(f"Missing required fields in {input_path}: {missing}")


# Boolean trajectory helpers


def first_true_layer(
    values: Sequence[bool],
    layers: Sequence[int],
) -> Optional[int]:
    for value, layer in zip(values, layers):
        if value:
            return int(layer)

    return None


def last_true_layer(
    values: Sequence[bool],
    layers: Sequence[int],
) -> Optional[int]:
    for value, layer in reversed(list(zip(values, layers))):
        if value:
            return int(layer)

    return None


def first_sustained_true_layer(
    values: Sequence[bool],
    layers: Sequence[int],
    min_consecutive: int,
) -> Optional[int]:
    """
    Return the first layer starting a run of at least min_consecutive
    True values.
    """
    if min_consecutive <= 0:
        raise ValueError("min_consecutive must be positive.")

    if len(values) != len(layers):
        raise ValueError("values and layers must have equal length.")

    if len(values) < min_consecutive:
        return None

    run_length = 0

    for index, value in enumerate(values):
        if value:
            run_length += 1
        else:
            run_length = 0

        if run_length >= min_consecutive:
            start_index = index - min_consecutive + 1
            return int(layers[start_index])

    return None


def first_stable_to_end_layer(
    values: Sequence[bool],
    layers: Sequence[int],
) -> Optional[int]:
    """
    Return the earliest layer from which the value stays True through
    the final layer.

    For Faithful-to-final@k, a value always exists because the final
    layer is compared with itself.
    """
    if len(values) != len(layers):
        raise ValueError("values and layers must have equal length.")

    suffix_all_true = True

    for index in range(len(values) - 1, -1, -1):
        suffix_all_true = suffix_all_true and values[index]

        if suffix_all_true:
            candidate = int(layers[index])
        else:
            break

    if values and values[-1]:
        return candidate

    return None


def max_true_run_length(values: Sequence[bool]) -> int:
    best = 0
    current = 0

    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def count_true_runs(values: Sequence[bool]) -> int:
    runs = 0
    previous = False

    for value in values:
        if value and not previous:
            runs += 1

        previous = value

    return runs


def true_then_lost(values: Sequence[bool]) -> bool:
    seen_true = False

    for value in values:
        if value:
            seen_true = True
        elif seen_true:
            return True

    return False


def compare_first_onsets(
    target_layer: Optional[int],
    final_layer: Optional[int],
) -> str:
    if target_layer is None and final_layer is None:
        return "neither_detected"

    if target_layer is None:
        return "final_only"

    if final_layer is None:
        return "gold_only"

    if target_layer < final_layer:
        return "gold_before_final"

    if final_layer < target_layer:
        return "final_before_gold"

    return "same_onset"


def compare_sustained_onsets(
    target_layer: Optional[int],
    final_layer: Optional[int],
) -> str:
    """
    Compare strict sustained Top-k onset layers.

    No fallback to first onset is used. A missing sustained onset means
    that the token never formed a run of ``min_consecutive`` True layers,
    even if it appeared transiently in one or more layers.
    """
    if target_layer is None and final_layer is None:
        return "neither_sustained"

    if target_layer is None:
        return "final_sustained_gold_not_sustained"

    if final_layer is None:
        return "gold_sustained_final_not_sustained"

    if target_layer < final_layer:
        return "gold_before_final"

    if final_layer < target_layer:
        return "final_before_gold"

    return "same_onset"


def make_pattern_label(
    final_exact_gold: bool,
    relation: str,
) -> str:
    if final_exact_gold:
        return "exact_gold_convergence"

    return relation


def layer_fraction(
    layer: Optional[int],
    final_layer: int,
) -> Optional[float]:
    if layer is None or final_layer <= 0:
        return None

    return float(layer / final_layer)


# Per-example analysis


def group_layer_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """
    Group raw rows by model, example ID, and language.

    The same example ID is expected to have separate English and
    Chinese trajectories.
    """
    buckets: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        key = (
            str(row["model_key"]),
            str(row["id"]),
            str(row["language"]),
        )
        buckets[key].append(row)

    return buckets


def validate_example_rows(
    example_rows: Sequence[dict[str, Any]],
) -> None:
    if not example_rows:
        raise ValueError("Cannot validate an empty example trajectory.")

    sorted_rows = sorted(
        example_rows,
        key=lambda row: int(row["layer"]),
    )

    layers = [int(row["layer"]) for row in sorted_rows]

    if len(layers) != len(set(layers)):
        raise AssertionError(
            "Duplicate layer rows for "
            f"model={sorted_rows[0]['model_key']}, "
            f"id={sorted_rows[0]['id']}, "
            f"language={sorted_rows[0]['language']}."
        )

    final_row = sorted_rows[-1]
    expected_final_layer = max(layers)

    if int(final_row["layer"]) != expected_final_layer:
        raise AssertionError("Final row is not the maximum layer.")

    final_is_target = as_bool(final_row["final_prediction_is_target"])

    if as_bool(final_row["target_in_top1"]) != final_is_target:
        raise AssertionError(
            "Final-layer Target@1 does not match "
            "final_prediction_is_target for "
            f"model={final_row['model_key']}, "
            f"id={final_row['id']}, "
            f"language={final_row['language']}."
        )

    for k in K_VALUES:
        faithful_key = f"faithful_final_at_{k}"

        if not as_bool(final_row[faithful_key]):
            raise AssertionError(
                f"Final-layer {faithful_key} must be True for "
                f"model={final_row['model_key']}, "
                f"id={final_row['id']}, "
                f"language={final_row['language']}."
            )

    if final_is_target:
        for row in sorted_rows:
            for k in K_VALUES:
                target_value = as_bool(row[f"target_in_top{k}"])
                faithful_value = as_bool(row[f"faithful_final_at_{k}"])

                if target_value != faithful_value:
                    raise AssertionError(
                        "Exact-gold trajectory mismatch: "
                        f"model={row['model_key']}, "
                        f"id={row['id']}, "
                        f"language={row['language']}, "
                        f"layer={row['layer']}, "
                        f"k={k}."
                    )


def analyze_one_example(
    example_rows: Sequence[dict[str, Any]],
    min_consecutive: int,
) -> dict[str, Any]:
    validate_example_rows(example_rows)

    rows = sorted(
        example_rows,
        key=lambda row: int(row["layer"]),
    )

    first_row = rows[0]
    final_row = rows[-1]

    layers = [int(row["layer"]) for row in rows]
    final_layer = max(layers)

    final_exact_gold = as_bool(final_row["final_prediction_is_target"])

    output: dict[str, Any] = {
        "model_key": str(first_row["model_key"]),
        "model_name": first_row.get("model_name"),
        "id": first_row["id"],
        "language": first_row["language"],
        "semantic_group": first_row["semantic_group"],
        "subgroup": first_row["subgroup"],
        "relation": first_row["relation"],
        "target": first_row["target"],
        "target_token_id": first_row["target_token_id"],
        "target_token": first_row.get("target_token"),
        "target_token_surface": first_row.get("target_token_surface"),
        "final_prediction_token_id": final_row["final_prediction_token_id"],
        "final_prediction_token": final_row.get("final_prediction_token"),
        "final_prediction_text": final_row.get("final_prediction_text"),
        "final_prediction_is_target": final_exact_gold,
        "final_match_status": (
            FINAL_EXACT_GOLD if final_exact_gold else FINAL_NON_GOLD
        ),
        "num_layers": final_layer,
        "n_layer_rows": len(rows),
        "min_consecutive": min_consecutive,
    }

    # Preserve optional dataset metadata.
    for field in OPTIONAL_METADATA_FIELDS:
        if field in first_row and field not in output:
            output[field] = first_row.get(field)

    for k in K_VALUES:
        target_key = f"target_in_top{k}"
        faithful_key = f"faithful_final_at_{k}"

        target_values = [as_bool(row[target_key]) for row in rows]
        final_values = [as_bool(row[faithful_key]) for row in rows]

        first_target = first_true_layer(
            target_values,
            layers,
        )
        first_final = first_true_layer(
            final_values,
            layers,
        )

        first_target_sustained = first_sustained_true_layer(
            target_values,
            layers,
            min_consecutive=min_consecutive,
        )
        first_final_sustained = first_sustained_true_layer(
            final_values,
            layers,
            min_consecutive=min_consecutive,
        )

        first_final_stable = first_stable_to_end_layer(
            final_values,
            layers,
        )

        onset_relation = compare_first_onsets(
            first_target,
            first_final,
        )
        sustained_onset_relation = compare_sustained_onsets(
            first_target_sustained,
            first_final_sustained,
        )

        coaccessible_values = [
            target_value and final_value
            for target_value, final_value in zip(
                target_values,
                final_values,
            )
        ]

        output.update(
            {
                f"target_top{k}_ever": any(target_values),
                f"target_top{k}_final": target_values[-1],
                f"target_top{k}_never": not any(target_values),
                f"first_target_top{k}_layer": first_target,
                f"first_target_top{k}_fraction": layer_fraction(
                    first_target,
                    final_layer,
                ),
                f"first_target_top{k}_sustained_layer": first_target_sustained,
                f"first_target_top{k}_sustained_fraction": layer_fraction(
                    first_target_sustained,
                    final_layer,
                ),
                f"last_target_top{k}_layer": last_true_layer(
                    target_values,
                    layers,
                ),
                f"target_top{k}_max_run": max_true_run_length(target_values),
                f"target_top{k}_num_runs": count_true_runs(target_values),
                f"target_top{k}_then_lost": true_then_lost(target_values),
                f"first_final_top{k}_layer": first_final,
                f"first_final_top{k}_fraction": layer_fraction(
                    first_final,
                    final_layer,
                ),
                f"first_final_top{k}_sustained_layer": first_final_sustained,
                f"first_final_top{k}_sustained_fraction": layer_fraction(
                    first_final_sustained,
                    final_layer,
                ),
                f"first_final_top{k}_stable_to_end_layer": first_final_stable,
                f"first_final_top{k}_stable_to_end_fraction": layer_fraction(
                    first_final_stable,
                    final_layer,
                ),
                f"final_top{k}_max_run": max_true_run_length(final_values),
                f"final_top{k}_num_runs": count_true_runs(final_values),
                f"final_top{k}_then_lost": true_then_lost(final_values),
                f"ever_coaccessible_top{k}": any(coaccessible_values),
                f"coaccessible_top{k}_layer_count": sum(coaccessible_values),
                f"onset_relation_top{k}": onset_relation,
                f"sustained_onset_relation_top{k}": sustained_onset_relation,
                f"first_gold_before_final_top{k}": onset_relation
                == "gold_before_final",
                f"first_final_before_gold_top{k}": onset_relation
                in {"final_before_gold", "final_only"},
                f"first_same_onset_top{k}": onset_relation == "same_onset",
                f"strict_sustained_gold_before_final_top{k}": sustained_onset_relation
                == "gold_before_final",
                f"strict_sustained_final_before_gold_top{k}": sustained_onset_relation
                in {
                    "final_before_gold",
                    "final_sustained_gold_not_sustained",
                },
                f"strict_sustained_same_onset_top{k}": sustained_onset_relation
                == "same_onset",
                # Backward-compatible aliases: these are FIRST-onset tags.
                f"gold_before_final_top{k}": onset_relation == "gold_before_final",
                f"final_before_gold_top{k}": onset_relation
                in {"final_before_gold", "final_only"},
                f"same_onset_top{k}": onset_relation == "same_onset",
            }
        )

    output["first_onset_pattern_top10"] = make_pattern_label(
        final_exact_gold=final_exact_gold,
        relation=output["onset_relation_top10"],
    )
    output["strict_sustained_pattern_top10"] = make_pattern_label(
        final_exact_gold=final_exact_gold,
        relation=output["sustained_onset_relation_top10"],
    )

    # Strict sustained onset only; no fallback to first onset.
    output["primary_pattern_top10"] = output["strict_sustained_pattern_top10"]

    # Convenient non-exclusive trajectory tags.
    output["gold_top1_then_lost"] = output["target_top1_then_lost"]
    output["gold_top10_then_lost"] = output["target_top10_then_lost"]
    output["gold_never_top10"] = output["target_top10_never"]
    output["gold_final_coaccessible_top10"] = output["ever_coaccessible_top10"]

    return output


def analyze_rows(
    rows: Sequence[dict[str, Any]],
    min_consecutive: int,
) -> list[dict[str, Any]]:
    buckets = group_layer_rows(rows)
    analyses: list[dict[str, Any]] = []

    for key in sorted(buckets):
        analyses.append(
            analyze_one_example(
                buckets[key],
                min_consecutive=min_consecutive,
            )
        )

    return analyses


# Aggregation of trajectory-level rows


def make_groupings() -> list[tuple[str, Callable[[dict[str, Any]], str]]]:
    return [
        (
            "overall",
            lambda row: "all",
        ),
        (
            "language",
            lambda row: str(row["language"]),
        ),
        (
            "semantic_group",
            lambda row: str(row["semantic_group"]),
        ),
        (
            "language_semantic_group",
            lambda row: f"{row['language']}|{row['semantic_group']}",
        ),
        (
            "final_match_status",
            lambda row: str(row["final_match_status"]),
        ),
        (
            "language_final_match_status",
            lambda row: f"{row['language']}|{row['final_match_status']}",
        ),
        (
            "semantic_group_final_match_status",
            lambda row: f"{row['semantic_group']}|{row['final_match_status']}",
        ),
        (
            "first_onset_pattern_top10",
            lambda row: str(row["first_onset_pattern_top10"]),
        ),
        (
            "strict_sustained_pattern_top10",
            lambda row: str(row["strict_sustained_pattern_top10"]),
        ),
        (
            "subgroup",
            lambda row: str(row["subgroup"]),
        ),
        (
            "relation",
            lambda row: str(row["relation"]),
        ),
    ]


def summarize_bucket(
    grouping: str,
    group_value: str,
    model_key: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "grouping": grouping,
        "group": group_value,
        "model_key": model_key,
        "n_examples": len(rows),
        "final_exact_gold_count": sum(
            as_bool(row["final_prediction_is_target"]) for row in rows
        ),
        "final_exact_gold_rate": rate(
            rows,
            "final_prediction_is_target",
        ),
        "final_non_gold_count": sum(
            not as_bool(row["final_prediction_is_target"]) for row in rows
        ),
    }

    for k in K_VALUES:
        boolean_metrics = [
            f"target_top{k}_ever",
            f"target_top{k}_final",
            f"target_top{k}_never",
            f"target_top{k}_then_lost",
            f"ever_coaccessible_top{k}",
            f"first_gold_before_final_top{k}",
            f"first_final_before_gold_top{k}",
            f"first_same_onset_top{k}",
            f"strict_sustained_gold_before_final_top{k}",
            f"strict_sustained_final_before_gold_top{k}",
            f"strict_sustained_same_onset_top{k}",
        ]

        for metric in boolean_metrics:
            metric_rate = rate(rows, metric)
            summary[f"{metric}_rate"] = metric_rate

        numeric_metrics = [
            f"first_target_top{k}_fraction",
            f"first_target_top{k}_sustained_fraction",
            f"first_final_top{k}_fraction",
            f"first_final_top{k}_sustained_fraction",
            f"first_final_top{k}_stable_to_end_fraction",
            f"target_top{k}_max_run",
            f"final_top{k}_max_run",
            f"coaccessible_top{k}_layer_count",
        ]

        for metric in numeric_metrics:
            values = [safe_float(row.get(metric)) for row in rows]
            summary[f"mean_{metric}"] = safe_mean(values)
            summary[f"median_{metric}"] = safe_median(values)

    return summary


def aggregate_trajectory_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for grouping_name, group_function in make_groupings():
        for row in rows:
            key = (
                grouping_name,
                group_function(row),
                str(row["model_key"]),
            )
            buckets[key].append(row)

    summaries: list[dict[str, Any]] = []

    for (
        grouping,
        group_value,
        model_key,
    ), bucket_rows in sorted(
        buckets.items(),
        key=lambda item: item[0],
    ):
        summaries.append(
            summarize_bucket(
                grouping=grouping,
                group_value=group_value,
                model_key=model_key,
                rows=bucket_rows,
            )
        )

    return summaries


def make_pattern_counts(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouping_specs = [
        ("overall", lambda row: "all"),
        ("language", lambda row: str(row["language"])),
        ("semantic_group", lambda row: str(row["semantic_group"])),
        (
            "language_semantic_group",
            lambda row: f"{row['language']}|{row['semantic_group']}",
        ),
        (
            "final_match_status",
            lambda row: str(row["final_match_status"]),
        ),
        (
            "language_semantic_group_final_match_status",
            lambda row: (
                f"{row['language']}|{row['semantic_group']}|{row['final_match_status']}"
            ),
        ),
    ]
    pattern_specs = [
        ("first_onset", "first_onset_pattern_top10"),
        ("strict_sustained", "strict_sustained_pattern_top10"),
    ]

    output: list[dict[str, Any]] = []

    for pattern_type, pattern_column in pattern_specs:
        for grouping_name, group_function in grouping_specs:
            buckets: dict[
                tuple[str, str, str],
                list[dict[str, Any]],
            ] = defaultdict(list)

            for row in rows:
                key = (
                    str(row["model_key"]),
                    group_function(row),
                    str(row[pattern_column]),
                )
                buckets[key].append(row)

            group_totals: dict[tuple[str, str], int] = defaultdict(int)
            for (model_key, group_value, _), pattern_rows in buckets.items():
                group_totals[(model_key, group_value)] += len(pattern_rows)

            for (model_key, group_value, pattern), pattern_rows in sorted(
                buckets.items(),
                key=lambda item: item[0],
            ):
                total = group_totals[(model_key, group_value)]
                output.append(
                    {
                        "pattern_type": pattern_type,
                        "pattern_column": pattern_column,
                        "grouping": grouping_name,
                        "group": group_value,
                        "model_key": model_key,
                        "pattern_top10": pattern,
                        "count": len(pattern_rows),
                        "rate_within_group": (
                            len(pattern_rows) / total if total else None
                        ),
                        "group_total": total,
                    }
                )

    return output


def _stable_sample_score(
    row: dict[str, Any],
    seed: int,
) -> str:
    payload = "|".join(
        [
            str(seed),
            str(row.get("model_key", "")),
            str(row.get("language", "")),
            str(row.get("semantic_group", "")),
            str(row.get("strict_sustained_pattern_top10", "")),
            str(row.get("final_match_status", "")),
            str(row.get("id", "")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_representative_examples(
    rows: Sequence[dict[str, Any]],
    examples_per_pattern: int,
    sampling_seed: int,
) -> list[dict[str, Any]]:
    """
    Select deterministic qualitative examples stratified by:

        model × language × semantic_group × strict_sustained_pattern
        × final_match_status

    This avoids the old semantic-group bias caused by sorting and head().
    """
    buckets: dict[
        tuple[str, str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        key = (
            str(row["model_key"]),
            str(row["language"]),
            str(row["semantic_group"]),
            str(row["strict_sustained_pattern_top10"]),
            str(row["final_match_status"]),
        )
        buckets[key].append(row)

    selected: list[dict[str, Any]] = []

    for key in sorted(buckets):
        candidates = sorted(
            buckets[key],
            key=lambda row: (_stable_sample_score(row, sampling_seed), str(row["id"])),
        )

        for row in candidates[:examples_per_pattern]:
            selected_row = {
                "model_key": row["model_key"],
                "id": row["id"],
                "language": row["language"],
                "semantic_group": row["semantic_group"],
                "subgroup": row["subgroup"],
                "relation": row["relation"],
                "target": row["target"],
                "final_prediction_text": row.get("final_prediction_text"),
                "final_match_status": row["final_match_status"],
                "first_onset_pattern_top10": row["first_onset_pattern_top10"],
                "strict_sustained_pattern_top10": row["strict_sustained_pattern_top10"],
                "first_target_top10_layer": row["first_target_top10_layer"],
                "first_final_top10_layer": row["first_final_top10_layer"],
                "first_target_top10_sustained_layer": row[
                    "first_target_top10_sustained_layer"
                ],
                "first_final_top10_sustained_layer": row[
                    "first_final_top10_sustained_layer"
                ],
                "first_final_top1_stable_to_end_layer": row[
                    "first_final_top1_stable_to_end_layer"
                ],
                "ever_coaccessible_top10": row["ever_coaccessible_top10"],
                "gold_top10_then_lost": row["gold_top10_then_lost"],
                "gold_never_top10": row["gold_never_top10"],
                "sampling_seed": sampling_seed,
            }

            for field in OPTIONAL_METADATA_FIELDS:
                if field in row:
                    selected_row[field] = row.get(field)

            selected.append(selected_row)

    return selected


# Reporting


def print_model_report(model_key: str, rows: Sequence[dict[str, Any]]) -> None:
    exact = sum(as_bool(row["final_prediction_is_target"]) for row in rows)
    print(
        f"{model_key}: {len(rows)} trajectories, {exact} exact-gold, "
        f"{len(rows) - exact} non-gold"
    )


# Main processing


def run_model(
    input_path: Path,
    output_dir: Path,
    min_consecutive: int,
    examples_per_pattern: int,
    sampling_seed: int,
    expected_model: Optional[str] = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw_rows = read_jsonl(input_path)
    validate_required_fields(raw_rows, input_path)

    trajectory_rows = analyze_rows(
        raw_rows,
        min_consecutive=min_consecutive,
    )

    model_keys = sorted({str(row["model_key"]) for row in trajectory_rows})

    if len(model_keys) != 1:
        raise ValueError(f"Expected one model in {input_path}, found: {model_keys}")

    model_key = model_keys[0]
    if expected_model is not None and model_key != expected_model:
        raise ValueError(
            f"Expected {expected_model} in {input_path}, found {model_key}"
        )
    summary_rows = aggregate_trajectory_rows(trajectory_rows)
    pattern_rows = make_pattern_counts(trajectory_rows)
    representative_rows = select_representative_examples(
        trajectory_rows,
        examples_per_pattern=examples_per_pattern,
        sampling_seed=sampling_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    write_tsv(
        output_dir / "layer_trajectory_examples.tsv",
        trajectory_rows,
    )
    write_jsonl(
        output_dir / "layer_trajectory_examples.jsonl",
        trajectory_rows,
    )
    write_tsv(
        output_dir / "layer_trajectory_summary.tsv",
        summary_rows,
    )
    write_tsv(
        output_dir / "trajectory_pattern_counts.tsv",
        pattern_rows,
    )
    write_tsv(
        output_dir / "representative_trajectory_examples.tsv",
        representative_rows,
    )

    print_model_report(
        model_key=model_key,
        rows=trajectory_rows,
    )

    return trajectory_rows, summary_rows, pattern_rows


# CLI


def check_outputs(output_dir: Path, filenames: Sequence[str], overwrite: bool) -> None:
    existing = [output_dir / name for name in filenames if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Existing outputs; use --overwrite or a different --output-dir: "
            + ", ".join(str(path) for path in existing)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze layer-wise Logit Lens trajectories."
    )
    parser.add_argument(
        "--model", "--models", nargs="+", default=["all"], choices=[*MODEL_KEYS, "all"]
    )
    parser.add_argument(
        "--input-dir",
        "--base_input_dir",
        "--base-input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        "--base_output_dir",
        "--base-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root (exact output directory in --input mode if customized).",
    )
    parser.add_argument("--input", type=Path, help="Analyze one explicit JSONL file.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files."
    )
    parser.add_argument("--input-filename", "--input_filename", default=INPUT_FILENAME)
    parser.add_argument(
        "--min-consecutive",
        "--min_consecutive",
        type=int,
        default=DEFAULT_MIN_CONSECUTIVE,
    )
    parser.add_argument(
        "--examples-per-pattern",
        "--examples_per_pattern",
        type=int,
        default=DEFAULT_EXAMPLES_PER_PATTERN,
    )
    parser.add_argument(
        "--sampling-seed", "--sampling_seed", type=int, default=DEFAULT_SAMPLING_SEED
    )
    parser.add_argument(
        "--require-all-models", "--require_all_models", action="store_true"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_consecutive < 1:
        parser.error("--min-consecutive must be positive")
    if args.examples_per_pattern < 0:
        parser.error("--examples-per-pattern must be non-negative")
    if "all" in args.model and args.model != ["all"]:
        parser.error("--model all cannot be combined with individual models")

    output_dir = args.output_dir
    nonstandard = (
        args.min_consecutive != DEFAULT_MIN_CONSECUTIVE
        or args.examples_per_pattern != DEFAULT_EXAMPLES_PER_PATTERN
        or args.sampling_seed != DEFAULT_SAMPLING_SEED
    )
    if nonstandard and (
        output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
        or DEFAULT_OUTPUT_DIR.resolve() in output_dir.resolve().parents
    ):
        parser.error("Nonstandard analysis requires a separate --output-dir.")

    settings = dict(
        min_consecutive=args.min_consecutive,
        examples_per_pattern=args.examples_per_pattern,
        sampling_seed=args.sampling_seed,
    )
    model_keys = [
        key for key in MODEL_KEYS if args.model == ["all"] or key in args.model
    ]
    if args.input is not None:
        observed = {str(row["model_key"]) for row in read_jsonl(args.input)}
        if len(observed) != 1 or not observed.issubset(model_keys):
            parser.error(
                f"Expected one selected model in --input, found {sorted(observed)}"
            )
        model_key = next(iter(observed))
        if output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
            output_dir = output_dir / model_key
        check_outputs(output_dir, OUTPUT_FILENAMES, args.overwrite)
        run_model(args.input, output_dir, expected_model=model_key, **settings)
        print(f"Output: {output_dir}")
        return 0

    inputs = [(key, args.input_dir / key / args.input_filename) for key in model_keys]
    missing = [path for _, path in inputs if not path.is_file()]
    if missing and (args.require_all_models or args.model != ["all"]):
        raise FileNotFoundError(f"Missing model inputs: {missing}")

    for model_key, input_path in inputs:
        if input_path.is_file():
            check_outputs(output_dir / model_key, OUTPUT_FILENAMES, args.overwrite)
    if set(model_keys) == set(MODEL_KEYS) and not missing:
        check_outputs(output_dir, COMBINED_OUTPUT_FILENAMES, args.overwrite)

    all_trajectory_rows = []
    all_summary_rows = []
    all_pattern_rows = []
    for model_key, input_path in inputs:
        if not input_path.is_file():
            print(f"Warning: missing input, skipping {model_key}: {input_path}")
            continue
        trajectories, summaries, patterns = run_model(
            input_path,
            args.output_dir / model_key,
            expected_model=model_key,
            **settings,
        )
        all_trajectory_rows.extend(trajectories)
        all_summary_rows.extend(summaries)
        all_pattern_rows.extend(patterns)

    if not all_trajectory_rows:
        raise FileNotFoundError("No model inputs were processed.")

    # Only a complete three-model run may replace the cross-model outputs.
    if set(model_keys) != set(MODEL_KEYS) or missing:
        print("Combined outputs unchanged (incomplete model selection).")
        print(f"Output: {output_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        output_dir / "all_models_layer_trajectory_examples.tsv", all_trajectory_rows
    )
    write_jsonl(
        output_dir / "all_models_layer_trajectory_examples.jsonl", all_trajectory_rows
    )
    write_tsv(output_dir / "all_models_layer_trajectory_summary.tsv", all_summary_rows)
    write_tsv(output_dir / "all_models_trajectory_pattern_counts.tsv", all_pattern_rows)
    representatives = select_representative_examples(
        all_trajectory_rows,
        args.examples_per_pattern,
        args.sampling_seed,
    )
    write_tsv(
        output_dir / "all_models_representative_trajectory_examples.tsv",
        representatives,
    )
    print(
        f"Total: {len(all_trajectory_rows)} trajectories across {len(model_keys)} models"
    )
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
