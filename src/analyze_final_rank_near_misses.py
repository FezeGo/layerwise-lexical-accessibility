"""
Recommended command
-------------------
python src/analyze_final_rank_near_misses.py \
  --base_input_dir data/logit_lens/full \
  --trajectory results/analysis/trajectory/all_models_layer_trajectory_examples.tsv \
  --output_dir results/analysis/final_rank_near_misses

You can also provide explicit raw files:

python src/analyze_final_rank_near_misses.py \
  --input path/to/0.6b.jsonl path/to/4b.jsonl path/to/14b.jsonl \
  --trajectory path/to/all_models_layer_trajectory_examples.tsv
_rank_near_miss_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence


# ============================================================
# Defaults
# ============================================================

DEFAULT_BASE_INPUT_DIR = Path("data/logit_lens/full")
DEFAULT_OUTPUT_DIR = Path("data/analysis/final_rank_near_misses")
DEFAULT_MODELS = [
    "qwen3_0_6b",
    "qwen3_4b",
    "qwen3_14b",
]
DEFAULT_INPUT_FILENAME = "logit_lens_by_layer.jsonl"

DEFAULT_TRAJECTORY_CANDIDATES = [
    Path("data/analysis/trajectory/all_models_layer_trajectory_examples.tsv"),
    Path("data/analysis/full/all_models_layer_trajectory_examples.tsv"),
]

RANK_MIN = 11
RANK_MAX = 30
MIN_CONSECUTIVE = 3
EXPECTED_FINAL_ROWS_PER_MODEL = 1040
DEFAULT_SAMPLE_PER_STRATUM = 1
DEFAULT_SAMPLING_SEED = 42

TRAJECTORY_FIELDS_TO_MERGE = (
    "final_match_status",
    "first_onset_pattern_top10",
    "strict_sustained_pattern_top10",
    "target_top10_ever",
    "target_top10_final",
    "target_top10_never",
    "first_target_top10_layer",
    "first_target_top10_fraction",
    "first_target_top10_sustained_layer",
    "first_target_top10_sustained_fraction",
    "last_target_top10_layer",
    "target_top10_max_run",
    "target_top10_num_runs",
    "target_top10_then_lost",
    "gold_top10_then_lost",
    "gold_never_top10",
    "ever_coaccessible_top10",
    "first_final_top10_layer",
    "first_final_top10_sustained_layer",
    "first_final_top1_stable_to_end_layer",
    # Preserve optional text metadata if an enriched trajectory table has it.
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


# ============================================================
# Generic I/O
# ============================================================


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            row["_source_file"] = str(path)
            row["_source_line"] = line_no
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"TSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def ordered_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key.startswith("_") or key in seen:
                continue
            seen.add(key)
            fields.append(key)
    return fields


def tsv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = ordered_fields(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: tsv_value(row.get(field, "")) for field in fields})


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    serializable_row(row),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            obj,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


# ============================================================
# Conversion helpers
# ============================================================


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
    raise ValueError(f"Cannot interpret Boolean value: {value!r}")


def optional_bool(value: Any) -> Optional[bool]:
    if value is None or str(value).strip() == "":
        return None
    return as_bool(value)


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number)


def normalise_id(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        return text[:-2]
    return text


def trajectory_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["model_key"]),
        normalise_id(row["id"]),
        str(row["language"]),
    )


def proportion(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


# ============================================================
# Raw trajectory validation
# ============================================================

REQUIRED_RAW_FIELDS = {
    "id",
    "model_key",
    "model_name",
    "language",
    "semantic_group",
    "subgroup",
    "relation",
    "layer",
    "num_layers",
    "layer_name",
    "target",
    "target_token_id",
    "target_rank",
    "target_in_top10",
    "final_prediction_token_id",
    "final_prediction_text",
    "final_prediction_is_target",
    "topk_token_ids",
    "topk_texts",
}


def validate_raw_schema(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        missing = REQUIRED_RAW_FIELDS - set(row)
        if missing:
            raise ValueError(
                f"Missing fields at {row.get('_source_file')}:"
                f"{row.get('_source_line')}: {sorted(missing)}"
            )


def group_raw_trajectories(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[trajectory_key(row)].append(row)

    for key, trajectory in grouped.items():
        trajectory.sort(key=lambda row: int(row["layer"]))

        layers = [int(row["layer"]) for row in trajectory]
        if len(layers) != len(set(layers)):
            raise ValueError(f"Duplicate layer rows for {key}")

        num_layers_values = {int(row["num_layers"]) for row in trajectory}
        if len(num_layers_values) != 1:
            raise ValueError(f"Inconsistent num_layers for {key}: {num_layers_values}")

        num_layers = next(iter(num_layers_values))
        expected = list(range(0, num_layers + 1))
        if layers != expected:
            raise ValueError(
                f"Incomplete/non-contiguous trajectory for {key}: "
                f"expected 0..{num_layers}, observed "
                f"{layers[:5]}...{layers[-5:]}"
            )

        final = trajectory[-1]
        if int(final["layer"]) != num_layers:
            raise ValueError(f"Final layer mismatch for {key}")

        if str(final.get("layer_name", "")) not in {
            "final",
            f"layer_{num_layers}",
        }:
            # Historical files may label the final state layer_N rather than
            # "final"; both are acceptable.
            raise ValueError(
                f"Unexpected final layer_name for {key}: {final.get('layer_name')!r}"
            )

        if as_bool(final["target_in_top10"]):
            if int(final["target_rank"]) > 10:
                raise ValueError(f"Final Target@10=True but target_rank>10 for {key}")

        if as_bool(final["final_prediction_is_target"]) != as_bool(
            final.get("target_in_top1", final["final_prediction_is_target"])
        ):
            raise ValueError(
                f"Final Target@1/final_prediction_is_target mismatch for {key}"
            )

    return dict(grouped)


# ============================================================
# Top-10 history
# ============================================================


def first_true_layer(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> Optional[int]:
    for row in rows:
        if as_bool(row[field]):
            return int(row["layer"])
    return None


def last_true_layer(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> Optional[int]:
    for row in reversed(rows):
        if as_bool(row[field]):
            return int(row["layer"])
    return None


def first_sustained_layer(
    rows: Sequence[dict[str, Any]],
    field: str,
    min_consecutive: int,
) -> Optional[int]:
    run = 0
    start: Optional[int] = None
    previous: Optional[int] = None

    for row in rows:
        layer = int(row["layer"])
        if as_bool(row[field]):
            if previous is not None and layer == previous + 1:
                run += 1
            else:
                start = layer
                run = 1
            previous = layer
            if run >= min_consecutive:
                return start
        else:
            run = 0
            start = None
            previous = None

    return None


def max_true_run(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> int:
    current = 0
    best = 0
    previous: Optional[int] = None

    for row in rows:
        layer = int(row["layer"])
        if as_bool(row[field]):
            current = (
                current + 1 if previous is not None and layer == previous + 1 else 1
            )
            previous = layer
            best = max(best, current)
        else:
            current = 0
            previous = None

    return best


def count_true_runs(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> int:
    runs = 0
    previous = False
    for row in rows:
        value = as_bool(row[field])
        if value and not previous:
            runs += 1
        previous = value
    return runs


def rank_bin(rank: int) -> str:
    if 11 <= rank <= 15:
        return "11-15"
    if 16 <= rank <= 20:
        return "16-20"
    if 21 <= rank <= 30:
        return "21-30"
    return "outside_default_bins"


def build_near_miss_case(
    trajectory: Sequence[dict[str, Any]],
    min_consecutive: int,
) -> dict[str, Any]:
    rows = list(trajectory)
    first = rows[0]
    final = rows[-1]

    final_rank = int(final["target_rank"])
    final_top10 = as_bool(final["target_in_top10"])

    ever_top10 = any(as_bool(row["target_in_top10"]) for row in rows)
    first_top10 = first_true_layer(rows, "target_in_top10")
    last_top10 = last_true_layer(rows, "target_in_top10")
    sustained_onset = first_sustained_layer(
        rows,
        "target_in_top10",
        min_consecutive,
    )

    if not ever_top10:
        history_category = "never_top10"
    elif sustained_onset is None:
        history_category = "transient_top10_no_sustained"
    else:
        # A final near miss is outside Top-10 by definition, so any sustained
        # Top-10 period must have been lost before the final layer.
        history_category = "sustained_top10_then_lost"

    rank_pairs = [(int(row["target_rank"]), int(row["layer"])) for row in rows]
    best_rank, best_rank_layer = min(rank_pairs, key=lambda item: item[0])

    topk_texts = list(final.get("topk_texts", []))
    topk_ids = list(final.get("topk_token_ids", []))
    topk_tokens = list(final.get("topk_tokens", []))
    topk_logits = list(final.get("topk_logits", []))

    case = {
        "model_key": str(final["model_key"]),
        "model_name": final.get("model_name", ""),
        "id": normalise_id(final["id"]),
        "language": str(final["language"]),
        "semantic_group": str(final["semantic_group"]),
        "subgroup": str(final["subgroup"]),
        "relation": str(final["relation"]),
        "target": final["target"],
        "target_token_id": int(final["target_token_id"]),
        "target_token": final.get("target_token", ""),
        "target_token_surface": final.get("target_token_surface", ""),
        "final_target_rank": final_rank,
        "rank_bin": rank_bin(final_rank),
        "distance_below_top10": final_rank - 10,
        "final_target_in_top10": final_top10,
        "final_target_in_top5": as_bool(final.get("target_in_top5", False)),
        "final_target_in_top1": as_bool(final.get("target_in_top1", False)),
        "final_target_logit": final.get("target_logit"),
        "final_target_prob": final.get("target_prob"),
        "final_prediction_token_id": int(final["final_prediction_token_id"]),
        "final_prediction_token": final.get("final_prediction_token", ""),
        "final_prediction_text": final.get("final_prediction_text", ""),
        "final_prediction_is_target": as_bool(final["final_prediction_is_target"]),
        "final_top10_token_ids": topk_ids[:10],
        "final_top10_tokens": topk_tokens[:10],
        "final_top10_texts": topk_texts[:10],
        "final_top10_logits": topk_logits[:10],
        "final_top10_texts_display": " | ".join(str(text) for text in topk_texts[:10]),
        "num_layers": int(final["num_layers"]),
        "n_layer_rows": len(rows),
        "ever_top10": ever_top10,
        "first_top10_layer": first_top10,
        "first_top10_fraction": (
            first_top10 / int(final["num_layers"])
            if first_top10 is not None and int(final["num_layers"]) > 0
            else None
        ),
        "last_top10_layer": last_top10,
        "last_top10_fraction": (
            last_top10 / int(final["num_layers"])
            if last_top10 is not None and int(final["num_layers"]) > 0
            else None
        ),
        "first_sustained_top10_layer": sustained_onset,
        "first_sustained_top10_fraction": (
            sustained_onset / int(final["num_layers"])
            if sustained_onset is not None and int(final["num_layers"]) > 0
            else None
        ),
        "top10_max_run": max_true_run(rows, "target_in_top10"),
        "top10_num_runs": count_true_runs(rows, "target_in_top10"),
        "history_category": history_category,
        "best_rank_across_layers": best_rank,
        "best_rank_layer": best_rank_layer,
        "best_rank_fraction": (
            best_rank_layer / int(final["num_layers"])
            if int(final["num_layers"]) > 0
            else None
        ),
        "final_minus_best_rank": final_rank - best_rank,
    }

    # Preserve text metadata if present in the raw extraction.
    for field in (
        "subject",
        "prefix",
        "full_sentence",
        "source",
    ):
        if field in first:
            case[field] = first.get(field)

    return case


# ============================================================
# Existing trajectory table merge / cross-check
# ============================================================


def find_default_trajectory() -> Optional[Path]:
    for path in DEFAULT_TRAJECTORY_CANDIDATES:
        if path.exists():
            return path
    return None


def build_trajectory_index(
    rows: Sequence[dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    required = {"model_key", "id", "language"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Trajectory TSV is missing required columns: {sorted(missing)}"
        )

    index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = trajectory_key(row)
        if key in index:
            raise ValueError(f"Duplicate trajectory TSV key: {key}")
        index[key] = row
    return index


def bool_equal(raw_value: bool, trajectory_value: Any) -> bool:
    parsed = optional_bool(trajectory_value)
    return parsed is None or raw_value == parsed


def merge_trajectory_metadata(
    cases: list[dict[str, Any]],
    trajectory_index: dict[tuple[str, str, str], dict[str, str]],
) -> dict[str, Any]:
    missing_keys: list[tuple[str, str, str]] = []
    mismatch_counts = Counter()

    for case in cases:
        key = trajectory_key(case)
        trajectory = trajectory_index.get(key)
        if trajectory is None:
            missing_keys.append(key)
            continue

        for field in TRAJECTORY_FIELDS_TO_MERGE:
            if field in trajectory and str(trajectory[field]).strip() != "":
                case[f"trajectory_{field}"] = trajectory[field]

        # Cross-check the fields most directly relevant to this analysis.
        checks = {
            "target_top10_ever": case["ever_top10"],
            "target_top10_never": not case["ever_top10"],
            "target_top10_then_lost": (
                case["ever_top10"] and not case["final_target_in_top10"]
            ),
            "gold_top10_then_lost": (
                case["ever_top10"] and not case["final_target_in_top10"]
            ),
            "gold_never_top10": not case["ever_top10"],
        }

        for field, raw_value in checks.items():
            if field in trajectory and str(trajectory[field]).strip() != "":
                if not bool_equal(raw_value, trajectory[field]):
                    mismatch_counts[field] += 1

        numeric_checks = {
            "first_target_top10_layer": case["first_top10_layer"],
            "first_target_top10_sustained_layer": case["first_sustained_top10_layer"],
            "last_target_top10_layer": case["last_top10_layer"],
        }

        for field, raw_value in numeric_checks.items():
            if field in trajectory and str(trajectory[field]).strip() != "":
                trajectory_value = safe_int(trajectory[field])
                if trajectory_value != raw_value:
                    mismatch_counts[field] += 1

    return {
        "missing_case_keys": len(missing_keys),
        "missing_case_keys_preview": [list(key) for key in missing_keys[:10]],
        "crosscheck_mismatch_counts": dict(mismatch_counts),
    }


# ============================================================
# Summaries
# ============================================================

GROUPING_SPECS: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
    ("overall", lambda row: "all"),
    ("model", lambda row: str(row["model_key"])),
    ("language", lambda row: str(row["language"])),
    ("semantic_group", lambda row: str(row["semantic_group"])),
    (
        "model_language",
        lambda row: f"{row['model_key']}|{row['language']}",
    ),
    (
        "model_semantic_group",
        lambda row: f"{row['model_key']}|{row['semantic_group']}",
    ),
    (
        "language_semantic_group",
        lambda row: f"{row['language']}|{row['semantic_group']}",
    ),
    (
        "model_language_semantic_group",
        lambda row: f"{row['model_key']}|{row['language']}|{row['semantic_group']}",
    ),
]


def make_summary(
    final_rows: Sequence[dict[str, Any]],
    near_miss_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for grouping, group_fn in GROUPING_SPECS:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in final_rows:
            buckets[group_fn(row)].append(row)

        for group, rows in sorted(buckets.items()):
            n = len(rows)
            failures = [row for row in rows if not as_bool(row["target_in_top10"])]
            near = [row for row in rows if trajectory_key(row) in near_miss_keys]

            output.append(
                {
                    "grouping": grouping,
                    "group": group,
                    "n_final_trajectories": n,
                    "final_target_at10_count": n - len(failures),
                    "final_target_at10_rate": proportion(n - len(failures), n),
                    "final_target_at10_failure_count": len(failures),
                    "final_target_at10_failure_rate": proportion(len(failures), n),
                    "near_miss_rank_11_30_count": len(near),
                    "near_miss_rank_11_30_rate_all": proportion(len(near), n),
                    "near_miss_share_of_target_at10_failures": proportion(
                        len(near),
                        len(failures),
                    ),
                }
            )

    return output


def make_rank_bin_summary(
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    specs = [
        ("overall", lambda row: "all"),
        ("model", lambda row: str(row["model_key"])),
        ("language", lambda row: str(row["language"])),
        ("semantic_group", lambda row: str(row["semantic_group"])),
        (
            "model_language",
            lambda row: f"{row['model_key']}|{row['language']}",
        ),
        (
            "model_semantic_group",
            lambda row: f"{row['model_key']}|{row['semantic_group']}",
        ),
    ]

    for grouping, group_fn in specs:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cases:
            buckets[group_fn(row)].append(row)

        for group, rows in sorted(buckets.items()):
            counts = Counter(str(row["rank_bin"]) for row in rows)
            for bin_name in ("11-15", "16-20", "21-30"):
                count = counts.get(bin_name, 0)
                output.append(
                    {
                        "grouping": grouping,
                        "group": group,
                        "rank_bin": bin_name,
                        "count": count,
                        "near_miss_total": len(rows),
                        "rate_within_near_misses": proportion(count, len(rows)),
                    }
                )

    return output


def make_history_summary(
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    history_order = (
        "never_top10",
        "transient_top10_no_sustained",
        "sustained_top10_then_lost",
    )

    specs = [
        ("overall", lambda row: "all"),
        ("model", lambda row: str(row["model_key"])),
        ("language", lambda row: str(row["language"])),
        ("semantic_group", lambda row: str(row["semantic_group"])),
        (
            "model_language",
            lambda row: f"{row['model_key']}|{row['language']}",
        ),
        (
            "model_semantic_group",
            lambda row: f"{row['model_key']}|{row['semantic_group']}",
        ),
    ]

    for grouping, group_fn in specs:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cases:
            buckets[group_fn(row)].append(row)

        for group, rows in sorted(buckets.items()):
            counts = Counter(str(row["history_category"]) for row in rows)
            for history in history_order:
                count = counts.get(history, 0)
                output.append(
                    {
                        "grouping": grouping,
                        "group": group,
                        "history_category": history,
                        "count": count,
                        "near_miss_total": len(rows),
                        "rate_within_near_misses": proportion(count, len(rows)),
                    }
                )

    return output


# ============================================================
# Deterministic qualitative sample
# ============================================================


def sample_score(row: dict[str, Any], seed: int) -> str:
    payload = "|".join(
        [
            str(seed),
            str(row["model_key"]),
            str(row["language"]),
            str(row["semantic_group"]),
            str(row["rank_bin"]),
            str(row["id"]),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def representative_sample(
    cases: Sequence[dict[str, Any]],
    per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    if per_stratum <= 0:
        return []

    buckets: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in cases:
        key = (
            str(row["model_key"]),
            str(row["language"]),
            str(row["semantic_group"]),
            str(row["rank_bin"]),
        )
        buckets[key].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        candidates = sorted(
            buckets[key],
            key=lambda row: (
                sample_score(row, seed),
                str(row["id"]),
            ),
        )
        for row in candidates[:per_stratum]:
            sampled = dict(row)
            sampled["sampling_seed"] = seed
            sampled["sampling_stratum"] = "|".join(key)
            selected.append(sampled)

    return selected


# ============================================================
# CLI
# ============================================================


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze final gold-target ranks 11--30 among main Logit Lens "
            "outputs and relate them to the operational Target@10 threshold."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        help="Explicit logit_lens_by_layer.jsonl files.",
    )
    parser.add_argument(
        "--base_input_dir",
        type=Path,
        default=DEFAULT_BASE_INPUT_DIR,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--input_filename",
        default=DEFAULT_INPUT_FILENAME,
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help=(
            "Optional all-model trajectory TSV. If omitted, common project "
            "locations are auto-detected."
        ),
    )
    parser.add_argument(
        "--no_trajectory_merge",
        action="store_true",
        help="Skip trajectory-table merge/cross-check.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--rank_min",
        type=int,
        default=RANK_MIN,
    )
    parser.add_argument(
        "--rank_max",
        type=int,
        default=RANK_MAX,
    )
    parser.add_argument(
        "--min_consecutive",
        type=int,
        default=MIN_CONSECUTIVE,
    )
    parser.add_argument(
        "--expected_final_rows_per_model",
        type=int,
        default=EXPECTED_FINAL_ROWS_PER_MODEL,
        help="Set to 0 to disable the 1040-row validation.",
    )
    parser.add_argument(
        "--sample_per_stratum",
        type=int,
        default=DEFAULT_SAMPLE_PER_STRATUM,
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=DEFAULT_SAMPLING_SEED,
    )

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(
            "Ignoring unrecognized runtime arguments: " + " ".join(unknown),
            file=sys.stderr,
        )
    return args


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [Path(path) for path in args.input]

    paths = [
        Path(args.base_input_dir) / model / args.input_filename for model in args.models
    ]

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw Logit Lens inputs:\n  "
            + "\n  ".join(str(path) for path in missing)
        )
    return paths


# ============================================================
# Main
# ============================================================


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    args = parse_args(argv)

    if args.rank_min <= 10:
        raise ValueError("--rank_min should be >10 for a Target@10 near miss.")
    if args.rank_max < args.rank_min:
        raise ValueError("--rank_max must be >= --rank_min.")
    if args.min_consecutive < 1:
        raise ValueError("--min_consecutive must be >=1.")

    input_paths = resolve_inputs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_raw_rows: list[dict[str, Any]] = []
    raw_rows_by_file: dict[str, int] = {}

    for path in input_paths:
        print(f"Loading {path} ...", file=sys.stderr)
        rows = load_jsonl(path)
        validate_raw_schema(rows)
        raw_rows_by_file[str(path)] = len(rows)
        all_raw_rows.extend(rows)

    trajectories = group_raw_trajectories(all_raw_rows)

    model_trajectory_counts = Counter(key[0] for key in trajectories)

    if args.expected_final_rows_per_model > 0:
        bad = {
            model: count
            for model, count in model_trajectory_counts.items()
            if count != args.expected_final_rows_per_model
        }
        if bad:
            raise ValueError(
                "Unexpected trajectory count(s) per model; expected "
                f"{args.expected_final_rows_per_model}: {bad}"
            )

    final_rows = [trajectory[-1] for _, trajectory in sorted(trajectories.items())]

    cases: list[dict[str, Any]] = []

    for key, trajectory in sorted(trajectories.items()):
        final = trajectory[-1]
        final_rank = int(final["target_rank"])

        if args.rank_min <= final_rank <= args.rank_max:
            if as_bool(final["target_in_top10"]):
                raise ValueError(
                    f"Near-miss candidate unexpectedly has Target@10=True: "
                    f"{key}, rank={final_rank}"
                )
            cases.append(
                build_near_miss_case(
                    trajectory,
                    min_consecutive=args.min_consecutive,
                )
            )

    near_miss_keys = {trajectory_key(row) for row in cases}

    # Optional existing trajectory merge.
    trajectory_path: Optional[Path] = None
    trajectory_merge_report: dict[str, Any] = {
        "used": False,
        "path": None,
    }

    if not args.no_trajectory_merge:
        if args.trajectory is not None:
            trajectory_path = Path(args.trajectory)
        else:
            trajectory_path = find_default_trajectory()

        if trajectory_path is not None:
            trajectory_rows = load_tsv(trajectory_path)
            trajectory_index = build_trajectory_index(trajectory_rows)
            merge_report = merge_trajectory_metadata(
                cases,
                trajectory_index,
            )
            trajectory_merge_report = {
                "used": True,
                "path": str(trajectory_path),
                "trajectory_rows": len(trajectory_rows),
                **merge_report,
            }

    # Summaries.
    summary_rows = make_summary(
        final_rows,
        near_miss_keys,
    )
    rank_bin_rows = make_rank_bin_summary(cases)
    history_rows = make_history_summary(cases)
    sample_rows = representative_sample(
        cases,
        per_stratum=args.sample_per_stratum,
        seed=args.sampling_seed,
    )

    # Stable sort for review tables.
    model_order = {
        "qwen3_0_6b": 0,
        "qwen3_4b": 1,
        "qwen3_14b": 2,
    }
    cases.sort(
        key=lambda row: (
            model_order.get(str(row["model_key"]), 99),
            str(row["language"]),
            str(row["semantic_group"]),
            int(row["final_target_rank"]),
            str(row["id"]),
        )
    )

    # Write outputs.
    write_tsv(
        output_dir / "final_rank_near_miss_cases.tsv",
        cases,
    )
    write_jsonl(
        output_dir / "final_rank_near_miss_cases.jsonl",
        cases,
    )
    write_tsv(
        output_dir / "final_rank_near_miss_summary.tsv",
        summary_rows,
    )
    write_tsv(
        output_dir / "final_rank_near_miss_rank_bins.tsv",
        rank_bin_rows,
    )
    write_tsv(
        output_dir / "final_rank_near_miss_history_summary.tsv",
        history_rows,
    )
    write_tsv(
        output_dir / "final_rank_near_miss_representative_sample.tsv",
        sample_rows,
    )

    # Headline report.
    total_final = len(final_rows)
    total_failures = sum(not as_bool(row["target_in_top10"]) for row in final_rows)
    total_near = len(cases)

    by_model = {}
    for model in sorted(model_trajectory_counts, key=lambda x: model_order.get(x, 99)):
        model_final = [row for row in final_rows if str(row["model_key"]) == model]
        model_cases = [row for row in cases if str(row["model_key"]) == model]
        model_failures = sum(not as_bool(row["target_in_top10"]) for row in model_final)
        by_model[model] = {
            "n_final_trajectories": len(model_final),
            "target_at10_failure_count": model_failures,
            "near_miss_rank_11_30_count": len(model_cases),
            "near_miss_rate_all": proportion(
                len(model_cases),
                len(model_final),
            ),
            "near_miss_share_of_target_at10_failures": proportion(
                len(model_cases),
                model_failures,
            ),
            "rank_bin_counts": dict(
                Counter(str(row["rank_bin"]) for row in model_cases)
            ),
            "language_counts": dict(
                Counter(str(row["language"]) for row in model_cases)
            ),
            "semantic_group_counts": dict(
                Counter(str(row["semantic_group"]) for row in model_cases)
            ),
            "history_counts": dict(
                Counter(str(row["history_category"]) for row in model_cases)
            ),
            "ever_top10_count": sum(bool(row["ever_top10"]) for row in model_cases),
            "strict_sustained_top10_count": sum(
                row["first_sustained_top10_layer"] is not None for row in model_cases
            ),
            "never_top10_count": sum(
                not bool(row["ever_top10"]) for row in model_cases
            ),
        }

    report = {
        "analysis": "final_gold_rank_near_misses",
        "interpretation": (
            "Final ranks 11--30 are treated as near misses relative to the "
            "operational Target@10 threshold, not as a semantic-correctness "
            "category."
        ),
        "near_miss_definition": {
            "rank_min": args.rank_min,
            "rank_max": args.rank_max,
            "final_target_at10_required_false": True,
        },
        "trajectory_history_definition": {
            "min_consecutive": args.min_consecutive,
            "never_top10": (
                "Gold target never belongs to actual Top-10 at any saved state."
            ),
            "transient_top10_no_sustained": (
                "Gold enters Top-10 at least once but never begins a run of "
                f"{args.min_consecutive} consecutive Top-10 layers."
            ),
            "sustained_top10_then_lost": (
                "Gold has a strict sustained Top-10 onset earlier but is "
                "outside Top-10 at the final layer."
            ),
        },
        "input_files": [str(path) for path in input_paths],
        "raw_rows_by_file": raw_rows_by_file,
        "model_trajectory_counts": dict(model_trajectory_counts),
        "headline": {
            "n_final_trajectories": total_final,
            "target_at10_failure_count": total_failures,
            "near_miss_rank_11_30_count": total_near,
            "near_miss_rate_all": proportion(total_near, total_final),
            "near_miss_share_of_target_at10_failures": proportion(
                total_near,
                total_failures,
            ),
            "rank_bin_counts": dict(Counter(str(row["rank_bin"]) for row in cases)),
            "language_counts": dict(Counter(str(row["language"]) for row in cases)),
            "semantic_group_counts": dict(
                Counter(str(row["semantic_group"]) for row in cases)
            ),
            "history_counts": dict(
                Counter(str(row["history_category"]) for row in cases)
            ),
        },
        "by_model": by_model,
        "trajectory_merge": trajectory_merge_report,
        "validation": {
            "raw_required_fields_checked": True,
            "complete_contiguous_layer_trajectories_checked": True,
            "one_final_row_per_trajectory_checked": True,
            "near_miss_target_at10_false_checked": True,
            "trajectory_table_crosschecked_when_available": (
                trajectory_merge_report.get("used", False)
            ),
        },
        "outputs": {
            "cases_tsv": "final_rank_near_miss_cases.tsv",
            "cases_jsonl": "final_rank_near_miss_cases.jsonl",
            "summary": "final_rank_near_miss_summary.tsv",
            "rank_bins": "final_rank_near_miss_rank_bins.tsv",
            "history_summary": "final_rank_near_miss_history_summary.tsv",
            "representative_sample": "final_rank_near_miss_representative_sample.tsv",
            "report": "final_rank_near_miss_report.json",
        },
    }

    write_json(
        output_dir / "final_rank_near_miss_report.json",
        report,
    )

    print("\n========================================")
    print("Final-rank near-miss analysis")
    print("========================================")
    print(f"Final trajectories:       {total_final}")
    print(f"Target@10 failures:       {total_failures}")
    print(f"Rank 11--30 near misses:  {total_near}")
    print(f"Near-miss rate overall:  {100 * total_near / total_final:.2f}%")
    print(f"Share of @10 failures:   {100 * total_near / total_failures:.2f}%")

    print("\nBy model:")
    for model, stats in by_model.items():
        print(
            f"  {model}: "
            f"near={stats['near_miss_rank_11_30_count']}, "
            f"failures={stats['target_at10_failure_count']}, "
            f"share="
            f"{100 * stats['near_miss_share_of_target_at10_failures']:.2f}%"
        )

    print("\nHistory among near misses:")
    for category, count in Counter(
        str(row["history_category"]) for row in cases
    ).items():
        print(f"  {category}: {count}")

    if trajectory_merge_report.get("used"):
        print(f"\nTrajectory merge: {trajectory_merge_report['path']}")
        print(
            f"  missing near-miss keys: {trajectory_merge_report['missing_case_keys']}"
        )
        print(
            "  cross-check mismatches: "
            f"{trajectory_merge_report['crosscheck_mismatch_counts']}"
        )

    print(f"\nOutputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
