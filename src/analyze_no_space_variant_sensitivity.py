from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence


DEFAULT_INPUT_DIR = Path("data/logit_lens/translation_equivalent")
DEFAULT_VALIDATION_PATH = Path(
    "results/analysis/translation_equivalent/token_variant_validation/"
    "translation_equivalent_token_variant_validation.tsv"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/analysis/translation_equivalent/no_space_sensitivity"
)

MODEL_KEYS = ("qwen3_0_6b", "qwen3_4b", "qwen3_14b")
INPUT_FILENAME = "translation_equivalent_by_layer.jsonl"
DEFAULT_MIN_CONSECUTIVE = 3
K_VALUES = (1, 5, 10)

OUTPUT_FILENAMES = (
    "no_space_sensitivity_layerwise.tsv",
    "no_space_sensitivity_trajectories.tsv",
    "no_space_sensitivity_trajectories.jsonl",
    "no_space_sensitivity_pattern_counts.tsv",
    "no_space_sensitivity_variant_comparison.tsv",
    "no_space_sensitivity_changed_cases.tsv",
    "no_space_sensitivity_report.json",
)

PATTERN_ORDER = (
    "english_before_chinese",
    "chinese_before_english",
    "same_onset",
    "english_only_sustained",
    "chinese_only_sustained",
    "neither_sustained",
)

REQUIRED_EXTRACTION_FIELDS = {
    "id",
    "model_key",
    "language",
    "layer",
    "num_layers",
    "semantic_group",
    "zh_gold",
    "zh_gold_token_id",
    "en_equivalent",
    "en_equivalent_token_id",
    "zh_gold_in_top1",
    "zh_gold_in_top5",
    "zh_gold_in_top10",
    "en_equivalent_in_top1",
    "en_equivalent_in_top5",
    "en_equivalent_in_top10",
    "topk_token_ids",
}

REQUIRED_VALIDATION_FIELDS = {
    "model_key",
    "id",
    "en_equivalent",
    "current_tracked_token_id",
    "no_space_context_single_token",
    "no_space_context_single_token_id",
    "tracked_matches_leading_space_context",
    "distinct_contextual_single_token_variants",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return int(float(text))


def read_validation_tsv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Validation TSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in validation TSV: {path}")

        missing = REQUIRED_VALIDATION_FIELDS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Validation TSV missing fields: {sorted(missing)}")

        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"No rows found in validation TSV: {path}")
    return rows


def ordered_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def clean_for_table(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = ordered_fieldnames(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_for_table(row.get(key, "")) for key in fields})


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def normalize_id(value: Any) -> str:
    return str(value)


def safe_fraction(layer: Optional[int], num_layers: int) -> Optional[float]:
    if layer is None or num_layers <= 0:
        return None
    return layer / num_layers


def proportion(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def first_true_layer(
    layer_rows: Sequence[dict[str, Any]],
    field: str,
) -> Optional[int]:
    for row in layer_rows:
        if bool(row[field]):
            return int(row["layer"])
    return None


def first_sustained_layer(
    layer_rows: Sequence[dict[str, Any]],
    field: str,
    min_consecutive: int,
) -> Optional[int]:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be >= 1")

    for start in range(len(layer_rows) - min_consecutive + 1):
        window = layer_rows[start : start + min_consecutive]
        layers = [int(row["layer"]) for row in window]

        if any(layers[i + 1] != layers[i] + 1 for i in range(len(layers) - 1)):
            continue

        if all(bool(row[field]) for row in window):
            return layers[0]

    return None


def max_consecutive_run(
    layer_rows: Sequence[dict[str, Any]],
    field: str,
) -> int:
    best = 0
    current = 0
    previous_layer: Optional[int] = None

    for row in layer_rows:
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


def classify_pattern(
    english_onset: Optional[int],
    chinese_onset: Optional[int],
) -> str:
    if english_onset is not None and chinese_onset is not None:
        if english_onset < chinese_onset:
            return "english_before_chinese"
        if chinese_onset < english_onset:
            return "chinese_before_english"
        return "same_onset"

    if english_onset is not None:
        return "english_only_sustained"
    if chinese_onset is not None:
        return "chinese_only_sustained"
    return "neither_sustained"


def classify_variant_change(
    leading_pattern: str,
    no_space_pattern: str,
) -> str:
    if leading_pattern == no_space_pattern:
        return "unchanged"

    directional = {
        "english_before_chinese": "english_earlier",
        "chinese_before_english": "chinese_earlier",
        "same_onset": "same",
        "english_only_sustained": "english_only",
        "chinese_only_sustained": "chinese_only",
        "neither_sustained": "neither",
    }

    leading_direction = directional[leading_pattern]
    no_space_direction = directional[no_space_pattern]

    if leading_direction == no_space_direction:
        return "same_direction_different_label"

    if (
        leading_direction == "english_earlier"
        and no_space_direction == "chinese_earlier"
    ) or (
        leading_direction == "chinese_earlier"
        and no_space_direction == "english_earlier"
    ):
        return "direction_reversal"

    return "changed"


def build_no_space_subset(
    validation_rows: Sequence[dict[str, Any]],
    model_keys: Sequence[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    subset: dict[tuple[str, str], dict[str, Any]] = {}
    allowed_models = set(model_keys)

    for row in validation_rows:
        model_key = str(row["model_key"])
        if model_key not in allowed_models:
            continue

        no_space_single = parse_bool(row["no_space_context_single_token"])
        distinct_variant = parse_bool(row["distinct_contextual_single_token_variants"])
        tracked_is_leading = parse_bool(row["tracked_matches_leading_space_context"])

        if not (no_space_single and distinct_variant):
            continue

        if not tracked_is_leading:
            raise ValueError(
                "Sensitivity subset contains a row whose primary tracked token "
                "does not match the leading-space contextual token: "
                f"model={model_key}, id={row['id']}"
            )

        no_space_token_id = optional_int(row["no_space_context_single_token_id"])
        tracked_token_id = optional_int(row["current_tracked_token_id"])

        if no_space_token_id is None or tracked_token_id is None:
            raise ValueError(f"Missing token ID for model={model_key}, id={row['id']}")

        if no_space_token_id == tracked_token_id:
            raise ValueError(
                "Expected distinct no-space and primary token IDs, but found "
                f"the same ID for model={model_key}, id={row['id']}"
            )

        key = (model_key, normalize_id(row["id"]))
        if key in subset:
            raise ValueError(f"Duplicate validation key: {key}")

        subset[key] = {
            "model_key": model_key,
            "id": normalize_id(row["id"]),
            "en_equivalent": str(row["en_equivalent"]),
            "leading_space_token_id": tracked_token_id,
            "no_space_token_id": no_space_token_id,
            "no_space_token_decoded": row.get(
                "no_space_context_single_token_decoded", ""
            ),
            "semantic_group": row.get("semantic_group", ""),
            "subgroup": row.get("subgroup", ""),
            "relation": row.get("relation", ""),
        }

    if not subset:
        raise ValueError("No valid no-space sensitivity cases were found.")

    return subset


def resolve_input_paths(
    args: argparse.Namespace, model_keys: Sequence[str]
) -> list[Path]:
    if args.input:
        return list(args.input)

    paths = [
        args.input_dir / model_key / args.input_filename for model_key in model_keys
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing EQ1 extraction files:\n  "
            + "\n  ".join(str(path) for path in missing)
        )
    return paths


def validate_extraction_row(row: dict[str, Any]) -> None:
    missing = REQUIRED_EXTRACTION_FIELDS.difference(row)
    if missing:
        raise ValueError(f"EQ1 extraction row missing fields: {sorted(missing)}")

    if str(row["language"]) != "zh":
        raise ValueError("Sensitivity analysis expects Chinese-prompt rows only.")

    if not isinstance(row["topk_token_ids"], list):
        raise TypeError("topk_token_ids must be a list")

    if len(row["topk_token_ids"]) < 10:
        raise ValueError("topk_token_ids must contain at least 10 IDs.")


def build_subset_layerwise_rows(
    all_rows: Sequence[dict[str, Any]],
    subset: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for raw in all_rows:
        validate_extraction_row(raw)

        key = (
            str(raw["model_key"]),
            normalize_id(raw["id"]),
        )
        if key not in subset:
            continue

        meta = subset[key]
        top_ids = [int(x) for x in raw["topk_token_ids"][:10]]

        # The existing EQ1 tracked English token should be exactly the
        # validation table's leading-space token.
        extraction_en_token_id = int(raw["en_equivalent_token_id"])
        if extraction_en_token_id != int(meta["leading_space_token_id"]):
            raise AssertionError(
                "Primary EQ1 token ID differs from validation token ID: "
                f"key={key}, extraction={extraction_en_token_id}, "
                f"validation={meta['leading_space_token_id']}"
            )

        # Revalidate the already saved leading-space flags against top-k IDs.
        expected_leading = {
            1: extraction_en_token_id == top_ids[0],
            5: extraction_en_token_id in top_ids[:5],
            10: extraction_en_token_id in top_ids[:10],
        }
        expected_zh = {
            1: int(raw["zh_gold_token_id"]) == top_ids[0],
            5: int(raw["zh_gold_token_id"]) in top_ids[:5],
            10: int(raw["zh_gold_token_id"]) in top_ids[:10],
        }

        for k in K_VALUES:
            if bool(raw[f"en_equivalent_in_top{k}"]) != expected_leading[k]:
                raise AssertionError(
                    f"Leading-space Top-{k} mismatch for key={key}, "
                    f"layer={raw['layer']}"
                )
            if bool(raw[f"zh_gold_in_top{k}"]) != expected_zh[k]:
                raise AssertionError(
                    f"Chinese-gold Top-{k} mismatch for key={key}, layer={raw['layer']}"
                )

        no_space_token_id = int(meta["no_space_token_id"])
        no_space_flags = {
            1: no_space_token_id == top_ids[0],
            5: no_space_token_id in top_ids[:5],
            10: no_space_token_id in top_ids[:10],
        }

        row = {
            "model_key": str(raw["model_key"]),
            "model_name": raw.get("model_name", ""),
            "id": normalize_id(raw["id"]),
            "semantic_group": raw.get("semantic_group", meta.get("semantic_group", "")),
            "subgroup": raw.get("subgroup", meta.get("subgroup", "")),
            "relation": raw.get("relation", meta.get("relation", "")),
            "layer": int(raw["layer"]),
            "layer_name": raw.get("layer_name", ""),
            "num_layers": int(raw["num_layers"]),
            "layer_fraction": raw.get(
                "layer_fraction",
                int(raw["layer"]) / int(raw["num_layers"]),
            ),
            "zh_gold": raw["zh_gold"],
            "zh_gold_token_id": int(raw["zh_gold_token_id"]),
            "en_equivalent": raw["en_equivalent"],
            "leading_space_token_id": extraction_en_token_id,
            "no_space_token_id": no_space_token_id,
            "no_space_token_decoded": meta.get("no_space_token_decoded", ""),
            "zh_gold_in_top1": bool(raw["zh_gold_in_top1"]),
            "zh_gold_in_top5": bool(raw["zh_gold_in_top5"]),
            "zh_gold_in_top10": bool(raw["zh_gold_in_top10"]),
            "leading_space_in_top1": bool(raw["en_equivalent_in_top1"]),
            "leading_space_in_top5": bool(raw["en_equivalent_in_top5"]),
            "leading_space_in_top10": bool(raw["en_equivalent_in_top10"]),
            "no_space_in_top1": no_space_flags[1],
            "no_space_in_top5": no_space_flags[5],
            "no_space_in_top10": no_space_flags[10],
            "topk_token_ids": top_ids,
        }

        output.append(row)

    if not output:
        raise ValueError(
            "No EQ1 extraction rows matched the no-space validation subset."
        )

    output.sort(
        key=lambda r: (
            str(r["model_key"]),
            normalize_id(r["id"]),
            int(r["layer"]),
        )
    )
    return output


def group_layerwise_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = (
            str(row["model_key"]),
            normalize_id(row["id"]),
        )
        grouped[key].append(row)

    for key, trajectory in grouped.items():
        trajectory.sort(key=lambda r: int(r["layer"]))
        layers = [int(r["layer"]) for r in trajectory]

        if len(layers) != len(set(layers)):
            raise ValueError(f"Duplicate layers for {key}")

        expected = list(range(0, int(trajectory[0]["num_layers"]) + 1))
        if layers != expected:
            raise ValueError(
                f"Incomplete/non-contiguous trajectory for {key}: "
                f"got {layers[:5]}...{layers[-5:]}, "
                f"expected 0...{expected[-1]}"
            )

    return dict(grouped)


def build_trajectory_summary(
    layer_rows: Sequence[dict[str, Any]],
    min_consecutive: int,
) -> dict[str, Any]:
    first = layer_rows[0]
    final = layer_rows[-1]
    num_layers = int(first["num_layers"])

    summary: dict[str, Any] = {
        "model_key": first["model_key"],
        "model_name": first.get("model_name", ""),
        "id": first["id"],
        "semantic_group": first.get("semantic_group", ""),
        "subgroup": first.get("subgroup", ""),
        "relation": first.get("relation", ""),
        "num_layers": num_layers,
        "zh_gold": first["zh_gold"],
        "zh_gold_token_id": first["zh_gold_token_id"],
        "en_equivalent": first["en_equivalent"],
        "leading_space_token_id": first["leading_space_token_id"],
        "no_space_token_id": first["no_space_token_id"],
        "no_space_token_decoded": first.get("no_space_token_decoded", ""),
        "min_consecutive": min_consecutive,
    }

    for k in K_VALUES:
        zh_field = f"zh_gold_in_top{k}"
        leading_field = f"leading_space_in_top{k}"
        no_space_field = f"no_space_in_top{k}"

        zh_first = first_true_layer(layer_rows, zh_field)
        leading_first = first_true_layer(layer_rows, leading_field)
        no_space_first = first_true_layer(layer_rows, no_space_field)

        zh_sustained = first_sustained_layer(layer_rows, zh_field, min_consecutive)
        leading_sustained = first_sustained_layer(
            layer_rows, leading_field, min_consecutive
        )
        no_space_sustained = first_sustained_layer(
            layer_rows, no_space_field, min_consecutive
        )

        leading_pattern = classify_pattern(leading_sustained, zh_sustained)
        no_space_pattern = classify_pattern(no_space_sustained, zh_sustained)

        summary.update(
            {
                f"zh_gold_ever_top{k}": any(bool(r[zh_field]) for r in layer_rows),
                f"leading_space_ever_top{k}": any(
                    bool(r[leading_field]) for r in layer_rows
                ),
                f"no_space_ever_top{k}": any(
                    bool(r[no_space_field]) for r in layer_rows
                ),
                f"zh_gold_final_top{k}": bool(final[zh_field]),
                f"leading_space_final_top{k}": bool(final[leading_field]),
                f"no_space_final_top{k}": bool(final[no_space_field]),
                f"first_zh_gold_top{k}_layer": zh_first,
                f"first_leading_space_top{k}_layer": leading_first,
                f"first_no_space_top{k}_layer": no_space_first,
                f"first_zh_gold_top{k}_fraction": safe_fraction(zh_first, num_layers),
                f"first_leading_space_top{k}_fraction": safe_fraction(
                    leading_first, num_layers
                ),
                f"first_no_space_top{k}_fraction": safe_fraction(
                    no_space_first, num_layers
                ),
                f"first_zh_gold_top{k}_sustained_layer": zh_sustained,
                f"first_leading_space_top{k}_sustained_layer": leading_sustained,
                f"first_no_space_top{k}_sustained_layer": no_space_sustained,
                f"first_zh_gold_top{k}_sustained_fraction": safe_fraction(
                    zh_sustained, num_layers
                ),
                f"first_leading_space_top{k}_sustained_fraction": safe_fraction(
                    leading_sustained, num_layers
                ),
                f"first_no_space_top{k}_sustained_fraction": safe_fraction(
                    no_space_sustained, num_layers
                ),
                f"zh_gold_max_consecutive_top{k}": max_consecutive_run(
                    layer_rows, zh_field
                ),
                f"leading_space_max_consecutive_top{k}": max_consecutive_run(
                    layer_rows, leading_field
                ),
                f"no_space_max_consecutive_top{k}": max_consecutive_run(
                    layer_rows, no_space_field
                ),
                f"leading_space_pattern_top{k}": leading_pattern,
                f"no_space_pattern_top{k}": no_space_pattern,
                f"variant_pattern_change_top{k}": classify_variant_change(
                    leading_pattern, no_space_pattern
                ),
                # Positive means the English variant starts earlier.
                f"leading_sustained_delta_zh_minus_en_top{k}": (
                    zh_sustained - leading_sustained
                    if (zh_sustained is not None and leading_sustained is not None)
                    else None
                ),
                f"no_space_sustained_delta_zh_minus_en_top{k}": (
                    zh_sustained - no_space_sustained
                    if (zh_sustained is not None and no_space_sustained is not None)
                    else None
                ),
            }
        )

    return summary


def aggregate_pattern_counts(
    trajectory_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    grouping_specs = [
        ("overall", lambda r: "all"),
        ("semantic_group", lambda r: str(r.get("semantic_group", ""))),
    ]

    for grouping_name, group_fn in grouping_specs:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for row in trajectory_rows:
            grouped[(str(row["model_key"]), str(group_fn(row)))].append(row)

        for (model_key, group_value), rows in sorted(grouped.items()):
            n = len(rows)

            for k in K_VALUES:
                for variant in ("leading_space", "no_space"):
                    field = f"{variant}_pattern_top{k}"
                    counts = Counter(str(r[field]) for r in rows)

                    for pattern in PATTERN_ORDER:
                        count = counts.get(pattern, 0)
                        output.append(
                            {
                                "grouping": grouping_name,
                                "group": group_value,
                                "model_key": model_key,
                                "k": k,
                                "english_variant": variant,
                                "pattern": pattern,
                                "count": count,
                                "n": n,
                                "rate": proportion(count, n),
                            }
                        )

    return output


def aggregate_variant_comparison(
    trajectory_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    grouping_specs = [
        ("overall", lambda r: "all"),
        ("semantic_group", lambda r: str(r.get("semantic_group", ""))),
    ]

    for grouping_name, group_fn in grouping_specs:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for row in trajectory_rows:
            grouped[(str(row["model_key"]), str(group_fn(row)))].append(row)

        for (model_key, group_value), rows in sorted(grouped.items()):
            n = len(rows)

            for k in K_VALUES:
                leading_pattern_field = f"leading_space_pattern_top{k}"
                no_space_pattern_field = f"no_space_pattern_top{k}"
                change_field = f"variant_pattern_change_top{k}"

                leading_counts = Counter(str(r[leading_pattern_field]) for r in rows)
                no_space_counts = Counter(str(r[no_space_pattern_field]) for r in rows)
                change_counts = Counter(str(r[change_field]) for r in rows)

                leading_en_sustained = sum(
                    r[f"first_leading_space_top{k}_sustained_layer"] is not None
                    for r in rows
                )
                no_space_en_sustained = sum(
                    r[f"first_no_space_top{k}_sustained_layer"] is not None
                    for r in rows
                )
                zh_sustained = sum(
                    r[f"first_zh_gold_top{k}_sustained_layer"] is not None for r in rows
                )

                output.append(
                    {
                        "grouping": grouping_name,
                        "group": group_value,
                        "model_key": model_key,
                        "k": k,
                        "n": n,
                        "leading_space_english_sustained_count": leading_en_sustained,
                        "leading_space_english_sustained_rate": proportion(
                            leading_en_sustained, n
                        ),
                        "no_space_english_sustained_count": no_space_en_sustained,
                        "no_space_english_sustained_rate": proportion(
                            no_space_en_sustained, n
                        ),
                        "chinese_sustained_count": zh_sustained,
                        "chinese_sustained_rate": proportion(zh_sustained, n),
                        "leading_english_before_chinese_count": leading_counts.get(
                            "english_before_chinese", 0
                        ),
                        "no_space_english_before_chinese_count": no_space_counts.get(
                            "english_before_chinese", 0
                        ),
                        "leading_chinese_before_english_count": leading_counts.get(
                            "chinese_before_english", 0
                        ),
                        "no_space_chinese_before_english_count": no_space_counts.get(
                            "chinese_before_english", 0
                        ),
                        "leading_same_onset_count": leading_counts.get("same_onset", 0),
                        "no_space_same_onset_count": no_space_counts.get(
                            "same_onset", 0
                        ),
                        "pattern_unchanged_count": change_counts.get("unchanged", 0),
                        "pattern_unchanged_rate": proportion(
                            change_counts.get("unchanged", 0), n
                        ),
                        "pattern_changed_count": n - change_counts.get("unchanged", 0),
                        "pattern_changed_rate": proportion(
                            n - change_counts.get("unchanged", 0), n
                        ),
                        "direction_reversal_count": change_counts.get(
                            "direction_reversal", 0
                        ),
                        "direction_reversal_rate": proportion(
                            change_counts.get("direction_reversal", 0),
                            n,
                        ),
                    }
                )

    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze no-space English token sensitivity from saved EQ1 Top-k outputs."
    )
    parser.add_argument(
        "--input", nargs="+", type=Path, help="Explicit extraction JSONL files."
    )
    parser.add_argument(
        "--input-dir",
        "--base_input_dir",
        "--base-input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument("--input-filename", "--input_filename", default=INPUT_FILENAME)
    parser.add_argument(
        "--validation-tsv",
        "--validation_tsv",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )
    parser.add_argument(
        "--output-dir", "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--model",
        "--models",
        nargs="+",
        choices=["all", *MODEL_KEYS],
        default=None,
        help="Model keys (default: all; inferred for explicit inputs).",
    )
    parser.add_argument(
        "--min-consecutive",
        "--min_consecutive",
        type=int,
        default=DEFAULT_MIN_CONSECUTIVE,
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_consecutive < 1:
        parser.error("--min-consecutive must be positive")
    explicit_models = args.model is not None
    if args.model is None or "all" in args.model:
        if args.model is not None and args.model != ["all"]:
            parser.error("--model all cannot be combined with individual models")
        model_keys = list(MODEL_KEYS)
    else:
        model_keys = [key for key in MODEL_KEYS if key in args.model]

    input_paths = resolve_input_paths(args, model_keys)
    all_extraction_rows = []
    for path in input_paths:
        all_extraction_rows.extend(read_jsonl(path))
    observed_models = {str(row["model_key"]) for row in all_extraction_rows}
    if args.input and not explicit_models:
        model_keys = [key for key in MODEL_KEYS if key in observed_models]
    if observed_models != set(model_keys):
        raise ValueError(
            f"Input models {sorted(observed_models)} do not match selected models {model_keys}"
        )

    # Keep partial runs separate from the canonical combined outputs.
    output_dir = args.output_dir
    if args.min_consecutive != DEFAULT_MIN_CONSECUTIVE and (
        output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
        or DEFAULT_OUTPUT_DIR.resolve() in output_dir.resolve().parents
    ):
        parser.error("Nonstandard analysis requires a separate --output-dir.")
    if set(model_keys) != set(MODEL_KEYS):
        output_dir = output_dir / "__".join(model_keys)

    existing = [
        output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        parser.error(
            "Existing outputs; use --overwrite or a different --output-dir: "
            + ", ".join(str(path) for path in existing)
        )

    validation_rows = read_validation_tsv(args.validation_tsv)
    subset = build_no_space_subset(
        validation_rows,
        model_keys,
    )

    subset_counts = Counter(model_key for model_key, _ in subset)

    layerwise_rows = build_subset_layerwise_rows(
        all_extraction_rows,
        subset,
    )
    grouped = group_layerwise_rows(layerwise_rows)

    # Every validation subset example should have one complete trajectory.
    missing_keys = set(subset).difference(grouped)
    unexpected_keys = set(grouped).difference(subset)

    if missing_keys:
        raise ValueError(
            f"{len(missing_keys)} validation subset cases are missing "
            "from EQ1 extraction. Examples: "
            f"{sorted(missing_keys)[:10]}"
        )
    if unexpected_keys:
        raise ValueError(f"Unexpected trajectory keys: {sorted(unexpected_keys)[:10]}")

    trajectory_rows = [
        build_trajectory_summary(
            trajectory,
            min_consecutive=args.min_consecutive,
        )
        for _, trajectory in sorted(grouped.items())
    ]

    pattern_counts = aggregate_pattern_counts(trajectory_rows)
    variant_comparison = aggregate_variant_comparison(trajectory_rows)

    changed_cases = [
        row
        for row in trajectory_rows
        if any(row[f"variant_pattern_change_top{k}"] != "unchanged" for k in K_VALUES)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    write_tsv(
        output_dir / "no_space_sensitivity_layerwise.tsv",
        layerwise_rows,
    )
    write_tsv(
        output_dir / "no_space_sensitivity_trajectories.tsv",
        trajectory_rows,
    )
    write_jsonl(
        output_dir / "no_space_sensitivity_trajectories.jsonl",
        trajectory_rows,
    )
    write_tsv(
        output_dir / "no_space_sensitivity_pattern_counts.tsv",
        pattern_counts,
    )
    write_tsv(
        output_dir / "no_space_sensitivity_variant_comparison.tsv",
        variant_comparison,
    )
    write_tsv(
        output_dir / "no_space_sensitivity_changed_cases.tsv",
        changed_cases,
    )

    top10_by_model: dict[str, Any] = {}

    for model_key in model_keys:
        rows = [r for r in trajectory_rows if str(r["model_key"]) == model_key]
        leading_counts = Counter(str(r["leading_space_pattern_top10"]) for r in rows)
        no_space_counts = Counter(str(r["no_space_pattern_top10"]) for r in rows)
        change_counts = Counter(str(r["variant_pattern_change_top10"]) for r in rows)

        top10_by_model[model_key] = {
            "n": len(rows),
            "leading_space_pattern_counts": dict(leading_counts),
            "no_space_pattern_counts": dict(no_space_counts),
            "pattern_change_counts": dict(change_counts),
            "leading_space_english_sustained_count": sum(
                r["first_leading_space_top10_sustained_layer"] is not None for r in rows
            ),
            "no_space_english_sustained_count": sum(
                r["first_no_space_top10_sustained_layer"] is not None for r in rows
            ),
            "chinese_sustained_count": sum(
                r["first_zh_gold_top10_sustained_layer"] is not None for r in rows
            ),
        }

    report = {
        "analysis": "no_space_translation_equivalent_sensitivity",
        "purpose": (
            "Test whether the primary EQ1 sustained Top-k conclusions depend "
            "on tracking the paired-English leading-space token by instead "
            "using the distinct no-space single-token alternative where it "
            "exists."
        ),
        "input_files": [str(path) for path in input_paths],
        "validation_tsv": str(args.validation_tsv),
        "models": list(model_keys),
        "min_consecutive": args.min_consecutive,
        "subset_definition": (
            "Examples where the no-space English equivalent is a clean "
            "single token and has a token ID distinct from the primary "
            "leading-space token."
        ),
        "subset_counts_by_model": dict(subset_counts),
        "trajectory_units": len(trajectory_rows),
        "rank_limitation": (
            "No-space exact full-vocabulary ranks are unavailable because "
            "only saved Top-10 token IDs are reused. This sensitivity analysis "
            "therefore evaluates Top-1/5/10 membership and onset only."
        ),
        "top10_by_model": top10_by_model,
        "changed_cases_any_k": len(changed_cases),
        "validation": {
            "primary_token_matches_validation_leading_space_token": True,
            "saved_primary_topk_membership_rechecked": True,
            "saved_chinese_topk_membership_rechecked": True,
            "all_subset_trajectories_complete_and_contiguous": True,
        },
        "outputs": {
            "layerwise": "no_space_sensitivity_layerwise.tsv",
            "trajectories_tsv": "no_space_sensitivity_trajectories.tsv",
            "trajectories_jsonl": "no_space_sensitivity_trajectories.jsonl",
            "pattern_counts": "no_space_sensitivity_pattern_counts.tsv",
            "variant_comparison": "no_space_sensitivity_variant_comparison.tsv",
            "changed_cases": "no_space_sensitivity_changed_cases.tsv",
        },
    }

    write_json(
        output_dir / "no_space_sensitivity_report.json",
        report,
    )

    for model_key in model_keys:
        info = top10_by_model[model_key]
        changed = sum(row["model_key"] == model_key for row in changed_cases)
        print(f"{model_key}: {info['n']} trajectories, {changed} changed at any k")
    print(
        f"Total: {len(trajectory_rows)} trajectories, {len(changed_cases)} changed at any k"
    )
    print(f"Output: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
