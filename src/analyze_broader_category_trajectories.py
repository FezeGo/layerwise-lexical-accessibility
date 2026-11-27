from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

DEFAULT_INPUT_DIR = Path("data/logit_lens/broader_category")
DEFAULT_OUTPUT_DIR = Path("results/analysis/broader_category")
MODEL_KEYS = ("qwen3_0_6b", "qwen3_4b", "qwen3_14b")
INPUT_FILENAME = "broader_category_by_layer.jsonl"
DEFAULT_MIN_CONSECUTIVE = 3
K_VALUES = (1, 5, 10)
PRIMARY_K = 10

OUTPUT_FILENAMES = (
    "broader_category_layerwise.tsv",
    "broader_category_trajectories.tsv",
    "broader_category_trajectories.jsonl",
    "broader_category_pattern_counts.tsv",
    "broader_category_onset_summary.tsv",
    "broader_category_layerwise_summary.tsv",
    "broader_category_final_layer_summary.tsv",
    "broader_before_gold_top10.tsv",
    "bilingual_pair_top10_comparison.tsv",
    "bilingual_language_comparison_summary.tsv",
    "broader_category_analysis_report.json",
)

PATTERN_ORDER = (
    "broader_before_gold",
    "gold_before_broader",
    "same_onset",
    "broader_only_sustained",
    "gold_only_sustained",
    "neither_sustained",
)

REQUIRED_FIELDS = {
    "id",
    "semantic_group",
    "subgroup",
    "relation",
    "mapping_rule",
    "mapping_status",
    "language",
    "model_key",
    "model_name",
    "layer",
    "num_layers",
    "layer_fraction",
    "subject",
    "prefix",
    "prediction_position",
    "gold",
    "gold_token_id",
    "gold_rank",
    "gold_logit",
    "gold_in_top1",
    "gold_in_top5",
    "gold_in_top10",
    "broader_category",
    "broader_token_id",
    "broader_rank",
    "broader_logit",
    "broader_in_top1",
    "broader_in_top5",
    "broader_in_top10",
    "rank_difference_broader_minus_gold",
    "final_prediction_token_id",
    "final_prediction_text",
    "final_prediction_is_gold",
    "final_prediction_is_broader_category",
}

META_FIELDS = (
    "model_key",
    "model_name",
    "id",
    "semantic_group",
    "subgroup",
    "relation",
    "mapping_rule",
    "mapping_status",
    "language",
    "subject",
    "prefix",
    "prediction_position",
    "num_layers",
    "gold",
    "gold_token_id",
    "gold_token",
    "gold_token_surface",
    "broader_category",
    "broader_token_id",
    "broader_token",
    "broader_token_surface",
)


# I/O


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            row["_source_file"] = str(path)
            row["_source_line"] = line_no
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n")


def write_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k.startswith("_") or k in seen:
                continue
            seen.add(k)
            fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                value = row.get(k, "")
                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(value, ensure_ascii=False)
                out[k] = value
            w.writerow(out)


# Generic helpers


def norm_id(x: Any) -> str:
    return str(x)


def prop(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def mean(values: Sequence[float | int]) -> Optional[float]:
    return None if not values else float(sum(values) / len(values))


def median(values: Sequence[float | int]) -> Optional[float]:
    return None if not values else float(statistics.median(values))


def frac(layer: Optional[int], num_layers: int) -> Optional[float]:
    return None if layer is None or num_layers <= 0 else layer / num_layers


# Validation / grouping


def validate_rows(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        loc = f"{row.get('_source_file')}:{row.get('_source_line')}"
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"Missing fields at {loc}: {sorted(missing)}")
        if str(row["language"]) not in {"en", "zh"}:
            raise ValueError(f"Unsupported language at {loc}: {row['language']!r}")
        if str(row["semantic_group"]) != "category":
            raise ValueError(f"Expected category group at {loc}")
        if str(row["mapping_status"]) != "manually_defined_broader_category":
            raise ValueError(f"Unexpected mapping_status at {loc}")
        gid = int(row["gold_token_id"])
        bid = int(row["broader_token_id"])
        if gid == bid:
            raise ValueError(f"Identical gold/broader token IDs at {loc}")
        gr = int(row["gold_rank"])
        br = int(row["broader_rank"])
        delta = int(row["rank_difference_broader_minus_gold"])
        if delta != br - gr:
            raise ValueError(
                f"Rank delta mismatch at {loc}: stored={delta}, expected={br - gr}"
            )

        # Validate actual saved torch.topk membership when available.
        if "topk_token_ids" in row:
            top = [int(x) for x in row["topk_token_ids"]]
            if len(top) < 10:
                raise ValueError(f"Need at least 10 top-k IDs at {loc}")
            expected = {
                "gold_in_top1": gid == top[0],
                "gold_in_top5": gid in top[:5],
                "gold_in_top10": gid in top[:10],
                "broader_in_top1": bid == top[0],
                "broader_in_top5": bid in top[:5],
                "broader_in_top10": bid in top[:10],
            }
            for field, value in expected.items():
                if bool(row[field]) != value:
                    raise ValueError(f"Top-k membership mismatch for {field} at {loc}")


def group_trajectories(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        key = (str(row["model_key"]), norm_id(row["id"]), str(row["language"]))
        grouped[key].append(row)

    constant_fields = (
        "semantic_group",
        "subgroup",
        "relation",
        "mapping_rule",
        "mapping_status",
        "subject",
        "prefix",
        "prediction_position",
        "gold",
        "gold_token_id",
        "broader_category",
        "broader_token_id",
        "final_prediction_token_id",
        "final_prediction_text",
    )

    for key, traj in grouped.items():
        traj.sort(key=lambda r: int(r["layer"]))
        layers = [int(r["layer"]) for r in traj]
        if len(layers) != len(set(layers)):
            raise ValueError(f"Duplicate layers for {key}")
        nums = {int(r["num_layers"]) for r in traj}
        if len(nums) != 1:
            raise ValueError(f"Inconsistent num_layers for {key}")
        n_layers = next(iter(nums))
        if layers != list(range(n_layers + 1)):
            raise ValueError(f"Incomplete/non-contiguous trajectory for {key}")
        for field in constant_fields:
            values = {str(r.get(field, "")) for r in traj}
            if len(values) != 1:
                raise ValueError(f"Inconsistent {field} for {key}: {values}")

        final = traj[-1]
        if "topk_token_ids" in final:
            top1 = int(final["topk_token_ids"][0])
            if int(final["final_prediction_token_id"]) != top1:
                raise ValueError(f"Final Top-1 mismatch for {key}")
            if bool(final["final_prediction_is_gold"]) != (
                int(final["gold_token_id"]) == top1
            ):
                raise ValueError(f"Final gold flag mismatch for {key}")
            if bool(final["final_prediction_is_broader_category"]) != (
                int(final["broader_token_id"]) == top1
            ):
                raise ValueError(f"Final broader flag mismatch for {key}")

    return dict(grouped)


# Onset / trajectory logic


def first_true(rows: Sequence[dict[str, Any]], field: str) -> Optional[int]:
    for row in rows:
        if bool(row[field]):
            return int(row["layer"])
    return None


def first_sustained(
    rows: Sequence[dict[str, Any]], field: str, min_consecutive: int
) -> Optional[int]:
    start = None
    run = 0
    prev = None
    for row in rows:
        layer = int(row["layer"])
        if bool(row[field]):
            if prev is not None and layer == prev + 1:
                run += 1
            else:
                start, run = layer, 1
            if run >= min_consecutive:
                return start
            prev = layer
        else:
            start, run, prev = None, 0, None
    return None


def max_run(rows: Sequence[dict[str, Any]], field: str) -> int:
    best = run = 0
    prev = None
    for row in rows:
        layer = int(row["layer"])
        if bool(row[field]):
            run = run + 1 if prev is not None and layer == prev + 1 else 1
            best = max(best, run)
            prev = layer
        else:
            run, prev = 0, None
    return best


def classify(broader: Optional[int], gold: Optional[int]) -> str:
    if broader is not None and gold is not None:
        if broader < gold:
            return "broader_before_gold"
        if gold < broader:
            return "gold_before_broader"
        return "same_onset"
    if broader is not None:
        return "broader_only_sustained"
    if gold is not None:
        return "gold_only_sustained"
    return "neither_sustained"


def direction(pattern: str) -> str:
    if pattern == "broader_before_gold":
        return "broader_first"
    if pattern == "gold_before_broader":
        return "gold_first"
    if pattern == "same_onset":
        return "same"
    return "one_or_neither_not_sustained"


def build_trajectory(
    rows: Sequence[dict[str, Any]], min_consecutive: int
) -> dict[str, Any]:
    first, final = rows[0], rows[-1]
    num_layers = int(first["num_layers"])
    out = {field: first.get(field, "") for field in META_FIELDS}

    gold_ranks = [int(r["gold_rank"]) for r in rows]
    broader_ranks = [int(r["broader_rank"]) for r in rows]
    deltas = [int(r["rank_difference_broader_minus_gold"]) for r in rows]

    out.update(
        {
            "min_consecutive": min_consecutive,
            "final_prediction_token_id": int(final["final_prediction_token_id"]),
            "final_prediction_text": str(final["final_prediction_text"]),
            "final_prediction_is_gold": bool(final["final_prediction_is_gold"]),
            "final_prediction_is_broader_category": bool(
                final["final_prediction_is_broader_category"]
            ),
            "gold_final_rank": int(final["gold_rank"]),
            "broader_final_rank": int(final["broader_rank"]),
            "final_rank_difference_broader_minus_gold": int(
                final["rank_difference_broader_minus_gold"]
            ),
            "gold_best_rank": min(gold_ranks),
            "broader_best_rank": min(broader_ranks),
            "layers_broader_rank_better_count": sum(x < 0 for x in deltas),
            "layers_gold_rank_better_count": sum(x > 0 for x in deltas),
            "layers_equal_rank_count": sum(x == 0 for x in deltas),
            "broader_rank_better_layer_rate": prop(
                sum(x < 0 for x in deltas), len(rows)
            ),
            "gold_rank_better_layer_rate": prop(sum(x > 0 for x in deltas), len(rows)),
        }
    )

    for k in K_VALUES:
        bf, gf = f"broader_in_top{k}", f"gold_in_top{k}"
        b_first, g_first = first_true(rows, bf), first_true(rows, gf)
        b_sus = first_sustained(rows, bf, min_consecutive)
        g_sus = first_sustained(rows, gf, min_consecutive)
        out.update(
            {
                f"broader_ever_top{k}": any(bool(r[bf]) for r in rows),
                f"gold_ever_top{k}": any(bool(r[gf]) for r in rows),
                f"broader_final_top{k}": bool(final[bf]),
                f"gold_final_top{k}": bool(final[gf]),
                f"first_broader_top{k}_layer": b_first,
                f"first_gold_top{k}_layer": g_first,
                f"first_broader_top{k}_fraction": frac(b_first, num_layers),
                f"first_gold_top{k}_fraction": frac(g_first, num_layers),
                f"first_broader_top{k}_sustained_layer": b_sus,
                f"first_gold_top{k}_sustained_layer": g_sus,
                f"first_broader_top{k}_sustained_fraction": frac(b_sus, num_layers),
                f"first_gold_top{k}_sustained_fraction": frac(g_sus, num_layers),
                f"broader_max_consecutive_top{k}": max_run(rows, bf),
                f"gold_max_consecutive_top{k}": max_run(rows, gf),
                f"sustained_pattern_top{k}": classify(b_sus, g_sus),
                # Positive => broader category became sustained earlier.
                f"sustained_onset_delta_gold_minus_broader_top{k}": (
                    g_sus - b_sus if b_sus is not None and g_sus is not None else None
                ),
            }
        )
    return out


# Aggregation


def grouping_specs() -> list[tuple[str, Callable[[dict[str, Any]], str]]]:
    return [
        ("overall", lambda r: "all"),
        ("language", lambda r: str(r.get("language", ""))),
        ("subgroup", lambda r: str(r.get("subgroup", ""))),
        (
            "language_x_subgroup",
            lambda r: f"{r.get('language', '')}|{r.get('subgroup', '')}",
        ),
    ]


def aggregate_patterns(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for grouping, fn in grouping_specs():
        groups = defaultdict(list)
        for row in rows:
            groups[(str(row["model_key"]), fn(row))].append(row)

        for (model, group), items in sorted(groups.items()):
            n = len(items)
            for k in K_VALUES:
                pf = f"sustained_pattern_top{k}"
                counts = Counter(str(r[pf]) for r in items)
                for pattern in PATTERN_ORDER:
                    count = counts.get(pattern, 0)
                    out.append(
                        {
                            "grouping": grouping,
                            "group": group,
                            "model_key": model,
                            "k": k,
                            "row_type": "pattern",
                            "pattern": pattern,
                            "count": count,
                            "n": n,
                            "rate": prop(count, n),
                        }
                    )

                bf = f"first_broader_top{k}_sustained_layer"
                gf = f"first_gold_top{k}_sustained_layer"
                both = [r for r in items if r[bf] is not None and r[gf] is not None]
                bc = sum(r[bf] is not None for r in items)
                gc = sum(r[gf] is not None for r in items)
                both_counts = Counter(str(r[pf]) for r in both)

                row = {
                    "grouping": grouping,
                    "group": group,
                    "model_key": model,
                    "k": k,
                    "row_type": "coverage",
                    "pattern": "__coverage_summary__",
                    "n": n,
                    "broader_sustained_count": bc,
                    "broader_sustained_rate": prop(bc, n),
                    "gold_sustained_count": gc,
                    "gold_sustained_rate": prop(gc, n),
                    "both_sustained_count": len(both),
                    "both_sustained_rate": prop(len(both), n),
                }
                for p in ("broader_before_gold", "gold_before_broader", "same_onset"):
                    c = both_counts.get(p, 0)
                    row[f"{p}_among_both_count"] = c
                    row[f"{p}_among_both_rate"] = prop(c, len(both))
                out.append(row)
    return out


def aggregate_onsets(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for grouping, fn in grouping_specs():
        groups = defaultdict(list)
        for row in rows:
            groups[(str(row["model_key"]), fn(row))].append(row)

        for (model, group), items in sorted(groups.items()):
            for k in K_VALUES:
                bl = f"first_broader_top{k}_sustained_layer"
                gl = f"first_gold_top{k}_sustained_layer"
                bf = f"first_broader_top{k}_sustained_fraction"
                gf = f"first_gold_top{k}_sustained_fraction"
                b_rows = [r for r in items if r[bl] is not None]
                g_rows = [r for r in items if r[gl] is not None]
                both = [r for r in items if r[bl] is not None and r[gl] is not None]
                layer_delta = [
                    int(r[f"sustained_onset_delta_gold_minus_broader_top{k}"])
                    for r in both
                ]
                frac_delta = [float(r[gf]) - float(r[bf]) for r in both]
                out.append(
                    {
                        "grouping": grouping,
                        "group": group,
                        "model_key": model,
                        "k": k,
                        "n": len(items),
                        "broader_sustained_n": len(b_rows),
                        "broader_sustained_rate": prop(len(b_rows), len(items)),
                        "broader_mean_onset_layer": mean([int(r[bl]) for r in b_rows]),
                        "broader_median_onset_layer": median(
                            [int(r[bl]) for r in b_rows]
                        ),
                        "broader_mean_onset_fraction": mean(
                            [float(r[bf]) for r in b_rows]
                        ),
                        "broader_median_onset_fraction": median(
                            [float(r[bf]) for r in b_rows]
                        ),
                        "gold_sustained_n": len(g_rows),
                        "gold_sustained_rate": prop(len(g_rows), len(items)),
                        "gold_mean_onset_layer": mean([int(r[gl]) for r in g_rows]),
                        "gold_median_onset_layer": median([int(r[gl]) for r in g_rows]),
                        "gold_mean_onset_fraction": mean(
                            [float(r[gf]) for r in g_rows]
                        ),
                        "gold_median_onset_fraction": median(
                            [float(r[gf]) for r in g_rows]
                        ),
                        "both_sustained_n": len(both),
                        "mean_delta_gold_minus_broader_layer_among_both": mean(
                            layer_delta
                        ),
                        "median_delta_gold_minus_broader_layer_among_both": median(
                            layer_delta
                        ),
                        "mean_delta_gold_minus_broader_fraction_among_both": mean(
                            frac_delta
                        ),
                        "median_delta_gold_minus_broader_fraction_among_both": median(
                            frac_delta
                        ),
                    }
                )
    return out


def aggregate_layerwise(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("overall", lambda r: "all"),
        ("language", lambda r: str(r["language"])),
        ("subgroup", lambda r: str(r["subgroup"])),
    ]
    out = []
    for grouping, fn in specs:
        groups = defaultdict(list)
        for row in rows:
            groups[(str(row["model_key"]), fn(row), int(row["layer"]))].append(row)

        for (model, group, layer), items in sorted(groups.items()):
            n = len(items)
            first = items[0]
            gr = [int(r["gold_rank"]) for r in items]
            br = [int(r["broader_rank"]) for r in items]
            d = [int(r["rank_difference_broader_minus_gold"]) for r in items]
            row = {
                "grouping": grouping,
                "group": group,
                "model_key": model,
                "layer": layer,
                "num_layers": int(first["num_layers"]),
                "layer_fraction": float(first["layer_fraction"]),
                "n": n,
                "mean_gold_rank": mean(gr),
                "median_gold_rank": median(gr),
                "mean_broader_rank": mean(br),
                "median_broader_rank": median(br),
                "mean_rank_difference_broader_minus_gold": mean(d),
                "median_rank_difference_broader_minus_gold": median(d),
                "broader_better_rank_count": sum(x < 0 for x in d),
                "broader_better_rank_rate": prop(sum(x < 0 for x in d), n),
                "gold_better_rank_count": sum(x > 0 for x in d),
                "gold_better_rank_rate": prop(sum(x > 0 for x in d), n),
            }
            for k in K_VALUES:
                gc = sum(bool(r[f"gold_in_top{k}"]) for r in items)
                bc = sum(bool(r[f"broader_in_top{k}"]) for r in items)
                row[f"gold_at_{k}"] = prop(gc, n)
                row[f"broader_at_{k}"] = prop(bc, n)
                row[f"broader_minus_gold_at_{k}"] = (bc - gc) / n
            out.append(row)
    return out


def aggregate_final(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for grouping, fn in grouping_specs():
        groups = defaultdict(list)
        for row in rows:
            groups[(str(row["model_key"]), fn(row))].append(row)

        for (model, group), items in sorted(groups.items()):
            n = len(items)
            gr = [int(r["gold_final_rank"]) for r in items]
            br = [int(r["broader_final_rank"]) for r in items]
            d = [int(r["final_rank_difference_broader_minus_gold"]) for r in items]
            row = {
                "grouping": grouping,
                "group": group,
                "model_key": model,
                "n": n,
                "mean_gold_final_rank": mean(gr),
                "median_gold_final_rank": median(gr),
                "mean_broader_final_rank": mean(br),
                "median_broader_final_rank": median(br),
                "mean_final_rank_difference_broader_minus_gold": mean(d),
                "median_final_rank_difference_broader_minus_gold": median(d),
                "broader_better_final_rank_count": sum(x < 0 for x in d),
                "broader_better_final_rank_rate": prop(sum(x < 0 for x in d), n),
                "gold_better_final_rank_count": sum(x > 0 for x in d),
                "gold_better_final_rank_rate": prop(sum(x > 0 for x in d), n),
                "final_prediction_is_gold_count": sum(
                    bool(r["final_prediction_is_gold"]) for r in items
                ),
                "final_prediction_is_gold_rate": prop(
                    sum(bool(r["final_prediction_is_gold"]) for r in items), n
                ),
                "final_prediction_is_broader_category_count": sum(
                    bool(r["final_prediction_is_broader_category"]) for r in items
                ),
                "final_prediction_is_broader_category_rate": prop(
                    sum(bool(r["final_prediction_is_broader_category"]) for r in items),
                    n,
                ),
            }
            for k in K_VALUES:
                gc = sum(bool(r[f"gold_final_top{k}"]) for r in items)
                bc = sum(bool(r[f"broader_final_top{k}"]) for r in items)
                row[f"gold_final_top{k}_count"] = gc
                row[f"gold_final_top{k}_rate"] = prop(gc, n)
                row[f"broader_final_top{k}_count"] = bc
                row[f"broader_final_top{k}_rate"] = prop(bc, n)
            out.append(row)
    return out


# Matched EN/ZH comparison


def bilingual_pairs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(dict)
    for row in rows:
        key = (str(row["model_key"]), norm_id(row["id"]))
        lang = str(row["language"])
        if lang in grouped[key]:
            raise ValueError(f"Duplicate {lang} row for matched pair {key}")
        grouped[key][lang] = row

    out = []
    pf = f"sustained_pattern_top{PRIMARY_K}"
    for (model, ex_id), pair in sorted(grouped.items()):
        if set(pair) != {"en", "zh"}:
            raise ValueError(f"Missing EN/ZH matched pair for {model}/{ex_id}")
        en, zh = pair["en"], pair["zh"]
        if str(en["subgroup"]) != str(zh["subgroup"]):
            raise ValueError(f"Subgroup mismatch for {model}/{ex_id}")

        ep, zp = str(en[pf]), str(zh[pf])
        row = {
            "model_key": model,
            "id": ex_id,
            "subgroup": en["subgroup"],
            "relation": en["relation"],
            "mapping_rule": en["mapping_rule"],
            "en_subject": en["subject"],
            "zh_subject": zh["subject"],
            "en_gold": en["gold"],
            "zh_gold": zh["gold"],
            "en_broader_category": en["broader_category"],
            "zh_broader_category": zh["broader_category"],
            "en_pattern_top10": ep,
            "zh_pattern_top10": zp,
            "en_direction_top10": direction(ep),
            "zh_direction_top10": direction(zp),
            "pattern_exact_match_en_zh": ep == zp,
            "direction_match_en_zh": direction(ep) == direction(zp),
        }
        for lang, r in (("en", en), ("zh", zh)):
            for target in ("broader", "gold"):
                key = f"first_{target}_top10_sustained_fraction"
                row[f"{lang}_{target}_sustained_onset_fraction_top10"] = r[key]
            bf = r["first_broader_top10_sustained_fraction"]
            gf = r["first_gold_top10_sustained_fraction"]
            row[f"{lang}_delta_gold_minus_broader_fraction_top10"] = (
                float(gf) - float(bf) if bf is not None and gf is not None else None
            )
        out.append(row)
    return out


def aggregate_bilingual(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("overall", lambda r: "all"),
        ("subgroup", lambda r: str(r["subgroup"])),
    ]
    out = []
    for grouping, fn in specs:
        groups = defaultdict(list)
        for row in rows:
            groups[(str(row["model_key"]), fn(row))].append(row)
        for (model, group), items in sorted(groups.items()):
            n = len(items)
            transitions = Counter(
                (str(r["en_direction_top10"]), str(r["zh_direction_top10"]))
                for r in items
            )
            en_d = [
                float(r["en_delta_gold_minus_broader_fraction_top10"])
                for r in items
                if r["en_delta_gold_minus_broader_fraction_top10"] is not None
            ]
            zh_d = [
                float(r["zh_delta_gold_minus_broader_fraction_top10"])
                for r in items
                if r["zh_delta_gold_minus_broader_fraction_top10"] is not None
            ]
            out.append(
                {
                    "grouping": grouping,
                    "group": group,
                    "model_key": model,
                    "n": n,
                    "exact_pattern_match_count": sum(
                        bool(r["pattern_exact_match_en_zh"]) for r in items
                    ),
                    "exact_pattern_match_rate": prop(
                        sum(bool(r["pattern_exact_match_en_zh"]) for r in items), n
                    ),
                    "direction_match_count": sum(
                        bool(r["direction_match_en_zh"]) for r in items
                    ),
                    "direction_match_rate": prop(
                        sum(bool(r["direction_match_en_zh"]) for r in items), n
                    ),
                    "en_broader_first_count": sum(
                        r["en_direction_top10"] == "broader_first" for r in items
                    ),
                    "zh_broader_first_count": sum(
                        r["zh_direction_top10"] == "broader_first" for r in items
                    ),
                    "en_gold_first_count": sum(
                        r["en_direction_top10"] == "gold_first" for r in items
                    ),
                    "zh_gold_first_count": sum(
                        r["zh_direction_top10"] == "gold_first" for r in items
                    ),
                    "en_mean_delta_gold_minus_broader_fraction_among_both": mean(en_d),
                    "zh_mean_delta_gold_minus_broader_fraction_among_both": mean(zh_d),
                    "direction_transition_counts_json": {
                        f"{a} -> {b}": c for (a, b), c in sorted(transitions.items())
                    },
                }
            )
    return out


# Compact layer-wise table


def compact_layerwise(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = (
        "model_key",
        "model_name",
        "id",
        "language",
        "semantic_group",
        "subgroup",
        "relation",
        "mapping_rule",
        "layer",
        "layer_name",
        "num_layers",
        "layer_fraction",
        "subject",
        "prefix",
        "gold",
        "gold_token_id",
        "broader_category",
        "broader_token_id",
        "gold_rank",
        "broader_rank",
        "rank_difference_broader_minus_gold",
        "gold_logit",
        "broader_logit",
        "logit_difference_broader_minus_gold",
        "gold_prob",
        "broader_prob",
        "prob_difference_broader_minus_gold",
        "gold_in_top1",
        "gold_in_top5",
        "gold_in_top10",
        "broader_in_top1",
        "broader_in_top5",
        "broader_in_top10",
        "final_prediction_token_id",
        "final_prediction_text",
        "final_prediction_is_gold",
        "final_prediction_is_broader_category",
    )
    out = [{k: row.get(k, "") for k in keep} for row in rows]
    out.sort(
        key=lambda r: (
            str(r["model_key"]),
            norm_id(r["id"]),
            0 if str(r["language"]) == "en" else 1,
            int(r["layer"]),
        )
    )
    return out


# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze specific-gold vs broader-category Logit Lens trajectories."
    )
    parser.add_argument("--input", nargs="+", type=Path)
    parser.add_argument(
        "--input-dir",
        "--base_input_dir",
        "--base-input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--model", "--models", nargs="+", choices=["all", *MODEL_KEYS], default=None
    )
    parser.add_argument("--input-filename", "--input_filename", default=INPUT_FILENAME)
    parser.add_argument(
        "--output-dir", "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--min-consecutive",
        "--min_consecutive",
        type=int,
        default=DEFAULT_MIN_CONSECUTIVE,
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing analysis outputs."
    )
    return parser


def resolve_inputs(args: argparse.Namespace, model_keys: Sequence[str]) -> list[Path]:
    if args.input:
        return list(args.input)
    paths = [
        args.input_dir / model_key / args.input_filename for model_key in model_keys
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing broader-category extraction files:\n  "
            + "\n  ".join(str(p) for p in missing)
        )
    return paths


# Main


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_consecutive < 1:
        parser.error("--min-consecutive must be positive")
    if args.model and "all" in args.model and args.model != ["all"]:
        parser.error("--model all cannot be combined with individual models")
    if (
        args.min_consecutive != DEFAULT_MIN_CONSECUTIVE
        and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
    ):
        parser.error("Nonstandard analysis requires a separate --output-dir")

    model_keys = [
        key for key in MODEL_KEYS if args.model in (None, ["all"]) or key in args.model
    ]
    inputs = resolve_inputs(args, model_keys)
    layer_rows = []
    for path in inputs:
        rows = load_jsonl(path)
        if not args.input and {str(row.get("model_key")) for row in rows} != {
            path.parent.name
        }:
            raise ValueError(f"Unexpected model key in {path}")
        layer_rows.extend(rows)

    validate_rows(layer_rows)
    observed_models = {str(row["model_key"]) for row in layer_rows}
    if args.input and args.model is None:
        model_keys = [key for key in MODEL_KEYS if key in observed_models]
    if observed_models != set(model_keys):
        raise ValueError(
            f"Input models {sorted(observed_models)} do not match selected models {model_keys}"
        )

    output_dir = args.output_dir
    if set(model_keys) != set(MODEL_KEYS):
        output_dir = output_dir / "__".join(model_keys)
    existing = [
        output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Existing analysis outputs; use --overwrite or a different --output-dir: "
            + ", ".join(str(path) for path in existing)
        )

    trajectories = group_trajectories(layer_rows)
    traj_rows = [
        build_trajectory(rows, args.min_consecutive)
        for _, rows in sorted(trajectories.items())
    ]
    traj_rows.sort(
        key=lambda r: (
            str(r["model_key"]),
            norm_id(r["id"]),
            0 if str(r["language"]) == "en" else 1,
        )
    )

    layer_compact = compact_layerwise(layer_rows)
    pattern_rows = aggregate_patterns(traj_rows)
    onset_rows = aggregate_onsets(traj_rows)
    layer_summary = aggregate_layerwise(layer_rows)
    final_summary = aggregate_final(traj_rows)
    broader_first = [
        r for r in traj_rows if r["sustained_pattern_top10"] == "broader_before_gold"
    ]
    pair_rows = bilingual_pairs(traj_rows)
    pair_summary = aggregate_bilingual(pair_rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    write_tsv(output_dir / "broader_category_layerwise.tsv", layer_compact)
    write_tsv(output_dir / "broader_category_trajectories.tsv", traj_rows)
    write_jsonl(output_dir / "broader_category_trajectories.jsonl", traj_rows)
    write_tsv(output_dir / "broader_category_pattern_counts.tsv", pattern_rows)
    write_tsv(output_dir / "broader_category_onset_summary.tsv", onset_rows)
    write_tsv(output_dir / "broader_category_layerwise_summary.tsv", layer_summary)
    write_tsv(output_dir / "broader_category_final_layer_summary.tsv", final_summary)
    write_tsv(output_dir / "broader_before_gold_top10.tsv", broader_first)
    write_tsv(output_dir / "bilingual_pair_top10_comparison.tsv", pair_rows)
    write_tsv(output_dir / "bilingual_language_comparison_summary.tsv", pair_summary)

    model_counts = Counter(str(r["model_key"]) for r in traj_rows)
    model_lang_counts = {
        model: dict(
            Counter(
                str(r["language"]) for r in traj_rows if str(r["model_key"]) == model
            )
        )
        for model in sorted(model_counts)
    }
    subgroup_counts = {
        model: dict(
            Counter(
                str(r["subgroup"])
                for r in traj_rows
                if str(r["model_key"]) == model and str(r["language"]) == "en"
            )
        )
        for model in sorted(model_counts)
    }
    top10_patterns = {
        model: {
            lang: dict(
                Counter(
                    str(r["sustained_pattern_top10"])
                    for r in traj_rows
                    if str(r["model_key"]) == model and str(r["language"]) == lang
                )
            )
            for lang in ("en", "zh")
        }
        for model in sorted(model_counts)
    }

    both = [
        r
        for r in traj_rows
        if r["first_broader_top10_sustained_layer"] is not None
        and r["first_gold_top10_sustained_layer"] is not None
    ]
    b_before = sum(r["sustained_pattern_top10"] == "broader_before_gold" for r in both)
    g_before = sum(r["sustained_pattern_top10"] == "gold_before_broader" for r in both)
    same = sum(r["sustained_pattern_top10"] == "same_onset" for r in both)

    report = {
        "analysis": "broader_category_accessibility",
        "research_question": (
            "For category-completion examples, does a manually specified broader "
            "category become accessible before the more specific predefined gold "
            "category under a plain Logit Lens?"
        ),
        "input_files": [str(p) for p in inputs],
        "counts": {
            "raw_layer_rows": len(layer_rows),
            "trajectory_units_model_id_language": len(traj_rows),
            "model_trajectory_counts": dict(model_counts),
            "model_language_counts": model_lang_counts,
            "bilingual_pairs_total": len(pair_rows),
            "bilingual_pairs_per_model": {
                model: sum(r["model_key"] == model for r in pair_rows)
                for model in sorted(model_counts)
            },
            "subgroup_counts_by_model": subgroup_counts,
        },
        "method": {
            "primary_k": 10,
            "min_consecutive": args.min_consecutive,
            "sustained_onset_definition": (
                f"Earliest layer beginning at least {args.min_consecutive} "
                "consecutive layers of Top-k membership."
            ),
            "rank_difference_definition": (
                "broader rank minus gold rank; negative means broader ranks better."
            ),
            "onset_delta_definition": (
                "gold sustained-onset layer minus broader sustained-onset layer; "
                "positive means broader became sustained earlier."
            ),
            "normalized_depth_definition": "layer / num_layers",
        },
        "primary_top10_summary_all_trajectories": {
            "both_sustained_count": len(both),
            "broader_before_gold_count": b_before,
            "broader_before_gold_rate_among_both": prop(b_before, len(both)),
            "gold_before_broader_count": g_before,
            "gold_before_broader_rate_among_both": prop(g_before, len(both)),
            "same_onset_count": same,
            "same_onset_rate_among_both": prop(same, len(both)),
        },
        "top10_pattern_counts_by_model_language": top10_patterns,
        "validation": {
            "required_fields_checked": True,
            "rank_difference_consistency_checked": True,
            "topk_membership_consistency_checked_when_available": True,
            "trajectory_contiguity_checked": True,
            "final_layer_prediction_consistency_checked_when_available": True,
            "matched_en_zh_pairs_checked": True,
        },
        "interpretation_note": (
            "Earlier/higher vocabulary-space accessibility under the plain Logit "
            "Lens does not establish a causal semantic-representation ordering."
        ),
        "outputs": {
            "layerwise": "broader_category_layerwise.tsv",
            "trajectories_tsv": "broader_category_trajectories.tsv",
            "trajectories_jsonl": "broader_category_trajectories.jsonl",
            "pattern_counts": "broader_category_pattern_counts.tsv",
            "onset_summary": "broader_category_onset_summary.tsv",
            "layerwise_summary": "broader_category_layerwise_summary.tsv",
            "final_layer_summary": "broader_category_final_layer_summary.tsv",
            "broader_before_gold_top10": "broader_before_gold_top10.tsv",
            "bilingual_pair_comparison": "bilingual_pair_top10_comparison.tsv",
            "bilingual_language_summary": "bilingual_language_comparison_summary.tsv",
        },
    }
    write_json(output_dir / "broader_category_analysis_report.json", report)

    for model_key in model_keys:
        print(
            f"{model_key}: {model_counts[model_key]} trajectories, "
            f"{sum(row['model_key'] == model_key for row in pair_rows)} bilingual pairs"
        )
    print(f"Total: {len(traj_rows)} trajectories, {len(pair_rows)} bilingual pairs")
    print(
        f"Top-10 among {len(both)} both-sustained: {b_before} broader first, {g_before} gold first, {same} same onset"
    )
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
