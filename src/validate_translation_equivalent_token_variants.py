from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_INPUT_DIR = Path("data/alignment")

DEFAULT_OUTPUT_DIR = Path(
    "results/analysis/translation_equivalent/token_variant_validation"
)

MODEL_REGISTRY = {
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_14b": "Qwen/Qwen3-14B",
}

MODEL_KEYS = tuple(MODEL_REGISTRY)

INPUT_FILENAME = "token_alignment.jsonl"
OUTPUT_FILENAMES = (
    "translation_equivalent_token_variant_validation.tsv",
    "translation_equivalent_token_variant_validation_report.json",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def ordered_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    seen = set()
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def write_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = ordered_fieldnames(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def pair_alignment_rows(
    rows: Sequence[dict[str, Any]],
    model_key: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in rows:
        if str(row.get("model_key")) != model_key:
            continue

        language = str(row.get("language"))
        if language not in {"en", "zh"}:
            continue

        example_id = str(row.get("id"))
        if language in by_id[example_id]:
            raise ValueError(
                f"Duplicate {language} row for model={model_key}, id={example_id}"
            )
        by_id[example_id][language] = row

    missing = {
        example_id: sorted({"en", "zh"} - set(lang_rows))
        for example_id, lang_rows in by_id.items()
        if set(lang_rows) != {"en", "zh"}
    }
    if missing:
        preview = list(missing.items())[:10]
        raise ValueError(
            f"{len(missing)} examples lack an EN/ZH pair for {model_key}. "
            f"Examples: {preview}"
        )

    def sort_key(x: str):
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    return [
        (by_id[example_id]["en"], by_id[example_id]["zh"])
        for example_id in sorted(by_id, key=sort_key)
    ]


def standalone_variant_info(tokenizer, text: str) -> dict[str, Any]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return {
        "token_ids": [int(x) for x in ids],
        "num_tokens": len(ids),
        "single_token": len(ids) == 1,
        "single_token_id": int(ids[0]) if len(ids) == 1 else None,
        "single_token_decoded": (tokenizer.decode([ids[0]]) if len(ids) == 1 else ""),
    }


def contextual_target_span_info(
    tokenizer,
    prefix: str,
    separator: str,
    target: str,
) -> dict[str, Any]:
    full_text = prefix + separator + target
    target_start = len(prefix) + len(separator)
    target_end = target_start + len(target)

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = [int(x) for x in encoded["input_ids"]]
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]

    overlapping_indices = []
    for i, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if start < target_end and end > target_start:
            overlapping_indices.append(i)

    target_ids = [input_ids[i] for i in overlapping_indices]
    target_offsets = [offsets[i] for i in overlapping_indices]

    if target_offsets:
        covered_start = min(start for start, _ in target_offsets)
        covered_end = max(end for _, end in target_offsets)
        covers_full_target = covered_start <= target_start and covered_end >= target_end
        boundary_clean = all(start >= len(prefix) for start, _ in target_offsets)
    else:
        covers_full_target = False
        boundary_clean = False

    return {
        "target_ids": target_ids,
        "target_offsets": target_offsets,
        "num_tokens": len(target_ids),
        "single_token": len(target_ids) == 1,
        "single_token_id": int(target_ids[0]) if len(target_ids) == 1 else None,
        "single_token_decoded": (
            tokenizer.decode([target_ids[0]]) if len(target_ids) == 1 else ""
        ),
        "boundary_clean": boundary_clean,
        "covers_full_target": covers_full_target,
    }


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def run_model(
    model_key: str,
    model_name: str,
    input_dir: Path,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    alignment_path = input_dir / model_key / INPUT_FILENAME
    rows = read_jsonl(alignment_path)
    pairs = pair_alignment_rows(rows, model_key)

    if not pairs:
        raise ValueError(
            f"No bilingual pairs found for {model_key} in {alignment_path}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"{model_name} tokenizer is not fast; offset mapping is required."
        )

    output = []

    for en_row, zh_row in pairs:
        example_id = str(en_row["id"])
        en_target = str(en_row["target"])
        zh_gold = str(zh_row["target"])
        zh_prefix = str(zh_row.get("prefix", ""))

        tracked_token_id = int(en_row["target_token_id"])
        tracked_token_decoded = tokenizer.decode([tracked_token_id])

        no_space_standalone = standalone_variant_info(tokenizer, en_target)
        leading_space_standalone = standalone_variant_info(tokenizer, " " + en_target)

        no_space_context = contextual_target_span_info(
            tokenizer,
            prefix=zh_prefix,
            separator="",
            target=en_target,
        )
        leading_space_context = contextual_target_span_info(
            tokenizer,
            prefix=zh_prefix,
            separator=" ",
            target=en_target,
        )

        no_space_context_id = no_space_context["single_token_id"]
        leading_space_context_id = leading_space_context["single_token_id"]

        tracked_matches_no_space_context = (
            no_space_context_id is not None
            and tracked_token_id == int(no_space_context_id)
        )
        tracked_matches_leading_space_context = (
            leading_space_context_id is not None
            and tracked_token_id == int(leading_space_context_id)
        )

        if tracked_matches_no_space_context and tracked_matches_leading_space_context:
            tracked_variant_class = "both_variants_same_token_id"
        elif tracked_matches_leading_space_context:
            tracked_variant_class = "leading_space_variant"
        elif tracked_matches_no_space_context:
            tracked_variant_class = "no_space_variant"
        else:
            tracked_variant_class = "neither_contextual_variant"

        distinct_contextual_single_token_variants = (
            no_space_context_id is not None
            and leading_space_context_id is not None
            and int(no_space_context_id) != int(leading_space_context_id)
        )

        output.append(
            {
                "model_key": model_key,
                "model_name": model_name,
                "id": example_id,
                "semantic_group": zh_row.get("semantic_group", ""),
                "subgroup": zh_row.get("subgroup", ""),
                "relation": zh_row.get("relation", ""),
                "zh_prefix": zh_prefix,
                "zh_gold": zh_gold,
                "en_equivalent": en_target,
                "current_tracked_token_id": tracked_token_id,
                "current_tracked_token": tokenizer.convert_ids_to_tokens(
                    [tracked_token_id]
                )[0],
                "current_tracked_decoded": tracked_token_decoded,
                "no_space_standalone_ids": json_cell(no_space_standalone["token_ids"]),
                "no_space_standalone_num_tokens": no_space_standalone["num_tokens"],
                "no_space_standalone_single_token": no_space_standalone["single_token"],
                "no_space_standalone_single_token_id": no_space_standalone[
                    "single_token_id"
                ],
                "no_space_standalone_single_token_decoded": no_space_standalone[
                    "single_token_decoded"
                ],
                "leading_space_standalone_ids": json_cell(
                    leading_space_standalone["token_ids"]
                ),
                "leading_space_standalone_num_tokens": leading_space_standalone[
                    "num_tokens"
                ],
                "leading_space_standalone_single_token": leading_space_standalone[
                    "single_token"
                ],
                "leading_space_standalone_single_token_id": leading_space_standalone[
                    "single_token_id"
                ],
                "leading_space_standalone_single_token_decoded": leading_space_standalone[
                    "single_token_decoded"
                ],
                "no_space_context_target_ids": json_cell(
                    no_space_context["target_ids"]
                ),
                "no_space_context_target_num_tokens": no_space_context["num_tokens"],
                "no_space_context_single_token": no_space_context["single_token"],
                "no_space_context_single_token_id": no_space_context_id,
                "no_space_context_single_token_decoded": no_space_context[
                    "single_token_decoded"
                ],
                "no_space_context_boundary_clean": no_space_context["boundary_clean"],
                "no_space_context_covers_full_target": no_space_context[
                    "covers_full_target"
                ],
                "no_space_context_offsets": json_cell(
                    no_space_context["target_offsets"]
                ),
                "leading_space_context_target_ids": json_cell(
                    leading_space_context["target_ids"]
                ),
                "leading_space_context_target_num_tokens": leading_space_context[
                    "num_tokens"
                ],
                "leading_space_context_single_token": leading_space_context[
                    "single_token"
                ],
                "leading_space_context_single_token_id": leading_space_context_id,
                "leading_space_context_single_token_decoded": leading_space_context[
                    "single_token_decoded"
                ],
                "leading_space_context_boundary_clean": leading_space_context[
                    "boundary_clean"
                ],
                "leading_space_context_covers_full_target": leading_space_context[
                    "covers_full_target"
                ],
                "leading_space_context_offsets": json_cell(
                    leading_space_context["target_offsets"]
                ),
                "tracked_matches_no_space_context": tracked_matches_no_space_context,
                "tracked_matches_leading_space_context": tracked_matches_leading_space_context,
                "tracked_variant_class": tracked_variant_class,
                "distinct_contextual_single_token_variants": distinct_contextual_single_token_variants,
            }
        )

    return output


def rate(count: int, n: int) -> Optional[float]:
    return count / n if n else None


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_key"])].append(row)

    model_summaries = {}

    for model_key, model_rows in sorted(by_model.items()):
        n = len(model_rows)
        class_counts = Counter(str(row["tracked_variant_class"]) for row in model_rows)

        def count_true(field: str) -> int:
            return sum(bool(row[field]) for row in model_rows)

        model_summaries[model_key] = {
            "n": n,
            "tracked_variant_class_counts": dict(class_counts),
            "no_space_context_single_token_count": count_true(
                "no_space_context_single_token"
            ),
            "no_space_context_single_token_rate": rate(
                count_true("no_space_context_single_token"), n
            ),
            "leading_space_context_single_token_count": count_true(
                "leading_space_context_single_token"
            ),
            "leading_space_context_single_token_rate": rate(
                count_true("leading_space_context_single_token"), n
            ),
            "tracked_matches_no_space_context_count": count_true(
                "tracked_matches_no_space_context"
            ),
            "tracked_matches_no_space_context_rate": rate(
                count_true("tracked_matches_no_space_context"), n
            ),
            "tracked_matches_leading_space_context_count": count_true(
                "tracked_matches_leading_space_context"
            ),
            "tracked_matches_leading_space_context_rate": rate(
                count_true("tracked_matches_leading_space_context"), n
            ),
            "distinct_contextual_single_token_variants_count": count_true(
                "distinct_contextual_single_token_variants"
            ),
            "distinct_contextual_single_token_variants_rate": rate(
                count_true("distinct_contextual_single_token_variants"), n
            ),
            "no_space_context_boundary_clean_count": count_true(
                "no_space_context_boundary_clean"
            ),
            "leading_space_context_boundary_clean_count": count_true(
                "leading_space_context_boundary_clean"
            ),
            "no_space_context_covers_full_target_count": count_true(
                "no_space_context_covers_full_target"
            ),
            "leading_space_context_covers_full_target_count": count_true(
                "leading_space_context_covers_full_target"
            ),
        }

    return {
        "analysis": "translation_equivalent_token_variant_validation",
        "purpose": (
            "Check whether the English token tracked in EQ1 corresponds to "
            "the no-space or leading-space English lexical variant after the "
            "Chinese prefix, and whether both contextual variants are clean "
            "single tokens."
        ),
        "num_rows": len(rows),
        "num_models": len(by_model),
        "models": sorted(by_model),
        "tracked_variant_class_counts_overall": dict(
            Counter(str(row["tracked_variant_class"]) for row in rows)
        ),
        "model_summaries": model_summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate English translation-equivalent token variants after Chinese prefixes."
    )
    parser.add_argument(
        "--model",
        "--models",
        nargs="+",
        choices=["all", *MODEL_KEYS],
        default=["all"],
        help="Models to validate (default: all).",
    )
    parser.add_argument(
        "--input-dir",
        "--alignment-dir",
        "--alignment_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--local-files-only",
        "--local_files_only",
        action="store_true",
        help="Use cached tokenizers only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if "all" in args.model and args.model != ["all"]:
        parser.error("--model all cannot be combined with individual models.")
    model_keys = (
        list(MODEL_KEYS)
        if args.model == ["all"]
        else [key for key in MODEL_KEYS if key in args.model]
    )
    output_dir = args.output_dir
    if set(model_keys) != set(MODEL_KEYS):
        output_dir = output_dir / "__".join(model_keys)

    detail_path, report_path = (output_dir / name for name in OUTPUT_FILENAMES)
    if not args.overwrite and any(path.exists() for path in (detail_path, report_path)):
        raise FileExistsError(
            f"Existing validation outputs in {output_dir}; use --overwrite."
        )

    for model_key in model_keys:
        path = args.input_dir / model_key / INPUT_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    all_rows = []
    for model_key in model_keys:
        model_rows = run_model(
            model_key, MODEL_REGISTRY[model_key], args.input_dir, args.local_files_only
        )
        all_rows.extend(model_rows)
        n = len(model_rows)
        distinct = sum(
            row["distinct_contextual_single_token_variants"] for row in model_rows
        )
        print(
            f"{model_key}: {n} validated, {distinct}/{n} distinct contextual single-token variants"
        )

    report = summarize(all_rows)
    report["outputs"] = {
        "detail_tsv": str(detail_path),
        "report_json": str(report_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(detail_path, all_rows)
    write_json(report_path, report)
    print(f"Total: {len(all_rows)} validated examples")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
