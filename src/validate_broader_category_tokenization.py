from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_INPUT_PATH = Path("data/broader_category/broader_category_mapping_review.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/alignment/broader_category")

MODEL_REGISTRY: dict[str, str] = {
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_14b": "Qwen/Qwen3-14B",
}

MODEL_KEYS = tuple(MODEL_REGISTRY)

LANGUAGES = ("en", "zh")
EXPECTED_MAPPING_ROWS = 55

OUTPUT_FILENAMES = (
    "broader_category_tokenization_validation.tsv",
    "broader_category_alignment.jsonl",
    "broader_category_alignment_compact.tsv",
    "broader_category_valid_pairs.tsv",
    "broader_category_valid_pairs.jsonl",
    "broader_category_tokenization_rejected.jsonl",
    "broader_category_tokenization_report.json",
)

VALIDATION_FIELDS = (
    "id",
    "semantic_group",
    "subgroup",
    "relation",
    "mapping_rule",
    "model_key",
    "model_name",
    "language",
    "subject",
    "prefix",
    "gold",
    "broader_category",
    "gold_token_id",
    "gold_token",
    "gold_token_surface",
    "gold_decoded_token",
    "broader_token_id",
    "broader_token",
    "broader_token_surface",
    "broader_decoded_token",
    "prediction_position",
    "prediction_token_id",
    "prediction_token",
    "contexts_match",
    "prediction_positions_match",
    "prediction_token_ids_match",
    "gold_broader_token_ids_distinct",
    "gold_prefix_matches_full_before_target",
    "broader_prefix_matches_full_before_target",
    "valid_for_eq2",
    "rejection_reason",
)

COMPACT_FIELDS = (
    "id",
    "subgroup",
    "mapping_rule",
    "model_key",
    "model_name",
    "language",
    "subject",
    "prefix",
    "gold",
    "gold_token_id",
    "gold_token",
    "gold_token_surface",
    "broader_category",
    "broader_token_id",
    "broader_token",
    "broader_token_surface",
    "prediction_position",
    "prediction_token_id",
    "prediction_token",
    "gold_prefix_matches_full_before_target",
    "broader_prefix_matches_full_before_target",
    "valid_for_eq2",
)

REQUIRED_FIELDS = {
    "id",
    "semantic_group",
    "subgroup",
    "relation",
    "en_subject",
    "en_gold",
    "en_broader_category",
    "en_prefix",
    "en_gold_sentence",
    "en_broader_sentence",
    "zh_subject",
    "zh_gold",
    "zh_broader_category",
    "zh_prefix",
    "zh_gold_sentence",
    "zh_broader_sentence",
    "mapping_rule",
    "mapping_status",
}

EXPLICIT_KEEP = {
    "yes",
    "y",
    "true",
    "1",
    "keep",
    "include",
    "included",
    "accept",
    "accepted",
}
EXPLICIT_REJECT = {
    "no",
    "n",
    "false",
    "0",
    "reject",
    "rejected",
    "exclude",
    "excluded",
    "drop",
    "dropped",
}


def normalize_id(value: Any) -> str:
    return str(value)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input mapping file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_no}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_no}")
                row["_line_idx"] = line_no
                rows.append(row)

    elif suffix in {".tsv", ".txt"}:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"No TSV header found in: {path}")
            rows = []
            for line_no, row in enumerate(reader, start=2):
                item = dict(row)
                item["_line_idx"] = line_no
                rows.append(item)

    else:
        raise ValueError(f"Unsupported mapping format {suffix!r}. Use .jsonl or .tsv.")

    if not rows:
        raise ValueError(f"No mapping rows found in: {path}")

    seen_ids: set[str] = set()
    for row in rows:
        missing = REQUIRED_FIELDS.difference(row)
        if missing:
            raise ValueError(
                f"Mapping row id={row.get('id')} is missing required fields: "
                f"{sorted(missing)}"
            )

        example_id = normalize_id(row["id"])
        if example_id in seen_ids:
            raise ValueError(f"Duplicate mapping id: {example_id}")
        seen_ids.add(example_id)

        if str(row["semantic_group"]) != "category":
            raise ValueError(
                f"Mapping id={example_id} has semantic_group="
                f"{row['semantic_group']!r}, expected 'category'."
            )

        if str(row["mapping_status"]) != "manually_defined_broader_category":
            raise ValueError(
                f"Mapping id={example_id} has mapping_status="
                f"{row['mapping_status']!r}, expected "
                "'manually_defined_broader_category'."
            )

    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {
                key: value for key, value in row.items() if not key.startswith("_")
            }
            handle.write(json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n")


def write_tsv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:

    if fieldnames is None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = [key for key in rows[0].keys() if not key.startswith("_")]

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            cleaned: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(value, ensure_ascii=False)
                cleaned[key] = value
            writer.writerow(cleaned)


def normalize_manual_keep(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().casefold()


def row_is_candidate(
    row: dict[str, Any],
    require_manual_keep_yes: bool,
) -> bool:
    value = normalize_manual_keep(row.get("manual_keep", ""))

    if require_manual_keep_yes:
        return value in EXPLICIT_KEEP

    if value in EXPLICIT_REJECT:
        return False

    # Blank or explicitly positive values are included by default.
    return True


def offsets_overlap(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def resolve_target_char_span(
    full_sentence: str,
    prefix: str,
    target: str,
) -> tuple[int, int]:
    """Resolve the full-sentence target span, with a unique prefix fallback."""
    if full_sentence.startswith(prefix + target):
        start = len(prefix)
        return start, start + len(target)

    candidates: list[tuple[int, int]] = []
    search_from = 0

    while True:
        idx = full_sentence.find(target, search_from)
        if idx == -1:
            break

        before = full_sentence[:idx]
        if before.strip() == prefix.strip():
            candidates.append((idx, idx + len(target)))

        search_from = idx + max(len(target), 1)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise ValueError(
            "target_span_not_found:"
            f" prefix={prefix!r}, target={target!r}, "
            f"full_sentence={full_sentence!r}"
        )

    raise ValueError(
        "ambiguous_target_span:"
        f" prefix={prefix!r}, target={target!r}, "
        f"full_sentence={full_sentence!r}"
    )


def find_single_target_token(
    tokenizer,
    full_sentence: str,
    target: str,
    target_start: int,
    target_end: int,
) -> dict[str, Any]:
    """Require one complete target token, allowing leading whitespace only."""
    enc = tokenizer(
        full_sentence,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in enc["offset_mapping"]]

    overlapping_indices: list[int] = []
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if offsets_overlap(start, end, target_start, target_end):
            overlapping_indices.append(idx)

    if len(overlapping_indices) != 1:
        raise ValueError(
            "target_not_single_token_in_fullspan:"
            f" target={target!r}, "
            f"target_span=({target_start},{target_end}), "
            f"overlap_indices={overlapping_indices}, "
            f"tokens={tokenizer.convert_ids_to_tokens(input_ids)}"
        )

    target_token_index = overlapping_indices[0]
    token_start, token_end = offsets[target_token_index]
    token_surface = full_sentence[token_start:token_end]

    # Same convention as the main experiment: allow only leading whitespace.
    if token_surface.lstrip() != target:
        raise ValueError(
            "target_token_surface_mismatch:"
            f" target={target!r}, "
            f"token_surface={token_surface!r}, "
            f"token_span=({token_start},{token_end})"
        )

    if target_token_index == 0:
        raise ValueError("target_token_is_first_token_no_prediction_position")

    target_token_id = int(input_ids[target_token_index])
    target_token = tokenizer.convert_ids_to_tokens([target_token_id])[0]
    decoded_token = tokenizer.decode(
        [target_token_id],
        clean_up_tokenization_spaces=False,
    )

    prediction_position = target_token_index - 1
    prediction_token_id = int(input_ids[prediction_position])
    prediction_token = tokenizer.convert_ids_to_tokens([prediction_token_id])[0]

    return {
        "full_input_ids": input_ids,
        "full_tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "offset_mapping": offsets,
        "target_token_index": target_token_index,
        "target_token_id": target_token_id,
        "target_token": target_token,
        "target_decoded_token": decoded_token,
        "target_token_surface": token_surface,
        "target_token_char_start": token_start,
        "target_token_char_end": token_end,
        "prediction_position": prediction_position,
        "prediction_token_id": prediction_token_id,
        "prediction_token": prediction_token,
    }


def build_alignment(
    tokenizer,
    *,
    prefix: str,
    target: str,
    full_sentence: str,
) -> dict[str, Any]:
    target_start, target_end = resolve_target_char_span(
        full_sentence=full_sentence,
        prefix=prefix,
        target=target,
    )

    alignment = find_single_target_token(
        tokenizer=tokenizer,
        full_sentence=full_sentence,
        target=target,
        target_start=target_start,
        target_end=target_end,
    )

    prefix_enc = tokenizer(
        prefix,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prefix_input_ids = [int(x) for x in prefix_enc["input_ids"]]
    prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_input_ids)

    if not prefix_input_ids:
        raise ValueError("empty_prefix_tokenization")

    full_ids_before_target = alignment["full_input_ids"][
        : alignment["target_token_index"]
    ]

    alignment["target_char_start"] = target_start
    alignment["target_char_end"] = target_end
    alignment["prefix_input_ids"] = prefix_input_ids
    alignment["prefix_tokens"] = prefix_tokens
    alignment["prefix_prediction_position"] = len(prefix_input_ids) - 1
    alignment["prefix_matches_full_before_target"] = (
        prefix_input_ids == full_ids_before_target
    )
    alignment["context_input_ids"] = full_ids_before_target
    alignment["context_tokens"] = tokenizer.convert_ids_to_tokens(
        full_ids_before_target
    )

    return alignment


def language_fields(
    row: dict[str, Any],
    language: str,
) -> dict[str, str]:
    if language == "en":
        return {
            "subject": str(row["en_subject"]),
            "prefix": str(row["en_prefix"]),
            "gold": str(row["en_gold"]),
            "broader": str(row["en_broader_category"]),
            "gold_sentence": str(row["en_gold_sentence"]),
            "broader_sentence": str(row["en_broader_sentence"]),
        }

    if language == "zh":
        return {
            "subject": str(row["zh_subject"]),
            "prefix": str(row["zh_prefix"]),
            "gold": str(row["zh_gold"]),
            "broader": str(row["zh_broader_category"]),
            "gold_sentence": str(row["zh_gold_sentence"]),
            "broader_sentence": str(row["zh_broader_sentence"]),
        }

    raise ValueError(f"Unsupported language: {language!r}")


def validate_one_language(
    row: dict[str, Any],
    tokenizer,
    *,
    model_key: str,
    model_name: str,
    language: str,
) -> dict[str, Any]:
    fields = language_fields(row, language)

    gold_alignment = build_alignment(
        tokenizer,
        prefix=fields["prefix"],
        target=fields["gold"],
        full_sentence=fields["gold_sentence"],
    )

    broader_alignment = build_alignment(
        tokenizer,
        prefix=fields["prefix"],
        target=fields["broader"],
        full_sentence=fields["broader_sentence"],
    )

    contexts_match = (
        gold_alignment["context_input_ids"] == broader_alignment["context_input_ids"]
    )

    prediction_positions_match = (
        gold_alignment["prediction_position"]
        == broader_alignment["prediction_position"]
    )

    prediction_token_ids_match = (
        gold_alignment["prediction_token_id"]
        == broader_alignment["prediction_token_id"]
    )

    token_ids_distinct = (
        gold_alignment["target_token_id"] != broader_alignment["target_token_id"]
    )

    if not contexts_match:
        raise ValueError(
            "gold_broader_context_tokenization_mismatch:"
            f" gold_context_ids={gold_alignment['context_input_ids']},"
            f" broader_context_ids={broader_alignment['context_input_ids']}"
        )

    if not prediction_positions_match:
        raise ValueError(
            "gold_broader_prediction_position_mismatch:"
            f" gold={gold_alignment['prediction_position']},"
            f" broader={broader_alignment['prediction_position']}"
        )

    if not prediction_token_ids_match:
        raise ValueError(
            "gold_broader_prediction_token_mismatch:"
            f" gold={gold_alignment['prediction_token_id']},"
            f" broader={broader_alignment['prediction_token_id']}"
        )

    if not token_ids_distinct:
        raise ValueError(
            "gold_and_broader_share_same_vocab_token_id:"
            f" token_id={gold_alignment['target_token_id']}"
        )

    return {
        "id": normalize_id(row["id"]),
        "semantic_group": str(row["semantic_group"]),
        "subgroup": str(row["subgroup"]),
        "relation": str(row["relation"]),
        "mapping_rule": str(row["mapping_rule"]),
        "mapping_status": str(row["mapping_status"]),
        "manual_keep": row.get("manual_keep", ""),
        "manual_note": row.get("manual_note", ""),
        "model_key": model_key,
        "model_name": model_name,
        "language": language,
        "subject": fields["subject"],
        "prefix": fields["prefix"],
        "gold": fields["gold"],
        "broader_category": fields["broader"],
        "gold_sentence": fields["gold_sentence"],
        "broader_sentence": fields["broader_sentence"],
        # Canonical context to feed into later EQ2 extraction.
        # We use the gold full-sentence tokenization, exactly as in the main
        # full-span setup. Causal masking prevents later target/suffix tokens
        # from affecting the prediction-position hidden state.
        "full_input_ids": gold_alignment["full_input_ids"],
        "full_tokens": gold_alignment["full_tokens"],
        "prediction_position": gold_alignment["prediction_position"],
        "prediction_token_id": gold_alignment["prediction_token_id"],
        "prediction_token": gold_alignment["prediction_token"],
        # Gold token.
        "gold_token_id": gold_alignment["target_token_id"],
        "gold_token": gold_alignment["target_token"],
        "gold_decoded_token": gold_alignment["target_decoded_token"],
        "gold_token_surface": gold_alignment["target_token_surface"],
        "gold_target_token_index": gold_alignment["target_token_index"],
        # Broader-category token.
        "broader_token_id": broader_alignment["target_token_id"],
        "broader_token": broader_alignment["target_token"],
        "broader_decoded_token": broader_alignment["target_decoded_token"],
        "broader_token_surface": broader_alignment["target_token_surface"],
        "broader_target_token_index": broader_alignment["target_token_index"],
        # Pairwise tokenizer checks.
        "contexts_match": contexts_match,
        "prediction_positions_match": prediction_positions_match,
        "prediction_token_ids_match": prediction_token_ids_match,
        "gold_broader_token_ids_distinct": token_ids_distinct,
        # Prefix-only tokenization is diagnostic only.
        "gold_prefix_matches_full_before_target": gold_alignment[
            "prefix_matches_full_before_target"
        ],
        "broader_prefix_matches_full_before_target": broader_alignment[
            "prefix_matches_full_before_target"
        ],
        "valid_for_eq2": True,
    }


def summarize_candidate_rows(
    rows: Sequence[dict[str, Any]],
    require_manual_keep_yes: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    manual_counter = Counter(
        normalize_manual_keep(row.get("manual_keep", "")) or "<blank>" for row in rows
    )

    candidates = [
        row
        for row in rows
        if row_is_candidate(
            row,
            require_manual_keep_yes=require_manual_keep_yes,
        )
    ]

    return candidates, manual_counter


def final_valid_pair_ids(
    candidates: Sequence[dict[str, Any]],
    successful_alignments: Sequence[dict[str, Any]],
) -> set[str]:
    expected_keys = {
        (model_key, language) for model_key in MODEL_KEYS for language in LANGUAGES
    }

    observed: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for row in successful_alignments:
        if not bool(row.get("valid_for_eq2")):
            continue
        observed[normalize_id(row["id"])].add(
            (str(row["model_key"]), str(row["language"]))
        )

    valid_ids = {
        normalize_id(row["id"])
        for row in candidates
        if observed.get(normalize_id(row["id"]), set()) == expected_keys
    }

    return valid_ids


def parse_reason_type(reason: str) -> str:
    if not reason:
        return "unknown"
    return reason.split(":", 1)[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate broader-category alignment across all three Qwen3 tokenizers."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--output-dir", "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--require-manual-keep-yes",
        "--require_manual_keep_yes",
        action="store_true",
        help="Include only explicitly approved mappings; otherwise include blanks and skip explicit rejects.",
    )
    parser.add_argument(
        "--local-files-only",
        "--local_files_only",
        action="store_true",
        help="Load tokenizers from the local cache only.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing validation outputs."
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    args = build_parser().parse_args(argv)

    input_path = args.input
    output_dir = args.output_dir

    mapping_rows = read_rows(input_path)
    candidates, manual_keep_counts = summarize_candidate_rows(
        mapping_rows,
        require_manual_keep_yes=bool(args.require_manual_keep_yes),
    )

    if len(mapping_rows) != EXPECTED_MAPPING_ROWS:
        print(
            f"WARNING: expected {EXPECTED_MAPPING_ROWS} mapping rows, "
            f"found {len(mapping_rows)}."
        )

    if not candidates:
        raise ValueError(
            "No candidate mapping rows remain after manual_keep filtering."
        )

    existing = [
        output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Existing validation outputs in {output_dir}; use --overwrite or another directory."
        )
    if (
        len(candidates) != len(mapping_rows)
        and output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
    ):
        raise ValueError("Filtered mappings require a separate --output-dir.")
    if (
        len(mapping_rows) != EXPECTED_MAPPING_ROWS
        and output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
    ):
        raise ValueError("Nonstandard mapping counts require a separate --output-dir.")

    from transformers import AutoTokenizer

    all_successes: list[dict[str, Any]] = []
    all_attempt_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for model_key in MODEL_KEYS:
        model_name = MODEL_REGISTRY[model_key]
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=bool(args.local_files_only),
        )

        if not tokenizer.is_fast:
            raise RuntimeError(
                f"Tokenizer for {model_name} is not fast. "
                "The canonical full-span validation requires "
                "offset_mapping support."
            )

        model_success = 0
        model_rejected = 0

        for item in candidates:
            for language in LANGUAGES:
                base_attempt = {
                    "id": normalize_id(item["id"]),
                    "semantic_group": str(item["semantic_group"]),
                    "subgroup": str(item["subgroup"]),
                    "relation": str(item["relation"]),
                    "mapping_rule": str(item["mapping_rule"]),
                    "model_key": model_key,
                    "model_name": model_name,
                    "language": language,
                }

                fields = language_fields(item, language)

                try:
                    alignment = validate_one_language(
                        item,
                        tokenizer,
                        model_key=model_key,
                        model_name=model_name,
                        language=language,
                    )

                    all_successes.append(alignment)
                    model_success += 1

                    all_attempt_rows.append(
                        {
                            **base_attempt,
                            "subject": fields["subject"],
                            "prefix": fields["prefix"],
                            "gold": fields["gold"],
                            "broader_category": fields["broader"],
                            "gold_token_id": alignment["gold_token_id"],
                            "gold_token": alignment["gold_token"],
                            "gold_token_surface": alignment["gold_token_surface"],
                            "gold_decoded_token": alignment["gold_decoded_token"],
                            "broader_token_id": alignment["broader_token_id"],
                            "broader_token": alignment["broader_token"],
                            "broader_token_surface": alignment["broader_token_surface"],
                            "broader_decoded_token": alignment["broader_decoded_token"],
                            "prediction_position": alignment["prediction_position"],
                            "prediction_token_id": alignment["prediction_token_id"],
                            "prediction_token": alignment["prediction_token"],
                            "contexts_match": alignment["contexts_match"],
                            "prediction_positions_match": alignment[
                                "prediction_positions_match"
                            ],
                            "prediction_token_ids_match": alignment[
                                "prediction_token_ids_match"
                            ],
                            "gold_broader_token_ids_distinct": alignment[
                                "gold_broader_token_ids_distinct"
                            ],
                            "gold_prefix_matches_full_before_target": alignment[
                                "gold_prefix_matches_full_before_target"
                            ],
                            "broader_prefix_matches_full_before_target": alignment[
                                "broader_prefix_matches_full_before_target"
                            ],
                            "valid_for_eq2": True,
                            "rejection_reason": "",
                        }
                    )

                except Exception as exc:
                    reason = str(exc)
                    model_rejected += 1

                    rejection_row = {
                        **base_attempt,
                        "subject": fields["subject"],
                        "prefix": fields["prefix"],
                        "gold": fields["gold"],
                        "broader_category": fields["broader"],
                        "gold_sentence": fields["gold_sentence"],
                        "broader_sentence": fields["broader_sentence"],
                        "reason": reason,
                        "reason_type": parse_reason_type(reason),
                        "manual_keep": item.get("manual_keep", ""),
                        "manual_note": item.get("manual_note", ""),
                    }
                    rejected.append(rejection_row)

                    all_attempt_rows.append(
                        {
                            **base_attempt,
                            "subject": fields["subject"],
                            "prefix": fields["prefix"],
                            "gold": fields["gold"],
                            "broader_category": fields["broader"],
                            "gold_token_id": "",
                            "gold_token": "",
                            "gold_token_surface": "",
                            "gold_decoded_token": "",
                            "broader_token_id": "",
                            "broader_token": "",
                            "broader_token_surface": "",
                            "broader_decoded_token": "",
                            "prediction_position": "",
                            "prediction_token_id": "",
                            "prediction_token": "",
                            "contexts_match": False,
                            "prediction_positions_match": False,
                            "prediction_token_ids_match": False,
                            "gold_broader_token_ids_distinct": "",
                            "gold_prefix_matches_full_before_target": "",
                            "broader_prefix_matches_full_before_target": "",
                            "valid_for_eq2": False,
                            "rejection_reason": reason,
                        }
                    )

        print(f"{model_key}: {model_success} aligned, {model_rejected} rejected")

    # A bilingual pair is final-valid only if it passes EN + ZH under all
    # three tokenizers.
    valid_ids = final_valid_pair_ids(
        candidates,
        all_successes,
    )

    final_mapping_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in candidates
        if normalize_id(row["id"]) in valid_ids
    ]

    canonical_alignments = [
        row for row in all_successes if normalize_id(row["id"]) in valid_ids
    ]

    # Stable ordering.
    model_order = {key: idx for idx, key in enumerate(MODEL_KEYS)}
    language_order = {"en": 0, "zh": 1}

    def id_sort_value(value: Any):
        text = normalize_id(value)
        return (0, int(text)) if text.isdigit() else (1, text)

    all_attempt_rows.sort(
        key=lambda row: (
            id_sort_value(row["id"]),
            model_order[str(row["model_key"])],
            language_order[str(row["language"])],
        )
    )
    canonical_alignments.sort(
        key=lambda row: (
            id_sort_value(row["id"]),
            model_order[str(row["model_key"])],
            language_order[str(row["language"])],
        )
    )
    final_mapping_rows.sort(key=lambda row: id_sort_value(row["id"]))
    rejected.sort(
        key=lambda row: (
            id_sort_value(row["id"]),
            model_order[str(row["model_key"])],
            language_order[str(row["language"])],
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    validation_tsv = output_dir / "broader_category_tokenization_validation.tsv"
    alignment_jsonl = output_dir / "broader_category_alignment.jsonl"
    alignment_compact_tsv = output_dir / "broader_category_alignment_compact.tsv"
    valid_pairs_tsv = output_dir / "broader_category_valid_pairs.tsv"
    valid_pairs_jsonl = output_dir / "broader_category_valid_pairs.jsonl"
    rejected_jsonl = output_dir / "broader_category_tokenization_rejected.jsonl"
    report_json = output_dir / "broader_category_tokenization_report.json"

    write_tsv(
        validation_tsv,
        all_attempt_rows,
        VALIDATION_FIELDS,
    )

    write_jsonl(
        alignment_jsonl,
        canonical_alignments,
    )

    write_tsv(
        alignment_compact_tsv,
        canonical_alignments,
        COMPACT_FIELDS,
    )

    write_tsv(
        valid_pairs_tsv,
        final_mapping_rows,
    )
    write_jsonl(
        valid_pairs_jsonl,
        final_mapping_rows,
    )
    write_jsonl(
        rejected_jsonl,
        rejected,
    )

    successes_by_model = Counter(row["model_key"] for row in all_successes)
    successes_by_model_language = Counter(
        (row["model_key"], row["language"]) for row in all_successes
    )
    rejections_by_reason = Counter(row["reason_type"] for row in rejected)

    final_subgroup_counts = Counter(row["subgroup"] for row in final_mapping_rows)

    candidate_ids = {normalize_id(row["id"]) for row in candidates}
    invalid_pair_ids = sorted(
        candidate_ids - valid_ids,
        key=id_sort_value,
    )

    # Prefix/full consistency is diagnostic only.
    prefix_diag_gold = Counter(
        str(row["gold_prefix_matches_full_before_target"]) for row in all_successes
    )
    prefix_diag_broader = Counter(
        str(row["broader_prefix_matches_full_before_target"]) for row in all_successes
    )

    report = {
        "analysis": "validate_broader_category_tokenization",
        "purpose": (
            "Validate gold and manually defined broader-category "
            "single-token directions using the same full-span "
            "alignment convention as the frozen main experiment."
        ),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "models": MODEL_REGISTRY,
        "languages": list(LANGUAGES),
        "validity_definition": {
            "target_is_single_token_in_full_sentence_span": True,
            "leading_whitespace_in_target_token_surface_allowed": True,
            "trailing_punctuation_or_extra_surface_rejected": True,
            "gold_and_broader_context_tokens_must_match": True,
            "prediction_position_must_match": True,
            "prediction_token_id_must_match": True,
            "gold_and_broader_token_ids_must_be_distinct": True,
            "prefix_only_tokenization_match_is_diagnostic_only": True,
        },
        "manual_review_filter": {
            "require_manual_keep_yes": bool(args.require_manual_keep_yes),
            "manual_keep_counts_in_input": dict(manual_keep_counts),
        },
        "counts": {
            "mapping_rows_loaded": len(mapping_rows),
            "candidate_mapping_rows": len(candidates),
            "expected_language_attempts": (
                len(candidates) * len(MODEL_KEYS) * len(LANGUAGES)
            ),
            "successful_language_alignments": len(all_successes),
            "rejected_language_alignments": len(rejected),
            "final_valid_bilingual_pairs_all_models": len(final_mapping_rows),
            "invalid_bilingual_pairs": len(invalid_pair_ids),
            "canonical_alignment_rows": len(canonical_alignments),
        },
        "successful_alignments_by_model": {
            model_key: successes_by_model.get(model_key, 0) for model_key in MODEL_KEYS
        },
        "successful_alignments_by_model_language": {
            model_key: {
                language: successes_by_model_language.get((model_key, language), 0)
                for language in LANGUAGES
            }
            for model_key in MODEL_KEYS
        },
        "final_valid_pairs_by_subgroup": dict(sorted(final_subgroup_counts.items())),
        "invalid_pair_ids": invalid_pair_ids,
        "rejections_by_reason_type": dict(sorted(rejections_by_reason.items())),
        "prefix_only_diagnostics": {
            "gold_prefix_matches_full_before_target": dict(prefix_diag_gold),
            "broader_prefix_matches_full_before_target": dict(prefix_diag_broader),
            "note": (
                "These prefix-only matches are diagnostics and are "
                "not part of the full-span validity criterion."
            ),
        },
        "outputs": {
            "validation_tsv": str(validation_tsv),
            "canonical_alignment_jsonl": str(alignment_jsonl),
            "canonical_alignment_compact_tsv": str(alignment_compact_tsv),
            "valid_pairs_tsv": str(valid_pairs_tsv),
            "valid_pairs_jsonl": str(valid_pairs_jsonl),
            "rejected_jsonl": str(rejected_jsonl),
            "report_json": str(report_json),
        },
    }

    write_json(report_json, report)

    print(
        f"Total: {len(final_mapping_rows)}/{len(candidates)} valid bilingual pairs, "
        f"{len(canonical_alignments)} canonical alignment rows"
    )
    if len(final_mapping_rows) != len(candidates):
        print(f"WARNING: {len(invalid_pair_ids)} pairs failed; see rejection report.")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
