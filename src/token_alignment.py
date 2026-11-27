import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_INPUT_PATH = Path("data/dataset/fullspan_dataset.jsonl")

DEFAULT_OUTPUT_DIR = Path("data/alignment")

MODEL_REGISTRY = {
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_14b": "Qwen/Qwen3-14B",
}

MODEL_KEYS = tuple(MODEL_REGISTRY)
LANGUAGES = ("en", "zh")

OUTPUT_FILENAMES = (
    "token_alignment.jsonl",
    "token_alignment_compact.tsv",
    "token_alignment.rejected.jsonl",
    "token_alignment_summary.json",
)

COMPACT_FIELDS = (
    "id",
    "semantic_group",
    "subgroup",
    "relation",
    "language",
    "model_key",
    "target",
    "target_token_id",
    "target_token",
    "target_token_surface",
    "target_token_index",
    "prediction_position",
    "prediction_token",
    "prefix_matches_full_before_target",
)


def read_jsonl(path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_idx}")
            item["_line_idx"] = line_idx
            items.append(item)
    if not items:
        raise ValueError(f"No rows found in {path}")
    return items


def write_jsonl(path, items):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_tsv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def require_field(item, field):
    if field not in item:
        raise ValueError(f"missing_field:{field}")

    value = item[field]

    if value is None:
        raise ValueError(f"none_field:{field}")

    if isinstance(value, str) and value == "":
        raise ValueError(f"empty_field:{field}")

    return value


def get_lang_fields(item, lang):
    subject = require_field(item, f"{lang}_subject")
    target = require_field(item, f"{lang}_target")
    prefix = require_field(item, f"{lang}_prefix")
    full_sentence = require_field(item, f"{lang}_full_sentence")

    return {
        "subject": subject,
        "target": target,
        "prefix": prefix,
        "full_sentence": full_sentence,
    }


def resolve_target_char_span(full_sentence, prefix, target):
    """Locate the target, allowing whitespace differences at the prefix boundary."""

    if full_sentence.startswith(prefix + target):
        start = len(prefix)
        end = start + len(target)
        return start, end

    candidates = []
    search_from = 0

    while True:
        idx = full_sentence.find(target, search_from)
        if idx == -1:
            break

        before = full_sentence[:idx]

        if before.strip() == str(prefix).strip():
            candidates.append((idx, idx + len(target)))

        search_from = idx + len(target)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise ValueError(
            "target_span_not_found:"
            f" prefix={repr(prefix)}, target={repr(target)}, "
            f"full_sentence={repr(full_sentence)}"
        )

    raise ValueError(
        "ambiguous_target_span:"
        f" prefix={repr(prefix)}, target={repr(target)}, "
        f"full_sentence={repr(full_sentence)}"
    )


def offsets_overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


def find_single_target_token(
    tokenizer, full_sentence, target, target_start, target_end
):
    """Require one complete target token, optionally with leading whitespace."""

    enc = tokenizer(
        full_sentence,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    overlapping_token_indices = []

    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue

        if offsets_overlap(s, e, target_start, target_end):
            overlapping_token_indices.append(i)

    if len(overlapping_token_indices) != 1:
        raise ValueError(
            "target_not_single_token_in_fullspan:"
            f" target={repr(target)}, "
            f"target_span=({target_start}, {target_end}), "
            f"overlap_indices={overlapping_token_indices}, "
            f"tokens={tokenizer.convert_ids_to_tokens(input_ids)}"
        )

    target_token_index = overlapping_token_indices[0]
    token_start, token_end = offsets[target_token_index]
    token_surface = full_sentence[token_start:token_end]

    if token_surface.lstrip() != target:
        raise ValueError(
            "target_token_surface_mismatch:"
            f" target={repr(target)}, "
            f"token_surface={repr(token_surface)}, "
            f"token_span=({token_start}, {token_end})"
        )

    target_token_id = input_ids[target_token_index]
    target_token = tokenizer.convert_ids_to_tokens([target_token_id])[0]

    if target_token_index == 0:
        raise ValueError("target_token_is_first_token_no_prediction_position")

    prediction_position = target_token_index - 1

    return {
        "full_input_ids": input_ids,
        "full_tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "offset_mapping": offsets,
        "target_token_index": target_token_index,
        "target_token_id": target_token_id,
        "target_token": target_token,
        "target_token_surface": token_surface,
        "target_token_char_start": token_start,
        "target_token_char_end": token_end,
        "prediction_position": prediction_position,
        "prediction_token_id": input_ids[prediction_position],
        "prediction_token": tokenizer.convert_ids_to_tokens(
            [input_ids[prediction_position]]
        )[0],
    }


def build_alignment_for_one(item, tokenizer, model_key, model_name, lang):
    fields = get_lang_fields(item, lang)

    subject = fields["subject"]
    target = fields["target"]
    prefix = fields["prefix"]
    full_sentence = fields["full_sentence"]

    target_start, target_end = resolve_target_char_span(
        full_sentence=full_sentence,
        prefix=prefix,
        target=target,
    )

    target_alignment = find_single_target_token(
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

    prefix_input_ids = prefix_enc["input_ids"]
    prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_input_ids)

    if len(prefix_input_ids) == 0:
        raise ValueError("empty_prefix_tokenization")

    prefix_prediction_position = len(prefix_input_ids) - 1

    # Prefix-only tokenization is diagnostic; experiments use the fullspan position.
    full_ids_before_target = target_alignment["full_input_ids"][
        : target_alignment["target_token_index"]
    ]

    prefix_matches_full_before_target = prefix_input_ids == full_ids_before_target

    output = {
        "id": item["id"],
        "semantic_group": item["semantic_group"],
        "subgroup": item["subgroup"],
        "relation": item["relation"],
        "language": lang,
        "model_key": model_key,
        "model_name": model_name,
        "subject": subject,
        "target": target,
        "prefix": prefix,
        "full_sentence": full_sentence,
        "target_char_start": target_start,
        "target_char_end": target_end,
        "full_input_ids": target_alignment["full_input_ids"],
        "full_tokens": target_alignment["full_tokens"],
        "offset_mapping": target_alignment["offset_mapping"],
        "target_token_index": target_alignment["target_token_index"],
        "target_token_id": target_alignment["target_token_id"],
        "target_token": target_alignment["target_token"],
        "target_token_surface": target_alignment["target_token_surface"],
        "target_token_char_start": target_alignment["target_token_char_start"],
        "target_token_char_end": target_alignment["target_token_char_end"],
        "prediction_position": target_alignment["prediction_position"],
        "prediction_token_id": target_alignment["prediction_token_id"],
        "prediction_token": target_alignment["prediction_token"],
        "prefix_input_ids": prefix_input_ids,
        "prefix_tokens": prefix_tokens,
        "prefix_prediction_position": prefix_prediction_position,
        "prefix_prediction_token_id": prefix_input_ids[prefix_prediction_position],
        "prefix_prediction_token": tokenizer.convert_ids_to_tokens(
            [prefix_input_ids[prefix_prediction_position]]
        )[0],
        "prefix_matches_full_before_target": prefix_matches_full_before_target,
    }

    return output


def run_model(items, model_key, model_name, args):
    from transformers import AutoTokenizer

    output_dir = args.output_dir / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment_jsonl, alignment_tsv, rejected_jsonl, summary_json = (
        output_dir / filename for filename in OUTPUT_FILENAMES
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            f"Tokenizer for {model_name} is not fast. "
            "Offset mapping requires a fast tokenizer."
        )

    model_alignments = []
    model_rejected = []

    for item in items:
        for lang in LANGUAGES:
            try:
                alignment = build_alignment_for_one(
                    item=item,
                    tokenizer=tokenizer,
                    model_key=model_key,
                    model_name=model_name,
                    lang=lang,
                )
                model_alignments.append(alignment)

            except Exception as e:
                model_rejected.append(
                    {
                        "id": item.get("id"),
                        "line_idx": item.get("_line_idx"),
                        "model_key": model_key,
                        "model_name": model_name,
                        "language": lang,
                        "reason": str(e),
                        "item": {
                            k: v for k, v in item.items() if not k.startswith("_")
                        },
                    }
                )

    write_jsonl(alignment_jsonl, model_alignments)
    write_jsonl(rejected_jsonl, model_rejected)

    write_tsv(alignment_tsv, model_alignments, COMPACT_FIELDS)

    summary = {
        "input_jsonl": str(args.input),
        "output_dir": str(output_dir),
        "model_key": model_key,
        "model_name": model_name,
        "num_input_rows": len(items),
        "languages": list(LANGUAGES),
        "num_alignment_rows_expected": len(items) * len(LANGUAGES),
        "num_alignment_rows_written": len(model_alignments),
        "num_rejected": len(model_rejected),
        "counts_by_language": dict(Counter(a["language"] for a in model_alignments)),
        "counts_by_semantic_group": dict(
            Counter(a["semantic_group"] for a in model_alignments)
        ),
        "counts_by_subgroup": dict(Counter(a["subgroup"] for a in model_alignments)),
        "counts_by_relation": dict(Counter(a["relation"] for a in model_alignments)),
        "prefix_matches_full_before_target": dict(
            Counter(
                str(a["prefix_matches_full_before_target"]) for a in model_alignments
            )
        ),
    }

    write_json(summary_json, summary)

    print(
        f"{model_key}: {len(model_alignments)} aligned, {len(model_rejected)} rejected"
    )
    if model_rejected:
        print(f"  Review rejected rows: {rejected_jsonl}")

    return len(model_alignments), len(model_rejected)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Align full-span targets for Qwen3 tokenizers."
    )
    parser.add_argument(
        "--model",
        "--models",
        nargs="+",
        choices=["all", *MODEL_KEYS],
        default=["all"],
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--output-dir", "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files."
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if "all" in args.model and args.model != ["all"]:
        parser.error("--model all cannot be combined with individual models.")
    model_keys = (
        MODEL_KEYS
        if args.model == ["all"]
        else tuple(key for key in MODEL_KEYS if key in args.model)
    )
    items = read_jsonl(args.input)
    output_dir = args.output_dir
    if not args.overwrite:
        existing = [
            output_dir / model_key / filename
            for model_key in model_keys
            for filename in OUTPUT_FILENAMES
            if (output_dir / model_key / filename).exists()
        ]
        if existing:
            parser.error(
                f"Output exists: {existing[0]}. Use --overwrite or a separate --output-dir."
            )

    total_aligned = total_rejected = 0
    for model_key in model_keys:
        aligned, rejected = run_model(items, model_key, MODEL_REGISTRY[model_key], args)
        total_aligned += aligned
        total_rejected += rejected

    print(f"Total: {total_aligned} aligned, {total_rejected} rejected")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
