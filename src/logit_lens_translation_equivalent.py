"""
python src/logit_lens_translation_equivalent.py \
    --models qwen3_0_6b qwen3_4b qwen3_14b

"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Config
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_ALIGNMENT_DIR = Path("data/alignment")
BASE_OUTPUT_DIR = Path("data/logit_lens/translation_equivalent")

MODELS = {
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_14b": "Qwen/Qwen3-14B",
}

# Keep the projection definition identical to the frozen main experiment.
APPLY_FINAL_NORM = True
TOP_K = 10
COMPUTE_PROB = True
STORE_TOPK = True

# Set to a small integer only for debugging. This limit is applied AFTER
# English/Chinese pairing, so a debug run still has valid pairs.
MAX_EXAMPLES: Optional[int] = None
PROGRESS_EVERY = 50

# Keep False first. If a large model does not fit on one GPU and accelerate is
# installed/configured, this can be changed to True.
USE_DEVICE_MAP_AUTO = False


# ============================================================
# I/O helpers
# ============================================================


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_idx}")
            row["_line_idx"] = line_idx
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_tsv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                cleaned[key] = value
            writer.writerow(cleaned)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


# ============================================================
# Device / model helpers
# ============================================================


def get_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float16

    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32

    return torch.device("cpu"), torch.float32


def load_model_and_tokenizer(model_name: str):
    device, dtype = get_device_and_dtype()

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if USE_DEVICE_MAP_AUTO:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)

    model.eval()
    return model, tokenizer


def get_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def get_final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm

    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f

    return None


def get_module_device(module) -> Optional[torch.device]:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def project_hidden_to_logits(
    hidden_state: torch.Tensor,
    final_norm,
    lm_head,
    layer_idx: int,
    last_layer_idx: int,
) -> torch.Tensor:
    """
    Plain Logit Lens projection, kept identical to the main experiment.

    - embedding layer: no final norm
    - intermediate layers: final norm + lm_head
    - final layer: handled separately with outputs.logits
    """
    h = hidden_state.unsqueeze(0).unsqueeze(0)

    should_apply_norm = (
        APPLY_FINAL_NORM
        and final_norm is not None
        and layer_idx != 0
        and layer_idx != last_layer_idx
    )

    if should_apply_norm:
        norm_device = get_module_device(final_norm)
        if norm_device is not None:
            h = h.to(norm_device)
        h = final_norm(h)

    head_device = get_module_device(lm_head)
    if head_device is not None:
        h = h.to(head_device)

    return lm_head(h).squeeze(0).squeeze(0).float()


# ============================================================
# Logit / rank helpers
# ============================================================


def rank_of_token(logits: torch.Tensor, token_id: int) -> tuple[int, float]:
    """
    1-based full-vocabulary rank, identical to the main experiment.

    rank = 1 means no vocabulary token has a strictly larger logit.
    Ties therefore receive the same numerical rank. Actual Target@k membership
    is computed separately from torch.topk.
    """
    token_logit = logits[int(token_id)]
    rank = int((logits > token_logit).sum().item()) + 1
    return rank, float(token_logit.item())


def get_topk_info(logits: torch.Tensor, tokenizer, k: int) -> dict[str, Any]:
    top_values, top_ids = torch.topk(logits, k=k)
    top_ids = [int(x) for x in top_ids.detach().cpu().tolist()]
    top_values = [float(x) for x in top_values.detach().cpu().tolist()]

    top_tokens = tokenizer.convert_ids_to_tokens(top_ids)
    top_texts = [tokenizer.decode([token_id]) for token_id in top_ids]

    return {
        "topk_token_ids": top_ids,
        "topk_tokens": top_tokens,
        "topk_texts": top_texts,
        "topk_logits": top_values,
    }


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


# ============================================================
# Pairing and tokenizer/alignment validation
# ============================================================


def normalize_id(value: Any) -> str:
    return str(value)


def build_language_pairs(
    rows: Sequence[dict[str, Any]],
    model_key: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Return (English row, Chinese row) pairs keyed by example id.

    The same canonical model-specific alignment file is used for both
    languages. Duplicate language rows for one id are treated as errors.
    """
    buckets: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in rows:
        if str(row.get("model_key")) != model_key:
            continue

        language = str(row.get("language", ""))
        if language not in {"en", "zh"}:
            continue

        example_id = normalize_id(row.get("id"))
        if language in buckets[example_id]:
            raise ValueError(
                f"Duplicate {language} alignment row for model={model_key}, "
                f"id={example_id}."
            )
        buckets[example_id][language] = row

    missing = {
        example_id: sorted({"en", "zh"} - set(language_rows))
        for example_id, language_rows in buckets.items()
        if set(language_rows) != {"en", "zh"}
    }
    if missing:
        preview = list(missing.items())[:10]
        raise ValueError(
            f"Found {len(missing)} example IDs without a complete EN/ZH pair "
            f"for {model_key}. Examples: {preview}"
        )

    pairs = [
        (buckets[example_id]["en"], buckets[example_id]["zh"])
        for example_id in sorted(buckets)
    ]

    if MAX_EXAMPLES is not None:
        pairs = pairs[:MAX_EXAMPLES]

    return pairs


def optional_single_token_span_check(
    row: dict[str, Any], token_id: int
) -> tuple[bool, str]:
    """
    Use any explicit target-span token-id field if the alignment file has one.

    Different versions of the alignment pipeline may use different names, so
    several common names are checked. If none is present, this check is marked
    as not available rather than treated as a failure.
    """
    candidate_fields = (
        "target_span_token_ids",
        "target_token_ids",
        "target_input_ids",
    )

    found_any = False
    for field in candidate_fields:
        value = row.get(field)
        if value is None:
            continue
        if not isinstance(value, list):
            continue

        found_any = True
        ids = [int(x) for x in value]
        if len(ids) != 1:
            return False, f"{field}_length={len(ids)}"
        if ids[0] != int(token_id):
            return False, f"{field}_id_mismatch"

    if found_any:
        return True, "explicit_single_token_span_confirmed"
    return True, "no_explicit_span_field_available"


def validate_english_equivalent_alignment(
    en_row: dict[str, Any],
    zh_row: dict[str, Any],
    tokenizer,
) -> dict[str, Any]:
    """
    Validate the paired English gold token that will operationalize the
    translation-equivalent vocabulary item under the Chinese prompt.

    Important operational definition:
    The English equivalent token ID is the canonical target_token_id from the
    paired English alignment row for the SAME model and example. We do not
    retokenize an artificial Chinese+English mixed sentence.
    """
    example_id = normalize_id(en_row.get("id"))
    if example_id != normalize_id(zh_row.get("id")):
        raise ValueError("English/Chinese pair has mismatched ids.")

    if str(en_row.get("model_key")) != str(zh_row.get("model_key")):
        raise ValueError("English/Chinese pair has mismatched model_key values.")

    en_token_id = int(en_row["target_token_id"])
    zh_token_id = int(zh_row["target_token_id"])

    vocab_size = len(tokenizer)
    id_in_vocab = 0 <= en_token_id < vocab_size

    en_input_ids = [int(x) for x in en_row["full_input_ids"]]
    prediction_position = int(en_row["prediction_position"])
    next_position = prediction_position + 1
    next_token_exists = 0 <= next_position < len(en_input_ids)
    next_token_matches = (
        next_token_exists and en_input_ids[next_position] == en_token_id
    )

    span_valid, span_validation_note = optional_single_token_span_check(
        en_row,
        en_token_id,
    )

    decoded = tokenizer.decode([en_token_id]) if id_in_vocab else ""
    token_string = tokenizer.convert_ids_to_tokens(en_token_id) if id_in_vocab else ""

    stored_surface = str(en_row.get("target_token_surface", ""))
    stored_token = str(en_row.get("target_token", ""))
    target_text = str(en_row.get("target", ""))

    decoded_nonempty = decoded != ""
    surface_matches_target_after_strip = (
        decoded.strip() == target_text.strip() if target_text.strip() != "" else False
    )

    valid = all(
        [
            id_in_vocab,
            next_token_exists,
            next_token_matches,
            span_valid,
            decoded_nonempty,
        ]
    )

    return {
        "model_key": en_row["model_key"],
        "id": example_id,
        "semantic_group": en_row.get(
            "semantic_group", zh_row.get("semantic_group", "")
        ),
        "subgroup": en_row.get("subgroup", zh_row.get("subgroup", "")),
        "relation": en_row.get("relation", zh_row.get("relation", "")),
        "zh_gold": zh_row.get("target", ""),
        "zh_gold_token_id": zh_token_id,
        "en_equivalent": target_text,
        "en_equivalent_token_id": en_token_id,
        "en_equivalent_stored_token": stored_token,
        "en_equivalent_stored_surface": stored_surface,
        "en_equivalent_decoded_single_token": decoded,
        "en_equivalent_tokenizer_token": token_string,
        "english_alignment_prediction_position": prediction_position,
        "english_alignment_next_token_id": (
            en_input_ids[next_position] if next_token_exists else None
        ),
        "token_id_in_vocab": id_in_vocab,
        "next_token_exists": next_token_exists,
        "next_token_matches_target_token_id": next_token_matches,
        "explicit_span_single_token_valid": span_valid,
        "span_validation_note": span_validation_note,
        "decoded_nonempty": decoded_nonempty,
        "decoded_matches_target_after_strip": surface_matches_target_after_strip,
        "same_vocab_token_as_zh_gold": en_token_id == zh_token_id,
        "valid_for_eq1": valid,
    }


def validate_pairs(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    tokenizer,
) -> list[dict[str, Any]]:
    rows = [
        validate_english_equivalent_alignment(en_row, zh_row, tokenizer)
        for en_row, zh_row in pairs
    ]

    invalid = [row for row in rows if not bool(row["valid_for_eq1"])]
    if invalid:
        preview = invalid[:10]
        raise ValueError(
            f"Found {len(invalid)} English equivalents that failed EQ1 "
            f"alignment/token validation. Examples: {preview}"
        )

    return rows


# ============================================================
# One-example EQ1 Logit Lens extraction
# ============================================================


@torch.inference_mode()
def run_translation_equivalent_logit_lens(
    zh_row: dict[str, Any],
    en_row: dict[str, Any],
    model,
    tokenizer,
    final_norm,
    lm_head,
) -> list[dict[str, Any]]:
    """
    Run the plain Logit Lens on ONE CHINESE prompt while tracking two tokens:

        1. the predefined Chinese gold token;
        2. the paired English gold token, treated as the English
           translation-equivalent vocabulary item.

    The projection and rank definitions are intentionally identical to the
    frozen main experiment. The only new measurement is the second tracked
    token under the same Chinese hidden state.
    """
    if str(zh_row.get("language")) != "zh":
        raise ValueError("EQ1 extraction must be run on a Chinese alignment row.")
    if str(en_row.get("language")) != "en":
        raise ValueError("Paired translation-equivalent row must be English.")
    if normalize_id(zh_row.get("id")) != normalize_id(en_row.get("id")):
        raise ValueError("Paired rows have different example ids.")
    if str(zh_row.get("model_key")) != str(en_row.get("model_key")):
        raise ValueError("Paired rows have different model keys.")
    if TOP_K < 10:
        raise ValueError("TOP_K must be at least 10.")

    input_ids_list = [int(x) for x in zh_row["full_input_ids"]]
    prediction_position = int(zh_row["prediction_position"])

    zh_gold_token_id = int(zh_row["target_token_id"])
    en_equivalent_token_id = int(en_row["target_token_id"])

    input_device = get_input_device(model)
    input_ids = torch.tensor(
        [input_ids_list],
        dtype=torch.long,
        device=input_device,
    )

    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states
    last_layer_idx = len(hidden_states) - 1
    num_layers = last_layer_idx

    final_logits = outputs.logits[0, prediction_position, :].float()
    final_topk = get_topk_info(final_logits, tokenizer, TOP_K)
    final_topk_token_ids = [int(x) for x in final_topk["topk_token_ids"]]
    final_prediction_token_id = int(final_topk_token_ids[0])
    final_prediction_token = final_topk["topk_tokens"][0]
    final_prediction_text = final_topk["topk_texts"][0]

    final_prediction_is_zh_gold = final_prediction_token_id == zh_gold_token_id
    final_prediction_is_en_equivalent = (
        final_prediction_token_id == en_equivalent_token_id
    )

    results: list[dict[str, Any]] = []

    for layer_idx, layer_hidden_states in enumerate(hidden_states):
        if layer_idx == last_layer_idx:
            logits = final_logits
        else:
            hidden = layer_hidden_states[0, prediction_position, :]
            logits = project_hidden_to_logits(
                hidden_state=hidden,
                final_norm=final_norm,
                lm_head=lm_head,
                layer_idx=layer_idx,
                last_layer_idx=last_layer_idx,
            )

        zh_rank, zh_logit = rank_of_token(logits, zh_gold_token_id)
        en_rank, en_logit = rank_of_token(logits, en_equivalent_token_id)

        topk = get_topk_info(logits, tokenizer, TOP_K)
        top_ids = [int(x) for x in topk["topk_token_ids"]]

        zh_in_top1 = zh_gold_token_id == top_ids[0]
        zh_in_top5 = zh_gold_token_id in top_ids[:5]
        zh_in_top10 = zh_gold_token_id in top_ids[:10]

        en_in_top1 = en_equivalent_token_id == top_ids[0]
        en_in_top5 = en_equivalent_token_id in top_ids[:5]
        en_in_top10 = en_equivalent_token_id in top_ids[:10]

        zh_prob = None
        en_prob = None
        if COMPUTE_PROB:
            probs = F.softmax(logits, dim=-1)
            zh_prob = float(probs[zh_gold_token_id].item())
            en_prob = float(probs[en_equivalent_token_id].item())

        if layer_idx == 0:
            layer_name = "embedding"
        elif layer_idx == last_layer_idx:
            layer_name = "final"
        else:
            layer_name = f"layer_{layer_idx}"

        layer_top1_token_id = int(top_ids[0])
        layer_top1_token = topk["topk_tokens"][0]
        layer_top1_text = topk["topk_texts"][0]
        layer_top1_logit = float(topk["topk_logits"][0])

        # Sanity checks at the model's actual final layer.
        if layer_idx == last_layer_idx:
            assert layer_top1_token_id == final_prediction_token_id
            assert zh_in_top1 == final_prediction_is_zh_gold
            assert en_in_top1 == final_prediction_is_en_equivalent

        result = {
            "id": zh_row["id"],
            "semantic_group": zh_row.get("semantic_group", ""),
            "subgroup": zh_row.get("subgroup", ""),
            "relation": zh_row.get("relation", ""),
            "language": "zh",
            "model_key": zh_row["model_key"],
            "model_name": zh_row["model_name"],
            "layer": layer_idx,
            "layer_name": layer_name,
            "num_layers": num_layers,
            "layer_fraction": layer_idx / num_layers if num_layers > 0 else None,
            # Chinese prompt / gold metadata
            "subject": zh_row.get("subject", ""),
            "prefix": zh_row.get("prefix", ""),
            "full_sentence": zh_row.get("full_sentence", ""),
            "prediction_position": prediction_position,
            "prediction_token_id": zh_row.get("prediction_token_id"),
            "prediction_token": zh_row.get("prediction_token", ""),
            "zh_gold": zh_row["target"],
            "zh_gold_token_id": zh_gold_token_id,
            "zh_gold_token": zh_row.get("target_token", ""),
            "zh_gold_token_surface": zh_row.get("target_token_surface", ""),
            # Paired English target defines the translation-equivalent token.
            "en_equivalent": en_row["target"],
            "en_equivalent_token_id": en_equivalent_token_id,
            "en_equivalent_token": en_row.get("target_token", ""),
            "en_equivalent_token_surface": en_row.get("target_token_surface", ""),
            "en_equivalent_decoded_token": tokenizer.decode([en_equivalent_token_id]),
            # Full-vocabulary diagnostics for Chinese gold.
            "zh_gold_logit": zh_logit,
            "zh_gold_prob": safe_float(zh_prob),
            "zh_gold_rank": zh_rank,
            "zh_gold_in_top1": zh_in_top1,
            "zh_gold_in_top5": zh_in_top5,
            "zh_gold_in_top10": zh_in_top10,
            # Full-vocabulary diagnostics for English equivalent.
            "en_equivalent_logit": en_logit,
            "en_equivalent_prob": safe_float(en_prob),
            "en_equivalent_rank": en_rank,
            "en_equivalent_in_top1": en_in_top1,
            "en_equivalent_in_top5": en_in_top5,
            "en_equivalent_in_top10": en_in_top10,
            # Pairwise diagnostics. Negative rank difference means the English
            # equivalent has the better (smaller) rank at this layer.
            "rank_difference_en_minus_zh": en_rank - zh_rank,
            "logit_difference_en_minus_zh": en_logit - zh_logit,
            # Current layer Top-1.
            "layer_top1_token_id": layer_top1_token_id,
            "layer_top1_token": layer_top1_token,
            "layer_top1_text": layer_top1_text,
            "layer_top1_logit": layer_top1_logit,
            # Actual final prediction under the Chinese prompt.
            "final_prediction_token_id": final_prediction_token_id,
            "final_prediction_token": final_prediction_token,
            "final_prediction_text": final_prediction_text,
            "final_prediction_is_zh_gold": final_prediction_is_zh_gold,
            "final_prediction_is_en_equivalent": final_prediction_is_en_equivalent,
        }

        # Compatibility aliases for downstream code that expects the main
        # experiment's gold-target field names. These aliases always refer to
        # the CHINESE gold token in this EQ1 file.
        result.update(
            {
                "target": zh_row["target"],
                "target_token_id": zh_gold_token_id,
                "target_token": zh_row.get("target_token", ""),
                "target_token_surface": zh_row.get("target_token_surface", ""),
                "target_logit": zh_logit,
                "target_prob": safe_float(zh_prob),
                "target_rank": zh_rank,
                "target_in_top1": zh_in_top1,
                "target_in_top5": zh_in_top5,
                "target_in_top10": zh_in_top10,
                "final_prediction_is_target": final_prediction_is_zh_gold,
            }
        )

        if STORE_TOPK:
            result.update(topk)

        results.append(result)

    return results


# ============================================================
# Per-model run
# ============================================================


def run_for_model(model_key: str, model_name: str) -> None:
    alignment_path = BASE_ALIGNMENT_DIR / model_key / "token_alignment.jsonl"
    output_dir = BASE_OUTPUT_DIR / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl = output_dir / "translation_equivalent_by_layer.jsonl"
    compact_tsv = output_dir / "translation_equivalent_by_layer_compact.tsv"
    validation_tsv = output_dir / "translation_equivalent_token_validation.tsv"
    rejected_jsonl = output_dir / "translation_equivalent.rejected.jsonl"
    report_json = output_dir / "translation_equivalent_extraction_report.json"

    alignment_rows = read_jsonl(alignment_path)
    pairs = build_language_pairs(alignment_rows, model_key)

    print("\n========================================")
    print(f"Model: {model_key} -> {model_name}")
    print(f"Alignment path: {alignment_path}")
    print(f"Complete EN/ZH pairs: {len(pairs)}")
    print(f"Chinese prompts to run: {len(pairs)}")
    print(f"Output dir: {output_dir}")
    print("========================================")

    model, tokenizer = load_model_and_tokenizer(model_name)
    final_norm = get_final_norm(model)
    lm_head = model.get_output_embeddings()

    if lm_head is None:
        raise RuntimeError("model.get_output_embeddings() returned None.")
    if APPLY_FINAL_NORM and final_norm is None:
        print("Warning: final norm not found. Using lm_head only.")

    # ------------------------------------------------------------
    # Validate the paired English equivalent token BEFORE extraction.
    # ------------------------------------------------------------
    validation_rows = validate_pairs(pairs, tokenizer)

    validation_fields = [
        "model_key",
        "id",
        "semantic_group",
        "subgroup",
        "relation",
        "zh_gold",
        "zh_gold_token_id",
        "en_equivalent",
        "en_equivalent_token_id",
        "en_equivalent_stored_token",
        "en_equivalent_stored_surface",
        "en_equivalent_decoded_single_token",
        "en_equivalent_tokenizer_token",
        "english_alignment_prediction_position",
        "english_alignment_next_token_id",
        "token_id_in_vocab",
        "next_token_exists",
        "next_token_matches_target_token_id",
        "explicit_span_single_token_valid",
        "span_validation_note",
        "decoded_nonempty",
        "decoded_matches_target_after_strip",
        "same_vocab_token_as_zh_gold",
        "valid_for_eq1",
    ]
    write_tsv(validation_tsv, validation_rows, validation_fields)

    all_results: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, (en_row, zh_row) in enumerate(pairs, start=1):
        try:
            rows = run_translation_equivalent_logit_lens(
                zh_row=zh_row,
                en_row=en_row,
                model=model,
                tokenizer=tokenizer,
                final_norm=final_norm,
                lm_head=lm_head,
            )
            all_results.extend(rows)
        except Exception as exc:
            rejected.append(
                {
                    "pair_index": idx - 1,
                    "id": zh_row.get("id"),
                    "model_key": model_key,
                    "semantic_group": zh_row.get("semantic_group"),
                    "zh_gold": zh_row.get("target"),
                    "en_equivalent": en_row.get("target"),
                    "reason": str(exc),
                }
            )

        if idx % PROGRESS_EVERY == 0:
            print(
                f"Processed {idx}/{len(pairs)} Chinese prompts; "
                f"layer rows so far: {len(all_results)}; "
                f"rejected: {len(rejected)}"
            )

        if torch.cuda.is_available() and idx % 100 == 0:
            torch.cuda.empty_cache()

    write_jsonl(raw_jsonl, all_results)
    write_jsonl(rejected_jsonl, rejected)

    compact_fields = [
        "id",
        "semantic_group",
        "subgroup",
        "relation",
        "language",
        "model_key",
        "layer",
        "layer_name",
        "num_layers",
        "layer_fraction",
        "zh_gold",
        "zh_gold_token_id",
        "en_equivalent",
        "en_equivalent_token_id",
        "zh_gold_rank",
        "en_equivalent_rank",
        "rank_difference_en_minus_zh",
        "zh_gold_logit",
        "en_equivalent_logit",
        "logit_difference_en_minus_zh",
        "zh_gold_prob",
        "en_equivalent_prob",
        "zh_gold_in_top1",
        "zh_gold_in_top5",
        "zh_gold_in_top10",
        "en_equivalent_in_top1",
        "en_equivalent_in_top5",
        "en_equivalent_in_top10",
        "layer_top1_text",
        "final_prediction_text",
        "final_prediction_is_zh_gold",
        "final_prediction_is_en_equivalent",
    ]
    write_tsv(compact_tsv, all_results, compact_fields)

    validation_invalid = sum(not bool(row["valid_for_eq1"]) for row in validation_rows)

    report = {
        "analysis": "translation_equivalent_accessibility_under_chinese_prompt",
        "model_key": model_key,
        "model_name": model_name,
        "alignment_path": str(alignment_path),
        "output_dir": str(output_dir),
        "num_complete_en_zh_pairs": len(pairs),
        "num_chinese_prompts_run": len(pairs),
        "num_layer_rows": len(all_results),
        "num_rejected": len(rejected),
        "num_invalid_translation_equivalent_tokens": validation_invalid,
        "english_equivalent_definition": (
            "The paired English gold target_token_id from the same model's "
            "canonical alignment row is tracked under the Chinese prompt."
        ),
        "single_token_validation": (
            "Checks that the paired English target_token_id is in vocabulary, "
            "is the next token at the stored English prediction position, and "
            "matches any explicit one-token target-span field when available."
        ),
        "projection": {
            "apply_final_norm": APPLY_FINAL_NORM,
            "final_layer_uses_outputs_logits": True,
            "rank_definition": (
                "1 + number of vocabulary logits strictly greater than the "
                "tracked token logit"
            ),
            "top_k": TOP_K,
            "topk_membership_uses_actual_torch_topk": True,
            "compute_prob": COMPUTE_PROB,
            "store_topk": STORE_TOPK,
        },
        "counts_by_semantic_group": dict(
            Counter(str(zh_row.get("semantic_group", "")) for _, zh_row in pairs)
        ),
        "outputs": {
            "raw_jsonl": str(raw_jsonl),
            "compact_tsv": str(compact_tsv),
            "token_validation_tsv": str(validation_tsv),
            "rejected_jsonl": str(rejected_jsonl),
        },
    }
    write_json(report_json, report)

    print("\nDone for model:", model_key)
    print(f"Chinese prompts: {len(pairs)}")
    print(f"Layer rows: {len(all_results)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Raw JSONL: {raw_jsonl}")
    print(f"Compact TSV: {compact_tsv}")
    print(f"Validation TSV: {validation_tsv}")
    print(f"Report JSON: {report_json}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# CLI
# ============================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run EQ1 Logit Lens extraction on Chinese prompts while tracking "
            "both the Chinese gold token and the paired English equivalent."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS),
        default=list(MODELS),
        help=(
            "Model keys to run. Default: all three Qwen3 models. "
            "For limited GPU memory, run one model at a time."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_key in args.models:
        run_for_model(model_key, MODELS[model_key])

    print("\n================ Overall ================")
    print(f"Active models: {', '.join(args.models)}")
    print(f"Base output dir: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
