from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

# Set before transformers/tokenizers initialization.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_INPUT_PATH = Path(
    "data/alignment/broader_category/broader_category_alignment.jsonl"
)

DEFAULT_OUTPUT_DIR = Path("data/logit_lens/broader_category")

MODEL_REGISTRY = {
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_14b": "Qwen/Qwen3-14B",
}
MODEL_KEYS = tuple(MODEL_REGISTRY)
MODEL_REVISION = "main"

OUTPUT_FILENAMES = (
    "broader_category_by_layer.jsonl",
    "broader_category_by_layer_compact.tsv",
    "broader_category.rejected.jsonl",
    "broader_category_extraction_report.json",
    "run_config.json",
)

EXPECTED_ROWS_PER_MODEL = 110
EXPECTED_ROWS_PER_LANGUAGE = 55

APPLY_FINAL_NORM = True
TOP_K = 10
COMPUTE_PROB = True
STORE_TOPK = True

DEFAULT_PROGRESS_EVERY = 100
DEFAULT_EMPTY_CACHE_EVERY = 25

REQUIRED_ALIGNMENT_FIELDS = {
    "id",
    "semantic_group",
    "subgroup",
    "relation",
    "mapping_rule",
    "mapping_status",
    "model_key",
    "model_name",
    "language",
    "subject",
    "prefix",
    "gold",
    "broader_category",
    "gold_sentence",
    "broader_sentence",
    "full_input_ids",
    "prediction_position",
    "prediction_token_id",
    "gold_token_id",
    "gold_token",
    "gold_token_surface",
    "broader_token_id",
    "broader_token",
    "broader_token_surface",
    "contexts_match",
    "prediction_positions_match",
    "prediction_token_ids_match",
    "gold_broader_token_ids_distinct",
    "valid_for_eq2",
}


COMPACT_FIELDS = [
    "id",
    "semantic_group",
    "subgroup",
    "relation",
    "mapping_rule",
    "language",
    "model_key",
    "model_name",
    "layer",
    "layer_name",
    "num_layers",
    "layer_fraction",
    "subject",
    "prefix",
    "prediction_position",
    "gold",
    "gold_token_id",
    "gold_token_surface",
    "broader_category",
    "broader_token_id",
    "broader_token_surface",
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
    "layer_top1_token_id",
    "layer_top1_text",
    "final_prediction_token_id",
    "final_prediction_text",
    "final_prediction_is_gold",
    "final_prediction_is_broader_category",
]


# I/O


def normalize_id(value: Any) -> str:
    return str(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Alignment JSONL not found: {path}")

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
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")

            row["_source_line"] = line_number
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in alignment file: {path}")

    return rows


def append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append one complete trajectory and flush immediately."""

    payload = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return data


def write_tsv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            clean: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                clean[key] = value
            writer.writerow(clean)


# Alignment validation


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalised = value.strip().casefold()
        if normalised in {"true", "yes", "1", "y"}:
            return True
        if normalised in {"false", "no", "0", "n", ""}:
            return False

    return bool(value)


def validate_alignment_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model_key: str,
    model_name: str,
    allow_nonstandard_count: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    for row in rows:
        if str(row.get("model_key")) != model_key:
            continue

        missing = REQUIRED_ALIGNMENT_FIELDS.difference(row)
        if missing:
            raise ValueError(
                f"Alignment id={row.get('id')} is missing fields: {sorted(missing)}"
            )

        stored_model_name = str(row["model_name"])
        if stored_model_name != model_name:
            raise ValueError(
                f"Alignment id={row['id']} has model_name="
                f"{stored_model_name!r}, expected {model_name!r}."
            )

        if str(row["semantic_group"]) != "category":
            raise ValueError(
                f"Alignment id={row['id']} has semantic_group="
                f"{row['semantic_group']!r}, expected 'category'."
            )

        if str(row["mapping_status"]) != ("manually_defined_broader_category"):
            raise ValueError(
                f"Alignment id={row['id']} has unexpected "
                f"mapping_status={row['mapping_status']!r}."
            )

        language = str(row["language"])
        if language not in {"en", "zh"}:
            raise ValueError(
                f"Alignment id={row['id']} has unsupported language={language!r}."
            )

        required_true = (
            "contexts_match",
            "prediction_positions_match",
            "prediction_token_ids_match",
            "gold_broader_token_ids_distinct",
            "valid_for_eq2",
        )

        bad_flags = [field for field in required_true if not as_bool(row[field])]
        if bad_flags:
            raise ValueError(
                f"Alignment id={row['id']} language={language} "
                f"failed required validation flags: {bad_flags}"
            )

        input_ids = row["full_input_ids"]
        if not isinstance(input_ids, list) or not input_ids:
            raise ValueError(
                f"Alignment id={row['id']} language={language} "
                "has empty/non-list full_input_ids."
            )

        prediction_position = int(row["prediction_position"])
        if not 0 <= prediction_position < len(input_ids):
            raise ValueError(
                f"Alignment id={row['id']} language={language}: "
                f"prediction_position={prediction_position} outside "
                f"input length={len(input_ids)}."
            )

        if int(row["gold_token_id"]) == int(row["broader_token_id"]):
            raise ValueError(
                f"Alignment id={row['id']} language={language} "
                "has identical gold and broader token IDs."
            )

        selected.append(row)

    if not selected:
        raise ValueError(f"No alignment rows found for model_key={model_key!r}.")

    seen: set[tuple[str, str]] = set()
    for row in selected:
        key = (
            normalize_id(row["id"]),
            str(row["language"]),
        )
        if key in seen:
            raise ValueError(
                f"Duplicate alignment trajectory for model={model_key}, "
                f"id={key[0]}, language={key[1]}."
            )
        seen.add(key)

    language_counts = Counter(str(row["language"]) for row in selected)

    if not allow_nonstandard_count:
        if len(selected) != EXPECTED_ROWS_PER_MODEL:
            raise ValueError(
                f"Expected {EXPECTED_ROWS_PER_MODEL} alignment rows "
                f"for {model_key}, found {len(selected)}. "
                "Use --allow-nonstandard-count only for deliberate "
                "debug/subset runs."
            )

        for language in ("en", "zh"):
            observed = language_counts.get(language, 0)
            if observed != EXPECTED_ROWS_PER_LANGUAGE:
                raise ValueError(
                    f"Expected {EXPECTED_ROWS_PER_LANGUAGE} {language} "
                    f"rows for {model_key}, found {observed}."
                )

    selected.sort(
        key=lambda row: (
            0 if str(row["id"]).isdigit() else 1,
            int(row["id"]) if str(row["id"]).isdigit() else str(row["id"]),
            0 if str(row["language"]) == "en" else 1,
        )
    )

    return selected


# Model/device helpers


def choose_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.strip().lower()

    if name == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16

        return torch.float32

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    if name not in mapping:
        raise ValueError(
            f"Unsupported dtype {dtype_name!r}. "
            "Choose auto, float32, float16, or bfloat16."
        )

    return mapping[name]


def default_single_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model_and_tokenizer(
    *,
    model_name: str,
    revision: str,
    dtype: torch.dtype,
    device_map: Optional[str],
    cache_dir: Optional[Path],
    attn_implementation: Optional[str],
):
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir_str,
        use_fast=True,
        trust_remote_code=True,
    )

    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "cache_dir": cache_dir_str,
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if device_map:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    if not device_map:
        device = default_single_device()
        model.to(device)

    model.eval()
    return model, tokenizer


def get_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def get_module_device(module) -> Optional[torch.device]:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def get_module_dtype(module) -> Optional[torch.dtype]:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return None


def move_to_module_device_and_dtype(
    tensor: torch.Tensor,
    module,
) -> torch.Tensor:
    device = get_module_device(module)
    dtype = get_module_dtype(module)

    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device

    if dtype is not None and tensor.is_floating_point():
        kwargs["dtype"] = dtype

    if kwargs:
        tensor = tensor.to(**kwargs)

    return tensor


def get_final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm

    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f

    return None


def project_hidden_to_logits(
    *,
    hidden_state: torch.Tensor,
    final_norm,
    lm_head,
    layer_idx: int,
    last_layer_idx: int,
) -> torch.Tensor:
    """Embedding: lm_head; intermediate: final norm + lm_head; final: outputs.logits."""
    hidden = hidden_state.unsqueeze(0).unsqueeze(0)

    should_apply_norm = (
        APPLY_FINAL_NORM
        and final_norm is not None
        and layer_idx != 0
        and layer_idx != last_layer_idx
    )

    if should_apply_norm:
        hidden = move_to_module_device_and_dtype(
            hidden,
            final_norm,
        )
        hidden = final_norm(hidden)

    hidden = move_to_module_device_and_dtype(
        hidden,
        lm_head,
    )

    logits = lm_head(hidden).squeeze(0).squeeze(0)
    return logits.float()


# Vocabulary metrics


def check_token_id(
    logits: torch.Tensor,
    token_id: int,
    name: str,
) -> None:
    vocab_size = int(logits.shape[-1])

    if not 0 <= token_id < vocab_size:
        raise IndexError(
            f"{name} token_id={token_id} outside vocabulary size={vocab_size}."
        )


def rank_and_logit(
    logits: torch.Tensor,
    token_id: int,
) -> tuple[int, float]:
    """One-based rank; tied logits share the same rank."""
    check_token_id(logits, token_id, "tracked")

    token_logit = logits[token_id]
    rank = int((logits > token_logit).sum().item()) + 1

    return rank, float(token_logit.item())


def probability_of_token(
    logits: torch.Tensor,
    token_id: int,
) -> float:
    """Compute probability using logsumexp without a full softmax tensor."""
    check_token_id(logits, token_id, "tracked")

    log_normalizer = torch.logsumexp(logits, dim=-1)
    probability = torch.exp(logits[token_id] - log_normalizer)

    return float(probability.item())


def get_topk_info(
    logits: torch.Tensor,
    tokenizer,
    k: int,
) -> dict[str, Any]:
    effective_k = min(
        int(k),
        int(logits.shape[-1]),
    )

    top_values, top_ids = torch.topk(
        logits,
        k=effective_k,
    )

    top_ids_list = [int(value) for value in top_ids.detach().cpu().tolist()]
    top_values_list = [float(value) for value in top_values.detach().cpu().tolist()]

    return {
        "topk_token_ids": top_ids_list,
        "topk_tokens": tokenizer.convert_ids_to_tokens(top_ids_list),
        "topk_texts": [
            tokenizer.decode(
                [token_id],
                clean_up_tokenization_spaces=False,
            )
            for token_id in top_ids_list
        ],
        "topk_logits": top_values_list,
    }


def safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    value = float(value)

    if math.isnan(value) or math.isinf(value):
        return None

    return value


# One-trajectory EQ2 extraction


@torch.inference_mode()
def run_broader_category_logit_lens(
    row: dict[str, Any],
    *,
    model,
    tokenizer,
    final_norm,
    lm_head,
) -> list[dict[str, Any]]:
    """
    Track the specific gold token and broader-category token
    from the same prediction-position hidden state.
    """
    if TOP_K < 10:
        raise ValueError("TOP_K must be at least 10 to compute Top-10.")

    input_ids_list = [int(value) for value in row["full_input_ids"]]
    prediction_position = int(row["prediction_position"])

    gold_token_id = int(row["gold_token_id"])
    broader_token_id = int(row["broader_token_id"])

    if gold_token_id == broader_token_id:
        raise ValueError("gold_and_broader_token_ids_are_identical")

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

    if hidden_states is None:
        raise RuntimeError("Model returned no hidden_states.")

    last_layer_idx = len(hidden_states) - 1
    num_layers = last_layer_idx

    configured_num_layers = getattr(
        model.config,
        "num_hidden_layers",
        None,
    )
    if configured_num_layers is not None and int(configured_num_layers) != num_layers:
        raise ValueError(
            "hidden_state_count_mismatch:"
            f" model.config.num_hidden_layers="
            f"{configured_num_layers}, "
            f"observed transformer outputs={num_layers}"
        )

    final_logits = outputs.logits[
        0,
        prediction_position,
        :,
    ].float()

    check_token_id(
        final_logits,
        gold_token_id,
        "gold",
    )
    check_token_id(
        final_logits,
        broader_token_id,
        "broader",
    )

    final_topk = get_topk_info(
        final_logits,
        tokenizer,
        TOP_K,
    )
    final_top_ids = [int(value) for value in final_topk["topk_token_ids"]]

    if len(final_top_ids) < 10:
        raise RuntimeError("Final Top-k returned fewer than 10 IDs.")

    final_prediction_token_id = int(final_top_ids[0])
    final_prediction_token = final_topk["topk_tokens"][0]
    final_prediction_text = final_topk["topk_texts"][0]

    final_prediction_is_gold = final_prediction_token_id == gold_token_id
    final_prediction_is_broader = final_prediction_token_id == broader_token_id

    results: list[dict[str, Any]] = []

    for layer_idx, layer_hidden_states in enumerate(hidden_states):
        if layer_idx == last_layer_idx:
            logits = final_logits
        else:
            hidden = layer_hidden_states[
                0,
                prediction_position,
                :,
            ]

            logits = project_hidden_to_logits(
                hidden_state=hidden,
                final_norm=final_norm,
                lm_head=lm_head,
                layer_idx=layer_idx,
                last_layer_idx=last_layer_idx,
            )

        gold_rank, gold_logit = rank_and_logit(
            logits,
            gold_token_id,
        )
        broader_rank, broader_logit = rank_and_logit(
            logits,
            broader_token_id,
        )

        topk = get_topk_info(
            logits,
            tokenizer,
            TOP_K,
        )
        top_ids = [int(value) for value in topk["topk_token_ids"]]

        # IMPORTANT: use actual torch.topk membership.
        gold_in_top1 = gold_token_id == top_ids[0]
        gold_in_top5 = gold_token_id in top_ids[:5]
        gold_in_top10 = gold_token_id in top_ids[:10]

        broader_in_top1 = broader_token_id == top_ids[0]
        broader_in_top5 = broader_token_id in top_ids[:5]
        broader_in_top10 = broader_token_id in top_ids[:10]

        gold_prob: Optional[float] = None
        broader_prob: Optional[float] = None

        if COMPUTE_PROB:
            gold_prob = probability_of_token(
                logits,
                gold_token_id,
            )
            broader_prob = probability_of_token(
                logits,
                broader_token_id,
            )

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

        # Final-layer consistency.
        if layer_idx == last_layer_idx:
            if layer_top1_token_id != final_prediction_token_id:
                raise AssertionError("final_top1_consistency_failed")

            if gold_in_top1 != final_prediction_is_gold:
                raise AssertionError("final_gold_top1_consistency_failed")

            if broader_in_top1 != final_prediction_is_broader:
                raise AssertionError("final_broader_top1_consistency_failed")

        result: dict[str, Any] = {
            "id": normalize_id(row["id"]),
            "semantic_group": str(row["semantic_group"]),
            "subgroup": str(row["subgroup"]),
            "relation": str(row["relation"]),
            "mapping_rule": str(row["mapping_rule"]),
            "mapping_status": str(row["mapping_status"]),
            "language": str(row["language"]),
            "model_key": str(row["model_key"]),
            "model_name": str(row["model_name"]),
            "layer": layer_idx,
            "layer_name": layer_name,
            "num_layers": num_layers,
            "layer_fraction": (layer_idx / num_layers if num_layers > 0 else 0.0),
            "subject": str(row.get("subject", "")),
            "prefix": str(row.get("prefix", "")),
            "gold_sentence": str(row.get("gold_sentence", "")),
            "broader_sentence": str(row.get("broader_sentence", "")),
            "prediction_position": (prediction_position),
            "prediction_token_id": int(row["prediction_token_id"]),
            "prediction_token": str(row.get("prediction_token", "")),
            "gold": str(row["gold"]),
            "gold_token_id": gold_token_id,
            "gold_token": str(row.get("gold_token", "")),
            "gold_token_surface": str(row.get("gold_token_surface", "")),
            "gold_decoded_token": str(
                row.get(
                    "gold_decoded_token",
                    tokenizer.decode(
                        [gold_token_id],
                        clean_up_tokenization_spaces=False,
                    ),
                )
            ),
            "gold_logit": gold_logit,
            "gold_prob": safe_float(gold_prob),
            "gold_rank": gold_rank,
            "gold_in_top1": gold_in_top1,
            "gold_in_top5": gold_in_top5,
            "gold_in_top10": gold_in_top10,
            "broader_category": str(row["broader_category"]),
            "broader_token_id": (broader_token_id),
            "broader_token": str(row.get("broader_token", "")),
            "broader_token_surface": str(
                row.get(
                    "broader_token_surface",
                    "",
                )
            ),
            "broader_decoded_token": str(
                row.get(
                    "broader_decoded_token",
                    tokenizer.decode(
                        [broader_token_id],
                        clean_up_tokenization_spaces=False,
                    ),
                )
            ),
            "broader_logit": broader_logit,
            "broader_prob": safe_float(broader_prob),
            "broader_rank": broader_rank,
            "broader_in_top1": (broader_in_top1),
            "broader_in_top5": (broader_in_top5),
            "broader_in_top10": (broader_in_top10),
            # Negative rank difference:
            # broader category ranks better.
            "rank_difference_broader_minus_gold": (broader_rank - gold_rank),
            "logit_difference_broader_minus_gold": (broader_logit - gold_logit),
            "prob_difference_broader_minus_gold": (
                safe_float(broader_prob - gold_prob)
                if (broader_prob is not None and gold_prob is not None)
                else None
            ),
            "layer_top1_token_id": (layer_top1_token_id),
            "layer_top1_token": (layer_top1_token),
            "layer_top1_text": (layer_top1_text),
            "layer_top1_logit": (layer_top1_logit),
            "final_prediction_token_id": (final_prediction_token_id),
            "final_prediction_token": (final_prediction_token),
            "final_prediction_text": (final_prediction_text),
            "final_prediction_is_gold": (final_prediction_is_gold),
            "final_prediction_is_broader_category": (final_prediction_is_broader),
        }

        # Compatibility aliases: target_* always means
        # the specific predefined gold category.
        result.update(
            {
                "target": str(row["gold"]),
                "target_token_id": gold_token_id,
                "target_token": str(row.get("gold_token", "")),
                "target_token_surface": str(
                    row.get(
                        "gold_token_surface",
                        "",
                    )
                ),
                "target_logit": gold_logit,
                "target_prob": safe_float(gold_prob),
                "target_rank": gold_rank,
                "target_in_top1": gold_in_top1,
                "target_in_top5": gold_in_top5,
                "target_in_top10": gold_in_top10,
                "final_prediction_is_target": (final_prediction_is_gold),
            }
        )

        if STORE_TOPK:
            result.update(topk)

        results.append(result)

    return results


# Resume / deduplication


def trajectory_key(
    row: dict[str, Any],
) -> tuple[str, str, str]:
    return (
        normalize_id(row["id"]),
        str(row["language"]),
        str(row["model_key"]),
    )


def read_existing_raw_safely(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception:
                # Ignore an incomplete final line after a disconnect.
                continue

            if isinstance(item, dict):
                rows.append(item)

    return rows


def completed_trajectory_keys(
    raw_rows: Sequence[dict[str, Any]],
    expected_layers: set[int],
) -> set[tuple[str, str, str]]:
    layers_by_key: dict[
        tuple[str, str, str],
        set[int],
    ] = defaultdict(set)

    for row in raw_rows:
        try:
            key = trajectory_key(row)
            layer = int(row["layer"])
        except Exception:
            continue

        layers_by_key[key].add(layer)

    return {key for key, layers in layers_by_key.items() if layers == expected_layers}


def deduplicate_layer_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the last record for each trajectory and layer."""
    last_by_key: dict[
        tuple[str, str, str, int],
        dict[str, Any],
    ] = {}

    for row in rows:
        try:
            key = (
                normalize_id(row["id"]),
                str(row["language"]),
                str(row["model_key"]),
                int(row["layer"]),
            )
        except Exception:
            continue

        last_by_key[key] = row

    def id_sort_key(value: Any):
        text = normalize_id(value)
        if text.isdigit():
            return (0, int(text))
        return (1, text)

    clean = list(last_by_key.values())
    clean.sort(
        key=lambda row: (
            id_sort_key(row["id"]),
            0 if str(row["language"]) == "en" else 1,
            int(row["layer"]),
        )
    )

    return clean


def validate_final_raw_coverage(
    rows: Sequence[dict[str, Any]],
    alignment_rows: Sequence[dict[str, Any]],
    expected_layers: set[int],
) -> None:
    layers_by_key: dict[
        tuple[str, str, str],
        set[int],
    ] = defaultdict(set)

    for row in rows:
        layers_by_key[trajectory_key(row)].add(int(row["layer"]))

    expected_keys = {trajectory_key(row) for row in alignment_rows}

    observed_keys = set(layers_by_key)

    missing_keys = expected_keys - observed_keys
    extra_keys = observed_keys - expected_keys

    if missing_keys:
        raise ValueError(
            "Missing completed trajectories after extraction: "
            f"{sorted(missing_keys)[:10]}"
        )

    if extra_keys:
        raise ValueError(
            f"Raw output contains unexpected trajectories: {sorted(extra_keys)[:10]}"
        )

    bad_layers = {
        key: sorted(layers)
        for key, layers in layers_by_key.items()
        if layers != expected_layers
    }

    if bad_layers:
        preview = list(bad_layers.items())[:5]
        raise ValueError(
            "Some trajectories do not contain the full "
            f"expected layer set {sorted(expected_layers)}. "
            f"Examples: {preview}"
        )


# Run config


def make_run_config(
    *,
    model_key: str,
    model_name: str,
    revision: str,
    alignment_path: Path,
    output_dir: Path,
    dtype_name: str,
    dtype_resolved: str,
    device_map: Optional[str],
    attn_implementation: Optional[str],
    max_rows: Optional[int],
) -> dict[str, Any]:
    return {
        "analysis": ("broader_category_logit_lens_extraction"),
        "model_key": model_key,
        "model_name": model_name,
        "revision_requested": revision,
        "alignment_path": str(alignment_path),
        "output_dir": str(output_dir),
        "dtype_requested": dtype_name,
        "dtype_resolved": dtype_resolved,
        "device_map": device_map,
        "attn_implementation": (attn_implementation),
        "apply_final_norm": APPLY_FINAL_NORM,
        "embedding_applies_final_norm": False,
        "intermediate_applies_final_norm": True,
        "final_layer_uses_outputs_logits": True,
        "top_k": TOP_K,
        "topk_membership_uses_actual_torch_topk": True,
        "compute_prob": COMPUTE_PROB,
        "store_topk": STORE_TOPK,
        "rank_definition": (
            "1 + number of vocabulary logits strictly greater "
            "than the tracked token logit"
        ),
        "rank_difference_definition": (
            "broader-category rank minus specific-gold rank; "
            "negative means broader ranks better"
        ),
        "max_rows": max_rows,
    }


def configs_compatible(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    keys = (
        "model_key",
        "model_name",
        "revision_requested",
        "alignment_path",
        "dtype_requested",
        "dtype_resolved",
        "device_map",
        "attn_implementation",
        "apply_final_norm",
        "embedding_applies_final_norm",
        "intermediate_applies_final_norm",
        "final_layer_uses_outputs_logits",
        "top_k",
        "topk_membership_uses_actual_torch_topk",
        "compute_prob",
        "store_topk",
        "rank_definition",
        "rank_difference_definition",
        "max_rows",
    )

    return all(previous.get(key) == current.get(key) for key in keys)


# Finalization / reporting


def final_layer_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["layer"]) == int(row["num_layers"])]


def make_report(
    *,
    model_key: str,
    model_name: str,
    alignment_path: Path,
    output_dir: Path,
    alignment_rows: Sequence[dict[str, Any]],
    layer_rows: Sequence[dict[str, Any]],
    rejected: Sequence[dict[str, Any]],
    dtype: torch.dtype,
    device_map: Optional[str],
    num_layers: int,
    raw_jsonl: Path,
    compact_tsv: Path,
    rejected_jsonl: Path,
    run_config_json: Path,
) -> dict[str, Any]:
    language_counts = Counter(str(row["language"]) for row in alignment_rows)
    subgroup_counts = Counter(
        str(row["subgroup"]) for row in alignment_rows if str(row["language"]) == "en"
    )

    finals = final_layer_rows(layer_rows)

    final_gold_top1 = sum(bool(row["final_prediction_is_gold"]) for row in finals)
    final_broader_top1 = sum(
        bool(row["final_prediction_is_broader_category"]) for row in finals
    )

    return {
        "analysis": ("broader_category_accessibility_extraction"),
        "model_key": model_key,
        "model_name": model_name,
        "alignment_path": str(alignment_path),
        "output_dir": str(output_dir),
        "trajectory_definition": (
            "One model-example-language trajectory; gold and "
            "broader-category vocabulary tokens are measured "
            "from the same causal prediction-position hidden state."
        ),
        "counts": {
            "alignment_trajectories_requested": len(alignment_rows),
            "alignment_by_language": dict(sorted(language_counts.items())),
            "bilingual_pair_count": (
                min(
                    language_counts.get("en", 0),
                    language_counts.get("zh", 0),
                )
            ),
            "bilingual_pairs_by_subgroup": dict(sorted(subgroup_counts.items())),
            "num_transformer_layers": num_layers,
            "hidden_states_per_trajectory": (num_layers + 1),
            "layer_rows": len(layer_rows),
            "rejected_trajectories": len(rejected),
        },
        "projection": {
            "apply_final_norm": APPLY_FINAL_NORM,
            "embedding_applies_final_norm": False,
            "intermediate_applies_final_norm": True,
            "final_layer_uses_outputs_logits": True,
            "rank_definition": (
                "1 + number of vocabulary logits strictly "
                "greater than the tracked token logit"
            ),
            "rank_difference_definition": (
                "broader rank minus gold rank; negative means broader ranks better"
            ),
            "top_k": TOP_K,
            "topk_membership_uses_actual_torch_topk": True,
            "compute_prob": COMPUTE_PROB,
            "store_topk": STORE_TOPK,
        },
        "runtime": {
            "dtype": str(dtype),
            "device_map": device_map,
        },
        "final_prediction_exact_match_counts": {
            "gold": final_gold_top1,
            "broader_category": (final_broader_top1),
        },
        "outputs": {
            "raw_jsonl": str(raw_jsonl),
            "compact_tsv": str(compact_tsv),
            "rejected_jsonl": str(rejected_jsonl),
            "run_config_json": str(run_config_json),
        },
    }


# Per-model execution


def run_model(model_key, args):
    model_name = MODEL_REGISTRY[model_key]
    revision = MODEL_REVISION

    alignment_path = args.input
    output_dir = args.output_dir / model_key

    raw_jsonl = output_dir / "broader_category_by_layer.jsonl"
    compact_tsv = output_dir / "broader_category_by_layer_compact.tsv"
    rejected_jsonl = output_dir / "broader_category.rejected.jsonl"
    report_json = output_dir / "broader_category_extraction_report.json"
    run_config_json = output_dir / "run_config.json"

    all_alignment_rows = read_jsonl(alignment_path)

    alignment_rows = validate_alignment_rows(
        all_alignment_rows,
        model_key=model_key,
        model_name=model_name,
        allow_nonstandard_count=(
            bool(args.allow_nonstandard_count) or args.max_rows is not None
        ),
    )

    if args.max_rows is not None:
        alignment_rows = alignment_rows[: args.max_rows]

    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = choose_dtype(args.dtype)
    print(f"{model_key}: processing {len(alignment_rows)} trajectories", flush=True)
    print(
        f"  Device: {args.device_map or default_single_device()}, dtype: {dtype}",
        flush=True,
    )

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        revision=revision,
        dtype=dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
        attn_implementation=(args.attn_implementation),
    )

    final_norm = get_final_norm(model)
    lm_head = model.get_output_embeddings()

    if lm_head is None:
        raise RuntimeError("model.get_output_embeddings() returned None.")

    if APPLY_FINAL_NORM and final_norm is None:
        raise RuntimeError(
            "Final normalization module was not found. "
            "Do not silently change the frozen Logit Lens "
            "definition."
        )

    num_layers = int(
        getattr(
            model.config,
            "num_hidden_layers",
            -1,
        )
    )
    if num_layers <= 0:
        raise RuntimeError("Could not determine model.config.num_hidden_layers.")

    expected_layers = set(range(num_layers + 1))

    vocab_size = int(model.get_output_embeddings().weight.shape[0])

    for row in alignment_rows:
        for field in (
            "gold_token_id",
            "broader_token_id",
            "prediction_token_id",
        ):
            token_id = int(row[field])
            if not 0 <= token_id < vocab_size:
                raise ValueError(
                    f"id={row['id']} "
                    f"language={row['language']} "
                    f"{field}={token_id} outside "
                    f"model vocab size={vocab_size}."
                )

    run_config = make_run_config(
        model_key=model_key,
        model_name=model_name,
        revision=revision,
        alignment_path=alignment_path,
        output_dir=output_dir,
        dtype_name=args.dtype,
        dtype_resolved=str(dtype),
        device_map=args.device_map,
        attn_implementation=(args.attn_implementation),
        max_rows=args.max_rows,
    )

    if args.overwrite:
        for path in (
            raw_jsonl,
            compact_tsv,
            rejected_jsonl,
            report_json,
            run_config_json,
        ):
            if path.exists():
                path.unlink()

    if raw_jsonl.exists() and not run_config_json.exists():
        raise RuntimeError(
            "Existing raw output has no run_config.json; use --overwrite."
        )

    if run_config_json.exists() and raw_jsonl.exists() and not args.no_resume:
        previous_config = load_json(run_config_json)

        if not configs_compatible(
            previous_config,
            run_config,
        ):
            raise RuntimeError(
                "Existing raw output was created with an "
                "incompatible run configuration. "
                "Use --overwrite or a different output path."
            )

    write_json(
        run_config_json,
        run_config,
    )

    existing_rows = read_existing_raw_safely(raw_jsonl) if not args.no_resume else []

    completed = completed_trajectory_keys(
        existing_rows,
        expected_layers,
    )

    requested_keys = {trajectory_key(row) for row in alignment_rows}
    completed &= requested_keys
    # Drop partial trajectories before retrying; rewrite to remove truncated lines.
    existing_rows = [
        row
        for row in deduplicate_layer_rows(existing_rows)
        if trajectory_key(row) in completed
    ]
    write_jsonl(raw_jsonl, existing_rows)
    if completed:
        print(f"  Resuming: {len(completed)} completed trajectories", flush=True)

    rejected: list[dict[str, Any]] = []
    if not args.overwrite and not args.no_resume and rejected_jsonl.exists():
        rejected = read_existing_raw_safely(rejected_jsonl)

    if completed and rejected:
        rejected = [
            item
            for item in rejected
            if (
                normalize_id(item.get("id")),
                str(item.get("language")),
                str(item.get("model_key")),
            )
            not in completed
        ]
        write_jsonl(
            rejected_jsonl,
            rejected,
        )

    processed_this_run = 0

    for index, row in enumerate(
        alignment_rows,
        start=1,
    ):
        key = trajectory_key(row)

        if not args.no_resume and key in completed:
            continue

        try:
            trajectory_rows = run_broader_category_logit_lens(
                row,
                model=model,
                tokenizer=tokenizer,
                final_norm=final_norm,
                lm_head=lm_head,
            )

            append_jsonl(
                raw_jsonl,
                trajectory_rows,
            )
            processed_this_run += 1
            completed.add(key)

            rejected = [
                item
                for item in rejected
                if (
                    normalize_id(item.get("id")) != normalize_id(row.get("id"))
                    or str(item.get("language")) != str(row.get("language"))
                    or str(item.get("model_key")) != model_key
                )
            ]
            write_jsonl(
                rejected_jsonl,
                rejected,
            )

        except torch.cuda.OutOfMemoryError:
            raise
        except OSError:
            raise
        except Exception as exc:
            rejected_row = {
                "alignment_index": index - 1,
                "id": normalize_id(row.get("id")),
                "model_key": model_key,
                "model_name": model_name,
                "language": row.get("language", ""),
                "subgroup": row.get("subgroup", ""),
                "gold": row.get("gold", ""),
                "broader_category": row.get("broader_category", ""),
                "reason": str(exc),
            }
            rejected.append(rejected_row)

            reject_map = {
                (
                    normalize_id(item["id"]),
                    str(item["language"]),
                    str(item["model_key"]),
                ): item
                for item in rejected
            }
            rejected = list(reject_map.values())
            write_jsonl(
                rejected_jsonl,
                rejected,
            )

        if args.progress_every and index % args.progress_every == 0:
            print(f"  {index}/{len(alignment_rows)} trajectories", flush=True)

        if torch.cuda.is_available() and index % args.empty_cache_every == 0:
            torch.cuda.empty_cache()

    raw_rows = read_existing_raw_safely(raw_jsonl)
    clean_rows = deduplicate_layer_rows(raw_rows)

    requested_keys = {trajectory_key(row) for row in alignment_rows}
    complete_keys = (
        completed_trajectory_keys(clean_rows, expected_layers) & requested_keys
    )
    clean_rows = [row for row in clean_rows if trajectory_key(row) in complete_keys]

    write_jsonl(
        raw_jsonl,
        clean_rows,
    )

    if not rejected:
        validate_final_raw_coverage(
            clean_rows,
            alignment_rows,
            expected_layers,
        )

    write_tsv(
        compact_tsv,
        clean_rows,
        COMPACT_FIELDS,
    )

    write_jsonl(
        rejected_jsonl,
        rejected,
    )

    report = make_report(
        model_key=model_key,
        model_name=model_name,
        alignment_path=alignment_path,
        output_dir=output_dir,
        alignment_rows=alignment_rows,
        layer_rows=clean_rows,
        rejected=rejected,
        dtype=dtype,
        device_map=args.device_map,
        num_layers=num_layers,
        raw_jsonl=raw_jsonl,
        compact_tsv=compact_tsv,
        rejected_jsonl=rejected_jsonl,
        run_config_json=run_config_json,
    )

    report["processed_this_run"] = processed_this_run

    completed_after = completed_trajectory_keys(
        clean_rows,
        expected_layers,
    )
    completed_requested = completed_after & requested_keys

    report["counts"]["completed_trajectories"] = len(completed_requested)

    report["validation"] = {
        "all_requested_trajectories_complete": (
            len(completed_requested) == len(alignment_rows) and len(rejected) == 0
        ),
        "all_saved_layers_contiguous": (
            all(
                {
                    int(item["layer"])
                    for item in clean_rows
                    if trajectory_key(item) == key
                }
                == expected_layers
                for key in completed_requested
            )
        ),
        "gold_and_broader_measured_from_same_hidden_state": True,
        "final_layer_prediction_consistency_checked_per_trajectory": True,
    }

    write_json(
        report_json,
        report,
    )

    print(
        f"{model_key}: {len(completed_requested)} processed, "
        f"{len(rejected)} rejected, {len(clean_rows)} layer rows",
        flush=True,
    )
    if rejected:
        print(f"  Rejections: {rejected_jsonl}", file=sys.stderr)
        return 2

    expected_row_count = len(alignment_rows) * (num_layers + 1)

    if len(clean_rows) != expected_row_count:
        raise AssertionError(
            "Unexpected final layer-row count: "
            f"expected {expected_row_count}, "
            f"observed {len(clean_rows)}."
        )

    return 0


# CLI


def build_parser():
    parser = argparse.ArgumentParser(
        description="Track specific gold and broader-category tokens under the same context."
    )
    parser.add_argument(
        "--model",
        "--models",
        "--model_key",
        nargs="+",
        choices=["all", *MODEL_KEYS],
        default=["qwen3_0_6b"],
    )
    parser.add_argument("--input", "--alignment", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        "--output_base",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--device-map", "--device_map", default=None)
    parser.add_argument(
        "--device-map-auto", action="store_const", const="auto", dest="device_map"
    )
    parser.add_argument("--attn-implementation", "--attn_implementation", default=None)
    parser.add_argument("--cache-dir", "--cache_dir", type=Path, default=None)
    parser.add_argument(
        "--progress-every",
        "--progress_every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Progress interval; 0 disables progress messages.",
    )
    parser.add_argument(
        "--empty-cache-every",
        "--empty_cache_every",
        type=int,
        default=DEFAULT_EMPTY_CACHE_EVERY,
    )
    parser.add_argument("--max-rows", "--max_rows", type=int, default=None)
    parser.add_argument(
        "--allow-nonstandard-count", "--allow_nonstandard_count", action="store_true"
    )
    parser.add_argument("--no-resume", "--no_resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        "--reset-output",
        "--reset_output",
        action="store_true",
        help="Discard existing outputs and start a new extraction.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-prob", action="store_true")
    parser.add_argument("--no-store-topk", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    if args.top_k < 10:
        parser.error("--top-k must be at least 10")
    if args.progress_every < 0 or args.empty_cache_every < 1:
        parser.error("Invalid progress or cache interval")
    if "all" in args.model and len(args.model) != 1:
        parser.error("Use all alone or specify individual model keys")
    if (
        args.max_rows is not None
        or args.allow_nonstandard_count
        or args.top_k != 10
        or args.no_prob
        or args.no_store_topk
    ) and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        parser.error("Use a separate --output-dir for debug runs")

    global TOP_K, COMPUTE_PROB, STORE_TOPK
    TOP_K = args.top_k
    COMPUTE_PROB = not args.no_prob
    STORE_TOPK = not args.no_store_topk
    model_keys = [
        key for key in MODEL_KEYS if args.model == ["all"] or key in args.model
    ]
    if not args.input.is_file():
        parser.error(f"Alignment file not found: {args.input}")
    for model_key in model_keys:
        output_dir = args.output_dir / model_key
        existing = [
            output_dir / name
            for name in OUTPUT_FILENAMES
            if (output_dir / name).exists()
        ]
        if args.overwrite or not existing:
            continue
        if args.no_resume:
            parser.error(
                f"Existing outputs for {model_key}; use --overwrite to replace them."
            )
        raw_path = output_dir / "broader_category_by_layer.jsonl"
        config_path = output_dir / "run_config.json"
        if not raw_path.exists() or not config_path.exists():
            parser.error(
                f"Existing outputs for {model_key} lack a resumable raw/config pair; use a different --output-dir or --overwrite."
            )
        report_path = output_dir / "broader_category_extraction_report.json"
        if (
            report_path.exists()
            and load_json(report_path)
            .get("validation", {})
            .get("all_requested_trajectories_complete")
            is True
        ):
            parser.error(
                f"Completed outputs for {model_key}; use --overwrite to rerun or select a different output directory."
            )
    status = 0
    for model_key in model_keys:
        try:
            status = max(status, run_model(model_key, args))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
