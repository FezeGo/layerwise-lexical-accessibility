import argparse
from pathlib import Path
import json
import csv
import os
import math
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Config
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_ALIGNMENT_DIR = Path("data/alignment")
BASE_OUTPUT_DIR = Path("data/logit_lens/full")

MODEL_REGISTRY = {
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_14b": "Qwen/Qwen3-14B",
}

# Configurable runtime selections; defaults match the original active run.
MODELS = {"qwen3_0_6b": MODEL_REGISTRY["qwen3_0_6b"]}

# Main logit-lens setting.
# For Qwen/Llama-style models, final logits are computed as:
#   lm_head(final_norm(hidden_state))
# We apply the same final norm + lm_head to intermediate hidden states.
APPLY_FINAL_NORM = True

TOP_K = 10
COMPUTE_PROB = True
STORE_TOPK = True

MAX_ALIGNMENT_ROWS = None  # set to a small number for debugging, e.g. 20
PROGRESS_EVERY = 50

# For large models, keep False first.
# If 14B does not fit on one GPU and you have accelerate installed,
# you can try setting this to True.
USE_DEVICE_MAP_AUTO = False


# ============================================================
# IO helpers
# ============================================================


def read_jsonl(path, max_rows=None):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_line_idx"] = line_idx
            items.append(item)
            if max_rows is not None and len(items) >= max_rows:
                break
    return items


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_tsv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ============================================================
# Device / model helpers
# ============================================================


def get_device_and_dtype():
    if torch.cuda.is_available():
        # bfloat16 is usually safer on newer GPUs.
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float16

    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32

    return torch.device("cpu"), torch.float32


def load_model_and_tokenizer(model_name):
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


def get_input_device(model):
    return model.get_input_embeddings().weight.device


def get_final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm

    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f

    return None


def get_module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def project_hidden_to_logits(
    hidden_state,
    final_norm,
    lm_head,
    layer_idx,
    last_layer_idx,
):
    """
    Plain logit lens projection.

    For Qwen/Llama-style models:
    - embedding layer: no final norm
    - intermediate layers: apply final norm before lm_head
    - final layer: should be handled separately using outputs.logits
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

    logits = lm_head(h).squeeze(0).squeeze(0)
    return logits.float()


# ============================================================
# Logit / rank helpers
# ============================================================


def rank_of_token(logits, token_id):
    """
    1-based rank.
    rank = 1 means highest logit.
    """
    token_logit = logits[token_id]
    rank = int((logits > token_logit).sum().item()) + 1
    return rank, float(token_logit.item())


def get_topk_info(logits, tokenizer, k):
    top_values, top_ids = torch.topk(logits, k=k)
    top_ids = top_ids.detach().cpu().tolist()
    top_values = [float(x) for x in top_values.detach().cpu().tolist()]

    top_tokens = tokenizer.convert_ids_to_tokens(top_ids)
    top_texts = [tokenizer.decode([tid]) for tid in top_ids]

    return {
        "topk_token_ids": top_ids,
        "topk_tokens": top_tokens,
        "topk_texts": top_texts,
        "topk_logits": top_values,
    }


def token_to_text(tokenizer, token_id):
    return tokenizer.decode([int(token_id)])


def safe_float(x):
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
    return float(x)


def topk_jaccard(layer_topk_ids, final_topk_ids):
    """
    Jaccard similarity between the layer's top-k token-id set and the
    final layer's top-k token-id set.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Token IDs are used rather than decoded strings so that distinct tokens
    with similar-looking text are not accidentally merged.
    """
    layer_set = {int(token_id) for token_id in layer_topk_ids}
    final_set = {int(token_id) for token_id in final_topk_ids}

    union = layer_set | final_set
    if not union:
        return 1.0

    return len(layer_set & final_set) / len(union)


# ============================================================
# One-example logit lens
# ============================================================

# @torch.inference_mode()
# def run_logit_lens_for_alignment(row, model, tokenizer, final_norm, lm_head):
#     """
#     One alignment row = one example-language-model item.

#     Uses:
#         full_input_ids
#         prediction_position
#         target_token_id

#     This matches the fullspan alignment setup.

#     Important:
#     - For intermediate layers, we apply final_norm + lm_head.
#     - For the final layer, we use outputs.logits directly.
#       This avoids accidentally applying the final norm twice.
#     """

#     input_ids_list = row["full_input_ids"]
#     prediction_position = int(row["prediction_position"])
#     target_token_id = int(row["target_token_id"])

#     input_device = get_input_device(model)

#     input_ids = torch.tensor(
#         [input_ids_list],
#         dtype=torch.long,
#         device=input_device,
#     )

#     outputs = model(
#         input_ids=input_ids,
#         output_hidden_states=True,
#         use_cache=False,
#         return_dict=True,
#     )

#     hidden_states = outputs.hidden_states

#     # hidden_states includes:
#     #   hidden_states[0]  = embedding output
#     #   hidden_states[1:] = transformer layer outputs
#     # For Qwen3-0.6B, this should be 29 hidden states:
#     #   0 embedding + 28 transformer layers
#     last_layer_idx = len(hidden_states) - 1
#     num_layers = last_layer_idx

#     # ------------------------------------------------------------
#     # Use the model's actual final logits for final prediction.
#     # This is the safest final-layer reference.
#     # ------------------------------------------------------------

#     final_logits = outputs.logits[0, prediction_position, :].float()

#     final_topk = get_topk_info(final_logits, tokenizer, TOP_K)
#     final_topk_token_ids = [
#         int(token_id)
#         for token_id in final_topk["topk_token_ids"]
#     ]

#     final_prediction_token_id = int(final_topk_token_ids[0])
#     final_prediction_token = final_topk["topk_tokens"][0]
#     final_prediction_text = final_topk["topk_texts"][0]

#     final_prediction_is_target = (
#         final_prediction_token_id == target_token_id
#     )

#     results = []

#     for layer_idx, layer_hidden_states in enumerate(hidden_states):
#         # --------------------------------------------------------
#         # Final layer:
#         # use outputs.logits directly, not final_norm + lm_head.
#         # --------------------------------------------------------
#         if layer_idx == last_layer_idx:
#             logits = final_logits

#         # --------------------------------------------------------
#         # Embedding / intermediate layers:
#         # use logit lens projection.
#         #
#         # The helper project_hidden_to_logits should apply final_norm
#         # only when appropriate. If your project_hidden_to_logits does
#         # not take layer_idx, use the updated helper below.
#         # --------------------------------------------------------
#         else:
#             hidden = layer_hidden_states[0, prediction_position, :]

#             logits = project_hidden_to_logits(
#                 hidden_state=hidden,
#                 final_norm=final_norm,
#                 lm_head=lm_head,
#                 layer_idx=layer_idx,
#                 last_layer_idx=last_layer_idx,
#             )

#         target_rank, target_logit = rank_of_token(
#             logits,
#             target_token_id,
#         )

#         final_rank, final_logit_at_layer = rank_of_token(
#             logits,
#             final_prediction_token_id,
#         )

#         topk = get_topk_info(logits, tokenizer, TOP_K)
#         top_ids = [
#             int(token_id)
#             for token_id in topk["topk_token_ids"]
#         ]

#         topk_overlap_count_with_final = len(
#             set(top_ids) & set(final_topk_token_ids)
#         )
#         topk_jaccard_with_final = topk_jaccard(
#             top_ids,
#             final_topk_token_ids,
#         )

#         target_prob = None
#         final_prediction_prob_at_layer = None

#         if COMPUTE_PROB:
#             probs = F.softmax(logits, dim=-1)
#             target_prob = float(probs[target_token_id].item())
#             final_prediction_prob_at_layer = float(
#                 probs[final_prediction_token_id].item()
#             )

#         layer_top1_token_id = int(top_ids[0])
#         layer_top1_token = topk["topk_tokens"][0]
#         layer_top1_text = topk["topk_texts"][0]
#         layer_top1_logit = float(topk["topk_logits"][0])

#         if layer_idx == 0:
#             layer_name = "embedding"
#         elif layer_idx == last_layer_idx:
#             layer_name = "final"
#         else:
#             layer_name = f"layer_{layer_idx}"

#         result = {
#             "id": row["id"],
#             "semantic_group": row["semantic_group"],
#             "subgroup": row["subgroup"],
#             "relation": row["relation"],
#             "language": row["language"],

#             "model_key": row["model_key"],
#             "model_name": row["model_name"],

#             "layer": layer_idx,
#             "layer_name": layer_name,
#             "num_layers": num_layers,
#             "layer_fraction": layer_idx / num_layers if num_layers > 0 else None,

#             "target": row["target"],
#             "target_token_id": target_token_id,
#             "target_token": row["target_token"],
#             "target_token_surface": row["target_token_surface"],

#             "prediction_position": prediction_position,
#             "prediction_token_id": row["prediction_token_id"],
#             "prediction_token": row["prediction_token"],

#             "target_logit": target_logit,
#             "target_prob": safe_float(target_prob),
#             "target_rank": target_rank,
#             "target_in_top1": target_rank <= 1,
#             "target_in_top5": target_rank <= 5,
#             "target_in_top10": target_rank <= 10,

#             "layer_top1_token_id": layer_top1_token_id,
#             "layer_top1_token": layer_top1_token,
#             "layer_top1_text": layer_top1_text,
#             "layer_top1_logit": layer_top1_logit,

#             "final_prediction_token_id": final_prediction_token_id,
#             "final_prediction_token": final_prediction_token,
#             "final_prediction_text": final_prediction_text,
#             "final_prediction_is_target": final_prediction_is_target,

#             "final_prediction_logit_at_layer": final_logit_at_layer,
#             "final_prediction_prob_at_layer": safe_float(
#                 final_prediction_prob_at_layer
#             ),
#             "final_prediction_rank_at_layer": final_rank,
#             "faithful_final_at_1": final_rank <= 1,
#             "faithful_final_at_5": final_rank <= 5,
#             "faithful_final_at_10": final_rank <= 10,

#             # Set-based overlap between this layer's top-k predictions
#             # and the true final layer's top-k predictions.
#             "topk_overlap_count_with_final":
#                 topk_overlap_count_with_final,
#             "topk_jaccard_with_final":
#                 float(topk_jaccard_with_final),
#         }

#         if STORE_TOPK:
#             result.update(topk)

#         results.append(result)

#     return results


@torch.inference_mode()
def run_logit_lens_for_alignment(
    row,
    model,
    tokenizer,
    final_norm,
    lm_head,
):
    """
    Run the plain Logit Lens for one aligned example.

    One alignment row corresponds to one
    example-language-model item.

    Required alignment fields:
        full_input_ids
        prediction_position
        target_token_id

    This function follows the full-span alignment setup.

    Projection rules:
    - Embedding and intermediate hidden states are projected using
      final_norm + lm_head.
    - The final layer uses outputs.logits directly.
      This avoids applying the model's final normalization twice.

    Metric definitions:
    - target_rank is retained as a separate rank-based diagnostic.
    - Target@1/5/10 is determined using membership in the actual
      torch.topk result, rather than target_rank <= k.
    - Faithful-to-final@1/5/10 is also determined using actual
      top-k membership.

    Using actual top-k membership avoids inconsistencies when several
    tokens have exactly tied logits.
    """

    if TOP_K < 10:
        raise ValueError(
            "TOP_K must be at least 10 to compute Target@10 and Faithful-to-final@10."
        )

    input_ids_list = row["full_input_ids"]
    prediction_position = int(row["prediction_position"])
    target_token_id = int(row["target_token_id"])

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

    # hidden_states contains:
    #
    # hidden_states[0]
    #     embedding output
    #
    # hidden_states[1:]
    #     transformer-layer outputs
    #
    # Therefore, if a model has L transformer layers,
    # len(hidden_states) should normally be L + 1.
    last_layer_idx = len(hidden_states) - 1
    num_layers = last_layer_idx

    # ------------------------------------------------------------
    # Final-layer reference
    # ------------------------------------------------------------
    #
    # Use the model's actual output logits as the final-layer
    # reference. Do not project the final hidden state again.
    # ------------------------------------------------------------

    final_logits = outputs.logits[
        0,
        prediction_position,
        :,
    ].float()

    final_topk = get_topk_info(
        final_logits,
        tokenizer,
        TOP_K,
    )

    final_topk_token_ids = [int(token_id) for token_id in final_topk["topk_token_ids"]]

    if len(final_topk_token_ids) < 10:
        raise ValueError("get_topk_info returned fewer than 10 token IDs.")

    final_prediction_token_id = int(final_topk_token_ids[0])

    final_prediction_token = final_topk["topk_tokens"][0]

    final_prediction_text = final_topk["topk_texts"][0]

    final_prediction_is_target = final_prediction_token_id == target_token_id

    results = []

    # ------------------------------------------------------------
    # Layer-wise Logit Lens
    # ------------------------------------------------------------

    for layer_idx, layer_hidden_states in enumerate(hidden_states):
        # --------------------------------------------------------
        # Final layer
        # --------------------------------------------------------
        #
        # Use outputs.logits directly.
        # --------------------------------------------------------

        if layer_idx == last_layer_idx:
            logits = final_logits

        # --------------------------------------------------------
        # Embedding and intermediate layers
        # --------------------------------------------------------
        #
        # Apply the plain Logit Lens projection.
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Rank-based diagnostics
        # --------------------------------------------------------
        #
        # These ranks remain useful for mean target rank and
        # final-prediction rank analyses.
        #
        # However, they are not used to define Target@k or
        # Faithful-to-final@k because tied logits can give multiple
        # tokens the same numerical rank.
        # --------------------------------------------------------

        target_rank, target_logit = rank_of_token(
            logits,
            target_token_id,
        )

        (
            final_rank,
            final_logit_at_layer,
        ) = rank_of_token(
            logits,
            final_prediction_token_id,
        )

        # --------------------------------------------------------
        # Actual top-k predictions
        # --------------------------------------------------------

        topk = get_topk_info(
            logits,
            tokenizer,
            TOP_K,
        )

        top_ids = [int(token_id) for token_id in topk["topk_token_ids"]]

        if len(top_ids) < 10:
            raise ValueError(
                f"get_topk_info returned fewer than 10 token IDs at layer {layer_idx}."
            )

        # --------------------------------------------------------
        # Exact Target@k membership
        # --------------------------------------------------------
        #
        # These are based on the actual ordered top-k list.
        # --------------------------------------------------------

        target_in_top1 = target_token_id == top_ids[0]

        target_in_top5 = target_token_id in top_ids[:5]

        target_in_top10 = target_token_id in top_ids[:10]

        # --------------------------------------------------------
        # Faithfulness to the final prediction
        # --------------------------------------------------------
        #
        # Check whether the model's actual final Top-1 prediction
        # is present in this layer's actual top-k list.
        # --------------------------------------------------------

        faithful_final_at_1 = final_prediction_token_id == top_ids[0]

        faithful_final_at_5 = final_prediction_token_id in top_ids[:5]

        faithful_final_at_10 = final_prediction_token_id in top_ids[:10]

        # --------------------------------------------------------
        # Top-k set overlap with final layer
        # --------------------------------------------------------

        topk_overlap_count_with_final = len(set(top_ids) & set(final_topk_token_ids))

        topk_jaccard_with_final = topk_jaccard(
            top_ids,
            final_topk_token_ids,
        )

        # --------------------------------------------------------
        # Token probabilities
        # --------------------------------------------------------

        target_prob = None
        final_prediction_prob_at_layer = None

        if COMPUTE_PROB:
            probs = F.softmax(
                logits,
                dim=-1,
            )

            target_prob = float(probs[target_token_id].item())

            final_prediction_prob_at_layer = float(
                probs[final_prediction_token_id].item()
            )

        # --------------------------------------------------------
        # Layer Top-1 information
        # --------------------------------------------------------

        layer_top1_token_id = int(top_ids[0])

        layer_top1_token = topk["topk_tokens"][0]

        layer_top1_text = topk["topk_texts"][0]

        layer_top1_logit = float(topk["topk_logits"][0])

        # --------------------------------------------------------
        # Layer name
        # --------------------------------------------------------

        if layer_idx == 0:
            layer_name = "embedding"

        elif layer_idx == last_layer_idx:
            layer_name = "final"

        else:
            layer_name = f"layer_{layer_idx}"

        # --------------------------------------------------------
        # Final-layer sanity checks
        # --------------------------------------------------------
        #
        # At the final layer:
        #
        # 1. The layer Top-1 must equal the stored final prediction.
        # 2. Target@1 must equal final_prediction_is_target.
        # 3. Faithfulness@1/5/10 must all equal True.
        # 4. The final top-k set must completely overlap with itself.
        # --------------------------------------------------------

        if layer_idx == last_layer_idx:
            assert layer_top1_token_id == final_prediction_token_id, (
                "Final-layer Top-1 does not match the "
                "stored final prediction: "
                f"id={row['id']}, "
                f"language={row['language']}, "
                f"layer_top1={layer_top1_token_id}, "
                f"final_prediction="
                f"{final_prediction_token_id}"
            )

            assert target_in_top1 == final_prediction_is_target, (
                "Final-layer Target@1 mismatch: "
                f"id={row['id']}, "
                f"language={row['language']}, "
                f"target_id={target_token_id}, "
                f"top1_id={layer_top1_token_id}, "
                f"target_in_top1={target_in_top1}, "
                f"final_prediction_is_target="
                f"{final_prediction_is_target}"
            )

            assert faithful_final_at_1, "Final-layer Faithful-to-final@1 must be True."

            assert faithful_final_at_5, "Final-layer Faithful-to-final@5 must be True."

            assert faithful_final_at_10, (
                "Final-layer Faithful-to-final@10 must be True."
            )

            assert topk_overlap_count_with_final == len(final_topk_token_ids), (
                "Final-layer top-k overlap count must equal TOP_K."
            )

            assert math.isclose(
                float(topk_jaccard_with_final),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ), "Final-layer top-k Jaccard must equal 1.0."

        # --------------------------------------------------------
        # Output record
        # --------------------------------------------------------

        result = {
            "id": row["id"],
            "semantic_group": row["semantic_group"],
            "subgroup": row["subgroup"],
            "relation": row["relation"],
            "language": row["language"],
            "model_key": row["model_key"],
            "model_name": row["model_name"],
            "layer": layer_idx,
            "layer_name": layer_name,
            "num_layers": num_layers,
            "layer_fraction": (layer_idx / num_layers if num_layers > 0 else None),
            "target": row["target"],
            "target_token_id": target_token_id,
            "target_token": row["target_token"],
            "target_token_surface": row["target_token_surface"],
            "prediction_position": prediction_position,
            "prediction_token_id": row["prediction_token_id"],
            "prediction_token": row["prediction_token"],
            # Gold-target diagnostics
            "target_logit": target_logit,
            "target_prob": safe_float(target_prob),
            "target_rank": target_rank,
            # Actual top-k membership
            "target_in_top1": target_in_top1,
            "target_in_top5": target_in_top5,
            "target_in_top10": target_in_top10,
            # Current layer Top-1
            "layer_top1_token_id": layer_top1_token_id,
            "layer_top1_token": layer_top1_token,
            "layer_top1_text": layer_top1_text,
            "layer_top1_logit": layer_top1_logit,
            # Final prediction
            "final_prediction_token_id": final_prediction_token_id,
            "final_prediction_token": final_prediction_token,
            "final_prediction_text": final_prediction_text,
            "final_prediction_is_target": final_prediction_is_target,
            # Final prediction at the current layer
            "final_prediction_logit_at_layer": final_logit_at_layer,
            "final_prediction_prob_at_layer": safe_float(
                final_prediction_prob_at_layer
            ),
            "final_prediction_rank_at_layer": final_rank,
            # Actual top-k faithfulness membership
            "faithful_final_at_1": faithful_final_at_1,
            "faithful_final_at_5": faithful_final_at_5,
            "faithful_final_at_10": faithful_final_at_10,
            # Set-based overlap between this layer's
            # top-k predictions and the true final
            # layer's top-k predictions.
            "topk_overlap_count_with_final": topk_overlap_count_with_final,
            "topk_jaccard_with_final": float(topk_jaccard_with_final),
        }

        if STORE_TOPK:
            result.update(topk)

        results.append(result)

    return results


# ============================================================
# Aggregation
# ============================================================

# ============================================================
# Aggregation
# ============================================================

FINAL_EXACT_GOLD = "final_exact_gold"
FINAL_NON_GOLD = "final_non_gold"


def mean(values):
    """
    Compute the arithmetic mean while ignoring None values.
    """
    values = [value for value in values if value is not None]

    if not values:
        return None

    return sum(values) / len(values)


def bool_rate(rows, key):
    """
    Compute the proportion of True values for a Boolean field.

    None values are ignored, although the current Logit Lens output
    should normally contain a Boolean value for all requested fields.
    """
    values = [bool(row[key]) for row in rows if row.get(key) is not None]

    if not values:
        return None

    return sum(values) / len(values)


def get_final_match_status(row):
    """
    Assign a neutral exact-match label to one result row.

    Important:
    - final_exact_gold:
        the model's final Top-1 token is exactly the predefined
        gold target token.

    - final_non_gold:
        the model's final Top-1 token is not the predefined
        gold target token.

    final_non_gold does not necessarily mean semantically incorrect.
    It may include translation equivalents, synonyms, direct
    hypernyms, or other plausible completions.
    """
    if bool(row["final_prediction_is_target"]):
        return FINAL_EXACT_GOLD

    return FINAL_NON_GOLD


def validate_exact_gold_rows(rows):
    """
    Sanity check for examples whose final prediction exactly matches
    the gold target.

    For these examples, the gold target token and the model's final
    prediction token are identical. Therefore, at every layer:

        Target@k == Faithful-to-final@k

    for k in {1, 5, 10}.
    """
    if not rows:
        return

    if not all(bool(row["final_prediction_is_target"]) for row in rows):
        return

    metric_pairs = [
        ("target_in_top1", "faithful_final_at_1"),
        ("target_in_top5", "faithful_final_at_5"),
        ("target_in_top10", "faithful_final_at_10"),
    ]

    for target_key, faithful_key in metric_pairs:
        for row in rows:
            if bool(row[target_key]) != bool(row[faithful_key]):
                raise AssertionError(
                    "Exact-gold sanity check failed: "
                    f"id={row.get('id')}, "
                    f"language={row.get('language')}, "
                    f"layer={row.get('layer')}, "
                    f"{target_key}={row.get(target_key)}, "
                    f"{faithful_key}={row.get(faithful_key)}"
                )


def aggregate_results(results):
    """
    Aggregate layer-wise Logit Lens results.

    Existing grouping dimensions:
        overall
        language
        semantic_group
        language_semantic_group
        subgroup
        relation

    New grouping dimensions:
        final_match_status
        language_final_match_status
        semantic_group_final_match_status
        language_semantic_group_final_match_status

    The final-match grouping distinguishes:

        final_exact_gold
        final_non_gold

    This is an exact-token distinction, not a semantic correctness
    judgment.
    """

    # Each grouping is represented by:
    #
    #   grouping name
    #   function that maps one raw result row to its group label
    #
    # Using functions makes it possible to define derived groupings
    # such as final_match_status without modifying the raw result rows.
    groupings = [
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
            "subgroup",
            lambda row: str(row["subgroup"]),
        ),
        (
            "relation",
            lambda row: str(row["relation"]),
        ),
        # --------------------------------------------------------
        # New final exact-match groupings
        # --------------------------------------------------------
        (
            "final_match_status",
            lambda row: get_final_match_status(row),
        ),
        (
            "language_final_match_status",
            lambda row: f"{row['language']}|{get_final_match_status(row)}",
        ),
        (
            "semantic_group_final_match_status",
            lambda row: f"{row['semantic_group']}|{get_final_match_status(row)}",
        ),
        (
            "language_semantic_group_final_match_status",
            lambda row: (
                f"{row['language']}|"
                f"{row['semantic_group']}|"
                f"{get_final_match_status(row)}"
            ),
        ),
    ]

    aggregate_rows = []

    for grouping_name, group_function in groupings:
        buckets = defaultdict(list)

        # --------------------------------------------------------
        # Assign each layer-wise result row to a bucket
        # --------------------------------------------------------

        for row in results:
            group_value = group_function(row)

            bucket_key = (
                grouping_name,
                group_value,
                row["model_key"],
                row["layer"],
            )

            buckets[bucket_key].append(row)

        # --------------------------------------------------------
        # Aggregate each bucket
        # --------------------------------------------------------

        sorted_buckets = sorted(
            buckets.items(),
            key=lambda item: (
                item[0][0],  # grouping name
                item[0][1],  # group value
                item[0][2],  # model key
                item[0][3],  # layer
            ),
        )

        for (
            grouping,
            group_value,
            model_key,
            layer,
        ), rows in sorted_buckets:
            # For an exact-gold subset, Target@k and Faithful@k
            # must be identical because they refer to the same token.
            validate_exact_gold_rows(rows)

            aggregate_row = {
                "grouping": grouping,
                "group": group_value,
                "model_key": model_key,
                "layer": layer,
                "n": len(rows),
                # ------------------------------------------------
                # Gold-target accessibility
                # ------------------------------------------------
                "target_at_1": bool_rate(
                    rows,
                    "target_in_top1",
                ),
                "target_at_5": bool_rate(
                    rows,
                    "target_in_top5",
                ),
                "target_at_10": bool_rate(
                    rows,
                    "target_in_top10",
                ),
                "mean_target_rank": mean(row["target_rank"] for row in rows),
                "mean_target_logit": mean(row["target_logit"] for row in rows),
                "mean_target_prob": mean(row["target_prob"] for row in rows),
                # ------------------------------------------------
                # Faithfulness to the model's actual final Top-1
                # ------------------------------------------------
                "faithful_final_at_1": bool_rate(
                    rows,
                    "faithful_final_at_1",
                ),
                "faithful_final_at_5": bool_rate(
                    rows,
                    "faithful_final_at_5",
                ),
                "faithful_final_at_10": bool_rate(
                    rows,
                    "faithful_final_at_10",
                ),
                "mean_final_prediction_rank_at_layer": mean(
                    row["final_prediction_rank_at_layer"] for row in rows
                ),
                # ------------------------------------------------
                # Final exact-gold match
                # ------------------------------------------------
                #
                # For final_match_status groups, this will be:
                #
                #   1.0 for final_exact_gold
                #   0.0 for final_non_gold
                #
                # This is expected and can serve as a sanity check.
                # ------------------------------------------------
                "final_prediction_is_target_rate": bool_rate(
                    rows,
                    "final_prediction_is_target",
                ),
                # ------------------------------------------------
                # Top-k convergence toward the final layer
                # ------------------------------------------------
                "mean_topk_overlap_count_with_final": mean(
                    row["topk_overlap_count_with_final"] for row in rows
                ),
                "mean_topk_jaccard_with_final": mean(
                    row["topk_jaccard_with_final"] for row in rows
                ),
            }

            aggregate_rows.append(aggregate_row)

    return aggregate_rows


# ============================================================
# Main per model
# ============================================================


def run_for_model(model_key, model_name):
    alignment_path = BASE_ALIGNMENT_DIR / model_key / "token_alignment.jsonl"
    output_dir = BASE_OUTPUT_DIR / model_key

    raw_jsonl = output_dir / "logit_lens_by_layer.jsonl"
    compact_tsv = output_dir / "logit_lens_by_layer_compact.tsv"
    summary_tsv = output_dir / "logit_lens_summary_by_layer.tsv"
    summary_json = output_dir / "logit_lens_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not alignment_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

    alignment_rows = read_jsonl(
        alignment_path,
        max_rows=MAX_ALIGNMENT_ROWS,
    )

    alignment_rows = [r for r in alignment_rows if r.get("model_key") == model_key]

    if not alignment_rows:
        raise ValueError(
            f"No alignment rows found for model_key={model_key} in {alignment_path}"
        )

    print("\n========================================")
    print(f"Model: {model_key} -> {model_name}")
    print(f"Alignment rows: {len(alignment_rows)}")
    print(f"Alignment path: {alignment_path}")
    print(f"Output dir: {output_dir}")
    print("========================================")

    model, tokenizer = load_model_and_tokenizer(model_name)

    final_norm = get_final_norm(model)
    lm_head = model.get_output_embeddings()

    if lm_head is None:
        raise RuntimeError("model.get_output_embeddings() returned None.")

    if APPLY_FINAL_NORM and final_norm is None:
        print("Warning: final norm not found. Using lm_head only.")

    all_results = []
    rejected = []

    for idx, row in enumerate(alignment_rows, start=1):
        try:
            results = run_logit_lens_for_alignment(
                row=row,
                model=model,
                tokenizer=tokenizer,
                final_norm=final_norm,
                lm_head=lm_head,
            )
            all_results.extend(results)

        except Exception as e:
            rejected.append(
                {
                    "alignment_row_index": idx - 1,
                    "id": row.get("id"),
                    "language": row.get("language"),
                    "model_key": model_key,
                    "reason": str(e),
                    "row": row,
                }
            )

        if idx % PROGRESS_EVERY == 0:
            print(
                f"Processed {idx}/{len(alignment_rows)} alignment rows; "
                f"logit-lens rows so far: {len(all_results)}; "
                f"rejected: {len(rejected)}"
            )

        if torch.cuda.is_available() and idx % 100 == 0:
            torch.cuda.empty_cache()

    write_jsonl(raw_jsonl, all_results)

    compact_fields = [
        "id",
        "semantic_group",
        "subgroup",
        "relation",
        "language",
        "model_key",
        "layer",
        "layer_name",
        "target",
        "target_token_id",
        "target_token_surface",
        "target_rank",
        "target_prob",
        "target_in_top1",
        "target_in_top5",
        "target_in_top10",
        "layer_top1_text",
        "final_prediction_text",
        "final_prediction_rank_at_layer",
        "faithful_final_at_1",
        "faithful_final_at_5",
        "faithful_final_at_10",
        "topk_overlap_count_with_final",
        "topk_jaccard_with_final",
        "final_prediction_is_target",
    ]

    write_tsv(compact_tsv, all_results, compact_fields)

    aggregate_rows = aggregate_results(all_results)

    summary_fields = [
        "grouping",
        "group",
        "model_key",
        "layer",
        "n",
        "target_at_1",
        "target_at_5",
        "target_at_10",
        "mean_target_rank",
        "mean_target_logit",
        "mean_target_prob",
        "faithful_final_at_1",
        "faithful_final_at_5",
        "faithful_final_at_10",
        "mean_final_prediction_rank_at_layer",
        "final_prediction_is_target_rate",
        "mean_topk_overlap_count_with_final",
        "mean_topk_jaccard_with_final",
    ]

    write_tsv(summary_tsv, aggregate_rows, summary_fields)

    rejected_jsonl = output_dir / "logit_lens.rejected.jsonl"
    write_jsonl(rejected_jsonl, rejected)

    summary = {
        "model_key": model_key,
        "model_name": model_name,
        "alignment_path": str(alignment_path),
        "output_dir": str(output_dir),
        "num_alignment_rows": len(alignment_rows),
        "num_logit_lens_rows": len(all_results),
        "num_rejected": len(rejected),
        "apply_final_norm": APPLY_FINAL_NORM,
        "top_k": TOP_K,
        "compute_prob": COMPUTE_PROB,
        "store_topk": STORE_TOPK,
        "jaccard_top_k": TOP_K,
        "jaccard_definition": (
            "set Jaccard between each layer's top-k token IDs "
            "and the final layer's top-k token IDs"
        ),
        "counts_by_language": dict(Counter(r["language"] for r in alignment_rows)),
        "counts_by_semantic_group": dict(
            Counter(r["semantic_group"] for r in alignment_rows)
        ),
        "num_layers_detected": (
            max(r["layer"] for r in all_results) if all_results else None
        ),
        "raw_jsonl": str(raw_jsonl),
        "compact_tsv": str(compact_tsv),
        "summary_tsv": str(summary_tsv),
        "rejected_jsonl": str(rejected_jsonl),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone for model:", model_key)
    print(f"Alignment rows: {len(alignment_rows)}")
    print(f"Logit-lens rows: {len(all_results)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Raw JSONL: {raw_jsonl}")
    print(f"Compact TSV: {compact_tsv}")
    print(f"Summary TSV: {summary_tsv}")
    print(f"Summary JSON: {summary_json}")

    if rejected:
        print("\nWarning: some rows failed. Check:")
        print(rejected_jsonl)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the thesis plain Logit Lens on precomputed token alignments."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_REGISTRY),
        default=["qwen3_0_6b"],
        metavar="MODEL",
        help="Model key(s): qwen3_0_6b, qwen3_4b, qwen3_14b (default: qwen3_0_6b).",
    )
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        default=BASE_ALIGNMENT_DIR,
        help="Directory containing <model_key>/token_alignment.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_OUTPUT_DIR,
        help="Output root; each model writes to <output-dir>/<model_key>/.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process at most N alignment rows per model (for smoke tests).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-scoring tokens to store; must be >= 10.",
    )
    parser.add_argument(
        "--no-prob",
        action="store_true",
        help="Skip softmax-based token probabilities.",
    )
    parser.add_argument(
        "--no-store-topk",
        action="store_true",
        help="Omit full top-k lists from each JSONL record.",
    )
    parser.add_argument(
        "--device-map-auto",
        action="store_true",
        help="Use Hugging Face device_map='auto' (requires accelerate).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N alignment rows (default: 50).",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be positive")
    if args.top_k < 10:
        raise SystemExit("--top-k must be >= 10 for Target@10 and Faithful@10")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")

    global BASE_ALIGNMENT_DIR, BASE_OUTPUT_DIR, MODELS
    global TOP_K, COMPUTE_PROB, STORE_TOPK
    global MAX_ALIGNMENT_ROWS, PROGRESS_EVERY, USE_DEVICE_MAP_AUTO

    BASE_ALIGNMENT_DIR = args.alignment_dir
    BASE_OUTPUT_DIR = args.output_dir
    MODELS = {key: MODEL_REGISTRY[key] for key in dict.fromkeys(args.models)}
    TOP_K = args.top_k
    COMPUTE_PROB = not args.no_prob
    STORE_TOPK = not args.no_store_topk
    MAX_ALIGNMENT_ROWS = args.max_rows
    PROGRESS_EVERY = args.progress_every
    USE_DEVICE_MAP_AUTO = args.device_map_auto

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_key, model_name in MODELS.items():
        run_for_model(model_key, model_name)

    print("\n================ Overall ================")
    print(f"Active models: {', '.join(MODELS.keys())}")
    print(f"Base output dir: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
