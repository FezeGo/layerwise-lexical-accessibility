import argparse
import csv
import gc
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_INPUT_DIR = Path("data/alignment")
DEFAULT_OUTPUT_DIR = Path("data/logit_lens/full")

MODEL_REGISTRY = {
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_14b": "Qwen/Qwen3-14B",
}

MODEL_KEYS = tuple(MODEL_REGISTRY)
INPUT_FILENAME = "token_alignment.jsonl"
OUTPUT_FILENAMES = (
    "logit_lens_by_layer.jsonl",
    "logit_lens_by_layer_compact.tsv",
    "logit_lens_summary_by_layer.tsv",
    "logit_lens_summary.json",
    "logit_lens.rejected.jsonl",
)

APPLY_FINAL_NORM = True

TOP_K = 10
COMPUTE_PROB = True
STORE_TOPK = True


COMPACT_FIELDS = (
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
)


SUMMARY_FIELDS = (
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
)


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


def write_json(path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, items):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def write_tsv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def get_device_and_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float16

    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32

    return torch.device("cpu"), torch.float32


def load_model_and_tokenizer(model_name, device_map_auto):
    device, dtype = get_device_and_dtype()

    print(f"  Device: {device}, dtype: {dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if device_map_auto:
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
    """Project embeddings directly and intermediate states through final norm and lm_head."""
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


def rank_of_token(logits, token_id):
    """Return the 1-based competition rank and token logit."""
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
    """Compute Jaccard similarity over token IDs."""
    layer_set = {int(token_id) for token_id in layer_topk_ids}
    final_set = {int(token_id) for token_id in final_topk_ids}

    union = layer_set | final_set
    if not union:
        return 1.0

    return len(layer_set & final_set) / len(union)


@torch.inference_mode()
def run_logit_lens_for_alignment(
    row,
    model,
    tokenizer,
    final_norm,
    lm_head,
):
    """Use full-span alignment and native final logits; Top-k metrics use actual membership, not rank thresholds."""

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

    last_layer_idx = len(hidden_states) - 1
    num_layers = last_layer_idx

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

        target_in_top1 = target_token_id == top_ids[0]

        target_in_top5 = target_token_id in top_ids[:5]

        target_in_top10 = target_token_id in top_ids[:10]

        faithful_final_at_1 = final_prediction_token_id == top_ids[0]

        faithful_final_at_5 = final_prediction_token_id in top_ids[:5]

        faithful_final_at_10 = final_prediction_token_id in top_ids[:10]

        topk_overlap_count_with_final = len(set(top_ids) & set(final_topk_token_ids))

        topk_jaccard_with_final = topk_jaccard(
            top_ids,
            final_topk_token_ids,
        )

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

        layer_top1_token_id = int(top_ids[0])

        layer_top1_token = topk["topk_tokens"][0]

        layer_top1_text = topk["topk_texts"][0]

        layer_top1_logit = float(topk["topk_logits"][0])

        if layer_idx == 0:
            layer_name = "embedding"

        elif layer_idx == last_layer_idx:
            layer_name = "final"

        else:
            layer_name = f"layer_{layer_idx}"

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
            "target_logit": target_logit,
            "target_prob": safe_float(target_prob),
            "target_rank": target_rank,
            "target_in_top1": target_in_top1,
            "target_in_top5": target_in_top5,
            "target_in_top10": target_in_top10,
            "layer_top1_token_id": layer_top1_token_id,
            "layer_top1_token": layer_top1_token,
            "layer_top1_text": layer_top1_text,
            "layer_top1_logit": layer_top1_logit,
            "final_prediction_token_id": final_prediction_token_id,
            "final_prediction_token": final_prediction_token,
            "final_prediction_text": final_prediction_text,
            "final_prediction_is_target": final_prediction_is_target,
            "final_prediction_logit_at_layer": final_logit_at_layer,
            "final_prediction_prob_at_layer": safe_float(
                final_prediction_prob_at_layer
            ),
            "final_prediction_rank_at_layer": final_rank,
            "faithful_final_at_1": faithful_final_at_1,
            "faithful_final_at_5": faithful_final_at_5,
            "faithful_final_at_10": faithful_final_at_10,
            "topk_overlap_count_with_final": topk_overlap_count_with_final,
            "topk_jaccard_with_final": float(topk_jaccard_with_final),
        }

        if STORE_TOPK:
            result.update(topk)

        results.append(result)

    return results


FINAL_EXACT_GOLD = "final_exact_gold"
FINAL_NON_GOLD = "final_non_gold"


def mean(values):
    """Compute the mean, ignoring None."""
    values = [value for value in values if value is not None]

    if not values:
        return None

    return sum(values) / len(values)


def bool_rate(rows, key):
    """Compute the proportion of true values, ignoring None."""
    values = [bool(row[key]) for row in rows if row.get(key) is not None]

    if not values:
        return None

    return sum(values) / len(values)


def get_final_match_status(row):
    """Label exact token matches, not semantic correctness."""
    if bool(row["final_prediction_is_target"]):
        return FINAL_EXACT_GOLD

    return FINAL_NON_GOLD


def validate_exact_gold_rows(rows):
    """Check that gold and final-token accessibility agree for exact matches."""
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
    """Aggregate layer-wise metrics by model and grouping."""

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

        for row in results:
            group_value = group_function(row)

            bucket_key = (
                grouping_name,
                group_value,
                row["model_key"],
                row["layer"],
            )

            buckets[bucket_key].append(row)

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
            validate_exact_gold_rows(rows)

            aggregate_row = {
                "grouping": grouping,
                "group": group_value,
                "model_key": model_key,
                "layer": layer,
                "n": len(rows),
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
                "final_prediction_is_target_rate": bool_rate(
                    rows,
                    "final_prediction_is_target",
                ),
                "mean_topk_overlap_count_with_final": mean(
                    row["topk_overlap_count_with_final"] for row in rows
                ),
                "mean_topk_jaccard_with_final": mean(
                    row["topk_jaccard_with_final"] for row in rows
                ),
            }

            aggregate_rows.append(aggregate_row)

    return aggregate_rows


def run_model(model_key, model_name, args):
    alignment_path = args.input_dir / model_key / INPUT_FILENAME
    output_dir = args.output_dir / model_key

    raw_jsonl = output_dir / "logit_lens_by_layer.jsonl"
    compact_tsv = output_dir / "logit_lens_by_layer_compact.tsv"
    summary_tsv = output_dir / "logit_lens_summary_by_layer.tsv"
    summary_json = output_dir / "logit_lens_summary.json"

    if not alignment_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

    alignment_rows = read_jsonl(
        alignment_path,
        max_rows=args.max_rows,
    )

    alignment_rows = [r for r in alignment_rows if r.get("model_key") == model_key]

    if not alignment_rows:
        raise ValueError(
            f"No alignment rows found for model_key={model_key} in {alignment_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{model_key}: processing {len(alignment_rows)} examples", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_name, args.device_map_auto)

    final_norm = get_final_norm(model)
    lm_head = model.get_output_embeddings()

    if lm_head is None:
        raise RuntimeError("model.get_output_embeddings() returned None.")

    if APPLY_FINAL_NORM and final_norm is None:
        raise RuntimeError("Final norm not found; refusing to change the readout.")

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

        except torch.cuda.OutOfMemoryError:
            raise
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

        if args.progress_every and idx % args.progress_every == 0:
            print(f"  {idx}/{len(alignment_rows)} examples", flush=True)

        if torch.cuda.is_available() and idx % 100 == 0:
            torch.cuda.empty_cache()

    if not all_results:
        write_jsonl(output_dir / "logit_lens.rejected.jsonl", rejected)
        raise RuntimeError(f"All examples failed for {model_key}.")

    write_jsonl(raw_jsonl, all_results)

    write_tsv(compact_tsv, all_results, COMPACT_FIELDS)

    aggregate_rows = aggregate_results(all_results)

    write_tsv(summary_tsv, aggregate_rows, SUMMARY_FIELDS)

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

    write_json(summary_json, summary)

    print(
        f"{model_key}: {len(alignment_rows) - len(rejected)} processed, "
        f"{len(rejected)} rejected, {len(all_results)} layer rows",
        flush=True,
    )
    if rejected:
        print(f"  Rejections: {rejected_jsonl}", file=sys.stderr)
    return len(alignment_rows) - len(rejected), len(rejected), len(all_results)


def build_parser():
    parser = argparse.ArgumentParser(description="Run the plain Logit Lens.")
    parser.add_argument(
        "--model",
        "--models",
        nargs="+",
        choices=["all", *MODEL_KEYS],
        default=["qwen3_0_6b"],
    )
    parser.add_argument(
        "--input-dir",
        "--alignment-dir",
        "--alignment_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output-dir", "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Smoke test: limit examples per model.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-prob", action="store_true")
    parser.add_argument("--no-store-topk", action="store_true")
    parser.add_argument("--device-map-auto", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Progress interval; 0 disables progress messages.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    if args.top_k < 10:
        parser.error("--top-k must be at least 10")
    if args.progress_every < 0:
        parser.error("--progress-every must be nonnegative")
    if "all" in args.model and len(args.model) != 1:
        parser.error("Use all alone or specify individual model keys")
    if (
        args.max_rows is not None
        or args.top_k != 10
        or args.no_prob
        or args.no_store_topk
    ) and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        parser.error(
            "Use a separate --output-dir for smoke tests or nonstandard extraction settings"
        )

    global TOP_K, COMPUTE_PROB, STORE_TOPK
    TOP_K = args.top_k
    COMPUTE_PROB = not args.no_prob
    STORE_TOPK = not args.no_store_topk
    model_keys = (
        list(MODEL_KEYS)
        if args.model == ["all"]
        else [key for key in MODEL_KEYS if key in args.model]
    )
    for model_key in model_keys:
        input_path = args.input_dir / model_key / INPUT_FILENAME
        if not input_path.is_file():
            parser.error(f"Alignment file not found: {input_path}")
        model_dir = args.output_dir / model_key
        existing = [
            model_dir / name for name in OUTPUT_FILENAMES if (model_dir / name).exists()
        ]
        if existing and not args.overwrite:
            parser.error(
                f"Output already exists: {existing[0]}. Use --overwrite or a separate --output-dir."
            )

    totals = [0, 0, 0]
    for model_key in model_keys:
        try:
            counts = run_model(model_key, MODEL_REGISTRY[model_key], args)
            totals = [a + b for a, b in zip(totals, counts)]
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
    if len(model_keys) > 1:
        print(
            f"Total: {totals[0]} processed, {totals[1]} rejected, {totals[2]} layer rows"
        )

    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
