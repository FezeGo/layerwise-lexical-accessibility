from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


DEFAULT_INPUT_PATH = Path(
    "results/analysis/trajectory/all_models_layer_trajectory_examples.tsv"
)
DEFAULT_OUTPUT_DIR = Path("results/analysis/gold_before_final/incidence_analysis")

BROADLY_ACCEPTABLE = {
    "translation_equivalent",
    "direct_hypernym",
    "direct_hyponym",
    "synonym",
    "surface_or_morphological_variant",
    "other_plausible",
}

MODEL_KEYS = ("qwen3_0_6b", "qwen3_4b", "qwen3_14b")
LANGUAGE_ORDER = ["en", "zh"]
GROUP_ORDER = ["attribute", "category", "material", "location"]

MODEL_CANDIDATES = ["model_key", "model", "model_name"]
LANGUAGE_CANDIDATES = ["language", "lang"]
GROUP_CANDIDATES = ["semantic_group", "semantic_category", "group"]
DEFAULT_EXPECTED_ANNOTATION_ROWS = 433
DEFAULT_UNIT_KEYS = ["model_key", "language", "id"]

PATTERN_CANDIDATES = [
    "strict_sustained_pattern_top10",
    "strict_pattern",
    "strict_trajectory_pattern",
]
INDICATOR_CANDIDATES = [
    "strict_gold_before_final",
]
JOIN_KEY_CANDIDATES = [
    ["annotation_id"],
    ["model_key", "language", "id"],
    ["model_key", "language", "semantic_group", "id"],
    ["model_key", "language", "example_id"],
    ["model_key", "language", "semantic_group", "example_id"],
    ["model_key", "language", "sample_id"],
    ["model_key", "language", "semantic_group", "sample_id"],
    ["model_key", "language", "row_index"],
    ["model_key", "language", "semantic_group", "row_index"],
    ["model_key", "language", "pair_id"],
    ["model_key", "language", "semantic_group", "pair_id"],
]
TRUTHY = {"1", "true", "t", "yes", "y"}

MODEL_DISPLAY = {
    "qwen3_0_6b": "Qwen3-0.6B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_14b": "Qwen3-14B",
}
LANGUAGE_DISPLAY = {"en": "EN", "zh": "ZH"}
GROUP_DISPLAY = {
    "attribute": "Attribute",
    "category": "Category",
    "material": "Material",
    "location": "Location",
}


GROUPING_SPECS: Mapping[str, Sequence[str]] = {
    "overall": [],
    "by_model": ["model_key"],
    "by_language": ["language"],
    "by_semantic_group": ["semantic_group"],
    "by_model_language": ["model_key", "language"],
    "by_model_semantic_group": ["model_key", "semantic_group"],
    "by_language_semantic_group": ["language", "semantic_group"],
    "by_model_language_semantic_group": [
        "model_key",
        "language",
        "semantic_group",
    ],
}

OUTPUT_FILENAMES = (
    *(f"gold_before_final_incidence_{name}.tsv" for name in GROUPING_SPECS),
    *(f"gold_before_final_outcome_incidence_{name}.tsv" for name in GROUPING_SPECS),
    "table5_gold_before_final_incidence.tsv",
    "table6_gold_before_final_outcome_incidence.tsv",
    "strict_gold_before_final_with_annotations.tsv",
    "gold_before_final_incidence_report.json",
    "table5_gold_before_final_incidence.tex",
    "table6_gold_before_final_outcome_incidence.tex",
)
FAILURE_FILENAME = "gold_before_final_incidence_failure.json"


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        df = pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig"
        )
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            for key in ("rows", "data", "records", "items"):
                if isinstance(raw.get(key), list):
                    df = pd.DataFrame(raw[key])
                    break
            else:
                raise ValueError(
                    f"JSON object in {path} has no row-list key "
                    "(rows/data/records/items)."
                )
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    else:
        raise ValueError(f"Unsupported extension: {suffix}")

    df.columns = [str(c).strip() for c in df.columns]
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()
    return df


def read_trajectory_files(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    schema: set[str] | None = None
    for path in paths:
        frame = read_table(path)
        current = set(frame.columns)
        if schema is None:
            schema = current
        elif current != schema:
            raise ValueError(
                f"Trajectory schemas differ. File {path} has columns not matching "
                "the first file."
            )
        frame["__source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, sep="\t", index=False, encoding="utf-8-sig")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            data, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str
        )
        handle.write("\n")


def find_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Sequence[str],
    role: str,
) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Specified {role} column '{explicit}' was not found.")
        return explicit
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not auto-detect {role} column. Tried: {', '.join(candidates)}"
    )


def rename_dimensions(
    df: pd.DataFrame,
    model_column: str,
    language_column: str,
    group_column: str,
) -> pd.DataFrame:
    result = df.copy()
    mapping = {
        model_column: "model_key",
        language_column: "language",
        group_column: "semantic_group",
    }
    for source, target in mapping.items():
        if source != target and target in result.columns:
            if not result[source].equals(result[target]):
                raise ValueError(
                    f"Both '{source}' and '{target}' exist but contain different values."
                )
            result = result.drop(columns=source)
    return result.rename(columns=mapping)


def harmonize_identifiers(
    trajectories: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectories = trajectories.copy()
    annotations = annotations.copy()

    if "id" not in trajectories.columns and "example_id" in trajectories.columns:
        trajectories["id"] = trajectories["example_id"]
    if "example_id" not in trajectories.columns and "id" in trajectories.columns:
        trajectories["example_id"] = trajectories["id"]

    if "id" not in annotations.columns and "example_id" in annotations.columns:
        annotations["id"] = annotations["example_id"]
    if "example_id" not in annotations.columns and "id" in annotations.columns:
        annotations["example_id"] = annotations["id"]

    required = {"model_key", "id", "language"}

    if "annotation_id" not in trajectories.columns and required.issubset(
        trajectories.columns
    ):
        trajectories["annotation_id"] = (
            trajectories["model_key"].astype(str).str.strip()
            + "|"
            + trajectories["id"].astype(str).str.strip()
            + "|"
            + trajectories["language"].astype(str).str.strip()
        )

    if "annotation_id" not in annotations.columns and required.issubset(
        annotations.columns
    ):
        annotations["annotation_id"] = (
            annotations["model_key"].astype(str).str.strip()
            + "|"
            + annotations["id"].astype(str).str.strip()
            + "|"
            + annotations["language"].astype(str).str.strip()
        )

    return trajectories, annotations


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    return re.sub(r"_+", "_", text)


def apply_eligibility(
    df: pd.DataFrame,
    column: str | None,
    values: Sequence[str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not column:
        return df.copy(), {
            "filter_applied": False,
            "input_rows": int(len(df)),
            "eligible_rows": int(len(df)),
        }
    if column not in df.columns:
        raise ValueError(f"Eligibility column '{column}' not found.")
    allowed = {str(v).strip().lower() for v in values} if values else TRUTHY
    mask = df[column].str.lower().isin(allowed)
    return df[mask].copy(), {
        "filter_applied": True,
        "column": column,
        "accepted_values": sorted(allowed),
        "input_rows": int(len(df)),
        "eligible_rows": int(mask.sum()),
        "excluded_rows": int((~mask).sum()),
    }


def detect_event(
    df: pd.DataFrame,
    pattern_column: str | None,
    indicator_column: str | None,
    gbf_label: str,
) -> tuple[pd.Series, dict[str, Any]]:
    if indicator_column:
        if indicator_column not in df.columns:
            raise ValueError(f"Indicator column '{indicator_column}' not found.")
        mask = df[indicator_column].str.lower().isin(TRUTHY)
        return mask, {
            "mode": "indicator",
            "column": indicator_column,
            "strict_rows": int(mask.sum()),
        }

    if pattern_column is None:
        pattern_matches = [c for c in PATTERN_CANDIDATES if c in df.columns]
        if pattern_matches:
            pattern_column = pattern_matches[0]
        else:
            indicator_matches = [c for c in INDICATOR_CANDIDATES if c in df.columns]
            if indicator_matches:
                column = indicator_matches[0]
                mask = df[column].str.lower().isin(TRUTHY)
                return mask, {
                    "mode": "auto_indicator",
                    "column": column,
                    "strict_rows": int(mask.sum()),
                }
            raise ValueError(
                "Could not detect a pattern or indicator column. Supply "
                "--pattern-column or --gbf-indicator-column."
            )

    if pattern_column not in df.columns:
        raise ValueError(f"Pattern column '{pattern_column}' not found.")

    normalized = df[pattern_column].map(normalize_label)
    target = normalize_label(gbf_label)
    mask = normalized.eq(target)
    auto_label = None
    if not mask.any():
        counts = normalized.value_counts().head(25).to_dict()
        raise ValueError(
            f"No strict rows found in '{pattern_column}' for '{target}'. "
            f"Observed labels include: {counts}"
        )
    return mask, {
        "mode": "pattern",
        "column": pattern_column,
        "requested_label": target,
        "auto_selected_label": auto_label,
        "strict_rows": int(mask.sum()),
    }


def key_series(df: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    return df[list(keys)].astype(str).agg("\x1f".join, axis=1)


def validate_keys(df: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"{name} missing key columns: {missing}")
    empty = pd.Series(False, index=df.index)
    for key in keys:
        empty |= df[key].eq("")
    if empty.any():
        raise ValueError(f"{name} has {int(empty.sum())} rows with empty key values.")


def is_unique(df: pd.DataFrame, keys: Sequence[str]) -> bool:
    if not all(k in df.columns for k in keys):
        return False
    validate_keys(df, keys, "Candidate-key data")
    return not key_series(df, keys).duplicated().any()


def auto_join_keys(strict_df: pd.DataFrame, annotations: pd.DataFrame) -> list[str]:
    diagnostics: list[dict[str, Any]] = []
    for keys in JOIN_KEY_CANDIDATES:
        if not all(k in strict_df.columns and k in annotations.columns for k in keys):
            continue
        strict_unique = is_unique(strict_df, keys)
        ann_unique = is_unique(annotations, keys)
        strict_keys = set(key_series(strict_df, keys))
        ann_keys = set(key_series(annotations, keys))
        overlap = len(strict_keys & ann_keys)
        diagnostics.append(
            {
                "keys": keys,
                "strict_unique": strict_unique,
                "annotation_unique": ann_unique,
                "overlap": overlap,
            }
        )
        if strict_unique and ann_unique and overlap > 0:
            return list(keys)
    raise ValueError(
        "Could not auto-detect unique join keys. Supply --join-keys. "
        f"Diagnostics: {diagnostics}"
    )


def validate_unit_keys(df: pd.DataFrame, keys: Sequence[str] | None) -> dict[str, Any]:
    if not keys:
        return {
            "checked": False,
            "message": "No --unit-keys supplied; one row is assumed to be one trajectory.",
        }
    validate_keys(df, keys, "Eligible trajectory data")
    duplicated = key_series(df, keys).duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, list(keys)].head(10).to_dict("records")
        raise ValueError(
            f"{int(duplicated.sum())} rows participate in duplicate trajectory "
            f"units under {list(keys)}. Examples: {examples}"
        )
    return {"checked": True, "unit_keys": list(keys), "duplicate_rows": 0}


def validate_annotations(df: pd.DataFrame, expected_rows: int) -> dict[str, Any]:
    required = {
        "annotation_id",
        "model_key",
        "language",
        "semantic_group",
        "final_completion_form",
        "final_semantic_relation",
    }
    errors: list[str] = []
    missing = sorted(required - set(df.columns))
    if missing:
        errors.append("Missing annotation columns: " + ", ".join(missing))
    if expected_rows > 0 and len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} annotations but found {len(df)}.")
    if "annotation_id" in df.columns:
        if df["annotation_id"].eq("").any():
            errors.append("Some annotation_id values are empty.")
        duplicate_rows = int(df["annotation_id"].duplicated(keep=False).sum())
        if duplicate_rows:
            errors.append(f"{duplicate_rows} rows have duplicate annotation_id values.")
    if "final_semantic_relation" in df.columns:
        allowed = BROADLY_ACCEPTABLE | {"incompatible_or_unrelated", "not_applicable"}
        unexpected = sorted(set(df["final_semantic_relation"]) - allowed)
        if unexpected:
            errors.append("Unexpected semantic labels: " + ", ".join(unexpected))
    return {
        "valid": not errors,
        "errors": errors,
        "rows": int(len(df)),
        "expected_rows": expected_rows if expected_rows > 0 else None,
    }


def outcome_class(relation: str) -> str:
    if relation == "not_applicable":
        return "not_evaluable"
    if relation == "incompatible_or_unrelated":
        return "incompatible_or_unrelated"
    if relation in BROADLY_ACCEPTABLE:
        return "broadly_acceptable"
    raise ValueError(f"Unexpected semantic relation: {relation}")


def merge_annotations(
    strict_df: pd.DataFrame,
    annotations: pd.DataFrame,
    join_keys: Sequence[str] | None,
    allow_partial: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = list(join_keys) if join_keys else auto_join_keys(strict_df, annotations)
    validate_keys(strict_df, keys, "Strict trajectory data")
    validate_keys(annotations, keys, "Annotation data")

    strict_ids = key_series(strict_df, keys)
    ann_ids = key_series(annotations, keys)
    if strict_ids.duplicated().any():
        raise ValueError(f"Strict data are not unique under join keys {keys}.")
    if ann_ids.duplicated().any():
        raise ValueError(f"Annotations are not unique under join keys {keys}.")

    strict_set, ann_set = set(strict_ids), set(ann_ids)
    matched = strict_set & ann_set
    unannotated = strict_set - ann_set
    unmatched_ann = ann_set - strict_set

    if not allow_partial and (unannotated or unmatched_ann):
        raise ValueError(
            "Annotation coverage is incomplete: "
            f"strict rows without annotation={len(unannotated)}, "
            f"annotations without strict row={len(unmatched_ann)}. "
            "Check event detection/join keys or deliberately use "
            "--allow-partial-coverage."
        )

    for column in ("model_key", "language", "semantic_group"):
        if column in keys:
            continue
        left = strict_df.assign(__join_key=strict_ids).set_index("__join_key")[column]
        right = annotations.assign(__join_key=ann_ids).set_index("__join_key")[column]
        common = sorted(matched)
        if not left.loc[common].equals(right.loc[common]):
            raise ValueError(f"Matched annotation dimension mismatch: {column}")

    annotation_columns = list(
        dict.fromkeys(
            list(keys)
            + ["annotation_id", "final_completion_form", "final_semantic_relation"]
        )
    )
    ann = annotations[annotation_columns].copy()
    ann["annotation_outcome"] = ann["final_semantic_relation"].map(outcome_class)

    merged = strict_df.merge(
        ann,
        on=list(keys),
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_annotation"),
    )
    merged["annotation_matched"] = merged["_merge"].eq("both")
    merged = merged.drop(columns="_merge")

    coverage = 100.0 * len(matched) / len(strict_set) if strict_set else 0.0
    return merged, {
        "join_keys": list(keys),
        "strict_rows": int(len(strict_df)),
        "annotation_rows": int(len(annotations)),
        "matched_rows": int(len(matched)),
        "strict_rows_without_annotation": int(len(unannotated)),
        "annotations_without_strict_row": int(len(unmatched_ann)),
        "coverage_percent": round(coverage, 4),
        "complete_coverage": not unannotated and not unmatched_ann,
    }


def safe_pct(n: int, d: int) -> float | None:
    return None if d == 0 else round(100.0 * n / d, 4)


def ordered_values(df: pd.DataFrame, column: str) -> list[str]:
    preferred = {
        "model_key": MODEL_KEYS,
        "language": LANGUAGE_ORDER,
        "semantic_group": GROUP_ORDER,
    }.get(column, [])
    observed = {v for v in df[column].astype(str).tolist() if v != ""}
    result = [v for v in preferred if v in observed]
    result.extend(sorted(observed - set(result)))
    return result


def group_combinations(
    df: pd.DataFrame, columns: Sequence[str]
) -> list[tuple[str, ...]]:
    if not columns:
        return [tuple()]
    combos = set(df[list(columns)].itertuples(index=False, name=None))
    rank = {c: {v: i for i, v in enumerate(ordered_values(df, c))} for c in columns}

    def sort_key(combo: tuple[str, ...]) -> tuple[Any, ...]:
        result: list[Any] = []
        for column, value in zip(columns, combo):
            result.extend([rank[column].get(value, 10**9), value])
        return tuple(result)

    return sorted(combos, key=sort_key)


def select_group(
    df: pd.DataFrame, columns: Sequence[str], combo: Sequence[str]
) -> pd.DataFrame:
    if not columns:
        return df
    mask = pd.Series(True, index=df.index)
    for column, value in zip(columns, combo):
        mask &= df[column].eq(value)
    return df[mask]


def aggregate_incidence(
    denominator: pd.DataFrame,
    strict: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for combo in group_combinations(denominator, columns):
        all_cell = select_group(denominator, columns, combo)
        strict_cell = select_group(strict, columns, combo)
        total_n = int(len(all_cell))
        gbf_n = int(len(strict_cell))
        annotated_n = int(strict_cell["annotation_matched"].sum())
        row = {c: v for c, v in zip(columns, combo)}
        row.update(
            {
                "total_eligible_trajectories": total_n,
                "gold_before_final_n": gbf_n,
                "gold_before_final_percent": safe_pct(gbf_n, total_n),
                "annotated_gold_before_final_n": annotated_n,
                "annotation_coverage_percent": safe_pct(annotated_n, gbf_n),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_outcomes(
    denominator: pd.DataFrame,
    strict: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outcomes = ["not_evaluable", "broadly_acceptable", "incompatible_or_unrelated"]
    for combo in group_combinations(denominator, columns):
        all_cell = select_group(denominator, columns, combo)
        strict_cell = select_group(strict, columns, combo)
        annotated = strict_cell[strict_cell["annotation_matched"]]
        total_n, gbf_n, annotated_n = len(all_cell), len(strict_cell), len(annotated)
        row = {c: v for c, v in zip(columns, combo)}
        row.update(
            {
                "total_eligible_trajectories": int(total_n),
                "gold_before_final_n": int(gbf_n),
                "gold_before_final_percent": safe_pct(int(gbf_n), int(total_n)),
                "annotated_gold_before_final_n": int(annotated_n),
                "annotation_coverage_percent": safe_pct(int(annotated_n), int(gbf_n)),
            }
        )
        for outcome in outcomes:
            n = int(annotated["annotation_outcome"].eq(outcome).sum())
            row[f"{outcome}_n"] = n
            row[f"{outcome}_percent_of_all_trajectories"] = safe_pct(n, int(total_n))
            row[f"{outcome}_percent_within_annotated_gold_before_final"] = safe_pct(
                n, int(annotated_n)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(value))


def fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.1f}\\%"


def render_table5(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Incidence of strict \texttt{gold\_before\_final} trajectories by model, language, and semantic group.}",
        r"    \label{tab:gold_before_final_incidence}",
        r"    \begin{tabular}{lllrrrr}",
        r"        \toprule",
        r"        Model & Lang. & Group & Eligible $n$ & GBF $n$ & GBF rate & Coverage \\",
        r"        \midrule",
    ]
    previous_model = None
    for _, row in df.iterrows():
        model = row["model_key"]
        if previous_model is not None and model != previous_model:
            lines.append(r"        \addlinespace")
        lines.append(
            "        "
            + latex_escape(MODEL_DISPLAY.get(model, model))
            + " & "
            + latex_escape(
                LANGUAGE_DISPLAY.get(row["language"], row["language"].upper())
            )
            + " & "
            + latex_escape(
                GROUP_DISPLAY.get(row["semantic_group"], row["semantic_group"].title())
            )
            + f" & {int(row['total_eligible_trajectories'])}"
            + f" & {int(row['gold_before_final_n'])}"
            + " & "
            + fmt_pct(row["gold_before_final_percent"])
            + " & "
            + fmt_pct(row["annotation_coverage_percent"])
            + r" \\"
        )
        previous_model = model
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \vspace{0.5em}",
            r"    \parbox{0.98\textwidth}{\footnotesize GBF rate uses all eligible trajectories in the same cell as the denominator. Coverage is the percentage of strict GBF rows matched to the finalized annotation file.}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_table6(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Absolute incidence of annotated strict \texttt{gold\_before\_final} outcomes.}",
        r"    \label{tab:gold_before_final_outcome_incidence}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{lllrrrrrrrr}",
        r"        \toprule",
        r"        Model & Lang. & Group & Eligible $n$ & GBF $n$ & \multicolumn{2}{c}{Not evaluable} & \multicolumn{2}{c}{Broadly acceptable} & \multicolumn{2}{c}{Incompatible} \\",
        r"        \cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r"        & & & & & $n$ & \% all & $n$ & \% all & $n$ & \% all \\",
        r"        \midrule",
    ]
    previous_model = None
    for _, row in df.iterrows():
        model = row["model_key"]
        if previous_model is not None and model != previous_model:
            lines.append(r"        \addlinespace")
        lines.append(
            "        "
            + latex_escape(MODEL_DISPLAY.get(model, model))
            + " & "
            + latex_escape(
                LANGUAGE_DISPLAY.get(row["language"], row["language"].upper())
            )
            + " & "
            + latex_escape(
                GROUP_DISPLAY.get(row["semantic_group"], row["semantic_group"].title())
            )
            + f" & {int(row['total_eligible_trajectories'])}"
            + f" & {int(row['gold_before_final_n'])}"
            + f" & {int(row['not_evaluable_n'])}"
            + " & "
            + fmt_pct(row["not_evaluable_percent_of_all_trajectories"])
            + f" & {int(row['broadly_acceptable_n'])}"
            + " & "
            + fmt_pct(row["broadly_acceptable_percent_of_all_trajectories"])
            + f" & {int(row['incompatible_or_unrelated_n'])}"
            + " & "
            + fmt_pct(row["incompatible_or_unrelated_percent_of_all_trajectories"])
            + r" \\"
        )
        previous_model = model
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"    \vspace{0.5em}",
            r"    \parbox{0.98\textwidth}{\footnotesize Percentages use all eligible trajectories in the same cell. Broadly acceptable combines translation equivalents, direct hypernyms, direct hyponyms, synonyms, surface or morphological variants, and other plausible alternatives.}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute strict gold-before-final incidence with full denominators."
    )
    parser.add_argument(
        "trajectory_files",
        nargs="*",
        type=Path,
        help="One or more complete trajectory files; files are concatenated.",
    )
    parser.add_argument("--input", nargs="+", type=Path)
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Finalized gold-before-final annotation file.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    event = parser.add_mutually_exclusive_group()
    event.add_argument("--pattern-column", default=None)
    event.add_argument("--gbf-indicator-column", default=None)
    parser.add_argument("--gbf-label", default="gold_before_final")

    parser.add_argument("--model-column", default=None)
    parser.add_argument("--language-column", default=None)
    parser.add_argument("--semantic-group-column", default=None)

    parser.add_argument("--eligible-column", default=None)
    parser.add_argument("--eligible-values", nargs="+", default=None)
    parser.add_argument("--join-keys", nargs="+", default=None)
    parser.add_argument("--unit-keys", nargs="+", default=None)
    parser.add_argument(
        "--expected-annotation-rows", type=int, default=DEFAULT_EXPECTED_ANNOTATION_ROWS
    )
    parser.add_argument("--allow-partial-coverage", action="store_true")
    parser.add_argument("--no-latex", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input and args.trajectory_files:
        parser.error("Use either --input or positional trajectory files, not both.")
    args.trajectory_files = args.input or args.trajectory_files or [DEFAULT_INPUT_PATH]
    if args.expected_annotation_rows < 0:
        parser.error("--expected-annotation-rows must be non-negative.")
    if args.eligible_values and not args.eligible_column:
        parser.error("--eligible-values requires --eligible-column.")
    output_dir = args.output_dir
    nonstandard = (
        args.allow_partial_coverage
        or args.expected_annotation_rows != DEFAULT_EXPECTED_ANNOTATION_ROWS
        or args.eligible_column is not None
        or args.gbf_label != "gold_before_final"
        or args.pattern_column is not None
        or args.gbf_indicator_column is not None
        or args.unit_keys is not None
        or args.join_keys is not None
        or args.model_column is not None
        or args.language_column is not None
        or args.semantic_group_column is not None
    )
    if nonstandard and output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        parser.error("Nonstandard analysis requires a separate --output-dir.")
    existing = [
        output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        parser.error(
            "Existing outputs; use --overwrite or a separate --output-dir: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        trajectories_raw = read_trajectory_files(args.trajectory_files)
        annotations = read_table(args.annotations)

        ann_validation = validate_annotations(
            annotations, args.expected_annotation_rows
        )
        if not ann_validation["valid"]:
            raise ValueError("; ".join(ann_validation["errors"]))

        model_column = find_column(
            trajectories_raw, args.model_column, MODEL_CANDIDATES, "model"
        )
        language_column = find_column(
            trajectories_raw, args.language_column, LANGUAGE_CANDIDATES, "language"
        )
        group_column = find_column(
            trajectories_raw,
            args.semantic_group_column,
            GROUP_CANDIDATES,
            "semantic-group",
        )
        trajectories = rename_dimensions(
            trajectories_raw, model_column, language_column, group_column
        )

        trajectories, annotations = harmonize_identifiers(
            trajectories=trajectories,
            annotations=annotations,
        )

        denominator, eligibility_report = apply_eligibility(
            trajectories, args.eligible_column, args.eligible_values
        )
        if denominator.empty:
            raise ValueError("No eligible trajectories remain after filtering.")
        unit_report = validate_unit_keys(
            denominator, args.unit_keys or DEFAULT_UNIT_KEYS
        )

        event_mask, event_report = detect_event(
            denominator,
            args.pattern_column,
            args.gbf_indicator_column,
            args.gbf_label,
        )
        strict = denominator[event_mask].copy()
        strict["is_gold_before_final"] = True

        strict_annotated, coverage_report = merge_annotations(
            strict,
            annotations,
            args.join_keys,
            args.allow_partial_coverage,
        )

        outputs: dict[str, str] = {}
        incidence_tables: dict[str, pd.DataFrame] = {}
        outcome_tables: dict[str, pd.DataFrame] = {}

        for name, columns in GROUPING_SPECS.items():
            incidence = aggregate_incidence(denominator, strict_annotated, columns)
            outcomes = aggregate_outcomes(denominator, strict_annotated, columns)
            incidence_path = output_dir / f"gold_before_final_incidence_{name}.tsv"
            outcome_path = (
                output_dir / f"gold_before_final_outcome_incidence_{name}.tsv"
            )
            write_tsv(incidence_path, incidence)
            write_tsv(outcome_path, outcomes)
            outputs[f"incidence_{name}"] = str(incidence_path)
            outputs[f"outcome_incidence_{name}"] = str(outcome_path)
            incidence_tables[name] = incidence
            outcome_tables[name] = outcomes

        table5 = incidence_tables["by_model_language_semantic_group"]
        table6 = outcome_tables["by_model_language_semantic_group"]
        table5_tsv = output_dir / "table5_gold_before_final_incidence.tsv"
        table6_tsv = output_dir / "table6_gold_before_final_outcome_incidence.tsv"
        write_tsv(table5_tsv, table5)
        write_tsv(table6_tsv, table6)
        outputs["table5_tsv"] = str(table5_tsv)
        outputs["table6_tsv"] = str(table6_tsv)

        matched_path = output_dir / "strict_gold_before_final_with_annotations.tsv"
        write_tsv(matched_path, strict_annotated)
        outputs["matched_strict_rows"] = str(matched_path)

        if not args.no_latex:
            table5_tex = output_dir / "table5_gold_before_final_incidence.tex"
            table6_tex = output_dir / "table6_gold_before_final_outcome_incidence.tex"
            table5_tex.write_text(render_table5(table5), encoding="utf-8")
            table6_tex.write_text(render_table6(table6), encoding="utf-8")
            outputs["table5_tex"] = str(table5_tex)
            outputs["table6_tex"] = str(table6_tex)

        overall_incidence = incidence_tables["overall"].iloc[0]
        overall_outcomes = outcome_tables["overall"].iloc[0]
        report = {
            "valid": True,
            "trajectory_files": [str(p) for p in args.trajectory_files],
            "annotation_file": str(args.annotations),
            "output_directory": str(output_dir),
            "dimension_columns": {
                "model": model_column,
                "language": language_column,
                "semantic_group": group_column,
            },
            "eligibility": eligibility_report,
            "unit_validation": unit_report,
            "event_detection": event_report,
            "annotation_validation": ann_validation,
            "annotation_coverage": coverage_report,
            "overall": {
                "total_eligible_trajectories": int(
                    overall_incidence["total_eligible_trajectories"]
                ),
                "gold_before_final_n": int(overall_incidence["gold_before_final_n"]),
                "gold_before_final_percent": overall_incidence[
                    "gold_before_final_percent"
                ],
                "not_evaluable_n": int(overall_outcomes["not_evaluable_n"]),
                "broadly_acceptable_n": int(overall_outcomes["broadly_acceptable_n"]),
                "incompatible_or_unrelated_n": int(
                    overall_outcomes["incompatible_or_unrelated_n"]
                ),
            },
            "interpretation_scope": (
                "GBF incidence uses all eligible trajectories as denominator. "
                "Outcome incidence divides annotated GBF outcomes by the same "
                "full denominator."
            ),
            "partial_coverage_allowed": bool(args.allow_partial_coverage),
            "latex_requirements": ["booktabs", "graphicx"],
            "outputs": outputs,
        }
        report_path = output_dir / "gold_before_final_incidence_report.json"
        write_json(report_path, report)
        if args.overwrite:
            (output_dir / FAILURE_FILENAME).unlink(missing_ok=True)
        overall = report["overall"]
        print(
            f"Gold-before-final: {overall['gold_before_final_n']}/"
            f"{overall['total_eligible_trajectories']} trajectories "
            f"({overall['gold_before_final_percent']:.4f}%)"
        )
        print(
            f"Annotation coverage: {coverage_report['matched_rows']}/"
            f"{coverage_report['strict_rows']} "
            f"({coverage_report['coverage_percent']:.4f}%)"
        )
        if not coverage_report["complete_coverage"]:
            print("WARNING: incomplete annotation coverage.", file=sys.stderr)
        print(f"Output: {output_dir}")
        return 0

    except Exception as exc:
        failure = {
            "valid": False,
            "error": str(exc),
            "trajectory_files": [str(p) for p in args.trajectory_files],
            "annotation_file": str(args.annotations),
        }
        failure_path = output_dir / FAILURE_FILENAME
        if failure_path.exists() and not args.overwrite:
            print(f"ERROR: {exc}", file=sys.stderr)
            print(f"Existing failure report preserved: {failure_path}", file=sys.stderr)
            return 1
        write_json(failure_path, failure)
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Failure report: {failure_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
