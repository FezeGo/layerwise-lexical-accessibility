from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results/analysis/gold_before_final/annotation_summary")
DEFAULT_EXPECTED_ROWS = 433

COMPLETION_ORDER = [
    "standalone_content_candidate",
    "modifier_before_gold",
    "incomplete_or_structural_continuation",
    "gold_surface_fragment",
    "blank_or_punctuation",
    "unclear_form",
]

SEMANTIC_ORDER = [
    "translation_equivalent",
    "direct_hypernym",
    "direct_hyponym",
    "synonym",
    "surface_or_morphological_variant",
    "other_plausible",
    "incompatible_or_unrelated",
    "not_applicable",
]

EVALUABLE_SEMANTIC_ORDER = [
    label for label in SEMANTIC_ORDER if label != "not_applicable"
]

SURFACE_VARIANT_ORDER = [
    "natural_shorter_variant",
    "incomplete_gold_fragment",
    "longer_surface_variant",
    "orthographic_or_morphological_variant",
]

BROAD_SEMANTIC_ALL_ORDER = [
    "broadly_acceptable",
    "incompatible_or_unrelated",
    "not_evaluable",
]

BROAD_SEMANTIC_EVALUABLE_ORDER = [
    "broadly_acceptable",
    "incompatible_or_unrelated",
]

BROADLY_ACCEPTABLE_RELATIONS = {
    "translation_equivalent",
    "direct_hypernym",
    "direct_hyponym",
    "synonym",
    "surface_or_morphological_variant",
    "other_plausible",
}

REQUIRED_COLUMNS = {
    "annotation_id",
    "model_key",
    "language",
    "semantic_group",
    "final_completion_form",
    "final_semantic_relation",
    "final_surface_variant_type",
}

DIMENSIONS = {
    "language": "language",
    "model": "model_key",
    "semantic_group": "semantic_group",
}


MODEL_KEYS = (
    "qwen3_0_6b",
    "qwen3_4b",
    "qwen3_14b",
)

LANGUAGE_ORDER = [
    "en",
    "zh",
]

SEMANTIC_GROUP_ORDER = [
    "attribute",
    "category",
    "material",
    "location",
]

VALUE_ORDERS = {
    "model_key": MODEL_KEYS,
    "language": LANGUAGE_ORDER,
    "semantic_group": SEMANTIC_GROUP_ORDER,
}


CROSS_GROUPS = (
    "by_model_language",
    "by_model_semantic_group",
    "by_language_semantic_group",
    "by_model_language_semantic_group",
)
DISTRIBUTION_STEMS = (
    "annotation_completion",
    "annotation_semantic_all",
    "annotation_semantic_evaluable",
    "annotation_broad_semantic_all",
    "annotation_broad_semantic_evaluable",
    "annotation_surface_variant",
)
OUTPUT_FILENAMES = (
    "annotation_analysis_validation.json",
    "annotation_analysis_report.json",
    "annotation_overview.tsv",
    *(
        f"annotation_sample_composition_{name}.tsv"
        for name in ("by_language", "by_model", "by_semantic_group", *CROSS_GROUPS)
    ),
    *(
        f"{stem}_{scope}.tsv"
        for stem in DISTRIBUTION_STEMS
        for scope in ("overall", *(f"by_{name}" for name in DIMENSIONS))
    ),
    *(
        f"annotation_{family}_{scope}.tsv"
        for family in ("completion", "semantic_evaluable")
        for scope in CROSS_GROUPS
    ),
)


def read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        sep = "\t"
    elif suffix == ".csv":
        sep = ","
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Use TSV, TXT, or CSV."
        )

    df = pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    df.columns = [str(column).strip() for column in df.columns]
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()

    return df


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, sep="\t", index=False, encoding="utf-8-sig")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def validate_input(
    df: pd.DataFrame,
    expected_rows: int | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(missing_columns))

    if expected_rows is not None and len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows but found {len(df)}.")

    if "annotation_id" in df.columns:
        missing_ids = int((df["annotation_id"] == "").sum())
        duplicate_ids = int(df["annotation_id"].duplicated(keep=False).sum())

        if missing_ids:
            errors.append(f"{missing_ids} rows have a missing annotation_id.")
        if duplicate_ids:
            errors.append(
                f"{duplicate_ids} rows participate in duplicate annotation IDs."
            )
    else:
        missing_ids = None
        duplicate_ids = None

    if "final_completion_form" in df.columns:
        observed = set(df["final_completion_form"]) - {""}
        unexpected = sorted(observed - set(COMPLETION_ORDER))
        missing = int((df["final_completion_form"] == "").sum())
        if missing:
            errors.append(f"{missing} rows have a missing final_completion_form.")
        if unexpected:
            errors.append("Unexpected completion labels: " + ", ".join(unexpected))

    if "final_semantic_relation" in df.columns:
        observed = set(df["final_semantic_relation"]) - {""}
        unexpected = sorted(observed - set(SEMANTIC_ORDER))
        missing = int((df["final_semantic_relation"] == "").sum())
        if missing:
            errors.append(f"{missing} rows have a missing final_semantic_relation.")
        if unexpected:
            errors.append("Unexpected semantic labels: " + ", ".join(unexpected))

    if "final_surface_variant_type" in df.columns:
        allowed_surface = set(SURFACE_VARIANT_ORDER) | {"not_applicable"}
        observed = set(df["final_surface_variant_type"]) - {""}
        unexpected = sorted(observed - allowed_surface)
        missing = int((df["final_surface_variant_type"] == "").sum())
        if missing:
            errors.append(f"{missing} rows have a missing final_surface_variant_type.")
        if unexpected:
            errors.append("Unexpected surface-variant labels: " + ", ".join(unexpected))

    for column, allowed in VALUE_ORDERS.items():
        if column in df.columns:
            unexpected = sorted(set(df[column]) - set(allowed))
            if unexpected:
                errors.append(f"Unexpected values in {column}: {unexpected}")

    unresolved_count = 0

    # Cross-field checks.
    if REQUIRED_COLUMNS.issubset(df.columns):
        non_evaluable_forms = {
            "modifier_before_gold",
            "incomplete_or_structural_continuation",
            "blank_or_punctuation",
            "unclear_form",
        }
        mismatch_1 = df[
            df["final_completion_form"].isin(non_evaluable_forms)
            & (df["final_semantic_relation"] != "not_applicable")
        ]
        if len(mismatch_1):
            errors.append(
                f"{len(mismatch_1)} non-evaluable completion rows have a "
                "semantic relation other than not_applicable."
            )

        missing_relation = df[
            (df["final_completion_form"] == "standalone_content_candidate")
            & (df["final_semantic_relation"] == "not_applicable")
        ]
        if len(missing_relation):
            errors.append(
                f"{len(missing_relation)} standalone content candidates lack an evaluable relation."
            )

        mismatch_2 = df[
            (df["final_completion_form"] == "gold_surface_fragment")
            & (df["final_semantic_relation"] != "surface_or_morphological_variant")
        ]
        if len(mismatch_2):
            errors.append(
                f"{len(mismatch_2)} gold_surface_fragment rows lack the "
                "surface_or_morphological_variant relation."
            )

        mismatch_3 = df[
            (df["final_semantic_relation"] == "surface_or_morphological_variant")
            & (df["final_surface_variant_type"].isin(["", "not_applicable"]))
        ]
        if len(mismatch_3):
            errors.append(
                f"{len(mismatch_3)} surface-variant rows lack a specific "
                "surface subtype."
            )

        mismatch_4 = df[
            (df["final_semantic_relation"] != "surface_or_morphological_variant")
            & (df["final_surface_variant_type"] != "not_applicable")
        ]
        if len(mismatch_4):
            errors.append(
                f"{len(mismatch_4)} non-surface rows have a specific surface subtype."
            )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "total_rows": int(len(df)),
        "expected_rows": expected_rows,
        "missing_annotation_id_count": missing_ids,
        "duplicate_annotation_id_row_count": duplicate_ids,
        "unresolved_row_count": unresolved_count,
    }


def add_broad_semantic_class(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    def classify(relation: str) -> str:
        if relation in BROADLY_ACCEPTABLE_RELATIONS:
            return "broadly_acceptable"
        if relation == "incompatible_or_unrelated":
            return "incompatible_or_unrelated"
        if relation == "not_applicable":
            return "not_evaluable"
        raise ValueError(f"Cannot broadly classify semantic relation: {relation}")

    result["broad_semantic_class"] = result["final_semantic_relation"].map(classify)
    return result


def tidy_distribution(
    df: pd.DataFrame,
    label_column: str,
    label_order: Sequence[str],
    group_column: str | None = None,
    group_name: str = "overall",
) -> pd.DataFrame:
    """Include zero-count labels in each distribution."""
    if group_column is None:
        denominator = int(len(df))
        counts = df[label_column].value_counts().to_dict()
        rows = [
            {
                group_name: "all",
                "label": label,
                "n": int(counts.get(label, 0)),
                "denominator": denominator,
                "percent": (
                    round(100.0 * counts.get(label, 0) / denominator, 4)
                    if denominator
                    else 0.0
                ),
            }
            for label in label_order
        ]
        return pd.DataFrame(rows)

    group_values = sorted(
        value for value in df[group_column].unique().tolist() if value != ""
    )
    rows: list[dict[str, Any]] = []

    for group_value in group_values:
        subset = df[df[group_column] == group_value]
        denominator = int(len(subset))
        counts = subset[label_column].value_counts().to_dict()

        for label in label_order:
            n = int(counts.get(label, 0))
            rows.append(
                {
                    group_name: group_value,
                    "label": label,
                    "n": n,
                    "denominator": denominator,
                    "percent": (
                        round(100.0 * n / denominator, 4) if denominator else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def ordered_values(df: pd.DataFrame, column: str) -> list[str]:
    observed = {value for value in df[column].astype(str).tolist() if value != ""}
    preferred = VALUE_ORDERS.get(column, [])
    ordered = [value for value in preferred if value in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def observed_group_combinations(
    df: pd.DataFrame,
    group_columns: Sequence[str],
) -> list[tuple[str, ...]]:
    combinations = {
        tuple(str(value) for value in row)
        for row in df[list(group_columns)].itertuples(index=False, name=None)
        if all(str(value) != "" for value in row)
    }

    rank_maps = {
        column: {value: index for index, value in enumerate(ordered_values(df, column))}
        for column in group_columns
    }

    def sort_key(combo: tuple[str, ...]) -> tuple[Any, ...]:
        key: list[Any] = []
        for column, value in zip(group_columns, combo):
            key.extend([rank_maps[column].get(value, 10**9), value])
        return tuple(key)

    return sorted(combinations, key=sort_key)


def tidy_distribution_multi(
    df: pd.DataFrame,
    label_column: str,
    label_order: Sequence[str],
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for combination in observed_group_combinations(df, group_columns):
        mask = pd.Series(True, index=df.index)
        group_values: dict[str, str] = {}

        for column, value in zip(group_columns, combination):
            mask &= df[column] == value
            group_values[column] = value

        subset = df[mask]
        denominator = int(len(subset))
        counts = subset[label_column].value_counts().to_dict()

        for label in label_order:
            n = int(counts.get(label, 0))
            row: dict[str, Any] = dict(group_values)
            row.update(
                {
                    "label": label,
                    "n": n,
                    "denominator": denominator,
                    "percent": (
                        round(100.0 * n / denominator, 4) if denominator else 0.0
                    ),
                }
            )
            rows.append(row)

    columns = list(group_columns) + [
        "label",
        "n",
        "denominator",
        "percent",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_completion_cross_tables(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    specifications = {
        "by_model_language": ["model_key", "language"],
        "by_model_semantic_group": ["model_key", "semantic_group"],
        "by_language_semantic_group": ["language", "semantic_group"],
        "by_model_language_semantic_group": [
            "model_key",
            "language",
            "semantic_group",
        ],
    }

    output_paths: dict[str, str] = {}
    for name, group_columns in specifications.items():
        table = tidy_distribution_multi(
            df=df,
            label_column="final_completion_form",
            label_order=COMPLETION_ORDER,
            group_columns=group_columns,
        )
        path = output_dir / f"annotation_completion_{name}.tsv"
        write_tsv(path, table)
        output_paths[name] = str(path)

    return output_paths


def write_semantic_evaluable_cross_tables(
    evaluable_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    """Use relation-evaluable rows in each cell as the denominator."""
    specifications = {
        "by_language_semantic_group": [
            "language",
            "semantic_group",
        ],
        "by_model_language": [
            "model_key",
            "language",
        ],
        "by_model_semantic_group": [
            "model_key",
            "semantic_group",
        ],
        "by_model_language_semantic_group": [
            "model_key",
            "language",
            "semantic_group",
        ],
    }

    output_paths: dict[str, str] = {}
    for name, group_columns in specifications.items():
        table = tidy_distribution_multi(
            df=evaluable_df,
            label_column="final_semantic_relation",
            label_order=EVALUABLE_SEMANTIC_ORDER,
            group_columns=group_columns,
        )
        path = output_dir / (f"annotation_semantic_evaluable_{name}.tsv")
        write_tsv(path, table)
        output_paths[name] = str(path)

    return output_paths


def write_distribution_family(
    df: pd.DataFrame,
    output_dir: Path,
    filename_stem: str,
    label_column: str,
    label_order: Sequence[str],
) -> dict[str, str]:
    output_paths: dict[str, str] = {}

    overall = tidy_distribution(
        df=df,
        label_column=label_column,
        label_order=label_order,
        group_column=None,
        group_name="scope",
    )
    overall_path = output_dir / f"{filename_stem}_overall.tsv"
    write_tsv(overall_path, overall)
    output_paths["overall"] = str(overall_path)

    for dimension_name, column_name in DIMENSIONS.items():
        table = tidy_distribution(
            df=df,
            label_column=label_column,
            label_order=label_order,
            group_column=column_name,
            group_name=dimension_name,
        )
        path = output_dir / (f"{filename_stem}_by_{dimension_name}.tsv")
        write_tsv(path, table)
        output_paths[f"by_{dimension_name}"] = str(path)

    return output_paths


def make_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    semantic_evaluable = df["final_semantic_relation"] != "not_applicable"
    standalone = df["final_completion_form"] == "standalone_content_candidate"
    surface_fragments = df["final_completion_form"] == "gold_surface_fragment"
    broadly_acceptable = df["broad_semantic_class"] == "broadly_acceptable"
    incompatible = df["broad_semantic_class"] == "incompatible_or_unrelated"

    rows = [
        ("total_annotated_rows", int(len(df))),
        ("semantic_evaluable_rows", int(semantic_evaluable.sum())),
        ("semantic_not_evaluable_rows", int((~semantic_evaluable).sum())),
        ("standalone_content_candidate_rows", int(standalone.sum())),
        ("gold_surface_fragment_rows", int(surface_fragments.sum())),
        (
            "broadly_acceptable_evaluable_rows",
            int((semantic_evaluable & broadly_acceptable).sum()),
        ),
        (
            "incompatible_evaluable_rows",
            int((semantic_evaluable & incompatible).sum()),
        ),
    ]

    evaluable_n = int(semantic_evaluable.sum())
    if evaluable_n:
        rows.extend(
            [
                (
                    "broadly_acceptable_percent_among_evaluable",
                    round(
                        100.0
                        * int((semantic_evaluable & broadly_acceptable).sum())
                        / evaluable_n,
                        4,
                    ),
                ),
                (
                    "incompatible_percent_among_evaluable",
                    round(
                        100.0
                        * int((semantic_evaluable & incompatible).sum())
                        / evaluable_n,
                        4,
                    ),
                ),
            ]
        )

    return pd.DataFrame(rows, columns=["metric", "value"])


def make_sample_composition(
    df: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    grouped = df.groupby(list(group_columns), dropna=False).size().reset_index(name="n")
    grouped["total"] = int(len(df))
    grouped["percent_of_annotated_subset"] = (100.0 * grouped["n"] / len(df)).round(4)
    return grouped.sort_values(list(group_columns)).reset_index(drop=True)


def dimension_short_name(column: str) -> str:
    return {
        "model_key": "model",
        "language": "language",
        "semantic_group": "semantic_group",
    }.get(column, column)


def make_two_way_sample_composition(
    df: pd.DataFrame,
    row_column: str,
    column_column: str,
) -> pd.DataFrame:
    total = int(len(df))
    row_short = dimension_short_name(row_column)
    column_short = dimension_short_name(column_column)

    counts = df.groupby([row_column, column_column], dropna=False).size().to_dict()
    row_totals = df.groupby(row_column, dropna=False).size().to_dict()
    column_totals = df.groupby(column_column, dropna=False).size().to_dict()

    rows: list[dict[str, Any]] = []
    for row_value, column_value in observed_group_combinations(
        df, [row_column, column_column]
    ):
        n = int(counts.get((row_value, column_value), 0))
        row_total = int(row_totals.get(row_value, 0))
        column_total = int(column_totals.get(column_value, 0))

        rows.append(
            {
                row_column: row_value,
                column_column: column_value,
                "n": n,
                f"{row_short}_total": row_total,
                f"percent_within_{row_short}": (
                    round(100.0 * n / row_total, 4) if row_total else 0.0
                ),
                f"{column_short}_total": column_total,
                f"percent_within_{column_short}": (
                    round(100.0 * n / column_total, 4) if column_total else 0.0
                ),
                "total": total,
                "percent_of_annotated_subset": (
                    round(100.0 * n / total, 4) if total else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def make_three_way_sample_composition(
    df: pd.DataFrame,
) -> pd.DataFrame:
    total = int(len(df))
    counts = (
        df.groupby(
            ["model_key", "language", "semantic_group"],
            dropna=False,
        )
        .size()
        .to_dict()
    )
    model_language_totals = (
        df.groupby(["model_key", "language"], dropna=False).size().to_dict()
    )

    rows: list[dict[str, Any]] = []
    for model_key, language, semantic_group in observed_group_combinations(
        df,
        ["model_key", "language", "semantic_group"],
    ):
        n = int(counts.get((model_key, language, semantic_group), 0))
        denominator = int(model_language_totals.get((model_key, language), 0))

        rows.append(
            {
                "model_key": model_key,
                "language": language,
                "semantic_group": semantic_group,
                "n": n,
                "model_language_total": denominator,
                "percent_within_model_language": (
                    round(100.0 * n / denominator, 4) if denominator else 0.0
                ),
                "total": total,
                "percent_of_annotated_subset": (
                    round(100.0 * n / total, 4) if total else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def counter_dict(
    series: pd.Series, order: Sequence[str] | None = None
) -> dict[str, int]:
    counts = Counter(series.tolist())
    if order is None:
        return {key: int(counts[key]) for key in sorted(counts)}
    return {label: int(counts.get(label, 0)) for label in order}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize finalized gold-before-final annotations."
    )
    parser.add_argument(
        "input_file", nargs="?", type=Path, help="Legacy positional input."
    )
    parser.add_argument("--input", type=Path, help="Finalized TSV or CSV annotations.")
    parser.add_argument(
        "--output-dir", "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing analysis outputs."
    )
    parser.add_argument(
        "--expected-rows", "--expected_rows", type=int, default=DEFAULT_EXPECTED_ROWS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.input is None) == (args.input_file is None):
        parser.error("Specify exactly one input: --input or the positional path.")
    args.input_file = args.input if args.input is not None else args.input_file
    if args.expected_rows <= 0:
        parser.error("--expected-rows must be positive.")
    output_dir = args.output_dir
    if args.expected_rows != DEFAULT_EXPECTED_ROWS and (
        output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
    ):
        parser.error("Nonstandard row counts require a separate --output-dir.")
    existing = [
        output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        parser.error(
            "Existing outputs; use --overwrite or a different --output-dir: "
            + ", ".join(str(path) for path in existing)
        )

    try:
        df = read_input(args.input_file)
        validation = validate_input(
            df=df,
            expected_rows=args.expected_rows,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "annotation_analysis_validation.json"
    write_json(validation_path, validation)

    if not validation["valid"]:
        for error in validation["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            "\nAnalysis stopped because input validation failed.",
            file=sys.stderr,
        )
        print(f"Validation report: {validation_path}", file=sys.stderr)
        return 1

    df = add_broad_semantic_class(df)

    output_files: dict[str, Any] = {
        "validation": str(validation_path),
    }

    # Overview and sample composition.
    overview = make_overview_table(df)
    overview_path = output_dir / "annotation_overview.tsv"
    write_tsv(overview_path, overview)
    output_files["overview"] = str(overview_path)

    for name, columns in {
        "by_language": ["language"],
        "by_model": ["model_key"],
        "by_semantic_group": ["semantic_group"],
    }.items():
        table = make_sample_composition(df, columns)
        path = output_dir / f"annotation_sample_composition_{name}.tsv"
        write_tsv(path, table)
        output_files[f"sample_composition_{name}"] = str(path)

    for name, row_column, column_column in [
        ("by_model_language", "model_key", "language"),
        (
            "by_model_semantic_group",
            "model_key",
            "semantic_group",
        ),
        (
            "by_language_semantic_group",
            "language",
            "semantic_group",
        ),
    ]:
        table = make_two_way_sample_composition(
            df=df,
            row_column=row_column,
            column_column=column_column,
        )
        path = output_dir / f"annotation_sample_composition_{name}.tsv"
        write_tsv(path, table)
        output_files[f"sample_composition_{name}"] = str(path)

    three_way_composition = make_three_way_sample_composition(df)
    three_way_composition_path = output_dir / (
        "annotation_sample_composition_by_model_language_semantic_group.tsv"
    )
    write_tsv(three_way_composition_path, three_way_composition)
    output_files["sample_composition_by_model_language_semantic_group"] = str(
        three_way_composition_path
    )

    # Completion-form distributions: denominator = all annotated rows.
    output_files["completion"] = write_distribution_family(
        df=df,
        output_dir=output_dir,
        filename_stem="annotation_completion",
        label_column="final_completion_form",
        label_order=COMPLETION_ORDER,
    )

    output_files["completion_cross_tables"] = write_completion_cross_tables(
        df=df,
        output_dir=output_dir,
    )

    # Fine-grained semantic distributions across all rows.
    output_files["semantic_all"] = write_distribution_family(
        df=df,
        output_dir=output_dir,
        filename_stem="annotation_semantic_all",
        label_column="final_semantic_relation",
        label_order=SEMANTIC_ORDER,
    )

    # Fine-grained semantic distributions among semantically evaluable rows.
    evaluable_df = df[df["final_semantic_relation"] != "not_applicable"].copy()
    output_files["semantic_evaluable"] = write_distribution_family(
        df=evaluable_df,
        output_dir=output_dir,
        filename_stem="annotation_semantic_evaluable",
        label_column="final_semantic_relation",
        label_order=EVALUABLE_SEMANTIC_ORDER,
    )

    output_files["semantic_evaluable_cross_tables"] = (
        write_semantic_evaluable_cross_tables(
            evaluable_df=evaluable_df,
            output_dir=output_dir,
        )
    )

    # Broad semantic grouping across all rows.
    output_files["broad_semantic_all"] = write_distribution_family(
        df=df,
        output_dir=output_dir,
        filename_stem="annotation_broad_semantic_all",
        label_column="broad_semantic_class",
        label_order=BROAD_SEMANTIC_ALL_ORDER,
    )

    # Broad semantic grouping among evaluable rows only.
    output_files["broad_semantic_evaluable"] = write_distribution_family(
        df=evaluable_df,
        output_dir=output_dir,
        filename_stem="annotation_broad_semantic_evaluable",
        label_column="broad_semantic_class",
        label_order=BROAD_SEMANTIC_EVALUABLE_ORDER,
    )

    # Surface subtype distribution among surface-variant rows only.
    surface_df = df[
        df["final_semantic_relation"] == "surface_or_morphological_variant"
    ].copy()
    output_files["surface_variant"] = write_distribution_family(
        df=surface_df,
        output_dir=output_dir,
        filename_stem="annotation_surface_variant",
        label_column="final_surface_variant_type",
        label_order=SURFACE_VARIANT_ORDER,
    )

    semantic_evaluable_count = int(len(evaluable_df))
    expected_evaluable_from_form = int(
        (
            df["final_completion_form"].isin(
                [
                    "standalone_content_candidate",
                    "gold_surface_fragment",
                ]
            )
        ).sum()
    )

    report: dict[str, Any] = {
        "analysis_scope": (
            "Descriptive composition of manually annotated gold-before-final "
            "trajectories only. These tables do not estimate event incidence "
            "in the full experimental dataset."
        ),
        "cross_table_denominators": {
            "completion_by_model_language": "exact model x language cell",
            "completion_by_model_semantic_group": "exact model x semantic-group cell",
            "completion_by_language_semantic_group": "exact language x semantic-group cell",
            "completion_by_model_language_semantic_group": "exact model x language x semantic-group cell",
            "semantic_evaluable_cross_tables": "semantically evaluable rows in the exact cross-tabulated "
            "cell; rows with final_semantic_relation=not_applicable "
            "are excluded before percentages are calculated",
        },
        "input_file": str(args.input_file),
        "output_directory": str(output_dir),
        "total_rows": int(len(df)),
        "semantic_evaluable_rows": semantic_evaluable_count,
        "semantic_not_evaluable_rows": int(len(df) - semantic_evaluable_count),
        "semantic_evaluable_alignment": {
            "semantic_relation_non_na_rows": semantic_evaluable_count,
            "standalone_plus_surface_fragment_rows": expected_evaluable_from_form,
            "matches": (semantic_evaluable_count == expected_evaluable_from_form),
        },
        "completion_counts": counter_dict(
            df["final_completion_form"], COMPLETION_ORDER
        ),
        "semantic_relation_counts": counter_dict(
            df["final_semantic_relation"], SEMANTIC_ORDER
        ),
        "broad_semantic_counts_all_rows": counter_dict(
            df["broad_semantic_class"], BROAD_SEMANTIC_ALL_ORDER
        ),
        "broad_semantic_counts_evaluable_rows": counter_dict(
            evaluable_df["broad_semantic_class"],
            BROAD_SEMANTIC_EVALUABLE_ORDER,
        ),
        "surface_variant_counts": counter_dict(
            surface_df["final_surface_variant_type"],
            SURFACE_VARIANT_ORDER,
        ),
        "model_counts": counter_dict(df["model_key"]),
        "language_counts": counter_dict(df["language"]),
        "semantic_group_counts": counter_dict(df["semantic_group"]),
        "output_files": output_files,
    }

    if semantic_evaluable_count:
        acceptable_n = int(
            (evaluable_df["broad_semantic_class"] == "broadly_acceptable").sum()
        )
        incompatible_n = int(
            (evaluable_df["broad_semantic_class"] == "incompatible_or_unrelated").sum()
        )
        report["semantic_evaluable_summary"] = {
            "broadly_acceptable_n": acceptable_n,
            "broadly_acceptable_percent": round(
                100.0 * acceptable_n / semantic_evaluable_count, 4
            ),
            "incompatible_n": incompatible_n,
            "incompatible_percent": round(
                100.0 * incompatible_n / semantic_evaluable_count, 4
            ),
        }

    report_path = output_dir / "annotation_analysis_report.json"
    write_json(report_path, report)

    print(
        f"Gold-before-final: {len(df)} annotated, "
        f"{semantic_evaluable_count} relation-evaluable"
    )
    print(
        f"  Broadly acceptable: {int((evaluable_df['broad_semantic_class'] == 'broadly_acceptable').sum())}"
    )
    print(
        f"  Incompatible or unrelated: {int((evaluable_df['broad_semantic_class'] == 'incompatible_or_unrelated').sum())}"
    )
    print(f"Output: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
