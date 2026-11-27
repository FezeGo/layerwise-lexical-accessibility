from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


DEFAULT_INPUT_PATH = Path("data/dataset/fullspan_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/broader_category")

OUTPUT_FILENAMES = (
    "broader_category_mapping_review.tsv",
    "broader_category_mapping_review.jsonl",
    "broader_category_mapping_build_report.json",
)

RULES: dict[str, dict[str, Any]] = {
    "animal_taxonomy": {
        "expected_count": 17,
        "en_map": {"bird": "animal", "fish": "animal", "insect": "animal"},
        "zh_map": {"鸟": "动物", "鱼": "动物", "昆虫": "动物"},
        "rule_label": "animal_taxonomy_to_animal",
    },
    "chemical_element_taxonomy": {
        "expected_count": 18,
        "en_map": {"element": "substance"},
        "zh_map": {"元素": "物质"},
        "rule_label": "chemical_element_to_substance",
    },
    "food_taxonomy": {
        "expected_count": 12,
        "en_map": {"fruit": "food", "vegetable": "food"},
        "zh_map": {"水果": "食物", "蔬菜": "食物"},
        "rule_label": "food_taxonomy_to_food",
    },
    "plant_taxonomy": {
        "expected_count": 8,
        "en_map": {"flower": "plant"},
        "zh_map": {"花": "植物"},
        "rule_label": "flower_to_plant",
    },
}

EXPECTED_TOTAL = sum(rule["expected_count"] for rule in RULES.values())

REQUIRED_FIELDS = {
    "id",
    "semantic_group",
    "subgroup",
    "relation",
    "en_subject",
    "en_target",
    "en_prefix",
    "en_full_sentence",
    "zh_subject",
    "zh_target",
    "zh_prefix",
    "zh_full_sentence",
}


def read_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
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

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            candidate_keys = [
                key
                for key in ("data", "rows", "examples", "items")
                if key in data and isinstance(data[key], list)
            ]
            if len(candidate_keys) != 1:
                raise ValueError(
                    "JSON object input must contain exactly one list under "
                    "one of: data, rows, examples, items."
                )
            rows = data[candidate_keys[0]]
        else:
            raise ValueError("JSON input must be a list or supported wrapper.")

        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("Every JSON dataset row must be an object.")

    elif suffix in {".tsv", ".txt"}:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"No TSV header found in {path}")
            rows = [dict(row) for row in reader]

    else:
        raise ValueError(
            f"Unsupported input format {suffix!r}; use .jsonl, .json, .tsv, or .txt."
        )

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return rows


def write_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def validate_required_fields(row: dict[str, Any], row_index: int) -> None:
    missing = REQUIRED_FIELDS.difference(row)
    if missing:
        raise ValueError(f"Row {row_index} missing required fields: {sorted(missing)}")


def replace_target_after_prefix(
    prefix: str,
    gold: str,
    full_sentence: str,
    broader: str,
    *,
    language: str,
    example_id: Any,
) -> tuple[str, str]:
    expected_start = prefix + gold

    if not full_sentence.startswith(expected_start):
        raise ValueError(
            f"{language} alignment mismatch for id={example_id}: "
            f"full_sentence does not start with prefix+gold.\n"
            f"prefix={prefix!r}\ngold={gold!r}\nfull={full_sentence!r}"
        )

    suffix = full_sentence[len(expected_start) :]
    broader_sentence = prefix + broader + suffix
    return broader_sentence, suffix


def build_mapping(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen_ids = set()

    for row_index, row in enumerate(rows, start=1):
        validate_required_fields(row, row_index)

        if str(row["semantic_group"]) != "category":
            continue

        subgroup = str(row["subgroup"])
        if subgroup not in RULES:
            continue

        if str(row["relation"]) != "is_a":
            raise ValueError(
                f"Included row id={row['id']} has relation={row['relation']!r}; expected 'is_a'."
            )

        example_id = str(row["id"])
        if example_id in seen_ids:
            raise ValueError(f"Duplicate included id: {example_id}")
        seen_ids.add(example_id)

        rule = RULES[subgroup]
        en_gold = str(row["en_target"])
        zh_gold = str(row["zh_target"])

        if en_gold not in rule["en_map"]:
            raise ValueError(
                f"Unexpected English gold {en_gold!r} for subgroup={subgroup}, id={example_id}"
            )
        if zh_gold not in rule["zh_map"]:
            raise ValueError(
                f"Unexpected Chinese gold {zh_gold!r} for subgroup={subgroup}, id={example_id}"
            )

        en_broader = rule["en_map"][en_gold]
        zh_broader = rule["zh_map"][zh_gold]

        en_broader_sentence, en_suffix = replace_target_after_prefix(
            str(row["en_prefix"]),
            en_gold,
            str(row["en_full_sentence"]),
            en_broader,
            language="English",
            example_id=example_id,
        )
        zh_broader_sentence, zh_suffix = replace_target_after_prefix(
            str(row["zh_prefix"]),
            zh_gold,
            str(row["zh_full_sentence"]),
            zh_broader,
            language="Chinese",
            example_id=example_id,
        )

        output.append(
            {
                "id": example_id,
                "semantic_group": "category",
                "subgroup": subgroup,
                "relation": "is_a",
                "en_subject": str(row["en_subject"]),
                "en_gold": en_gold,
                "en_broader_category": en_broader,
                "en_prefix": str(row["en_prefix"]),
                "en_gold_sentence": str(row["en_full_sentence"]),
                "en_broader_sentence": en_broader_sentence,
                "en_preserved_suffix": en_suffix,
                "zh_subject": str(row["zh_subject"]),
                "zh_gold": zh_gold,
                "zh_broader_category": zh_broader,
                "zh_prefix": str(row["zh_prefix"]),
                "zh_gold_sentence": str(row["zh_full_sentence"]),
                "zh_broader_sentence": zh_broader_sentence,
                "zh_preserved_suffix": zh_suffix,
                "mapping_rule": str(rule["rule_label"]),
                "mapping_status": "manually_defined_broader_category",
                "manual_keep": "",
                "manual_note": "",
            }
        )

    def sort_key(row: dict[str, Any]):
        value = row["id"]
        return (0, int(value)) if str(value).isdigit() else (1, str(value))

    output.sort(key=sort_key)
    return output


def validate_output(mapping_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["subgroup"] for row in mapping_rows)
    expected_counts = {
        subgroup: int(rule["expected_count"]) for subgroup, rule in RULES.items()
    }

    problems = []

    for subgroup, expected in expected_counts.items():
        observed = counts.get(subgroup, 0)
        if observed != expected:
            problems.append(f"{subgroup}: expected {expected}, observed {observed}")

    if len(mapping_rows) != EXPECTED_TOTAL:
        problems.append(
            f"Total: expected {EXPECTED_TOTAL}, observed {len(mapping_rows)}"
        )

    identical = [
        row["id"]
        for row in mapping_rows
        if row["en_gold"] == row["en_broader_category"]
        or row["zh_gold"] == row["zh_broader_category"]
    ]
    if identical:
        problems.append(f"Gold equals broader category for ids: {identical}")

    if problems:
        raise ValueError(
            "Broader-category mapping validation failed:\n  " + "\n  ".join(problems)
        )

    return {
        "expected_total": EXPECTED_TOTAL,
        "observed_total": len(mapping_rows),
        "counts_by_subgroup": {subgroup: counts.get(subgroup, 0) for subgroup in RULES},
        "expected_counts_by_subgroup": expected_counts,
        "all_counts_match": True,
        "gold_broader_distinct": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the manually specified broader-category mapping review dataset for EQ2."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Frozen bilingual dataset (.jsonl, .json, or .tsv).",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files, including manual review fields.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    dataset_rows = read_dataset(args.input)
    mapping_rows = build_mapping(dataset_rows)
    validation = validate_output(mapping_rows)

    output_dir = args.output_dir
    review_tsv, review_jsonl, report_json = (
        output_dir / filename for filename in OUTPUT_FILENAMES
    )

    existing = [
        output_dir / filename
        for filename in OUTPUT_FILENAMES
        if (output_dir / filename).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Existing mapping outputs; use a different --output-dir or "
            "explicitly pass --overwrite: " + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(review_tsv, mapping_rows)
    write_jsonl(review_jsonl, mapping_rows)

    report = {
        "analysis": "build_broader_category_mapping",
        "purpose": (
            "Create the manually specified broader-category mapping review dataset "
            "for EQ2 without automatic hypernym inference."
        ),
        "input": str(args.input),
        "included_subgroups": list(RULES.keys()),
        "mapping_status": "manually_defined_broader_category",
        "mapping_rules": {
            subgroup: {
                "expected_count": rule["expected_count"],
                "en_map": rule["en_map"],
                "zh_map": rule["zh_map"],
                "rule_label": rule["rule_label"],
            }
            for subgroup, rule in RULES.items()
        },
        "validation": validation,
        "manual_review_required": True,
        "manual_review_fields": ["manual_keep", "manual_note"],
        "outputs": {
            "review_tsv": str(review_tsv),
            "review_jsonl": str(review_jsonl),
            "build_report_json": str(report_json),
        },
    }
    write_json(report_json, report)

    print(
        f"Broader-category mapping: {len(mapping_rows)}/{len(dataset_rows)} dataset rows"
    )
    print(f"Output: {output_dir}")
    print("Review manual_keep and manual_note before freezing the mapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
