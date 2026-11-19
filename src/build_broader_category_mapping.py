from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


DEFAULT_OUTPUT_DIR = Path("data/analysis/full/broader_category")

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


def load_dataset(path: Path) -> list[dict[str, Any]]:
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


def write_tsv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
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


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the manually specified broader-category mapping review dataset for EQ2."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Frozen bilingual dataset (.jsonl, .json, or .tsv).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    dataset_rows = load_dataset(Path(args.input))
    mapping_rows = build_mapping(dataset_rows)
    validation = validate_output(mapping_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    review_tsv = output_dir / "broader_category_mapping_review.tsv"
    review_jsonl = output_dir / "broader_category_mapping_review.jsonl"
    report_json = output_dir / "broader_category_mapping_build_report.json"

    write_tsv(mapping_rows, review_tsv)
    write_jsonl(mapping_rows, review_jsonl)

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
    write_json(report, report_json)

    print("\nBroader-category mapping build complete.")
    print(f"Input dataset rows: {len(dataset_rows)}")
    print(f"EQ2 review rows: {len(mapping_rows)}")
    for subgroup in RULES:
        print(f"  {subgroup}: {validation['counts_by_subgroup'][subgroup]}")

    print(f"\nReview TSV: {review_tsv}")
    print(f"Review JSONL: {review_jsonl}")
    print(f"Report: {report_json}")
    print("\nNext: manually review all 55 rows and fill manual_keep/manual_note.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
