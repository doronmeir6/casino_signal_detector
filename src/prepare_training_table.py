"""
Flatten extracted JSONL into a model-ready training table.

Input format:
  JSONL rows from `src/extract_with_llm.py` containing:
    - extraction (canonical schema)
    - labels
    - conversation
    - metadata

Outputs:
  - table CSV with label_* and feat_* columns
  - summary JSON with class balance and source composition
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow running as `python src/prepare_training_table.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction_features import extractions_to_table
from src.signal_schema import SIGNAL_CATEGORIES


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_training_table(
    input_jsonl: str,
    output_csv: str,
    summary_json: str,
    min_label_positives: int = 0,
) -> pd.DataFrame:
    input_path = Path(input_jsonl)
    rows = _read_jsonl(input_path)

    extractions = [r.get("extraction", {}) for r in rows]
    df = extractions_to_table(extractions)
    df["record_id"] = [r.get("id") for r in rows]
    df["source"] = [r.get("source", "unknown") for r in rows]
    df["cache_key"] = [r.get("cache_key") for r in rows]
    df["conversation_turns"] = [len(r.get("conversation", [])) for r in rows]

    # Keep full guest text for embeddings later
    df["guest_text"] = df["guest_text"].fillna("").astype(str)
    df = df[df["guest_text"].str.len() > 0].reset_index(drop=True)

    # Optional class-balance gate
    label_cols = [f"label_{cat}" for cat in SIGNAL_CATEGORIES]
    if min_label_positives > 0:
        # Remove rows where every label is 0 only if we have enough negatives.
        # This keeps weak positive coverage in smaller datasets.
        all_zero_mask = (df[label_cols].sum(axis=1) == 0)
        if all_zero_mask.sum() > len(df) * 0.5:
            drop_n = int(all_zero_mask.sum() - len(df) * 0.5)
            drop_idx = df[all_zero_mask].head(max(drop_n, 0)).index
            df = df.drop(drop_idx).reset_index(drop=True)

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    label_counts = {cat: int(df[f"label_{cat}"].sum()) for cat in SIGNAL_CATEGORIES}
    source_counts = df["source"].value_counts().to_dict()
    summary = {
        "input_jsonl": str(input_path),
        "output_csv": str(out_csv),
        "rows": int(len(df)),
        "label_counts": label_counts,
        "source_counts": source_counts,
        "feature_columns": [c for c in df.columns if c.startswith("feat_")],
        "label_columns": label_cols,
    }
    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved training table -> {out_csv}")
    print(f"Rows: {len(df)}")
    print(f"Label counts: {label_counts}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training table from extracted JSONL")
    parser.add_argument("--input_jsonl", default="data/processed/extractions.jsonl")
    parser.add_argument("--output_csv", default="data/processed/training_table.csv")
    parser.add_argument("--summary_json", default="data/processed/training_table_summary.json")
    parser.add_argument("--min_label_positives", type=int, default=0)
    args = parser.parse_args()

    build_training_table(
        input_jsonl=args.input_jsonl,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        min_label_positives=args.min_label_positives,
    )
