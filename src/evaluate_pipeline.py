"""
Combine extraction and model evaluation into a pipeline-level report.

Inputs:
  - data/processed/extraction_report.json
  - models/evaluation.csv

Output:
  - data/processed/pipeline_evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_pipeline_report(
    extraction_report_path: str = "data/processed/extraction_report.json",
    model_eval_csv: str = "models/evaluation.csv",
    output_path: str = "data/processed/pipeline_evaluation.json",
) -> dict:
    extraction_path = Path(extraction_report_path)
    model_eval_path = Path(model_eval_csv)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    extraction_report = {}
    if extraction_path.exists():
        with open(extraction_path, encoding="utf-8") as f:
            extraction_report = json.load(f)

    model_eval = pd.DataFrame()
    if model_eval_path.exists():
        model_eval = pd.read_csv(model_eval_path)

    extraction_stats = extraction_report.get("stats", {})
    total = int(extraction_stats.get("total", 0) or 0)
    ok = int(extraction_stats.get("ok", 0) or 0)
    extraction_success_rate = (ok / total) if total > 0 else 0.0

    if not model_eval.empty:
        cls_metrics = {
            "macro_f1": float(model_eval["f1"].mean()),
            "macro_roc_auc": float(model_eval["roc_auc"].mean()),
            "macro_brier": float(model_eval["brier"].mean()) if "brier" in model_eval.columns else None,
            "macro_logloss": float(model_eval["logloss"].dropna().mean())
            if "logloss" in model_eval.columns and model_eval["logloss"].notna().any()
            else None,
        }
    else:
        cls_metrics = {
            "macro_f1": None,
            "macro_roc_auc": None,
            "macro_brier": None,
            "macro_logloss": None,
        }

    failure_examples = extraction_report.get("failure_examples", [])
    reason_counts = {}
    for f in failure_examples:
        reason = f.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    report = {
        "extraction": {
            "stats": extraction_stats,
            "success_rate": extraction_success_rate,
            "label_counts": extraction_report.get("label_counts", {}),
            "failure_reason_counts_sample": reason_counts,
        },
        "classification": cls_metrics,
        "pipeline_summary": {
            "overall_health": "good"
            if extraction_success_rate >= 0.8 and (cls_metrics["macro_f1"] or 0.0) >= 0.6
            else "needs_improvement",
            "notes": [
                "Extraction failure slices are sampled from failure_examples.",
                "Classification metrics are computed on held-out test split.",
                "Domain mismatch should be monitored via source-wise metrics."
            ],
        },
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved pipeline report -> {out}")
    if cls_metrics["macro_f1"] is not None:
        print(
            f"macro_f1={cls_metrics['macro_f1']:.4f} "
            f"macro_roc_auc={cls_metrics['macro_roc_auc']:.4f}"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build pipeline-level evaluation report")
    parser.add_argument("--extraction_report", default="data/processed/extraction_report.json")
    parser.add_argument("--model_eval_csv", default="models/evaluation.csv")
    parser.add_argument("--output", default="data/processed/pipeline_evaluation.json")
    args = parser.parse_args()

    build_pipeline_report(
        extraction_report_path=args.extraction_report,
        model_eval_csv=args.model_eval_csv,
        output_path=args.output,
    )
