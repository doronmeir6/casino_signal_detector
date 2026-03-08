"""
Run the full assignment pipeline (OpenAI extraction path).

This orchestrates:
1) MultiWOZ conversion
2) Optional synthetic top-up generation
3) Merge
4) LLM extraction (OpenAI)
5) Flatten to training CSV
6) Train XGBoost + Optuna
7) Build pipeline evaluation report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def run_all(args: argparse.Namespace) -> None:
    py = sys.executable
    root = Path(args.project_root).resolve()

    mwz_path = root / "data" / "raw" / "conversations_multiwoz.json"
    synth_path = root / "data" / "raw" / "conversations_synth.json"
    merged_path = root / "data" / "raw" / "conversations_merged.json"
    extraction_jsonl = root / "data" / "processed" / "extractions.jsonl"
    extraction_cache = root / "data" / "processed" / "extraction_cache.jsonl"
    extraction_report = root / "data" / "processed" / "extraction_report.json"
    training_csv = root / "data" / "processed" / "training_table.csv"
    training_summary = root / "data" / "processed" / "training_table_summary.json"
    pipeline_report = root / "data" / "processed" / "pipeline_evaluation.json"
    final_eval_dir = root / "data" / "processed" / "final_eval"

    if args.fresh_extraction_cache and extraction_cache.exists():
        extraction_cache.unlink()
        print(f"Deleted extraction cache for fresh run: {extraction_cache}")

    # 1) MultiWOZ base
    _run(
        [
            py,
            str(root / "src" / "build_dataset_multiwoz.py"),
            "--multiwoz_dir",
            str(root / "data" / "MultiWOZ_2.2"),
            "--output",
            str(mwz_path),
            "--services",
            "hotel",
            "restaurant",
            "--splits",
            "train",
            "dev",
            "test",
            "--max_records",
            str(args.multiwoz_records),
        ]
    )

    # 2) Optional synthetic top-up
    inputs = [str(mwz_path)]
    if args.synthetic_records > 0:
        _run(
            [
                py,
                str(root / "src" / "generate_data.py"),
                "--n_total",
                str(args.synthetic_records),
                "--output",
                str(synth_path),
                "--model",
                args.synthetic_model,
            ]
        )
        inputs.append(str(synth_path))

    # 3) Merge
    _run(
        [
            py,
            str(root / "src" / "merge_datasets.py"),
            "--inputs",
            *inputs,
            "--output",
            str(merged_path),
        ]
    )

    # 4) LLM extraction
    extract_cmd = [
        py,
        str(root / "src" / "extract_with_llm.py"),
        "--input",
        str(merged_path),
        "--output_jsonl",
        str(extraction_jsonl),
        "--cache_path",
        str(extraction_cache),
        "--report_path",
        str(extraction_report),
        "--model",
        args.extract_model,
        "--min_parser_confidence",
        str(args.min_parser_confidence),
        "--retries",
        str(args.retries),
    ]
    _run(extract_cmd)

    # 5) Flatten table
    _run(
        [
            py,
            str(root / "src" / "prepare_training_table.py"),
            "--input_jsonl",
            str(extraction_jsonl),
            "--output_csv",
            str(training_csv),
            "--summary_json",
            str(training_summary),
        ]
    )

    # 6) Train
    train_cmd = [
        py,
        str(root / "src" / "train.py"),
        "--table_csv",
        str(training_csv),
        "--n_trials",
        str(args.n_trials),
        "--output_dir",
        str(root / "models"),
    ]
    if args.tabular_only:
        train_cmd.append("--tabular_only")
    _run(train_cmd)

    # 7) Pipeline report
    _run(
        [
            py,
            str(root / "src" / "evaluate_pipeline.py"),
            "--extraction_report",
            str(extraction_report),
            "--model_eval_csv",
            str(root / "models" / "evaluation.csv"),
            "--output",
            str(pipeline_report),
        ]
    )

    # 8) Final held-out inference evaluation artifacts
    if args.run_final_eval:
        _run(
            [
                py,
                str(root / "src" / "final_eval.py"),
                "--table_csv",
                str(training_csv),
                "--model_dir",
                str(root / "models"),
                "--splits",
                str(root / "data" / "processed" / "splits.npz"),
                "--output_dir",
                str(final_eval_dir),
                "--sample_size",
                str(args.sample_size),
            ]
        )

    print("\nPipeline completed successfully.")
    print(f"- Merged dataset: {merged_path}")
    print(f"- Extraction JSONL: {extraction_jsonl}")
    print(f"- Training CSV: {training_csv}")
    print(f"- Model eval: {root / 'models' / 'evaluation.csv'}")
    print(f"- Pipeline report: {pipeline_report}")
    if args.run_final_eval:
        print(f"- Final eval summary: {final_eval_dir / 'final_eval_summary.json'}")
        print(f"- Final eval category metrics: {final_eval_dir / 'category_metrics.csv'}")
        print(f"- Final eval samples: {final_eval_dir / 'sample_predictions.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end local pipeline")
    parser.add_argument("--project_root", default=".")
    # Kept for backward compatibility; only OpenAI path is supported.
    parser.add_argument("--provider", default="openai", choices=["openai"])
    parser.add_argument("--extract_model", default="gpt-4o-mini")
    parser.add_argument("--synthetic_model", default="gpt-4o-mini")
    parser.add_argument("--multiwoz_records", type=int, default=1200)
    parser.add_argument("--synthetic_records", type=int, default=800)
    parser.add_argument("--min_parser_confidence", type=float, default=0.45)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=60)
    parser.add_argument("--tabular_only", action="store_true")
    parser.add_argument(
        "--fresh_extraction_cache",
        action="store_true",
        help="Delete extraction cache before extraction to force relabeling",
    )
    parser.add_argument(
        "--run_final_eval",
        action="store_true",
        help="Run final held-out inference evaluation after training",
    )
    parser.add_argument("--sample_size", type=int, default=10)

    args = parser.parse_args()

    run_all(args)
