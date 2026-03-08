"""
Final held-out inference evaluation for submission-ready reporting.

This script evaluates trained models on a held-out split and writes:
  - per-category metrics CSV
  - source-slice macro metrics CSV
  - summary JSON
  - sample predictions JSON (with per-signal confidence scores)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, classification_report, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Allow running as `python src/final_eval.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FeatureExtractor
from src.signal_schema import SIGNAL_CATEGORIES

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_project_path(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _build_feature_matrix(df: pd.DataFrame, metadata: dict, model_dir: Path) -> np.ndarray:
    parts: list[np.ndarray] = []
    use_embeddings = bool(metadata.get("use_embeddings", True))

    if use_embeddings:
        if "guest_text" not in df.columns:
            raise ValueError("guest_text column is required when embeddings are enabled.")
        extractor_path = model_dir / "feature_extractor.joblib"
        if not extractor_path.exists():
            raise FileNotFoundError(f"Missing feature extractor config: {extractor_path}")
        extractor = FeatureExtractor.load(str(extractor_path))
        texts = df["guest_text"].fillna("").astype(str).tolist()
        emb = extractor.transform(texts)
        parts.append(emb.astype(np.float32))

    tab_cols = metadata.get("tabular_feature_columns", []) or []
    if tab_cols:
        tab = df.reindex(columns=tab_cols, fill_value=0.0).fillna(0.0).values.astype(np.float32)
        parts.append(tab)

    if not parts:
        raise ValueError("No feature blocks available for evaluation.")
    return np.hstack(parts).astype(np.float32)


def _load_test_indices(
    n_rows: int,
    splits_path: Path,
    test_size: float = 0.2,
) -> np.ndarray:
    if splits_path.exists():
        payload = np.load(splits_path)
        if "test_idx" in payload:
            return payload["test_idx"]
    idx = np.arange(n_rows)
    _, test_idx = train_test_split(
        idx, test_size=test_size, random_state=RANDOM_STATE, shuffle=True
    )
    return test_idx


def run_final_eval(
    table_csv: str = "data/processed/training_table.csv",
    model_dir: str = "models",
    splits_path: str = "data/processed/splits.npz",
    output_dir: str = "data/processed/final_eval",
    sample_size: int = 10,
) -> dict:
    model_dir_path = _resolve_project_path(model_dir)
    table_path = _resolve_project_path(table_csv)
    out_dir = _resolve_project_path(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        out_dir = PROJECT_ROOT / "data" / "processed" / "final_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[warn] output_dir not writable; using fallback: {out_dir}")

    metadata_path = model_dir_path / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    df = pd.read_csv(table_path)
    label_cols = [f"label_{cat}" for cat in SIGNAL_CATEGORIES]
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in table: {missing}")

    X = _build_feature_matrix(df, metadata=metadata, model_dir=model_dir_path)
    test_idx = _load_test_indices(n_rows=len(df), splits_path=_resolve_project_path(splits_path))
    test_idx = np.array(test_idx, dtype=int)
    test_idx = test_idx[(test_idx >= 0) & (test_idx < len(df))]
    if len(test_idx) == 0:
        raise ValueError("No valid test indices found.")

    X_test = X[test_idx]
    df_test = df.iloc[test_idx].reset_index(drop=True)
    y_test = df_test[label_cols].copy()
    y_test.columns = SIGNAL_CATEGORIES
    threshold = float(metadata.get("threshold_default", 0.5))

    per_cat_rows: list[dict] = []
    prob_map: dict[str, np.ndarray] = {}
    pred_map: dict[str, np.ndarray] = {}

    for cat in SIGNAL_CATEGORIES:
        model_path = model_dir_path / f"xgb_{cat}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)

        y_true = y_test[cat].values.astype(int)
        y_prob = model.predict_proba(X_test)[:, 1].astype(float)
        y_pred = (y_prob >= threshold).astype(int)
        prob_map[cat] = y_prob
        pred_map[cat] = y_pred

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        brier = brier_score_loss(y_true, y_prob)
        ll = log_loss(y_true, y_prob, labels=[0, 1]) if len(np.unique(y_true)) > 1 else float("nan")
        per_cat_rows.append(
            {
                "category": cat,
                "precision": round(report.get("1", {}).get("precision", 0.0), 4),
                "recall": round(report.get("1", {}).get("recall", 0.0), 4),
                "f1": round(report.get("1", {}).get("f1-score", 0.0), 4),
                "roc_auc": round(roc, 4) if not np.isnan(roc) else np.nan,
                "brier": round(brier, 4),
                "logloss": round(ll, 4) if not np.isnan(ll) else np.nan,
                "support_pos": int(y_true.sum()),
                "support_neg": int((y_true == 0).sum()),
            }
        )

    metrics_df = pd.DataFrame(per_cat_rows)
    metrics_csv = out_dir / "category_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Per-row prediction table for notebook diagnostics (confusion matrices, ROC, histograms)
    pred_table = pd.DataFrame(
        {
            "record_id": df_test.get("record_id", pd.Series([None] * len(df_test))),
            "source": df_test.get("source", pd.Series(["unknown"] * len(df_test))),
            "guest_text": df_test.get("guest_text", pd.Series([""] * len(df_test))),
        }
    )
    for cat in SIGNAL_CATEGORIES:
        if cat not in prob_map:
            continue
        pred_table[f"y_true_{cat}"] = y_test[cat].values.astype(int)
        pred_table[f"y_prob_{cat}"] = prob_map[cat]
        pred_table[f"y_pred_{cat}"] = pred_map[cat]
    pred_table_csv = out_dir / "prediction_table.csv"
    pred_table.to_csv(pred_table_csv, index=False)

    # Source-slice macro F1
    source_rows: list[dict] = []
    if "source" in df_test.columns:
        for source_name, g in df_test.groupby("source"):
            idx_local = g.index.to_numpy()
            f1_vals: list[float] = []
            for cat in SIGNAL_CATEGORIES:
                if cat not in prob_map:
                    continue
                y_true = y_test.loc[idx_local, cat].values.astype(int)
                y_pred = pred_map[cat][idx_local]
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                f1_vals.append(float(report.get("1", {}).get("f1-score", 0.0)))
            source_rows.append(
                {
                    "source": source_name,
                    "n_rows": int(len(idx_local)),
                    "macro_f1": round(float(np.mean(f1_vals)) if f1_vals else 0.0, 4),
                }
            )
    source_df = pd.DataFrame(source_rows)
    if not source_df.empty and "n_rows" in source_df.columns:
        source_df = source_df.sort_values(by="n_rows", ascending=False).reset_index(drop=True)
    source_csv = out_dir / "source_macro_metrics.csv"
    source_df.to_csv(source_csv, index=False)

    # Sample predictions (submission/demo friendly)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_n = min(max(int(sample_size), 1), len(df_test))
    sample_indices = np.sort(rng.choice(len(df_test), size=sample_n, replace=False))
    samples = []
    for i in sample_indices:
        row = df_test.iloc[i]
        scores = {}
        labels = {}
        for cat in SIGNAL_CATEGORIES:
            if cat not in prob_map:
                continue
            conf = float(prob_map[cat][i])
            scores[cat] = {
                "confidence": round(conf, 4),
                "triggered": bool(conf >= threshold),
            }
            labels[cat] = int(y_test.iloc[i][cat])
        samples.append(
            {
                "record_id": row.get("record_id"),
                "source": row.get("source", "unknown"),
                "guest_text": str(row.get("guest_text", ""))[:320],
                "ground_truth": labels,
                "predictions": scores,
            }
        )

    samples_path = out_dir / "sample_predictions.json"
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    summary = {
        "table_csv": str(table_path),
        "model_dir": str(model_dir_path),
        "test_rows": int(len(df_test)),
        "threshold": threshold,
        "classification": {
            "macro_f1": float(metrics_df["f1"].mean()) if not metrics_df.empty else None,
            "macro_roc_auc": float(metrics_df["roc_auc"].dropna().mean())
            if not metrics_df.empty and metrics_df["roc_auc"].notna().any()
            else None,
            "macro_brier": float(metrics_df["brier"].mean()) if not metrics_df.empty else None,
            "macro_logloss": float(metrics_df["logloss"].dropna().mean())
            if not metrics_df.empty and metrics_df["logloss"].notna().any()
            else None,
        },
        "artifacts": {
            "category_metrics_csv": str(metrics_csv),
            "source_macro_metrics_csv": str(source_csv),
            "sample_predictions_json": str(samples_path),
            "prediction_table_csv": str(pred_table_csv),
        },
    }
    summary_path = out_dir / "final_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved category metrics -> {metrics_csv}")
    print(f"Saved source metrics   -> {source_csv}")
    print(f"Saved sample outputs   -> {samples_path}")
    print(f"Saved prediction table -> {pred_table_csv}")
    print(f"Saved summary          -> {summary_path}")
    if summary["classification"]["macro_f1"] is not None:
        print(
            "Macro metrics: "
            f"F1={summary['classification']['macro_f1']:.4f}, "
            f"ROC-AUC={summary['classification']['macro_roc_auc']:.4f}, "
            f"Brier={summary['classification']['macro_brier']:.4f}"
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run final held-out inference evaluation")
    parser.add_argument("--table_csv", default="data/processed/training_table.csv")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--splits", default="data/processed/splits.npz")
    parser.add_argument("--output_dir", default="data/processed/final_eval")
    parser.add_argument("--sample_size", type=int, default=10)
    args = parser.parse_args()

    run_final_eval(
        table_csv=args.table_csv,
        model_dir=args.model_dir,
        splits_path=args.splits,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
    )
