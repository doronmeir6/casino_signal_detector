"""
Train XGBoost classifiers with Optuna hyperparameter optimisation.

Supports two input modes:
  1) Legacy matrix mode: features.npz + labels.csv
  2) LLM-table mode: training_table.csv (label_* + feat_* + guest_text)

In LLM-table mode, training features are:
  - Sentence embeddings from guest_text
  - Tabular extraction features (feat_*)
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

# Allow running as `python src/train.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constant_model import ConstantProbabilityModel
from src.features import FeatureExtractor
from src.signal_schema import SIGNAL_CATEGORIES

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
_LEAKY_TABULAR_PATTERNS = (
    re.compile(r"^feat_(intent|value|sentiment|life_event|competitive)_(detected|confidence|evidence_len_norm)$"),
    re.compile(r"^feat_(num_detected|mean_confidence|max_confidence)$"),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    features_path: str = "data/processed/features.npz",
    labels_path: str = "data/processed/labels.csv",
) -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(features_path)["X"]
    y = pd.read_csv(labels_path)
    return X, y


def load_data_from_table(
    table_csv: str,
    use_embeddings: bool = True,
    use_tabular_features: bool = True,
    allow_leaky_tabular_features: bool = False,
    embedding_model: str = FeatureExtractor.DEFAULT_MODEL,
    feature_extractor_path: str = "models/feature_extractor.joblib",
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Build X/y from extracted training table.

    Expected columns:
      - guest_text
      - label_<category> for each category
      - feat_* numeric columns (optional but recommended)
    """
    df = pd.read_csv(table_csv)
    label_cols = [f"label_{cat}" for cat in SIGNAL_CATEGORIES if f"label_{cat}" in df.columns]
    if len(label_cols) != len(SIGNAL_CATEGORIES):
        missing = [c for c in SIGNAL_CATEGORIES if f"label_{c}" not in df.columns]
        raise ValueError(f"Missing label columns in table: {missing}")

    y = df[label_cols].copy()
    y.columns = SIGNAL_CATEGORIES
    y = y.astype(int)

    all_feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if use_tabular_features:
        if allow_leaky_tabular_features:
            feat_cols = all_feat_cols
            excluded_feat_cols = []
        else:
            feat_cols = [
                c for c in all_feat_cols
                if not any(p.match(c) for p in _LEAKY_TABULAR_PATTERNS)
            ]
            excluded_feat_cols = [c for c in all_feat_cols if c not in feat_cols]
    else:
        feat_cols = []
        excluded_feat_cols = all_feat_cols

    X_parts = []
    metadata = {
        "input_mode": "table",
        "use_embeddings": use_embeddings,
        "embedding_model": embedding_model if use_embeddings else None,
        "tabular_feature_columns": feat_cols,
        "excluded_tabular_feature_columns": excluded_feat_cols,
        "allow_leaky_tabular_features": allow_leaky_tabular_features,
    }

    if use_embeddings:
        if "guest_text" not in df.columns:
            raise ValueError("Table mode with embeddings requires `guest_text` column.")
        extractor = FeatureExtractor(model_name=embedding_model)
        texts = df["guest_text"].fillna("").astype(str).tolist()
        emb = extractor.transform(texts)
        X_parts.append(emb)
        extractor.save(feature_extractor_path)
        metadata["embedding_dim"] = int(emb.shape[1])
    else:
        metadata["embedding_dim"] = 0

    if feat_cols:
        tab = df[feat_cols].fillna(0.0).values.astype(np.float32)
        X_parts.append(tab)
        metadata["tabular_dim"] = int(tab.shape[1])
        print(f"Tabular features kept: {len(feat_cols)}")
    else:
        metadata["tabular_dim"] = 0
        if use_tabular_features:
            print("Tabular features kept: 0 (all filtered as leaky or missing)")
    if excluded_feat_cols:
        print(f"Tabular features excluded: {len(excluded_feat_cols)} (leakage prevention)")

    if not X_parts:
        raise ValueError("No features available from table. Provide guest_text and/or feat_* columns.")
    X = np.hstack(X_parts).astype(np.float32)
    metadata["total_dim"] = int(X.shape[1])
    return X, y, metadata


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(X_train: np.ndarray, y_train: np.ndarray):
    """Return an Optuna objective function that maximises CV F1."""
    unique, counts = np.unique(y_train, return_counts=True)
    class_count = {int(k): int(v) for k, v in zip(unique, counts)}
    min_class_count = min(class_count.values()) if class_count else 0
    n_splits = max(2, min(5, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 9),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            # Handle class imbalance per category
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 6.0),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            error_score=np.nan,
        )
        mean_score = float(np.nanmean(scores))
        # Optuna cannot handle NaN objective values.
        if np.isnan(mean_score):
            return -1.0
        return mean_score

    return objective


# ---------------------------------------------------------------------------
# Per-category training
# ---------------------------------------------------------------------------

def train_category(
    X_train: np.ndarray,
    y_train: np.ndarray,
    category: str,
    n_trials: int = 60,
    output_dir: str = "models",
) -> dict:
    """Tune, retrain, and calibrate one XGBoost classifier."""
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    print(f"\n{'─'*56}")
    print(f"  {category.upper():20s}  pos={pos}  neg={neg}")
    print(f"{'─'*56}")

    # Edge case: one class only OR too few minority examples for stable CV.
    unique_vals = np.unique(y_train)
    min_class_count = int(min(pos, neg))
    if len(unique_vals) < 2 or min_class_count < 2:
        # Use majority class as constant fallback.
        constant_class = 1 if pos >= neg else 0
        const_model = ConstantProbabilityModel(positive_class=constant_class)
        const_model.fit(X_train, y_train)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(const_model, Path(output_dir) / f"xgb_{category}.joblib")
        return {
            "category": category,
            "best_cv_f1": None,
            "best_params": {"model_type": "constant", "constant_class": constant_class},
            "n_train_pos": pos,
            "n_train_neg": neg,
            "n_trials": 0,
            "note": "insufficient_minority_class_fallback",
        }

    study = optuna.create_study(
        direction="maximize",
        study_name=f"xgb_{category}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )
    study.optimize(
        make_objective(X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and not np.isnan(t.value)
    ]
    if not completed:
        # Robust fallback if all trials become invalid.
        constant_class = 1 if pos >= neg else 0
        const_model = ConstantProbabilityModel(positive_class=constant_class)
        const_model.fit(X_train, y_train)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(const_model, Path(output_dir) / f"xgb_{category}.joblib")
        return {
            "category": category,
            "best_cv_f1": None,
            "best_params": {"model_type": "constant", "constant_class": constant_class},
            "n_train_pos": pos,
            "n_train_neg": neg,
            "n_trials": n_trials,
            "note": "all_optuna_trials_invalid_fallback",
        }

    best_params = study.best_params
    print(f"  Best CV F1 : {study.best_value:.4f}")

    # Build final model with best params
    final_params = {
        **best_params,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    base_model = XGBClassifier(**final_params)

    # Platt-scale probability calibration
    # Calibration CV also needs enough samples in each class.
    min_class_count = int(min((y_train == 0).sum(), (y_train == 1).sum()))
    calib_cv = max(2, min(5, min_class_count))
    calibrated = CalibratedClassifierCV(base_model, cv=calib_cv, method="sigmoid")
    calibrated.fit(X_train, y_train)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, Path(output_dir) / f"xgb_{category}.joblib")
    joblib.dump(study, Path(output_dir) / f"study_{category}.joblib")

    return {
        "category": category,
        "best_cv_f1": round(study.best_value, 4),
        "best_params": best_params,
        "n_train_pos": pos,
        "n_train_neg": neg,
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    X_test: np.ndarray,
    y_test: pd.DataFrame,
    model_dir: str = "models",
) -> pd.DataFrame:
    """Evaluate all saved models on the held-out test split."""
    rows = []
    for cat in SIGNAL_CATEGORIES:
        path = Path(model_dir) / f"xgb_{cat}.joblib"
        if not path.exists():
            print(f"  [skip] {cat}: model not found at {path}")
            continue

        model = joblib.load(path)
        y_true = y_test[cat].values
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        brier = brier_score_loss(y_true, y_prob)
        # log_loss requires two classes in y_true
        if len(np.unique(y_true)) < 2:
            ll = float("nan")
        else:
            ll = log_loss(y_true, y_prob, labels=[0, 1])

        rows.append({
            "category": cat,
            "precision": round(report.get("1", {}).get("precision", 0.0), 4),
            "recall": round(report.get("1", {}).get("recall", 0.0), 4),
            "f1": round(report.get("1", {}).get("f1-score", 0.0), 4),
            "roc_auc": round(roc, 4),
            "brier": round(brier, 4),
            "logloss": round(ll, 4) if not np.isnan(ll) else np.nan,
            "support_pos": int(y_true.sum()),
            "support_neg": int((y_true == 0).sum()),
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_all(
    features_path: str = "data/processed/features.npz",
    labels_path: str = "data/processed/labels.csv",
    table_csv: str = "",
    splits_path: str = "data/processed/splits.npz",
    n_trials: int = 60,
    use_embeddings: bool = True,
    use_tabular_features: bool = True,
    allow_leaky_tabular_features: bool = False,
    embedding_model: str = FeatureExtractor.DEFAULT_MODEL,
    output_dir: str = "models",
) -> None:
    print("=" * 56)
    print("Casino Signal Detector — XGBoost + Optuna Training")
    print("=" * 56)

    feature_extractor_path = str(Path(output_dir) / "feature_extractor.joblib")
    if table_csv:
        X, y, input_meta = load_data_from_table(
            table_csv=table_csv,
            use_embeddings=use_embeddings,
            use_tabular_features=use_tabular_features,
            allow_leaky_tabular_features=allow_leaky_tabular_features,
            embedding_model=embedding_model,
            feature_extractor_path=feature_extractor_path,
        )
    else:
        X, y = load_data(features_path, labels_path)
        input_meta = {
            "input_mode": "matrix",
            "use_embeddings": True,
            "embedding_model": FeatureExtractor.DEFAULT_MODEL,
            "embedding_dim": int(X.shape[1]),
            "tabular_dim": 0,
            "tabular_feature_columns": [],
            "total_dim": int(X.shape[1]),
        }
    print(f"\nFeatures : {X.shape}")
    print(f"Labels   : {y.shape}")
    print(f"\nClass distribution (positives):\n{y.sum().to_string()}\n")

    # Single shared train/test split
    idx = np.arange(len(X))
    # Stratify on a combined multi-label key (first label as proxy)
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    # Persist split indices for notebook reproducibility
    Path(splits_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(splits_path, train_idx=train_idx, test_idx=test_idx)
    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}")
    print(f"Split indices saved → {splits_path}\n")

    # Train one model per category
    training_results = []
    for cat in SIGNAL_CATEGORIES:
        result = train_category(
            X_train, y_train[cat].values,
            cat, n_trials=n_trials, output_dir=output_dir,
        )
        training_results.append(result)

    # Save training summary
    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(training_results, f, indent=2)
    print(f"\nTraining summary → {summary_path}")

    model_metadata = {
        "signal_categories": SIGNAL_CATEGORIES,
        "threshold_default": 0.5,
        **input_meta,
    }
    meta_path = Path(output_dir) / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(model_metadata, f, indent=2)
    print(f"Model metadata → {meta_path}")

    # Held-out evaluation
    print("\n" + "=" * 56)
    print("HELD-OUT TEST EVALUATION")
    print("=" * 56)
    eval_df = evaluate_all(X_test, y_test, model_dir=output_dir)
    print(eval_df.to_string(index=False))
    eval_path = Path(output_dir) / "evaluation.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"\nEvaluation results → {eval_path}")

    # Macro averages
    print(f"\nMacro avg  F1      : {eval_df['f1'].mean():.4f}")
    print(f"Macro avg  ROC-AUC : {eval_df['roc_auc'].mean():.4f}")
    print(f"Macro avg  Brier   : {eval_df['brier'].mean():.4f}")
    print(f"Macro avg  LogLoss : {eval_df['logloss'].dropna().mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train casino signal detectors")
    parser.add_argument("--features", default="data/processed/features.npz")
    parser.add_argument("--labels", default="data/processed/labels.csv")
    parser.add_argument("--table_csv", default="", help="Training table with label_* and feat_* columns")
    parser.add_argument("--splits", default="data/processed/splits.npz")
    parser.add_argument("--n_trials", type=int, default=60)
    parser.add_argument("--tabular_only", action="store_true", help="In table mode, disable text embeddings")
    parser.add_argument(
        "--disable_tabular_features",
        action="store_true",
        help="Ignore all feat_* table columns during training",
    )
    parser.add_argument(
        "--allow_leaky_tabular_features",
        action="store_true",
        help="Include leak-prone feat_* extraction columns (not recommended)",
    )
    parser.add_argument("--embedding_model", default=FeatureExtractor.DEFAULT_MODEL)
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()

    train_all(
        features_path=args.features,
        labels_path=args.labels,
        table_csv=args.table_csv,
        splits_path=args.splits,
        n_trials=args.n_trials,
        use_embeddings=not args.tabular_only,
        use_tabular_features=not args.disable_tabular_features,
        allow_leaky_tabular_features=args.allow_leaky_tabular_features,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir,
    )
