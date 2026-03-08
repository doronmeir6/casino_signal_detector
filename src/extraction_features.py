"""
Shared feature engineering from canonical LLM extraction payloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signal_schema import SIGNAL_CATEGORIES, extraction_to_labels


def extraction_feature_columns() -> list[str]:
    cols: list[str] = []
    for cat in SIGNAL_CATEGORIES:
        cols.extend(
            [
                f"feat_{cat}_detected",
                f"feat_{cat}_confidence",
                f"feat_{cat}_evidence_len_norm",
            ]
        )
    cols.extend(
        [
            "feat_num_detected",
            "feat_mean_confidence",
            "feat_max_confidence",
            "feat_parser_confidence",
        ]
    )
    return cols


def extraction_to_feature_dict(extraction: dict) -> dict[str, float]:
    feats: dict[str, float] = {}
    confidences: list[float] = []
    detected_count = 0

    for cat in SIGNAL_CATEGORIES:
        block = extraction.get("signals", {}).get(cat, {})
        detected = 1.0 if bool(block.get("detected", False)) else 0.0
        conf = float(block.get("confidence", 0.0) or 0.0)
        evidence = str(block.get("evidence", "") or "")
        evidence_len_norm = min(len(evidence), 240) / 240.0

        feats[f"feat_{cat}_detected"] = detected
        feats[f"feat_{cat}_confidence"] = min(max(conf, 0.0), 1.0)
        feats[f"feat_{cat}_evidence_len_norm"] = evidence_len_norm

        confidences.append(feats[f"feat_{cat}_confidence"])
        detected_count += int(detected)

    feats["feat_num_detected"] = detected_count / max(len(SIGNAL_CATEGORIES), 1)
    feats["feat_mean_confidence"] = float(np.mean(confidences)) if confidences else 0.0
    feats["feat_max_confidence"] = float(np.max(confidences)) if confidences else 0.0
    feats["feat_parser_confidence"] = float(extraction.get("parser_confidence", 0.0) or 0.0)
    feats["feat_parser_confidence"] = min(max(feats["feat_parser_confidence"], 0.0), 1.0)
    return feats


def extraction_to_feature_array(extraction: dict) -> np.ndarray:
    feats = extraction_to_feature_dict(extraction)
    cols = extraction_feature_columns()
    return np.array([feats[c] for c in cols], dtype=np.float32)


def extractions_to_table(extractions: list[dict]) -> pd.DataFrame:
    rows = []
    feat_cols = extraction_feature_columns()

    for ex in extractions:
        labels = extraction_to_labels(ex)
        row = {
            "conversation_id": ex.get("conversation_id"),
            "guest_text": ex.get("guest_text", ""),
            "parser_confidence": ex.get("parser_confidence", 0.0),
        }
        row.update(extraction_to_feature_dict(ex))
        for cat, val in labels.items():
            row[f"label_{cat}"] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    for cat in SIGNAL_CATEGORIES:
        c = f"label_{cat}"
        if c not in df.columns:
            df[c] = 0
    return df
