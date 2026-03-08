"""Inference pipeline with optional LLM extraction + XGBoost scoring."""

import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction_features import extraction_to_feature_array, extraction_to_feature_dict
from src.features import SIGNAL_CATEGORIES, FeatureExtractor, extract_guest_text
from src.infer_extract_llm import LLMConversationExtractor

DEFAULT_THRESHOLD = 0.45


class SignalDetector:
    """
    End-to-end signal detection pipeline.

    Loads trained XGBoost models and the feature extractor once on first call
    (lazy loading), then exposes a simple `detect()` method.

    Parameters
    ----------
    model_dir   : directory containing xgb_<category>.joblib files
    threshold   : minimum confidence to mark a signal as "triggered"
    """

    def __init__(
        self,
        model_dir: str = "models",
        threshold: float = DEFAULT_THRESHOLD,
        use_llm_extractor: bool = False,
        llm_model: str = "gpt-4o-mini",
    ):
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.use_llm_extractor = use_llm_extractor
        self.llm_model = llm_model
        self._extractor: Optional[FeatureExtractor] = None
        self._models: Optional[dict] = None
        self._metadata: Optional[dict] = None
        self._llm_extractor: Optional[LLMConversationExtractor] = None

    # -- lazy loaders --------------------------------------------------------

    @property
    def extractor(self) -> FeatureExtractor:
        if self._extractor is None:
            path = self.model_dir / "feature_extractor.joblib"
            self._extractor = FeatureExtractor.load(str(path))
        return self._extractor

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            path = self.model_dir / "model_metadata.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self._metadata = json.load(f)
            else:
                # Backward-compat fallback for earlier training artifacts.
                self._metadata = {
                    "input_mode": "matrix",
                    "use_embeddings": True,
                    "embedding_dim": 398,
                    "tabular_dim": 0,
                }
        return self._metadata

    @property
    def llm_extractor(self) -> Optional[LLMConversationExtractor]:
        if not self.use_llm_extractor:
            return None
        if self._llm_extractor is None:
            self._llm_extractor = LLMConversationExtractor(model=self.llm_model)
        return self._llm_extractor

    @property
    def models(self) -> dict:
        if self._models is None:
            self._models = {}
            for cat in SIGNAL_CATEGORIES:
                path = self.model_dir / f"xgb_{cat}.joblib"
                if path.exists():
                    self._models[cat] = joblib.load(path)
        return self._models

    # -- public API ----------------------------------------------------------

    def _build_feature_row_from_guest_text_and_extraction(
        self,
        guest_text: str,
        extraction: Optional[dict],
    ) -> np.ndarray:
        """Build feature row from guest text and optional canonical extraction."""
        guest_text = (guest_text or "").strip()

        if not guest_text.strip():
            # Build a safe zero-vector based on metadata dimensions.
            emb_dim = int(self.metadata.get("embedding_dim", 398))
            tab_dim = int(self.metadata.get("tabular_dim", 0))
            return np.zeros((1, emb_dim + tab_dim), dtype=np.float32)

        # Step 2: build embedding block (same feature extractor as training)
        emb_dim = int(self.metadata.get("embedding_dim", 398))
        use_embeddings = emb_dim > 0
        parts = []
        if use_embeddings:
            emb = self.extractor.transform([guest_text])
            parts.append(emb)

        # Step 3: optional tabular LLM extraction block
        tab_dim = int(self.metadata.get("tabular_dim", 0))
        if tab_dim > 0:
            if extraction is not None:
                feature_cols = self.metadata.get("tabular_feature_columns", [])
                if feature_cols:
                    feat_map = extraction_to_feature_dict(extraction)
                    tab = np.array([[float(feat_map.get(c, 0.0)) for c in feature_cols]], dtype=np.float32)
                else:
                    # Backward compatibility for older metadata without named tabular columns.
                    tab = extraction_to_feature_array(extraction).reshape(1, -1)
            else:
                tab = np.zeros((1, tab_dim), dtype=np.float32)
            if tab.shape[1] != tab_dim:
                # Defensive alignment for metadata mismatch.
                if tab.shape[1] > tab_dim:
                    tab = tab[:, :tab_dim]
                else:
                    pad = np.zeros((1, tab_dim - tab.shape[1]), dtype=np.float32)
                    tab = np.hstack([tab, pad])
            parts.append(tab.astype(np.float32))

        X = np.hstack(parts).astype(np.float32) if parts else np.zeros((1, 0), dtype=np.float32)
        return X

    def _build_feature_row(self, conversation: list[dict]) -> tuple[np.ndarray, Optional[dict]]:
        extraction = None

        # Step 1: get text + optional LLM extraction
        guest_text = extract_guest_text(conversation)
        if self.llm_extractor is not None:
            extraction = self.llm_extractor.extract(conversation)
            if extraction.get("guest_text", "").strip():
                guest_text = extraction["guest_text"]

        X = self._build_feature_row_from_guest_text_and_extraction(
            guest_text=guest_text,
            extraction=extraction,
        )
        return X, extraction

    def detect(self, conversation: list[dict], include_evidence: bool = True) -> list[dict]:
        """Detect signals in a conversation."""
        X, extraction = self._build_feature_row(conversation)
        return self._score_feature_row(X, extraction=extraction, include_evidence=include_evidence)

    def detect_from_extraction(
        self,
        extraction: dict,
        include_evidence: bool = True,
    ) -> list[dict]:
        """
        Score directly from a canonical extraction payload.

        Useful for explicit Extractor->Validator->Scorer workflows where
        extraction is already computed by a separate agent.
        """
        guest_text = extraction.get("guest_text", "")
        X = self._build_feature_row_from_guest_text_and_extraction(
            guest_text=guest_text,
            extraction=extraction,
        )
        return self._score_feature_row(X, extraction=extraction, include_evidence=include_evidence)

    def _score_feature_row(
        self,
        X: np.ndarray,
        extraction: Optional[dict],
        include_evidence: bool,
    ) -> list[dict]:
        """Run all category models against a prepared feature row."""
        detections = []
        for cat, model in self.models.items():
            prob = float(model.predict_proba(X)[0, 1])
            row = {
                "category": cat,
                "confidence": round(prob, 4),
                "triggered": prob >= self.threshold,
            }
            if include_evidence and extraction is not None:
                sig = extraction.get("signals", {}).get(cat, {})
                row["evidence"] = sig.get("evidence", "")
                row["extract_confidence"] = round(float(sig.get("confidence", 0.0)), 4)
            detections.append(row)

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def detect_triggered(self, conversation: list[dict]) -> list[dict]:
        """Return only signals that exceed the confidence threshold."""
        return [d for d in self.detect(conversation) if d["triggered"]]

    def detect_batch(self, conversations: list[list[dict]]) -> list[list[dict]]:
        """Run detection on multiple conversations (iterative for LLM compatibility)."""
        all_results = []
        for conv in conversations:
            all_results.append(self.detect(conv))
        return all_results


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

_DEMO_CASES = [
    {
        "description": "Intent + Value",
        "conversation": [
            {"role": "host", "content": "Hey Marco! Great to hear from you. What can we do for you?"},
            {"role": "guest", "content": "Hey! I'm thinking about coming in March with my group of eight. We usually book a penthouse suite."},
            {"role": "host", "content": "Absolutely, we'd love to have you back! Any special requests?"},
            {"role": "guest", "content": "Just want to make sure the baccarat tables are open late. Money's not an issue."},
        ],
    },
    {
        "description": "Competitive + Negative Sentiment",
        "conversation": [
            {"role": "host", "content": "Hi Lisa! How's everything going?"},
            {"role": "guest", "content": "Honestly, I was a bit disappointed with my last stay. Wynn offered me a comp suite last week and I'm reconsidering where to go."},
        ],
    },
    {
        "description": "Life Event",
        "conversation": [
            {"role": "host", "content": "Hi Sarah! Planning anything special?"},
            {"role": "guest", "content": "Actually yes — it's our 25th anniversary next month and I'd love to do something really memorable."},
        ],
    },
    {
        "description": "Neutral (no signals)",
        "conversation": [
            {"role": "host", "content": "Good morning! Is there anything I can help with today?"},
            {"role": "guest", "content": "Just calling to confirm my reservation number. That's all."},
        ],
    },
]


def _print_detection(tc: dict, results: list[dict]) -> None:
    width = 62
    print(f"\n{'═' * width}")
    print(f"  {tc['description']}")
    print(f"{'─' * width}")
    for r in results:
        bar = "█" * int(r["confidence"] * 20)
        flag = "✓" if r["triggered"] else "·"
        print(f"  [{flag}] {r['category']:15s}  {r['confidence']:.3f}  {bar}")
    print(f"{'═' * width}")


if __name__ == "__main__":
    print("Casino Signal Detector — inference demo\n")
    detector = SignalDetector(use_llm_extractor=False)

    for tc in _DEMO_CASES:
        results = detector.detect(tc["conversation"])
        _print_detection(tc, results)
