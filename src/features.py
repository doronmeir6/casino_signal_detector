"""
Feature extraction for casino signal detection.

Produces per-conversation feature vectors by combining:
  1. Sentence-transformer embeddings of all guest messages (384-dim, L2-normalised)
  2. Handcrafted keyword features — casino vocabulary, negation, intensity (14-dim)

Total feature vector: 398 dimensions.

Typical usage:
    from src.features import FeatureExtractor, load_dataset, build_features

    extractor = FeatureExtractor()
    texts, labels, records = load_dataset("data/raw/conversations.json")
    X = extractor.transform(texts)
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm

SIGNAL_CATEGORIES = ["intent", "value", "sentiment", "life_event", "competitive"]

# ---------------------------------------------------------------------------
# Keyword dictionaries — deliberately broad to capture paraphrases
# ---------------------------------------------------------------------------
KEYWORD_SIGNALS: dict[str, list[str]] = {
    "intent": [
        "book", "reserve", "reservation", "plan", "planning", "visit", "trip",
        "stay", "come", "coming", "schedule", "want to", "looking to",
        "thinking about", "need to", "dining", "restaurant", "table", "room",
        "suite", "show", "ticket", "check in", "interested in",
    ],
    "value": [
        "suite", "penthouse", "villa", "high roller", "vip", "baccarat",
        "blackjack", "budget", "spend", "money", "thousand", "k a hand",
        "group", "people", "premium", "luxury", "upgrade", "no limit",
        "no problem", "not an issue", "first class", "private",
    ],
    "sentiment": [
        "amazing", "great", "excellent", "love", "loved", "fantastic",
        "wonderful", "terrible", "disappointed", "awful", "unhappy", "upset",
        "frustrated", "incredible", "perfect", "let down", "expected more",
        "couldn't be happier", "not happy", "best experience", "worst",
        "blown away", "very happy", "really enjoyed",
    ],
    "life_event": [
        "anniversary", "birthday", "promotion", "wedding", "engaged",
        "engagement", "graduation", "retire", "retiring", "honeymoon",
        "celebrate", "celebrating", "special occasion", "milestone", "married",
        "new job", "new baby", "newborn", "proposal", "bachelorette", "bachelor",
    ],
    "competitive": [
        "wynn", "cosmo", "cosmopolitan", "mgm", "bellagio", "caesars",
        "palazzo", "venetian", "aria", "mandalay", "harrahs", "encore",
        "other casino", "another property", "offered me", "gave me",
        "treated me", "better deal", "free room", "comped", "comp from",
        "usually go to", "normally stay at", "first choice was",
    ],
}

_NEGATION_WORDS = frozenset(
    ["not", "no", "never", "don't", "doesn't", "didn't", "won't",
     "can't", "couldn't", "haven't", "isn't", "wasn't", "aren't"]
)
_INTENSITY_WORDS = frozenset(
    ["very", "really", "extremely", "absolutely", "totally", "always",
     "definitely", "certainly", "incredibly", "quite", "especially",
     "completely", "utterly", "just", "so"]
)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def extract_guest_text(conversation: list[dict]) -> str:
    """Concatenate all guest turns into a single string."""
    return " ".join(
        msg["content"]
        for msg in conversation
        if msg.get("role") == "guest"
    )


def keyword_features(text: str) -> np.ndarray:
    """
    Returns a 14-dimensional keyword feature vector.

    Layout (per signal category, 5 cats × 2 = 10 dims):
      [keyword_count_norm, negation_flag]   × 5 categories

    Global features (4 dims):
      [word_count_norm, sentence_count_norm, negation_density, intensity_density]
    """
    text_lower = text.lower()
    words = text_lower.split()
    n_words = max(len(words), 1)

    features: list[float] = []

    for cat in SIGNAL_CATEGORIES:
        keywords = KEYWORD_SIGNALS[cat]
        count = sum(1 for kw in keywords if kw in text_lower)

        # Check if any keyword is negated (negation within 4 words before keyword)
        negated = 0.0
        for kw in keywords:
            idx = text_lower.find(kw)
            if idx > 0:
                window = text_lower[max(0, idx - 35) : idx].split()
                if any(w in _NEGATION_WORDS for w in window[-4:]):
                    negated = 1.0
                    break

        features.append(min(count, 10) / 10.0)
        features.append(negated)

    # Global features
    n_sentences = max(
        text.count(".") + text.count("!") + text.count("?"), 1
    )
    neg_count = sum(1 for w in words if w in _NEGATION_WORDS)
    int_count = sum(1 for w in words if w in _INTENSITY_WORDS)

    features.extend([
        min(n_words, 300) / 300.0,
        min(n_sentences, 15) / 15.0,
        min(neg_count, 6) / 6.0,
        min(int_count, 6) / 6.0,
    ])

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Sentence-transformer embeddings + keyword features.

    Feature dimensions:
      • all-MiniLM-L6-v2 embeddings: 384
      • keyword features:            14
      ─────────────────────────────────
      Total:                        398
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._encoder = None

    # Lazy-load the heavy model
    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading sentence transformer: {self.model_name}")
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def transform(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of guest-text strings into feature vectors.

        Returns np.ndarray of shape (n_samples, 398), dtype float32.
        """
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        embeddings = normalize(embeddings).astype(np.float32)

        kw = np.vstack([keyword_features(t) for t in tqdm(texts, desc="Keyword features")])
        return np.hstack([embeddings, kw])

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_name": self.model_name}, path)
        print(f"FeatureExtractor config saved → {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        data = joblib.load(path)
        return cls(model_name=data["model_name"])


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(
    path: str,
) -> tuple[list[str], pd.DataFrame, list[dict]]:
    """
    Load the raw conversations JSON.

    Returns
    -------
    texts     : guest text per conversation (n_samples,)
    labels_df : binary label DataFrame (n_samples × 5)
    records   : original JSON records for reference
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    texts = [extract_guest_text(r["conversation"]) for r in records]
    labels = pd.DataFrame([r["labels"] for r in records], columns=SIGNAL_CATEGORIES)
    return texts, labels, records


# ---------------------------------------------------------------------------
# End-to-end feature build
# ---------------------------------------------------------------------------

def build_features(
    data_path: str = "data/raw/conversations.json",
    features_path: str = "data/processed/features.npz",
    labels_path: str = "data/processed/labels.csv",
    extractor_path: str = "models/feature_extractor.joblib",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load raw data → extract features → save to disk. Returns (X, labels)."""
    print(f"Loading conversations from {data_path}")
    texts, labels, records = load_dataset(data_path)
    print(f"  {len(texts)} conversations loaded")
    print(f"  Label distribution:\n{labels.sum().to_string()}\n")

    extractor = FeatureExtractor()
    X = extractor.transform(texts)

    Path(features_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(features_path, X=X)
    labels.to_csv(labels_path, index=False)
    extractor.save(extractor_path)

    print(f"\nFeature matrix saved: {X.shape} → {features_path}")
    print(f"Labels saved → {labels_path}")
    return X, labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build feature matrix from conversations")
    parser.add_argument("--data", type=str, default="data/raw/conversations.json")
    parser.add_argument("--features", type=str, default="data/processed/features.npz")
    parser.add_argument("--labels", type=str, default="data/processed/labels.csv")
    parser.add_argument("--extractor", type=str, default="models/feature_extractor.joblib")
    args = parser.parse_args()

    build_features(
        data_path=args.data,
        features_path=args.features,
        labels_path=args.labels,
        extractor_path=args.extractor,
    )
