"""
Simple constant-probability classifier used when a label has only one class.
"""

from __future__ import annotations

import numpy as np


class ConstantProbabilityModel:
    """
    API-compatible minimal model for binary `predict_proba`.

    If positive_class=1, returns probability [0.0, 1.0] for every sample.
    If positive_class=0, returns probability [1.0, 0.0] for every sample.
    """

    def __init__(self, positive_class: int):
        self.positive_class = int(positive_class)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        n = len(X)
        if self.positive_class == 1:
            return np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (n, 1))
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (n, 1))
