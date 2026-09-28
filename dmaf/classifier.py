"""
Deep threat classification — Sec. III-D of the paper (Eq. 17-20).

Wraps an MLP classifier (stand-in for the paper's "hybrid deep
authentication classifier ... residual fully connected layers") that
takes the fused multimodal representation and outputs a
legitimate/malicious decision with a confidence score, matching
Eq. 17-18 and the secure-decision threshold of Eq. 23.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)


@dataclass
class ThreatClassifier:
    model: MLPClassifier
    threshold: float = 0.5  # tau, the adaptive authentication threshold (Eq. 23)

    @classmethod
    def fit(cls, X_train: np.ndarray, y_train: np.ndarray, seed: int = 0) -> "ThreatClassifier":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,               # L2 regularization, mirrors Eq. 24's lambda ||theta||^2
            max_iter=300,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        return cls(model=model)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]  # confidence C = max(Y), Eq. 22

    def decide(self, X: np.ndarray) -> np.ndarray:
        """Secure authentication decision: malicious if confidence >= tau (Eq. 23)."""
        return (self.predict_proba(X) >= self.threshold).astype(int)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
