"""
Explainable risk assessment — Sec. III-D (the "explainable attention
interpretation module") and the paper's use of ablation-style
explainability alongside attention weights.

Two complementary views are produced:
  1. Attention-weight summary: how much each modality attended to each
     other modality on average (from fusion.cross_modal_attention).
  2. Modality-ablation importance: how much the classifier's confidence
     shifts, on average, when a given modality's contribution to the
     fused vector is zeroed out — a simple, model-agnostic proxy for
     "which modality mattered most for this decision".
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .classifier import ThreatClassifier


def attention_summary(weights: np.ndarray, names: List[str]) -> Dict[str, float]:
    """Average attention each modality *receives* from all others, across the dataset."""
    avg = weights.mean(axis=0)          # (n_mod, n_mod)
    received = avg.mean(axis=0)         # average column -> how much attention each modality gets
    return {name: float(received[i]) for i, name in enumerate(names)}


def modality_ablation_importance(clf: ThreatClassifier, fused: np.ndarray,
                                  names: List[str], d_per_modality: int) -> Dict[str, float]:
    base_conf = clf.predict_proba(fused)
    importances = {}
    for i, name in enumerate(names):
        ablated = fused.copy()
        start, end = i * d_per_modality, (i + 1) * d_per_modality
        ablated[:, start:end] = 0.0
        ablated_conf = clf.predict_proba(ablated)
        importances[name] = float(np.mean(np.abs(base_conf - ablated_conf)))
    total = sum(importances.values()) or 1.0
    return {k: v / total for k, v in importances.items()}  # normalize to sum to 1
