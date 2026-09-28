"""
Cross-modal attention fusion — Sec. III-C of the paper (Eq. 13-16).

For each sample, every modality embedding attends to every other
modality embedding via scaled dot-product attention, producing a
per-modality context vector that is then combined with a confidence-
weighted sum ("adaptive modality weighting", Eq. 15) into a single
fused representation.

This is deliberately implemented with plain NumPy (no learned attention
weights beyond the deterministic scaled dot-product) so the whole
pipeline runs anywhere without a deep learning framework installed. The
`confidences` argument is where an adaptive/learned weighting network
would plug in.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

MODALITY_ORDER = ["I", "T", "U", "M", "B"]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def cross_modal_attention(encoded: Dict[str, np.ndarray]) -> np.ndarray:
    """
    encoded: modality name -> (n_samples, d) embedding matrix, same d for all modalities.
    Returns: (n_samples, n_modalities * d) fused representation (attention context
             for each modality concatenated, matching Eq. 16's F = concat(...)).
    """
    names = [m for m in MODALITY_ORDER if m in encoded]
    n_samples, d = encoded[names[0]].shape
    stacked = np.stack([encoded[m] for m in names], axis=1)  # (n, n_mod, d)
    scale = np.sqrt(d)

    # scaled dot-product attention scores between every pair of modalities, per sample
    scores = np.einsum("nqd,nkd->nqk", stacked, stacked) / scale     # (n, n_mod, n_mod)
    weights = _softmax(scores, axis=-1)                              # Eq. 13
    context = np.einsum("nqk,nkd->nqd", weights, stacked)            # Eq. 14

    fused = context.reshape(n_samples, -1)                           # Eq. 16 (concat)
    return fused, weights, names


def modality_confidence_weights(encoded: Dict[str, np.ndarray]) -> np.ndarray:
    """
    A simple stand-in for Eq. 15's confidence score C_m: modalities whose
    embedding is further from the population mean (i.e. more "informative"
    / anomalous relative to typical traffic) get a higher weight.
    """
    names = [m for m in MODALITY_ORDER if m in encoded]
    dists = []
    for m in names:
        X = encoded[m]
        centroid = X.mean(axis=0, keepdims=True)
        dists.append(np.linalg.norm(X - centroid, axis=1))
    dists = np.stack(dists, axis=1)  # (n, n_mod)
    return _softmax(dists, axis=1)
