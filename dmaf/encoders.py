"""
Modality-specific encoders — Sec. III-B of the paper (Eq. 7-12).

Each modality is standardized and projected into a shared-dimensional
latent space by its own encoder, so that heterogeneous feature vectors
(different lengths / scales) become comparable embeddings that the
cross-modal attention fusion stage can operate over.

This reference implementation uses PCA-based linear encoders (fit per
modality) rather than a deep CNN / transformer / GNN, since those need
real images, pretrained language models, and a GPU. The encoder
interface below is the natural place to swap in:
  - I: a pretrained ViT/CNN feature extractor
  - T: a sentence-transformer / MobileBERT encoder
  - U: a graph or transformer-based domain-trust encoder
  - M, B: small MLPs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

LATENT_DIM = 8


@dataclass
class ModalityEncoder:
    name: str
    scaler: StandardScaler
    reducer: PCA

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.reducer.transform(Xs)


def fit_encoder(name: str, X: np.ndarray, latent_dim: int = LATENT_DIM) -> ModalityEncoder:
    n_components = min(latent_dim, X.shape[1])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    reducer = PCA(n_components=n_components, random_state=0).fit(Xs)
    return ModalityEncoder(name=name, scaler=scaler, reducer=reducer)


def fit_all_encoders(modality_matrices: Dict[str, np.ndarray]) -> Dict[str, ModalityEncoder]:
    return {name: fit_encoder(name, X) for name, X in modality_matrices.items()}


def encode_all(encoders: Dict[str, ModalityEncoder],
                modality_matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Encode every modality and pad to a common latent width so shapes align for attention."""
    encoded = {name: enc.transform(modality_matrices[name]) for name, enc in encoders.items()}
    width = max(arr.shape[1] for arr in encoded.values())
    for name, arr in encoded.items():
        if arr.shape[1] < width:
            pad = np.zeros((arr.shape[0], width - arr.shape[1]))
            encoded[name] = np.concatenate([arr, pad], axis=1)
    return encoded
