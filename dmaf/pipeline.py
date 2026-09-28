"""End-to-end DMAF pipeline: data -> features -> encoders -> fusion -> classifier -> explain."""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np
from sklearn.model_selection import train_test_split

from .data_generation import generate_dataset
from .features import build_modality_matrices
from .encoders import fit_all_encoders, encode_all
from .fusion import cross_modal_attention, modality_confidence_weights
from .classifier import ThreatClassifier, evaluate
from .explain import attention_summary, modality_ablation_importance


def run(n_samples: int = 4000, test_size: float = 0.25, seed: int = 42, verbose: bool = True) -> dict:
    t0 = time.time()

    if verbose:
        print(f"[1/6] Generating synthetic multimodal dataset (n={n_samples}) ...")
    samples = generate_dataset(n_samples=n_samples, seed=seed)
    modality_matrices, y = build_modality_matrices(samples)

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    train_matrices = {m: X[idx_train] for m, X in modality_matrices.items()}
    test_matrices = {m: X[idx_test] for m, X in modality_matrices.items()}
    y_train, y_test = y[idx_train], y[idx_test]

    if verbose:
        print("[2/6] Fitting per-modality encoders (hierarchical feature representation) ...")
    encoders = fit_all_encoders(train_matrices)
    encoded_train = encode_all(encoders, train_matrices)
    encoded_test = encode_all(encoders, test_matrices)

    if verbose:
        print("[3/6] Running cross-modal attention fusion ...")
    fused_train, weights_train, names = cross_modal_attention(encoded_train)
    fused_test, weights_test, _ = cross_modal_attention(encoded_test)
    d_per_modality = fused_train.shape[1] // len(names)

    if verbose:
        print("[4/6] Training deep threat classifier ...")
    clf = ThreatClassifier.fit(fused_train, y_train, seed=seed)

    if verbose:
        print("[5/6] Evaluating on held-out test set ...")
    y_prob = clf.predict_proba(fused_test)
    y_pred = clf.decide(fused_test)
    metrics = evaluate(y_test, y_pred, y_prob)

    inference_start = time.time()
    _ = clf.predict_proba(fused_test)
    inference_ms_per_sample = (time.time() - inference_start) / max(len(fused_test), 1) * 1000

    if verbose:
        print("[6/6] Computing explainability report ...")
    att_summary = attention_summary(weights_test, names)
    ablation_importance = modality_ablation_importance(clf, fused_test, names, d_per_modality)

    total_time = time.time() - t0

    report = {
        "dataset": {"n_samples": n_samples, "n_train": len(idx_train), "n_test": len(idx_test)},
        "metrics": metrics,
        "latency": {
            "total_pipeline_seconds": round(total_time, 3),
            "inference_ms_per_sample": round(inference_ms_per_sample, 4),
        },
        "explainability": {
            "modality_order": names,
            "average_attention_received": att_summary,
            "modality_ablation_importance": ablation_importance,
        },
    }

    if verbose:
        print("\n=== DMAF Results ===")
        print(json.dumps(report, indent=2))

    return report


if __name__ == "__main__":
    run()
