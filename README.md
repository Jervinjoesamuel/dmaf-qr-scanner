# DMAF — Deep Multimodal Authentication Framework
### Reference implementation for *"Deep Multimodal Authentication Framework for Detecting Malicious QR Codes in Mobile Payment Systems"*

This is a runnable, from-scratch Python implementation of the DMAF pipeline
described in the paper, mapped stage-for-stage onto the five phases in the
paper's Figure 1:

| Paper stage | Module | Paper section |
|---|---|---|
| 1. Multimodal QR code acquisition & preprocessing | `dmaf/data_generation.py`, `dmaf/features.py` | III-A, Eq. 1-4 |
| 2. Hierarchical feature representation learning | `dmaf/encoders.py` | III-B, Eq. 7-12 |
| 3. Cross-modal attention fusion | `dmaf/fusion.py` | III-C, Eq. 13-16 |
| 4. Deep threat classification & explainable risk assessment | `dmaf/classifier.py`, `dmaf/explain.py` | III-D, Eq. 17-20 |
| 5. Adaptive continual learning & secure decision optimization | `dmaf/classifier.py` (threshold `tau`), `dmaf/pipeline.py` | III-E, Eq. 21-24 |

The five modalities from Eq. (1), `D = {I, T, U, M, B}`, are all represented:

- **I** — QR code image features
- **T** — decoded QR text payload
- **U** — URL / domain information
- **M** — transaction metadata
- **B** — mobile behavioral features

## Why synthetic data?

The paper trains on a 200,000-image Kaggle "Benign and Malicious QR Codes"
dataset plus real transaction/behavioral logs, none of which are available
here. So this project ships a **synthetic multimodal data generator**
(`dmaf/data_generation.py`) that produces QR-code-style samples with the
same *kind* of signal the paper describes — younger domain age, invalid
certificates, IP-based/obfuscated URLs, suspicious payload keywords, and
atypical scan behavior for malicious samples — with realistic noise and a
small amount of label-inconsistent data so the task isn't trivial.

This lets you run the **entire architecture end-to-end today**, and gives
you a clear seam to swap in real data later (see "Using real data" below).

## What's implemented vs. simplified

| Paper component | This implementation |
|---|---|
| Vision Transformer image encoder | Modality-specific PCA encoder (`encoders.py`) operating on a simulated image-embedding vector. Swap in a real pretrained ViT/CNN's output vector. |
| MobileBERT / contextual language encoder for text & URL | Handcrafted lexical features (entropy, keyword hits, HTTPS/IP checks, length) + PCA encoder. Swap in a sentence-transformer embedding. |
| Graph-based domain/transaction encoder | Numeric metadata features + PCA encoder. |
| Learned cross-modal attention (trained weights) | Deterministic scaled dot-product attention (Eq. 13) computed directly on encoder outputs — same math, but the projection isn't learned jointly with the classifier. |
| Hybrid residual FC threat classifier | `sklearn.neural_network.MLPClassifier` (two hidden layers, L2-regularized) — a real trained neural classifier. |
| Adaptive continual learning | Not implemented (would need an online/incremental training loop); the threshold `tau` and regularization objective from Eq. 23-24 are represented. |
| Explainable attention module | `dmaf/explain.py`: (a) average attention received per modality, (b) modality-ablation importance (zero out a modality, measure the shift in the classifier's confidence). |

## Quickstart

```bash
pip install -r requirements.txt
python main.py                        # runs with 4,000 synthetic samples
python main.py --n-samples 8000 --out report.json
```

Expected output: a JSON report with dataset size, held-out test metrics
(accuracy / precision / recall / F1 / ROC-AUC / confusion matrix),
latency, and the explainability summary. On the default synthetic data
this typically lands around 98-99% accuracy — comparable to, but not
directly compared against, the paper's reported 99.18% (that number
comes from a real 200k-image dataset, not this synthetic one).

## Project layout

```
dmaf_project/
├── main.py                  # CLI entry point
├── requirements.txt
├── README.md
└── dmaf/
    ├── data_generation.py   # Stage 1a: synthetic D = {I,T,U,M,B} generator
    ├── features.py          # Stage 1b: raw -> fixed-length feature vectors
    ├── encoders.py           # Stage 2: per-modality PCA encoders
    ├── fusion.py             # Stage 3: cross-modal attention fusion
    ├── classifier.py         # Stage 4: MLP threat classifier + metrics
    └── explain.py            # Stage 4: explainability (attention + ablation)
```

## Using real data

To move from the synthetic demo to a real dataset:

1. **Images (I):** run a pretrained CNN or ViT (e.g. via `torchvision` or
   `timm`) over your QR code images and use its penultimate-layer output
   as the feature vector, replacing `_simulate_image_embedding()` in
   `data_generation.py`.
2. **Text/URL (T, U):** either keep the handcrafted lexical features in
   `features.py` (they're real, not simulated) or add a pretrained
   language-model embedding (e.g. a sentence-transformer) alongside them.
3. **Metadata/behavioral (M, B):** point `RawSample` construction at your
   real transaction and device-telemetry fields.
4. Everything downstream (`encoders.py`, `fusion.py`, `classifier.py`,
   `explain.py`) works unchanged on any set of fixed-length modality
   feature matrices.

## Extending

- **Multi-class threat typing:** the paper's classifier distinguishes
  phishing, malware, payment fraud, fake-merchant identity, and
  credential harvesting. This demo is binary (benign/malicious); to
  extend, add a `threat_type` label to `RawSample` and switch
  `ThreatClassifier` to multi-class.
- **Learned attention:** replace the deterministic attention in
  `fusion.py` with a small trainable projection (e.g. a PyTorch
  `nn.MultiheadAttention` layer) trained jointly with the classifier.
- **Continual learning:** wrap `ThreatClassifier.fit` in an
  incremental-training loop (e.g. `partial_fit`) that periodically
  retrains on newly labelled samples without catastrophic forgetting.
