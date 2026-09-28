"""
Per-modality feature extraction.

Implements the preprocessing stage of the paper (Sec. III-A, Eq. 2-4):
each raw modality is turned into a fixed-length numeric vector before
being passed to the modality-specific encoders in `encoders.py`.

  I -> qr_image_features         (already a vector; see data_generation)
  T -> text_features()           lexical features of the decoded payload
  U -> url_features()            lexical / structural URL features
  M -> metadata_features()       transaction metadata
  B -> behavioral_features()     mobile behavioral signals
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

import numpy as np

from .data_generation import RawSample, SUSPICIOUS_KEYWORDS

_IP_RE = re.compile(r"^\w+://\d{1,3}(\.\d{1,3}){3}")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def url_features(url: str) -> np.ndarray:
    is_https = 1.0 if url.startswith("https://") else 0.0
    is_ip = 1.0 if _IP_RE.match(url) else 0.0
    length = float(len(url))
    num_digits = float(sum(ch.isdigit() for ch in url))
    num_special = float(sum(not ch.isalnum() for ch in url))
    subdomain_count = float(url.split("//")[-1].split("/")[0].count("."))
    keyword_hits = float(sum(k in url.lower() for k in SUSPICIOUS_KEYWORDS))
    entropy = _shannon_entropy(url)
    return np.array([
        is_https, is_ip, length, num_digits, num_special,
        subdomain_count, keyword_hits, entropy,
    ], dtype=np.float64)


def text_features(payload_text: str) -> np.ndarray:
    length = float(len(payload_text))
    digit_ratio = (sum(ch.isdigit() for ch in payload_text) / length) if length else 0.0
    keyword_hits = float(sum(k in payload_text.lower() for k in SUSPICIOUS_KEYWORDS))
    entropy = _shannon_entropy(payload_text)
    has_upi_scheme = 1.0 if payload_text.startswith("upi://") else 0.0
    return np.array([length, digit_ratio, keyword_hits, entropy, has_upi_scheme], dtype=np.float64)


def metadata_features(s: RawSample) -> np.ndarray:
    return np.array([
        s.domain_age_days,
        float(s.cert_valid),
        s.tx_amount,
        float(s.prior_tx_count),
    ], dtype=np.float64)


def behavioral_features(s: RawSample) -> np.ndarray:
    return np.array([
        s.scan_to_pay_latency_s,
        s.tap_pressure_var,
        s.device_trust_score,
    ], dtype=np.float64)


def build_modality_matrices(samples: List[RawSample]):
    """Returns a dict of modality name -> (n_samples, n_features) matrix, plus labels."""
    I = np.stack([s.qr_image_features for s in samples])
    T = np.stack([text_features(s.payload_text) for s in samples])
    U = np.stack([url_features(s.url) for s in samples])
    M = np.stack([metadata_features(s) for s in samples])
    B = np.stack([behavioral_features(s) for s in samples])
    y = np.array([s.label for s in samples], dtype=np.int64)
    return {"I": I, "T": T, "U": U, "M": M, "B": B}, y
