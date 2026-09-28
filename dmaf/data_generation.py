"""
Synthetic multimodal dataset generator.

There is no bundled real-world QR image / URL corpus in this environment,
so this module generates a *labelled synthetic dataset* whose statistical
structure mimics the malicious-vs-benign QR code problem described in the
paper: malicious samples tend to have younger domains, IP-based or
obfuscated URLs, suspicious payload text, invalid certificates, and
atypical scan behaviour, all with realistic noise so the problem isn't
trivially separable.

Swap `generate_dataset()` for a loader over the real Kaggle
"Benign and Malicious QR Codes" dataset (or your own data) to run DMAF on
real data — see README.md.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import List

import numpy as np

SUSPICIOUS_KEYWORDS = [
    "verify", "login", "secure", "update", "confirm", "wallet-reset",
    "bonus", "urgent", "refund", "kyc",
]

BENIGN_DOMAINS = [
    "paytm.com", "phonepe.com", "gpay.google.com", "amazon.in",
    "flipkart.com", "irctc.co.in", "sbi.co.in", "hdfcbank.com",
]


@dataclass
class RawSample:
    qr_image_features: np.ndarray   # simulated deep-CNN embedding for the QR image, shape (32,)
    payload_text: str               # decoded QR payload / URL text
    url: str                        # extracted URL
    domain_age_days: float
    cert_valid: int                 # 0/1
    tx_amount: float
    prior_tx_count: int
    scan_to_pay_latency_s: float    # behavioral: time between scan and payment confirm
    tap_pressure_var: float         # behavioral: variance in touch pressure
    device_trust_score: float       # behavioral: 0..1 score from device fingerprinting
    label: int                      # 0 = benign, 1 = malicious


def _random_string(n: int, alphabet: str = string.ascii_lowercase + string.digits) -> str:
    return "".join(random.choice(alphabet) for _ in range(n))


def _make_url(malicious: bool) -> str:
    if not malicious:
        domain = random.choice(BENIGN_DOMAINS)
        path = _random_string(random.randint(4, 10))
        return f"https://{domain}/pay/{path}"
    # malicious URLs: IP-based, extra subdomains, suspicious keywords, http (no TLS)
    style = random.choice(["ip", "lookalike", "keyword", "shortener"])
    if style == "ip":
        ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
        scheme = random.choice(["http", "https"])
        return f"{scheme}://{ip}/{random.choice(SUSPICIOUS_KEYWORDS)}"
    if style == "lookalike":
        base = random.choice(BENIGN_DOMAINS).split(".")[0]
        return f"http://{base}-{_random_string(4)}.{_random_string(3)}.com/{random.choice(SUSPICIOUS_KEYWORDS)}"
    if style == "keyword":
        return f"http://{_random_string(6)}.com/{random.choice(SUSPICIOUS_KEYWORDS)}-{_random_string(3)}"
    return f"http://bit.ly/{_random_string(7)}"


def _simulate_image_embedding(malicious: bool, rng: np.random.Generator) -> np.ndarray:
    """
    Stand-in for a real vision encoder's (e.g. ViT / CNN) output embedding
    for the QR code image. Malicious samples are drawn from a shifted
    distribution to represent subtle visual artifacts (pixel tampering,
    logo overlay, module misalignment) that a real vision model would
    pick up on.
    """
    mean = 0.35 if malicious else 0.0
    return rng.normal(loc=mean, scale=1.0, size=32)


def generate_dataset(n_samples: int = 4000, malicious_ratio: float = 0.5,
                      seed: int = 42) -> List[RawSample]:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    samples: List[RawSample] = []

    for _ in range(n_samples):
        malicious = random.random() < malicious_ratio
        # 6% of URLs are generated with the "wrong" style relative to the true label,
        # so the problem isn't trivially separable on URL style alone.
        url_style_malicious = malicious if random.random() > 0.06 else not malicious
        url = _make_url(url_style_malicious)
        payload_text = url if random.random() < 0.7 else f"upi://pay?url={url}&note={_random_string(6)}"

        domain_age_days = (rng.exponential(scale=150) if malicious
                            else rng.normal(loc=1000, scale=500))
        domain_age_days = float(max(0, domain_age_days))

        cert_valid = int(rng.random() > (0.65 if malicious else 0.08))

        tx_amount = float(np.clip(rng.normal(loc=1300 if malicious else 700, scale=900), 1, 50000))
        prior_tx_count = int(max(0, rng.normal(loc=3 if malicious else 35, scale=8)))

        scan_to_pay_latency_s = float(max(0.1, rng.normal(loc=1.8 if malicious else 3.7, scale=1.8)))
        tap_pressure_var = float(max(0.0, rng.normal(loc=0.28 if malicious else 0.15, scale=0.1)))
        device_trust_score = float(np.clip(rng.normal(loc=0.45 if malicious else 0.78, scale=0.2), 0, 1))

        samples.append(RawSample(
            qr_image_features=_simulate_image_embedding(malicious, rng),
            payload_text=payload_text,
            url=url,
            domain_age_days=domain_age_days,
            cert_valid=cert_valid,
            tx_amount=tx_amount,
            prior_tx_count=prior_tx_count,
            scan_to_pay_latency_s=scan_to_pay_latency_s,
            tap_pressure_var=tap_pressure_var,
            device_trust_score=device_trust_score,
            label=int(malicious),
        ))

    random.shuffle(samples)
    return samples
