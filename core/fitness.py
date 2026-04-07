"""
Fitness functions.  All inputs are uint8 BGR NumPy arrays (OpenCV native).
Returns scores in [0, 1] where 1 = perfect camouflage.

scikit-image is used for SSIM if available; falls back to a pure-numpy
normalised cross-correlation so the evolution thread never crashes on import.
"""
from __future__ import annotations
import numpy as np
import cv2
from config.defaults import EVOLUTION

# ── optional skimage ──────────────────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as _ssim_fn
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False


# ── individual metrics ────────────────────────────────────────────────────────

def color_score(pattern: np.ndarray, background: np.ndarray) -> float:
    pat_lab = cv2.cvtColor(pattern,    cv2.COLOR_BGR2LAB)
    bg_lab  = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
    scores  = []
    for ch in range(3):
        h_p = cv2.calcHist([pat_lab], [ch], None, [64], [0, 256])
        h_b = cv2.calcHist([bg_lab],  [ch], None, [64], [0, 256])
        cv2.normalize(h_p, h_p)
        cv2.normalize(h_b, h_b)
        scores.append(cv2.compareHist(h_p, h_b, cv2.HISTCMP_INTERSECT))
    return float(np.mean(scores))


def texture_score(pattern: np.ndarray, background: np.ndarray) -> float:
    p_gray = cv2.cvtColor(pattern,    cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    if p_gray.shape != b_gray.shape:
        b_gray = cv2.resize(b_gray, (p_gray.shape[1], p_gray.shape[0]))

    if _HAVE_SKIMAGE:
        try:
            val, _ = _ssim_fn(p_gray, b_gray, full=True)
            return (float(val) + 1.0) / 2.0
        except Exception:
            pass

    # Numpy fallback: normalised cross-correlation
    pf = p_gray.astype(np.float32)
    bf = b_gray.astype(np.float32)
    pf -= pf.mean(); pf_s = pf.std()
    bf -= bf.mean(); bf_s = bf.std()
    if pf_s < 1e-6 or bf_s < 1e-6:
        return 0.5
    ncc = float((pf * bf).mean()) / (pf_s * bf_s)
    return (np.clip(ncc, -1.0, 1.0) + 1.0) / 2.0


def disruption_score(pattern: np.ndarray, background: np.ndarray) -> float:
    def edge_energy(img):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(edges.mean())

    diff = abs(edge_energy(pattern) - edge_energy(background))
    return max(0.0, 1.0 - diff / 255.0)


def neural_score(pattern: np.ndarray, background: np.ndarray) -> float:
    """STUB – MobileNetV2 feature similarity. Requires torch (optional)."""
    return 0.5


# ── composite ─────────────────────────────────────────────────────────────────

def composite_fitness(
    pattern: np.ndarray,
    background: np.ndarray,
    weights: dict | None = None,
    use_neural: bool = False,
) -> dict:
    if weights is None:
        weights = EVOLUTION["fitness_weights"]

    # Ensure pattern is 3-channel BGR (drop alpha if present)
    if pattern.ndim == 3 and pattern.shape[2] == 4:
        pattern = cv2.cvtColor(pattern, cv2.COLOR_BGRA2BGR)

    h, w = background.shape[:2]
    if pattern.shape[:2] != (h, w):
        pattern = cv2.resize(pattern, (w, h))

    try:
        c = color_score(pattern, background)
    except Exception:
        c = 0.5
    try:
        t = texture_score(pattern, background)
    except Exception:
        t = 0.5
    try:
        d = disruption_score(pattern, background)
    except Exception:
        d = 0.5

    total = np.clip(
        weights["color"] * c + weights["texture"] * t + weights["disruption"] * d,
        0.0, 1.0,
    )
    result = {"color": c, "texture": t, "disruption": d, "total": float(total)}
    if use_neural:
        result["neural"] = neural_score(pattern, background)
    return result
