"""
Improved fitness functions for camouflage evaluation.

Enhancements:
- Color: global + local (patch-based)
- Texture: frequency-domain (radial FFT spectrum)
- Disruption:
    • edge misalignment (low correlation = good)
    • edge orientation divergence (distribution mismatch = good)
- Multi-scale evaluation
- Nonlinear aggregation (geometric mean)

All inputs: uint8 BGR NumPy arrays.
Outputs in [0, 1].
"""

from __future__ import annotations
import numpy as np
import cv2

EPS = 1e-8


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def _resize_match(a, b):
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return a, b


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _safe_norm(x):
    s = np.sum(x)
    return x / (s + EPS)


# ─────────────────────────────────────────────────────────────
# Color (global + local)
# ─────────────────────────────────────────────────────────────

def _lab_hist(img, bins=32):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hists = []
    for ch in range(3):
        h = cv2.calcHist([lab], [ch], None, [bins], [0, 256])
        h = _safe_norm(h.flatten())
        hists.append(h)
    return hists


def _hist_intersection(h1, h2):
    return float(np.sum(np.minimum(h1, h2)))


def color_score(pattern, background, patch_size=64, samples=8):
    pattern, background = _resize_match(pattern, background)

    # global
    h_p = _lab_hist(pattern)
    h_b = _lab_hist(background)
    global_score = np.mean([_hist_intersection(p, b) for p, b in zip(h_p, h_b)])

    # local patches
    H, W = pattern.shape[:2]
    local_scores = []

    for _ in range(samples):
        x = np.random.randint(0, max(1, W - patch_size))
        y = np.random.randint(0, max(1, H - patch_size))

        p_patch = pattern[y:y+patch_size, x:x+patch_size]
        b_patch = background[y:y+patch_size, x:x+patch_size]

        hp = _lab_hist(p_patch)
        hb = _lab_hist(b_patch)

        local_scores.append(np.mean([
            _hist_intersection(a, b) for a, b in zip(hp, hb)
        ]))

    local_score = float(np.mean(local_scores)) if local_scores else global_score

    return 0.5 * global_score + 0.5 * local_score


# ─────────────────────────────────────────────────────────────
# Texture (frequency-domain)
# ─────────────────────────────────────────────────────────────

def _fft_spectrum(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    return np.log1p(mag)


def _radial_profile(mag):
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)

    r_max = min(cx, cy)
    profile = np.zeros(r_max)
    counts = np.zeros(r_max)

    for i in range(h):
        for j in range(w):
            ri = r[i, j]
            if ri < r_max:
                profile[ri] += mag[i, j]
                counts[ri] += 1

    profile /= (counts + EPS)
    return _safe_norm(profile)


def texture_score(pattern, background):
    pattern, background = _resize_match(pattern, background)

    p = _to_gray(pattern)
    b = _to_gray(background)

    sp = _fft_spectrum(p)
    sb = _fft_spectrum(b)

    rp = _radial_profile(sp)
    rb = _radial_profile(sb)

    return _hist_intersection(rp, rb)


# ─────────────────────────────────────────────────────────────
# Disruption (edge + orientation)
# ─────────────────────────────────────────────────────────────

def _edge_map(gray):
    return cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0


def _edge_orientation_hist(gray, bins=16):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)  # [-pi, pi]

    # map to [0, pi] (orientation, not direction)
    ang = np.abs(ang)

    # weight by gradient magnitude
    hist, _ = np.histogram(
        ang,
        bins=bins,
        range=(0, np.pi),
        weights=mag
    )

    return _safe_norm(hist)


def disruption_score(pattern, background):
    pattern, background = _resize_match(pattern, background)

    p_gray = _to_gray(pattern)
    b_gray = _to_gray(background)

    # --- Edge misalignment (NCC inverted) ---
    p_edge = _edge_map(p_gray)
    b_edge = _edge_map(b_gray)

    p_edge -= p_edge.mean()
    b_edge -= b_edge.mean()

    ps = p_edge.std()
    bs = b_edge.std()

    if ps < EPS or bs < EPS:
        corr_score = 0.5
    else:
        corr = float((p_edge * b_edge).mean() / (ps * bs))
        corr_score = 1.0 - (corr + 1.0) / 2.0  # invert

    # --- Orientation divergence ---
    hp = _edge_orientation_hist(p_gray)
    hb = _edge_orientation_hist(b_gray)

    orient_similarity = _hist_intersection(hp, hb)
    orient_score = 1.0 - orient_similarity  # divergence = good

    # --- Combine ---
    # weight misalignment more than orientation
    score = 0.7 * corr_score + 0.3 * orient_score

    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# Multi-scale
# ─────────────────────────────────────────────────────────────

def _multi_scale(metric_fn, pattern, background, scales=(1.0, 0.5, 0.25)):
    scores = []

    for s in scales:
        if s != 1.0:
            p = cv2.resize(pattern, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            b = cv2.resize(background, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        else:
            p, b = pattern, background

        try:
            scores.append(metric_fn(p, b))
        except Exception:
            scores.append(0.5)

    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
# Composite
# ─────────────────────────────────────────────────────────────

def composite_fitness(
    pattern: np.ndarray,
    background: np.ndarray,
    weights: dict | None = None,
) -> dict:

    if weights is None:
        weights = {
            "color": 1.0,
            "texture": 1.0,
            "disruption": 1.0,
        }

    # ensure BGR
    if pattern.ndim == 3 and pattern.shape[2] == 4:
        pattern = cv2.cvtColor(pattern, cv2.COLOR_BGRA2BGR)

    pattern, background = _resize_match(pattern, background)

    # multi-scale metrics
    c = _multi_scale(color_score, pattern, background)
    t = _multi_scale(texture_score, pattern, background)
    d = _multi_scale(disruption_score, pattern, background)

    wc = weights.get("color", 1.0)
    wt = weights.get("texture", 1.0)
    wd = weights.get("disruption", 1.0)

    # geometric mean aggregation
    total = (
        (c ** wc) *
        (t ** wt) *
        (d ** wd)
    ) ** (1.0 / (wc + wt + wd + EPS))

    total = float(np.clip(total, 0.0, 1.0))

    return {
        "color": float(c),
        "texture": float(t),
        "disruption": float(d),
        "total": total,
    }
