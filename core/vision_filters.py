"""
Animal Vision Filters – colour-transform approximations of non-human vision.

Each filter takes a BGR uint8 numpy array and returns a BGR uint8 array of
the same shape representing what the scene looks like to that animal.

Scientific basis
────────────────
Most mammalian game animals are dichromats (two cone types) rather than the
trichromats humans are.  Their colour perception can be approximated by
projecting the three-dimensional human colour space onto the two-dimensional
subspace available to that animal, then mapping back to a human-visible BGR
image.

The simulation matrices below are derived from:
  Viénot, Brettel & Mollon (1999) – "Digital video colourmaps for checking
    the legibility of displays by dichromats."  Color Research & Application.
  Machado, Oliveira & Fernandes (2009) – "A Physiologically-based Model for
    Simulation of Color Vision Deficiency."  IEEE TVCG.
  Neitz & Jacobs (1989) – dichromat modelling for domestic and wild animals.

We work directly on gamma-corrected sRGB (which is what uint8 BGR images are)
rather than linearising first.  This is a small approximation that avoids
expensive gamma operations while remaining visually accurate enough for
camouflage evaluation purposes.

Animals provided
────────────────
  none        Human trichromat (identity — no transform).
  deer        Protanopic dichromat (missing L-cones / red receptor).
              Deer are maximally sensitive around 450 nm (blue/UV) and
              540 nm (green-yellow).  Reds and oranges appear very dark.
              UV component modelled by boosting the blue channel slightly.
  dog         Deuteranopic dichromat (missing M-cones / green receptor).
              Dogs have peak sensitivities around 430 nm and 555 nm.
              They can distinguish blue from yellow but not red from green.
  black_bear  Deuteranope like dogs; slightly different sensitivity curve.
  elk         Close to deer (protanopia); elk may have slightly more
              sensitivity in the blue-green range.
  wild_boar   Near-monochromat or extreme dichromat with very limited hue
              discrimination; modelled as strong desaturation with a faint
              blue-yellow bias.
  turkey      Tetrachromat with UV sensitivity.  Can see ultraviolet
              (~380 nm).  Modelled by shifting some near-UV/blue stimulus
              to a separate "violet" channel and increasing overall
              saturation — a rough perceptual approximation since we cannot
              add genuine UV information to a standard RGB image.
"""
from __future__ import annotations
import numpy as np
import cv2


# ── colour matrix helper ──────────────────────────────────────────────────────

def _apply_matrix(img_bgr: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    Apply a 3×3 matrix (row = output channel, col = input channel) to
    an H×W×3 BGR uint8 image.  Channel order: B=0 G=1 R=2.
    """
    f   = img_bgr.astype(np.float32) / 255.0
    out = np.einsum('ij,...j->...i', mat, f)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _boost_saturation(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    """Scale saturation in HSV space."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ── Viénot 1999 dichromacy matrices (RGB order, then transposed for BGR) ─────
#
# These matrices are defined for RGB channels (R=0, G=1, B=2).
# We permute rows/cols for BGR (B=0, G=1, R=2).

def _rgb_mat_to_bgr(m_rgb: np.ndarray) -> np.ndarray:
    """
    Permute a 3×3 matrix defined in RGB order (R=0,G=1,B=2) to BGR order.
    """
    perm = np.array([2, 1, 0])   # R↔B swap
    return m_rgb[np.ix_(perm, perm)]


# Protanopia (missing L / red cone) — Viénot 1999
_PROTAN_RGB = np.array([
    [0.56667, 0.43333, 0.00000],
    [0.55833, 0.44167, 0.00000],
    [0.00000, 0.24167, 0.75833],
], dtype=np.float32)

# Deuteranopia (missing M / green cone) — Viénot 1999
_DEUTAN_RGB = np.array([
    [0.62500, 0.37500, 0.00000],
    [0.70000, 0.30000, 0.00000],
    [0.00000, 0.30000, 0.70000],
], dtype=np.float32)

_PROTAN_BGR = _rgb_mat_to_bgr(_PROTAN_RGB)
_DEUTAN_BGR = _rgb_mat_to_bgr(_DEUTAN_RGB)

# Near-monochromat (wild boar): heavily desaturated, faint blue-yellow bias
_MONO_BGR = np.array([
    [0.30, 0.59, 0.11],   # B out: luminance
    [0.30, 0.59, 0.11],   # G out: luminance (same)
    [0.30, 0.59, 0.11],   # R out: luminance
], dtype=np.float32)
# Add faint blue tint (boar may retain minimal S-cone response)
_BOAR_BGR = 0.85 * _MONO_BGR + 0.15 * np.array([
    [0.50, 0.30, 0.20],
    [0.30, 0.50, 0.20],
    [0.10, 0.20, 0.70],
], dtype=np.float32)


# ── filter functions ──────────────────────────────────────────────────────────

def _filter_none(img: np.ndarray) -> np.ndarray:
    """Identity – human trichromat view."""
    return img


def _filter_deer(img: np.ndarray) -> np.ndarray:
    """
    Deer – protanopic dichromat with mild UV boost.
    Red/orange regions appear very dark.  Blue and blue-green
    remain salient.  We add a small boost to the blue channel
    to approximate UV sensitivity that deer have but cameras don't record.
    """
    out = _apply_matrix(img, _PROTAN_BGR)
    # Mild UV-blue boost: add ~10% of B channel back
    f   = out.astype(np.float32)
    f[:, :, 0] = np.clip(f[:, :, 0] * 1.12, 0, 255)   # boost B
    return f.astype(np.uint8)


def _filter_dog(img: np.ndarray) -> np.ndarray:
    """Dog – deuteranopic dichromat.  Blue and yellow salient; reds and greens confused."""
    return _apply_matrix(img, _DEUTAN_BGR)


def _filter_black_bear(img: np.ndarray) -> np.ndarray:
    """
    Black bear – deuteranope with slightly weaker colour discrimination.
    Similar to dog but with additional mild desaturation.
    """
    out = _apply_matrix(img, _DEUTAN_BGR)
    return _boost_saturation(out, 0.80)


def _filter_elk(img: np.ndarray) -> np.ndarray:
    """
    Elk – protanopic like deer but with slightly more green-yellow sensitivity.
    Modelled as protanopia with mild green channel enhancement.
    """
    out = _apply_matrix(img, _PROTAN_BGR)
    f   = out.astype(np.float32)
    f[:, :, 1] = np.clip(f[:, :, 1] * 1.08, 0, 255)   # slight G boost
    return f.astype(np.uint8)


def _filter_wild_boar(img: np.ndarray) -> np.ndarray:
    """
    Wild boar – near-monochromat with very limited colour vision.
    Scene appears almost greyscale with a faint blue-yellow bias.
    """
    return _apply_matrix(img, _BOAR_BGR)


def _filter_turkey(img: np.ndarray) -> np.ndarray:
    """
    Turkey – tetrachromat with UV sensitivity.
    True UV information is absent from standard images so we simulate
    enhanced colour perception via increased saturation and a slight
    violet shift in the deep-blue channel (approximating UV → visible violet).
    This is an artistic approximation rather than a strict simulation.
    """
    # Boost saturation significantly
    out = _boost_saturation(img, 1.45)
    # Shift deepest blues toward violet (UV proxy): add red to pure-blue pixels
    f      = out.astype(np.float32)
    b_ch   = f[:, :, 0]
    r_ch   = f[:, :, 2]
    g_ch   = f[:, :, 1]
    # Where blue dominates over red and green, add a violet tint
    blue_dom = (b_ch > r_ch * 1.5) & (b_ch > g_ch * 1.2)
    f[blue_dom, 2] = np.clip(r_ch[blue_dom] + b_ch[blue_dom] * 0.25, 0, 255)
    return f.astype(np.uint8)


# ── registry ──────────────────────────────────────────────────────────────────

VISION_FILTERS: dict[str, tuple[str, callable]] = {
    "none":        ("Human (none)",       _filter_none),
    "deer":        ("Deer",               _filter_deer),
    "dog":         ("Dog",                _filter_dog),
    "black_bear":  ("Black Bear",         _filter_black_bear),
    "elk":         ("Elk",                _filter_elk),
    "wild_boar":   ("Wild Boar",          _filter_wild_boar),
    "turkey":      ("Turkey (tetrachromat)", _filter_turkey),
}


def apply_vision_filter(img_bgr: np.ndarray, filter_name: str) -> np.ndarray:
    """
    Apply the named animal vision filter to an BGR uint8 image.
    filter_name must be a key of VISION_FILTERS; silently returns unchanged
    image if name is unknown.
    """
    entry = VISION_FILTERS.get(filter_name)
    if entry is None:
        return img_bgr
    _, fn = entry
    return fn(img_bgr)
