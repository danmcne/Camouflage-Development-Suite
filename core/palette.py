"""
ColorPalette – colour management with presets and k-means extraction.
"""
from __future__ import annotations
import json
import math
import random
import colorsys
from typing import Sequence
import numpy as np


def hex_to_rgb(h: str) -> tuple[int,int,int]:
    h = h.lstrip("#")
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def rgb_to_hex(r,g,b) -> str:
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

def rgb_to_lab(rgb) -> np.ndarray:
    import cv2
    arr = np.array([[list(rgb)]], dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0,0].astype(float)

def delta_e(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a,float) - np.asarray(b,float)))

def _similar_color(hex_color: str, variation=0.09) -> str:
    r,g,b = hex_to_rgb(hex_color)
    h,s,v = colorsys.rgb_to_hsv(r/255,g/255,b/255)
    h = (h + random.uniform(-variation, variation)) % 1.0
    s = max(0.0, min(1.0, s + random.uniform(-variation*0.5, variation*0.5)))
    v = max(0.1, min(0.95, v + random.uniform(-variation*0.5, variation*0.5)))
    rn,gn,bn = colorsys.hsv_to_rgb(h,s,v)
    return rgb_to_hex(int(rn*255),int(gn*255),int(bn*255))


class ColorPalette:
    def __init__(self, colors: Sequence[str] | None = None):
        self._colors: list[str] = list(colors) if colors else []
        self._locked: list[bool] = [False] * len(self._colors)
        self.source_image: str | None = None

    def __len__(self):   return len(self._colors)
    def __getitem__(self,i): return self._colors[i]
    def __iter__(self):  return iter(self._colors)

    def append(self, h: str):
        self._colors.append(h); self._locked.append(False)

    def remove(self, idx: int):
        self._colors.pop(idx); self._locked.pop(idx)

    def set_color(self, idx: int, h: str):   self._colors[idx] = h
    def set_locked(self, idx: int, v: bool): self._locked[idx] = v
    def is_locked(self, idx: int) -> bool:   return self._locked[idx]

    def resize_to(self, n: int):
        current = len(self._colors)
        if n > current:
            for i in range(n - current):
                src = self._colors[i % current]
                self.append(_similar_color(src))
        elif n < current:
            while len(self._colors) > n:
                for j in range(len(self._colors)-1, -1, -1):
                    if not self._locked[j]:
                        self.remove(j); break
                else:
                    self.remove(len(self._colors)-1)

    def as_rgb(self):  return [hex_to_rgb(c) for c in self._colors]
    def as_bgr(self):  return [(b,g,r) for r,g,b in self.as_rgb()]
    def as_lab(self):  return [rgb_to_lab(c) for c in self.as_rgb()]
    def as_numpy_rgb(self): return np.array(self.as_rgb(), dtype=np.uint8)
    def as_qcolors(self):
        from PyQt6.QtGui import QColor
        return [QColor(c) for c in self._colors]

    @classmethod
    def from_image_kmeans(cls, image_path: str, n_colors=5, sample=2000):
        import cv2
        from sklearn.cluster import KMeans
        img = cv2.imread(image_path)
        if img is None: raise FileNotFoundError(f"Cannot open: {image_path}")
        pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1,3)
        if len(pixels) > sample:
            pixels = pixels[np.random.choice(len(pixels), sample, replace=False)]
        km = KMeans(n_clusters=n_colors, n_init="auto", random_state=0)
        km.fit(pixels)
        colors = [rgb_to_hex(int(r),int(g),int(b))
                  for r,g,b in km.cluster_centers_.astype(np.uint8)]
        pal = cls(colors)
        pal.source_image = image_path
        return pal



    @classmethod
    def from_images_histogram_peaks(cls, image_paths: list, n_colors: int = 5):
        """
        Find palette by quantising pixels into coarse colour bins, ranking bins
        by occupancy, then returning the median pixel from each top-N bin.

        Unlike k-means, the result colours are *actual pixel values* — no
        averaging — so they never appear washed out.  Works well when the image
        has distinct, saturated colour regions.

        Bin resolution: 32 levels per channel (32^3 = 32768 buckets).
        """
        import cv2
        n = max(1, len(image_paths))
        per_image = max(150, int(round(2000 / math.sqrt(n))))
        all_pixels = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
            if len(pixels) > per_image:
                idx = np.random.choice(len(pixels), per_image, replace=False)
                pixels = pixels[idx]
            all_pixels.append(pixels)
        if not all_pixels:
            raise ValueError("No readable images found.")
        combined = np.vstack(all_pixels).astype(np.int32)

        BINS = 32
        STEP = 256 // BINS
        # Quantise each channel
        qr = (combined[:, 0] // STEP).clip(0, BINS-1)
        qg = (combined[:, 1] // STEP).clip(0, BINS-1)
        qb = (combined[:, 2] // STEP).clip(0, BINS-1)
        keys = qr * BINS * BINS + qg * BINS + qb

        # Count occupancy
        unique_keys, counts = np.unique(keys, return_counts=True)
        order = np.argsort(counts)[::-1]

        colors = []
        for k in unique_keys[order]:
            if len(colors) >= n_colors:
                break
            mask = keys == k
            bucket_pixels = combined[mask]
            median_px = np.median(bucket_pixels, axis=0).astype(np.uint8)
            colors.append(rgb_to_hex(int(median_px[0]), int(median_px[1]), int(median_px[2])))
        # Pad if we ran out of buckets
        while len(colors) < n_colors:
            colors.append(colors[-1] if colors else "#808080")
        return cls(colors)

    @classmethod
    def from_images_median_cut(cls, image_paths: list, n_colors: int = 5):
        """
        Classic median-cut colour quantisation.

        Recursively splits the colour cloud along its longest axis at the
        median value.  Produces palettes with good coverage of the colour
        space — tends to capture both dark and light tones even when k-means
        would merge them.  Returns the mean colour of each final partition.
        """
        import cv2
        n = max(1, len(image_paths))
        per_image = max(150, int(round(2000 / math.sqrt(n))))
        all_pixels = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
            if len(pixels) > per_image:
                idx = np.random.choice(len(pixels), per_image, replace=False)
                pixels = pixels[idx]
            all_pixels.append(pixels)
        if not all_pixels:
            raise ValueError("No readable images found.")
        combined = np.vstack(all_pixels).astype(np.float32)

        def _median_cut(pixels, depth):
            """Return list of pixel groups."""
            if depth == 0 or len(pixels) == 0:
                return [pixels]
            # Split on the channel with the widest range
            ranges = pixels.max(axis=0) - pixels.min(axis=0)
            axis   = int(np.argmax(ranges))
            median = np.median(pixels[:, axis])
            lo = pixels[pixels[:, axis] <= median]
            hi = pixels[pixels[:, axis] >  median]
            if len(lo) == 0 or len(hi) == 0:
                return [pixels]
            return _median_cut(lo, depth-1) + _median_cut(hi, depth-1)

        depth  = int(math.ceil(math.log2(max(n_colors, 2))))
        groups = _median_cut(combined, depth)

        # Sort groups by size descending; take top n_colors
        groups.sort(key=len, reverse=True)
        colors = []
        for g in groups[:n_colors]:
            mean = g.mean(axis=0).clip(0, 255).astype(np.uint8)
            colors.append(rgb_to_hex(int(mean[0]), int(mean[1]), int(mean[2])))
        while len(colors) < n_colors:
            colors.append(colors[-1] if colors else "#808080")
        return cls(colors)

    @classmethod
    def from_images_perceptual(cls, image_paths: list, n_colors: int = 5):
        """
        K-means in CIE L*a*b* colour space.

        L*a*b* is perceptually uniform — equal Euclidean distances correspond
        to approximately equal perceived colour differences.  The resulting
        palette tends to be more visually distinct than RGB k-means, and
        avoids the 'muddy middle' that appears when RGB averaging blends
        very different hues.
        """
        import cv2
        from sklearn.cluster import KMeans
        n = max(1, len(image_paths))
        per_image = max(150, int(round(2000 / math.sqrt(n))))
        all_pixels_lab = []
        all_pixels_rgb = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = rgb.reshape(-1, 3)
            if len(pixels) > per_image:
                idx    = np.random.choice(len(pixels), per_image, replace=False)
                pixels = pixels[idx]
            # Convert sampled pixels to L*a*b*
            swatch = pixels.reshape(1, -1, 3).astype(np.uint8)
            lab    = cv2.cvtColor(swatch, cv2.COLOR_RGB2Lab).reshape(-1, 3)
            all_pixels_lab.append(lab.astype(np.float32))
            all_pixels_rgb.append(pixels)
        if not all_pixels_lab:
            raise ValueError("No readable images found.")
        lab_combined = np.vstack(all_pixels_lab)
        rgb_combined = np.vstack(all_pixels_rgb)

        km = KMeans(n_clusters=n_colors, n_init="auto", random_state=0)
        km.fit(lab_combined)
        labels = km.labels_
        colors = []
        for ci in range(n_colors):
            members = rgb_combined[labels == ci]
            if len(members) == 0:
                colors.append("#808080")
            else:
                # Use median RGB of the cluster (avoids averaging artefacts)
                med = np.median(members, axis=0).astype(np.uint8)
                colors.append(rgb_to_hex(int(med[0]), int(med[1]), int(med[2])))
        return cls(colors)

    @classmethod
    def from_images_kmeans(cls, image_paths: list, n_colors: int = 5):
        """
        Extract a palette by running k-means over pixels sampled from multiple images.

        Sampling strategy – equal weight per image, sublinear total growth:
          per_image = max(150, round(2000 / sqrt(n_images)))

          n=1 → 2000, n=4 → 1000, n=10 → 632  (total ~2000–6300, always manageable)

        Locked colours are handled by the caller after this returns.
        """
        import cv2
        from sklearn.cluster import KMeans
        n = max(1, len(image_paths))
        per_image = max(150, int(round(2000 / math.sqrt(n))))
        all_pixels = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
            if len(pixels) > per_image:
                idx = np.random.choice(len(pixels), per_image, replace=False)
                pixels = pixels[idx]
            all_pixels.append(pixels)
        if not all_pixels:
            raise ValueError("No readable images found.")
        combined = np.vstack(all_pixels)
        km = KMeans(n_clusters=n_colors, n_init="auto", random_state=0)
        km.fit(combined)
        colors = [rgb_to_hex(int(r), int(g), int(b))
                  for r, g, b in km.cluster_centers_.astype(np.uint8)]
        pal = cls(colors)
        return pal

    @classmethod
    def random(cls, n=5):
        return cls([rgb_to_hex(random.randint(30,210),
                               random.randint(30,210),
                               random.randint(30,210)) for _ in range(n)])

    # ── presets (8 colours) ───────────────────────────────────────────────────

    @classmethod
    def military_preset(cls):
        return cls(["#4B5320","#78866B","#8B7355","#2E3B1E",
                    "#A0956B","#5A6328","#3D4A2E","#6B7A45"])

    @classmethod
    def desert_preset(cls):
        return cls(["#C2A06E","#A0784A","#8B6340","#D4C5A9",
                    "#6B5A3E","#B8946A","#D9C080","#7A5C3A"])

    @classmethod
    def urban_preset(cls):
        return cls(["#808080","#A9A9A9","#696969","#C0C0C0",
                    "#2F2F2F","#B0B0B0","#555555","#909090"])

    @classmethod
    def warm_urban_preset(cls):
        """Bricks, rust, terracotta, stone, weathered wood."""
        return cls(["#8B3A2A","#A0522D","#C47A3A","#7A6652",
                    "#8C7B6B","#6B5B45","#9E8B72","#4E3D2F"])

    @classmethod
    def woodland_preset(cls):
        return cls(["#355E3B","#4F7942","#6B8E50","#8FBC8F",
                    "#2D4A1E","#7A6A3A","#5C4A2A","#3A5A28"])

    @classmethod
    def arctic_preset(cls):
        return cls(["#E8EEF0","#C8D8E0","#A0B8C8","#7090A8",
                    "#F0F4F8","#B0C8D8","#8090A0","#D0E0EC"])

    @classmethod
    def cool_contrast_preset(cls):
        """Snow, deep shadow, lichen, wet rock, dark spruce — high contrast cool."""
        return cls(["#F2F4F8","#1A1F2E","#6B8070","#2C3A28",
                    "#8BA0A8","#384050","#C0CCD4","#4A5840"])

    @classmethod
    def warm_contrast_preset(cls):
        """Dry grass, charred wood, ochre sand, red soil, bright sky hole — high contrast warm."""
        return cls(["#E8D890","#1C1008","#C87820","#5A2810",
                    "#D4B060","#3C2010","#F0E8C0","#804010"])
                    
    @classmethod
    def neon_preset(cls):
        """Bright high-contrast neon colors."""
        return cls(["#000000","#FFFFFF","#FF0000","#FFFF00",
                    "#00FF00","#0000FF","#FF00FF","#00FFFF"])

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self):
        return {"colors":self._colors,"locked":self._locked,"source_image":self.source_image}

    def to_json(self): return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d):
        pal = cls(d["colors"])
        pal._locked = d.get("locked",[False]*len(pal))
        pal.source_image = d.get("source_image")
        return pal

    @classmethod
    def from_json(cls, s): return cls.from_dict(json.loads(s))
