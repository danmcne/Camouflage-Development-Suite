"""
ColorPalette – colour management with presets and k-means extraction.
"""
from __future__ import annotations
import json
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
