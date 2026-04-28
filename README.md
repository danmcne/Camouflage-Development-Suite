# Camouflage Development Tool

A desktop application for designing, generating, and evolving seamless camouflage patterns. Built with Python and PyQt6.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

---

## Overview

This tool provides a complete workflow for camouflage pattern development:

**Generate** patterns using six different algorithmic methods → **Preview** them against real background images → **Evolve** them interactively or automatically toward better concealment → **Export** at high resolution.

All patterns are seamlessly tileable (toroidal topology throughout).

---

## Features

### Six Pattern Generators

| Generator | Description |
|---|---|
| **Procedural Noise** | Seamlessly tiling Perlin noise with threshold or Voronoi colour assignment |
| **Reaction-Diffusion** | Gray-Scott two-chemical system producing spots, labyrinths, stripes, coral, and mitosis patterns |
| **L-System** | Multiple turtle-geometry trees scattered across the canvas, each with independent angle, step size, width, and colour |
| **Recursive Fractal** | Multi-scale Voronoi pyramid — coarse blobs with progressively finer detail layered on top |
| **Urban Geometric** | Tiled hexagons, triangles, diamonds, or offset grids with multiple interleaved size variants |
| **Collage** | Stamp PNG or JPG shapes (or built-in procedural shapes) across the canvas with palette tinting |

All generators use toroidal boundary conditions so every output tiles seamlessly.

### Colour Palette Editor

- 2–10 colours with per-swatch colour picker
- Lock individual colours so evolution cannot change them
- Extract a palette from any image using k-means clustering — changing the colour count re-runs the extraction, it does not add random colours
- Six built-in presets: Military, Desert, Urban, Warm Urban (brick/rust/stone/wood), Woodland, Arctic
- Extending a preset fills new slots with perceptually similar variants rather than random colours

### Second Generator Layer

Any two generators can be composited with five blend modes (Normal, Multiply, Screen, Overlay, Soft Light) at adjustable opacity. Generators that support `transparent_bg` (Urban Geometric, Collage) can be used as overlays without a solid background bleeding through.

### Evolution Lab

Patterns are displayed as moth thumbnails scattered over the full background image — the natural metaphor for camouflage selection.

**Interactive mode:** click moths you can see easily to kill them; survivors breed the next generation.  
**Automatic mode:** fitness is computed from colour histogram similarity, texture (SSIM / normalised cross-correlation), and edge disruption against the background.

- Adjustable moth size (50–250 px)
- Configurable population size and fitness weights
- Background image collection (folder or individual files) with random selection during automatic evolution
- Generates in a background thread; Stop button cleanly aborts (including long-running fractal computations)

### Export

- Single pattern PNG/JPEG/TIFF at up to 2048×2048
- Colour swatch sheet (contact sheet of all candidates)

---

## Installation

```bash
git clone https://github.com/your-username/camouflage-dev.git
cd camouflage-dev

python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
python main.py
```

### Ubuntu / Debian — additional system packages

If PyQt6 cannot find its display backend:

```bash
sudo apt install libxcb-cursor0 libxcb-xinerama0
```

### Dependencies

| Package | Purpose |
|---|---|
| PyQt6 | GUI framework |
| numpy | Array operations throughout |
| opencv-python | Image processing, convolution, colour conversion |
| Pillow | Export and file I/O |
| scikit-learn | k-means palette extraction |
| scikit-image | SSIM texture metric (falls back to numpy NCC if unavailable) |
| noise | Perlin noise generation |
| pyswarms | Particle swarm optimiser (evolution backend) |

---

## Quick Start

1. **Choose a palette** on the Palette tab — start with a preset or extract colours from a reference image.
2. **Pick a generator** on the Generator tab, adjust parameters, and click **Generate**.
3. **Add a background image** via the Backgrounds menu (or the Evolution tab's left panel).
4. **Switch to the Evolution tab** to scatter moths over your background. Click **Seed & Run**, then kill the visible ones and click **Next generation**.
5. Click any moth to send its pattern to the preview pane. Use **File → Export** to save at full resolution.

---

## Generator Parameter Guide

### Reaction-Diffusion

Use the **Pattern preset** dropdown first — it sets feed and kill automatically:

| Preset | feed | kill | Result |
|---|---|---|---|
| Spots | 0.035 | 0.065 | Isolated blobs on a plain field |
| Labyrinth | 0.037 | 0.060 | Connected maze-like channels |
| Stripes | 0.060 | 0.062 | Elongated directional stripes |
| Coral | 0.062 | 0.061 | Branching coral / fingerprint |
| Mitosis | 0.028 | 0.062 | Dividing cells / leopard spots |

Set **Anisotropy** above 1 to stretch patterns horizontally (more stripe-like), below 1 to stretch vertically. **Work resolution** trades speed for detail — 200 is fast, 320 is richer.

### Procedural Noise

**Tile periods** is the most important control: low values (1–3) give large blobs, high values (6–12) give fine texture. **Turbulence** warps coordinates for organic distortion.

### L-System

The default rule `F->FF+[+F-F-F]-[-F+F+F]` produces plant-like branching. Some alternatives:

```
Sierpinski:   F->F+G+F ; G->G-F-G        angle=60
Dragon:       F->F+G ; G->F-G            angle=90
Fractal bush: F->FF ; X->F[+X]F[-X]+X   axiom=X
```

Increase **Number of trees** for denser coverage. Enable **Per-tree colour** for colour variation across the canvas.

### Recursive Fractal

Keep **Recursion depth** at 3–4. Higher values multiply the seed count quickly and will hit the safety cap (800 seeds/level). **Level blend opacity** controls how visibly finer levels overlay coarser ones — lower values preserve the large-patch structure.

### Urban Geometric / Collage as a Second Layer

Enable **Transparent background** on either generator, then use it as the second layer in the Generator tab. Only the tiles or shapes will appear over the base pattern, with no solid fill in the gaps.

---

## Project Structure

```
camouflage_dev/
├── main.py                     Entry point
├── config/
│   └── defaults.py             All generator parameter schemas and defaults
├── core/
│   ├── palette.py              ColorPalette class and k-means extraction
│   ├── pattern.py              CamoPattern dataclass
│   └── fitness.py              Colour, texture, disruption, and composite metrics
├── generators/
│   ├── base.py                 Abstract BaseGenerator (generate / mutate / crossover)
│   ├── procedural_noise.py
│   ├── reaction_diffusion.py
│   ├── l_system.py
│   ├── recursive_fractal.py
│   ├── urban_geometric.py
│   └── collage.py
├── evolution/
│   ├── population.py           Population manager (selection, crossover, mutation)
│   └── background_manager.py  Background image collection and caching
├── ui/
│   ├── main_window.py          Top-level window and signal wiring
│   ├── color_panel.py          Palette tab
│   ├── generator_panel.py      Generator tab with second-layer controls
│   ├── evolution_panel.py      Evolution tab with moth canvas
│   └── preview_canvas.py       Always-visible live preview
└── utils/
    ├── rendering.py            NumPy ↔ QPixmap conversion, superimpose
    └── image_ops.py            Export, tiling, swatch sheet
```

---

## Extending the Tool

### Adding a new generator

1. Create `generators/your_generator.py` inheriting from `BaseGenerator`.
2. Implement `generate()`, `get_param_schema()`, and optionally `mutate()`.
3. Add the parameter schema to `config/defaults.py`.
4. Register it in `generators/__init__.py`.

The UI will automatically build a parameter form from the schema and make the generator available in both the Generator and Evolution tabs.

### Adding a new fitness metric

Add a function to `core/fitness.py` returning a float in [0, 1], then include it in `composite_fitness()` with a weight. The evolution panel's weight sliders only need a new entry if you want user control over the weight.

---

## Known Limitations

- The Procedural Noise generator iterates pixel-by-pixel in Python; at 512×512 it takes a few seconds. Numba or a compiled noise library would speed this up significantly.
- Reaction-Diffusion simulation runs at a reduced work resolution (default 200×200) and is upscaled. Increase **Work resolution** for finer patterns at the cost of speed.
- The neural fitness metric (MobileNetV2 feature cosine similarity) is stubbed. Uncomment and install PyTorch to enable it.

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
