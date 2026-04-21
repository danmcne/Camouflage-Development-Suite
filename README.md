# Camouflage Development Tool

A desktop application for designing, generating, and evolving seamless camouflage patterns. Built with Python and PyQt6.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

---

## Overview

This tool provides a complete workflow for camouflage pattern development:

**Generate** patterns using nine different algorithmic methods → **Preview** them against real background images → **Evolve** them interactively or automatically toward better concealment → **Export** at 512x512, 1024x1024, 2048x2048, 4096x4096 or 8192x8192 resolution, in PNG, JPG or TIF format.

Images exported in PNG can have the background color exported as transparent.

All patterns use toroidal (wrap-around) geometry throughout, so every output tiles seamlessly at any size.

---

## Features

### Nine Pattern Generators

| Generator | Description |
|---|---|
| Procedural Noise | Seamlessly tiling Perlin noise with threshold or Voronoi colour assignment |
| Blur-Sharp | Iterative anisotropic Gaussian blur + unsharp mask — produces spots, stripes, and labyrinthine channels |
| Reaction-Diffusion | Gray-Scott two-chemical system producing spots, labyrinths, stripes, coral, and mitosis patterns |
| L-System | Multiple turtle-geometry trees scattered across the canvas, each with independent angle, step size, width, and colour |
| Recursive Fractal | Multi-scale Voronoi pyramid — coarse blobs with progressively finer detail layered on top |
| Urban Geometric | Tiled or scattered hexagons, triangles, diamonds, or offset grids — three placement modes |
| Collage | Stamp PNG or JPG shapes across the canvas with palette tinting; optional field-driven placement for correlated structure |
| Dazzle | WWI Razzle-Dazzle inspired: canvas divided into toroidal Voronoi zones, each with its own stripe direction and density |
| Plaid | Multiscale plaid — nested horizontal/vertical or diagonal stripe octaves, perfectly seamless at 0° or 45° |

All generators use toroidal boundary conditions so every output tiles seamlessly.

---

### Field-Driven Placement (Collage and Urban Geometric)

The Collage and Urban Geometric generators both support a **field-driven placement** mode that replaces uniform random scatter with spatially correlated structure:

- A **density field** (low-frequency Perlin noise) drives importance-sampled position clustering — shapes appear in patches with empty gaps between, like real environmental texture
- An **orientation field** drives local rotation alignment — nearby shapes lean the same way, as grass blades or fallen leaves do
- A **scale field** modulates shape size across the canvas

This is a significant perceptual upgrade over IID placement. Enable it with *Field-driven placement* in the Collage params, or set *Placement mode* to `field_driven` in Urban Geometric.

---

### Dazzle Camouflage

Unlike the other generators, Dazzle does not try to match the background — it tries to confuse depth and heading estimation. The canvas is divided into Voronoi zones using **toroidal** nearest-neighbour distance (seeds replicated at all 9 torus offsets), so zones wrap correctly across every edge.

Each zone independently draws stripes in one of four directions:

- `vertical` — `floor(xn × n) % 2`
- `horizontal` — `floor(yn × n) % 2`
- `diagonal+` — `floor((xn + yn) × n) % 2` (n forced even for seamless wrap)
- `diagonal-` — `floor(((xn − yn) mod 1) × n) % 2` (seamless for any n)

All formulas use normalised coordinates so tiling is exact regardless of canvas size.

---

### Plaid Generator

Produces nested stripe octaves in two crossing axes. **Seamless by construction**: stripe counts are always integers, and the normalised-coordinate formulas tile exactly without any work-size rounding.

- **0°** — axis-aligned horizontal + vertical stripes (classic tartan)
- **45°** — two crossing diagonal axes

Each octave composites three palette colours: one for each axis and a third for their intersection, producing the characteristic lighter or darker crossing zone of woven fabric. `scale_factor` multiplies the stripe count at each finer octave; keeping it an integer guarantees every level nests seamlessly within the coarser one.

---

### Urban Geometric — Three Placement Modes

| Mode | Behaviour |
|---|---|
| `tiled` | Original grid + per-cell jitter; 9-copy toroidal seam fix |
| `random` | Scatter N shapes at random positions, sizes, and rotations |
| `field_driven` | Positions importance-sampled from density field; rotation follows orientation field |

---

### Colour Palette Editor

- 2–10 colours with per-swatch colour picker, from either a standard palette or from images
- Lock individual colours so evolution further palette modification doesn't alter them
- Extract a palette from a collection of images using k-means clustering
- Nine built-in presets: Military, Desert, Urban, Warm Urban, Woodland, Arctic, Cool Hi-Contrast, Warm Hi-Contrast and Random
- Extending a preset fills new slots with perceptually similar variants

---

### Second Generator Layer

Any two generators can be composited with five blend modes (Normal, Multiply, Screen, Overlay, Soft Light) at adjustable opacity. Generators support `transparent_bg` and can be used as overlays without a solid background bleeding through.

---

### Evolution Lab

Patterns are displayed as moth thumbnails scattered over the full background image — the natural metaphor for camouflage selection.

- **Interactive mode** — click moths you can see easily to kill them; survivors breed the next generation
- **Automatic mode** — fitness computed from colour histogram similarity, texture (SSIM / normalised cross-correlation), and edge disruption against the background
- **2-Layer evolution** — tick *Evolve 2-layer stack* to jointly evolve both generator layers. The worker generates and blends both layers before fitness evaluation; mutation and crossover are applied independently to each layer's parameters. Inspect mode populates both layers in the Generator tab.
- **Inspect mode** — click a moth without killing it to send its full parameters (both layers if applicable) to the Generator tab for manual tweaking
- Adjustable moth size (50–250 px), population size, and per-metric fitness weights
- Background image collection with random selection during automatic evolution
- Generation runs in a background thread; Stop button cleanly aborts

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

---

## Linux Usage

After installation, run the application from the project root:

```bash
source .venv/bin/activate
python main.py
```

### Wayland / X11 notes

On some Linux systems (especially Wayland-based desktops), PyQt6 may default to a backend that causes rendering or input issues. If you encounter problems (blank window, crashes, or missing UI elements), try forcing X11:

```bash
QT_QPA_PLATFORM=xcb python main.py
```

### High-DPI scaling

If the UI appears too small or too large on high-resolution displays:

```bash
QT_SCALE_FACTOR=1.5 python main.py
```

Adjust the scale factor as needed (e.g. `1.25`, `2`).

### Common issues

* **Missing Qt platform plugin ("xcb")**
  Install required system libraries:

  ```bash
  sudo apt install libxcb-cursor0 libxcb-xinerama0
  ```

* **Slow performance (especially generators)**
  Ensure you are using the virtual environment and that `numpy`, `scipy`, and `opencv-python` are properly installed with native extensions.

---

## Windows Usage

### Running the application

From PowerShell or Command Prompt:

```powershell
cd camouflage-dev
.venv\Scripts\activate
python main.py
```

### PowerShell execution policy (first run only)

If activation fails with a script execution error:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then retry:

```powershell
.venv\Scripts\activate
```

### Optional: create a double-click launcher

Create a file `run.bat` in the project root:

```bat
@echo off
call .venv\Scripts\activate
python main.py
pause
```

This allows launching the app without opening a terminal manually.

### Common issues

* **"Python not found"**
  Ensure Python 3.9+ is installed and added to PATH.

* **Qt platform plugin errors**
  Reinstall dependencies inside the virtual environment:

  ```powershell
  pip install --upgrade --force-reinstall pyqt6
  ```

* **Blurry UI on high-DPI displays**
  Try:

  ```powershell
  set QT_SCALE_FACTOR=1.5
  python main.py
  ```
  
---

## Dependencies

| Package | Purpose |
|---|---|
| PyQt6 | GUI framework |
| numpy | Array operations throughout |
| opencv-python | Image processing, convolution, colour conversion |
| Pillow | Export and file I/O |
| scikit-learn | k-means palette extraction |
| scikit-image | SSIM texture metric (falls back to numpy NCC if unavailable) |
| noise | Perlin noise generation (also used for field-driven placement warp) |
| scipy | Fast toroidal Voronoi via cKDTree (Dazzle; falls back to numpy if unavailable) |

---

## Quick Start

1. Choose a palette on the **Palette** tab — start with a preset or extract colours from a reference image.
2. Pick a generator on the **Generator** tab, adjust parameters, and click **Generate**.
3. Add a background image via the **Backgrounds** menu or the Evolution tab's left panel.
4. Switch to the **Evolution** tab and click **Seed & Run**. Kill the moths you can see and click **Next generation**.
5. Click any moth to send its pattern (and parameters) to the Generator tab for inspection or tweaking.
6. Use **File → Export** to save at full resolution.

---

## Generator Parameter Guide

### Dazzle

| Parameter | Effect |
|---|---|
| Zone count | More zones = smaller, more fragmented areas of differing stripe direction |
| Stripe density | Base number of stripe periods per zone |
| Density variation | How much stripe density varies between zones |
| Directions | `mixed` = all four; `striped` = H+V only; `diagonal` = diagonals only |
| Zone outline | Dark border drawn between zones — adds graphic punch |
| High contrast | Alternates between the two halves of the palette for maximum colour contrast |

### Plaid

| Parameter | Effect |
|---|---|
| Grid angle | `0` = axis-aligned tartan; `45` = two crossing diagonals |
| Coarse stripe count | Integer number of stripe periods across the canvas at the coarsest scale |
| Octave count | 1 = simple grid; 3 = tartan; 5+ = complex weave |
| Scale factor | Period multiplier per octave (integer; keeps all scales seamlessly nested) |
| H/V balance | 0 = only axis-A stripes; 0.5 = symmetric plaid; 1 = only axis-B stripes |
| Edge softness | 0 = hard crisp tartan lines; 0.45 = soft colour wash |

### Reaction-Diffusion

Use the **Pattern preset** dropdown first — it sets feed and kill rates automatically:

| Preset | feed | kill | Result |
|---|---|---|---|
| Spots | 0.035 | 0.065 | Isolated blobs |
| Labyrinth | 0.037 | 0.060 | Connected maze-like channels |
| Stripes | 0.060 | 0.062 | Elongated directional stripes |
| Coral | 0.062 | 0.061 | Branching coral / fingerprint |
| Mitosis | 0.028 | 0.062 | Dividing cells / leopard spots |

Set *Anisotropy* above 1 to stretch patterns horizontally; below 1 to stretch vertically.

### Collage (Field-Driven)

Enable *Field-driven placement* to activate the three spatial field controls:

| Parameter | Effect |
|---|---|
| Field scale | Spatial scale of the fields. Low = large cluster regions |
| Density contrast | 0 = uniform scatter; 1 = tight clusters with empty gaps |
| Orientation coherence | 0 = fully random rotation; 1 = fully aligned to local field direction |
| Scale field variation | How strongly the scale field modulates shape sizes across the canvas |

### Urban Geometric — Placement Modes

Set *Placement mode* to `random` or `field_driven` to unlock scatter-based controls (*Shape count*, *Scale min/max*, *Rotation range*). In `field_driven` mode the same *Field scale*, *Density contrast*, and *Orientation coherence* controls apply as in field-driven Collage.

---

## Project Structure

```
camouflage_dev/
├── main.py                     Entry point
├── config/
│   └── defaults.py             All generator parameter schemas and defaults
├── core/
│   ├── palette.py              ColorPalette class and k-means extraction
│   ├── pattern.py              CamoPattern dataclass (supports 2-layer stack)
│   └── fitness.py              Colour, texture, disruption, and composite metrics
├── generators/
│   ├── base.py                 Abstract BaseGenerator (generate / mutate / crossover)
│   ├── procedural_noise.py
│   ├── blur_sharp.py
│   ├── reaction_diffusion.py
│   ├── l_system.py
│   ├── recursive_fractal.py
│   ├── urban_geometric.py      Three placement modes: tiled / random / field_driven
│   ├── collage.py              Field-driven placement with density/orient/scale fields
│   ├── dazzle.py               Toroidal Voronoi zones with per-zone stripe patterns
│   └── plaid.py                Multiscale plaid at 0° or 45°; seamless by construction
├── evolution/
│   ├── population.py           Population manager; supports 1- and 2-layer evolution
│   └── background_manager.py   Background image collection and caching
├── ui/
│   ├── main_window.py          Top-level window and signal wiring
│   ├── color_panel.py          Palette tab
│   ├── generator_panel.py      Generator tab with second-layer controls
│   ├── evolution_panel.py      Evolution tab with moth canvas and 2-layer toggle
│   └── preview_canvas.py       Always-visible live preview
└── utils/
    ├── rendering.py            NumPy ↔ QPixmap conversion
    └── image_ops.py            Export, tiling, swatch sheet
```

---

## Extending the Tool

### Adding a new generator

1. Create `generators/your_generator.py` inheriting from `BaseGenerator`.
2. Implement `generate()` and `get_param_schema()`. Override `mutate()` if needed.
3. Add the parameter schema block to `config/defaults.py` under `GENERATORS`.
4. Register it in `generators/__init__.py`.

The UI will automatically build a parameter form from the schema and make the generator available in both the Generator and Evolution tabs (including 2-layer evolution).

For seamless output, use one of the proven toroidal stripe formulas from `dazzle.py` / `plaid.py`, or the 9-copy toroidal stamp pattern from `collage.py` / `urban_geometric.py`.

### Adding a new fitness metric

Add a function to `core/fitness.py` returning a float in [0, 1], then include it in `composite_fitness()` with a weight. Add a corresponding entry to the `fitness_weights` dict in `config/defaults.py` if you want a user-facing weight slider in the Evolution tab.

---

## Known Limitations

- The Procedural Noise generator iterates pixel-by-pixel in Python; at 512×512 it takes a few seconds. A compiled noise library (Numba, or the `noise` C extension with vectorised calls) would help significantly.
- Reaction-Diffusion runs at a reduced work resolution (default 200×200) and is upscaled. Increase *Work resolution* for finer patterns at the cost of speed.
- Dazzle's toroidal Voronoi uses `scipy.spatial.cKDTree` when available for fast nearest-neighbour queries. If scipy is not installed it falls back to a pure-numpy loop which is noticeably slower for high zone counts.
- The 2-layer evolution evolves both generators' parameters jointly but does not evolve the blend mode or opacity between them — these are fixed from the Generator tab when the population is seeded.

---

## License

MIT License

Copyright (c) 2026

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
