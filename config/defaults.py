"""
Default parameters for every generator and global app settings.
"""

APP = {
    "preview_size": (512, 512),
    "export_size": (2048, 2048),
    "max_palette_colors": 10,
    "default_palette_colors": 5,
    "thread_pool_workers": 4,
}

GENERATORS = {
    # ── Reaction-Diffusion (Gray-Scott) ───────────────────────────────────────
    # Parameter guide:
    #   Du, Dv  – diffusion rates. Keep Du > Dv (typically 2×). Rarely need changing.
    #   feed    – how fast U is replenished. Controls overall density.
    #   kill    – how fast V is removed. Together with feed, determines pattern type.
    #   Presets (feed, kill):
    #     Spots      (0.035, 0.065)  – isolated blobs on a field
    #     Labyrinth  (0.037, 0.060)  – connected maze-like channels
    #     Stripes    (0.060, 0.062)  – elongated stripe pattern
    #     Coral      (0.062, 0.061)  – coral/fingerprint branching
    #     Mitosis    (0.028, 0.062)  – dividing cells / leopard spots
    "reaction_diffusion": {
        "preset":      {"default": "Labyrinth",
                        "options": ["Spots", "Labyrinth", "Stripes", "Coral", "Mitosis", "Custom"],
                        "label": "Pattern preset",    "tip": "Sets feed+kill automatically. Choose Custom to edit them manually.",
                        "evolvable": False},
        "Du":          {"default": 0.16,  "min": 0.05, "max": 0.5,  "step": 0.01,
                        "label": "Du (activator diff.)","tip": "Activator diffusion. Keep > Dv. Usually 0.10–0.20."},
        "Dv":          {"default": 0.08,  "min": 0.01, "max": 0.3,  "step": 0.01,
                        "label": "Dv (inhibitor diff.)","tip": "Inhibitor diffusion. Keep < Du. Usually 0.04–0.10."},
        "feed":        {"default": 0.037, "min": 0.01, "max": 0.10, "step": 0.001,
                        "label": "Feed rate",          "tip": "Spots≈0.035, Labyrinth≈0.037, Stripes≈0.060, Coral≈0.062."},
        "kill":        {"default": 0.060, "min": 0.04, "max": 0.07, "step": 0.001,
                        "label": "Kill rate",          "tip": "Spots≈0.065, Labyrinth≈0.060, Stripes≈0.062, Coral≈0.061."},
        "time_steps":  {"default": 3000,  "min": 500,  "max": 15000,"step": 500,
                        "label": "Time steps",         "tip": "More steps = more evolved pattern. 2000–5000 is usually enough."},
        "work_size":   {"default": 200,   "min": 64,   "max": 384,  "step": 32,
                        "label": "Work resolution",    "tip": "Simulation runs at this size then is upscaled. Lower = faster."},
        "noise_scale": {"default": 0.05,  "min": 0.01, "max": 0.3,  "step": 0.01,
                        "label": "Seed noise",         "tip": "Initial perturbation amplitude. Smaller = more uniform start."},
        "anisotropy":  {"default": 1.0,   "min": 0.2,  "max": 3.0,  "step": 0.1,
                        "label": "Anisotropy X/Y",     "tip": "Values ≠ 1 stretch the pattern horizontally (>1) or vertically (<1)."},
        "seed":        {"default": 42,    "min": 0,    "max": 99999, "step": 1,
                        "label": "Random seed",        "tip": "Reproducibility."},
    },

    # ── Procedural Noise ──────────────────────────────────────────────────────
    "procedural_noise": {
        "noise_type":  {"default": "seamless_perlin",
                        "options": ["seamless_perlin", "seamless_simplex"],
                        "label": "Noise type",         "tip": "Both are seamlessly tiling. Simplex is slightly smoother."},
        "octaves":     {"default": 6,    "min": 1,    "max": 12,   "step": 1,
                        "label": "Octaves",            "tip": "Layers of detail. More = richer texture."},
        "persistence": {"default": 0.5,  "min": 0.1,  "max": 1.0,  "step": 0.05,
                        "label": "Persistence",        "tip": "How much each octave contributes."},
        "lacunarity":  {"default": 2.0,  "min": 1.0,  "max": 4.0,  "step": 0.1,
                        "label": "Lacunarity",         "tip": "Frequency multiplier between octaves."},
        "periods":     {"default": 3,    "min": 1,    "max": 12,   "step": 1,
                        "label": "Tile periods",       "tip": "How many noise periods fit across the canvas. Lower = larger blobs."},
        "turbulence":  {"default": 0.0,  "min": 0.0,  "max": 1.0,  "step": 0.05,
                        "label": "Turbulence",         "tip": "Warps coordinates for organic distortion."},
        "color_mode":  {"default": "threshold", "options": ["threshold", "voronoi"],
                        "label": "Color assignment",   "tip": "Threshold = equal bands; Voronoi = nearest-centroid."},
        "seed":        {"default": 42,   "min": 0,    "max": 99999, "step": 1,
                        "label": "Random seed",        "tip": "Reproducibility."},
    },

    # ── L-System ──────────────────────────────────────────────────────────────
    "l_system": {
        "axiom":       {"default": "F",  "type": "str",
                        "label": "Axiom",              "tip": "Starting string."},
        "rules":       {"default": "F->FF+[+F-F-F]-[-F+F+F]", "type": "str",
                        "label": "Rules (semicolon-sep)","tip": "e.g.  F->FF+X ; X->F-X"},
        "angle":       {"default": 25.0, "min": 5.0,  "max": 90.0, "step": 0.5,
                        "label": "Base angle (°)",     "tip": "Central turning angle for +/– commands."},
        "angle_var":   {"default": 8.0,  "min": 0.0,  "max": 45.0, "step": 0.5,
                        "label": "Angle variation (°)","tip": "Each tree gets base_angle ± random(angle_var)."},
        "iterations":  {"default": 4,    "min": 1,    "max": 7,    "step": 1,
                        "label": "Iterations",         "tip": "More = finer detail, much slower."},
        "num_trees":   {"default": 9,    "min": 1,    "max": 50,   "step": 1,
                        "label": "Number of trees",    "tip": "Trees placed at random positions across the canvas."},
        "step_min":    {"default": 3,    "min": 1,    "max": 40,   "step": 1,
                        "label": "Step size min (px)", "tip": "Shortest turtle step."},
        "step_max":    {"default": 8,    "min": 1,    "max": 60,   "step": 1,
                        "label": "Step size max (px)", "tip": "Longest turtle step."},
        "width_min":   {"default": 1,    "min": 1,    "max": 8,    "step": 1,
                        "label": "Line width min (px)","tip": "Thinnest stroke."},
        "width_max":   {"default": 2,    "min": 1,    "max": 12,   "step": 1,
                        "label": "Line width max (px)","tip": "Thickest stroke."},
        "color_per_tree":{"default": True, "type": "bool",
                        "label": "Per-tree colour",    "tip": "Each tree picks a random palette colour."},
        "seed":        {"default": 42,   "min": 0,    "max": 99999, "step": 1,
                        "label": "Random seed",        "tip": "Reproducibility."},
    },

    # ── Recursive Fractal ─────────────────────────────────────────────────────
    "recursive_fractal": {
        "depth":          {"default": 3,   "min": 1,  "max": 5,   "step": 1,
                           "label": "Recursion depth",   "tip": "Voronoi levels. Each adds finer cells over coarser ones. >4 can be slow."},
        "base_seeds":     {"default": 6,   "min": 3,  "max": 16,  "step": 1,
                           "label": "Base seed count",   "tip": "Coarsest level cell count."},
        "seed_multiplier":{"default": 3,   "min": 2,  "max": 5,   "step": 1,
                           "label": "Seed multiplier",   "tip": "Each finer level has multiplier × more seeds. Capped at 800 total."},
        "level_opacity":  {"default": 0.55,"min": 0.1,"max": 0.9, "step": 0.05,
                           "label": "Level blend opacity","tip": "How strongly finer levels show through coarser ones."},
        "edge_sharpness": {"default": 0.0, "min": 0.0,"max": 20.0,"step": 0.5,
                           "label": "Edge sharpening",   "tip": "Sharpens cell boundaries. 0 = off."},
        "seed":           {"default": 42,  "min": 0,  "max": 99999,"step": 1,
                           "label": "Random seed",       "tip": "Reproducibility."},
    },

    # ── Urban Geometric ───────────────────────────────────────────────────────
    "urban_geometric": {
        "primitive":     {"default": "hexagon",
                          "options": ["hexagon", "triangle", "offset_grid", "diamond"],
                          "label": "Primitive shape",    "tip": "Base tile geometry."},
        "cell_size":     {"default": 40,   "min": 10,   "max": 200,  "step": 5,
                          "label": "Cell size (px)",     "tip": "Largest tile size."},
        "size_count":    {"default": 2,    "min": 1,    "max": 5,    "step": 1,
                          "label": "Size variants",      "tip": "Number of distinct tile sizes (multi-scale)."},
        "size_ratio":    {"default": 0.5,  "min": 0.2,  "max": 0.9,  "step": 0.05,
                          "label": "Size ratio",         "tip": "Each smaller size = previous × ratio."},
        "jitter":        {"default": 0.12, "min": 0.0,  "max": 0.5,  "step": 0.01,
                          "label": "Jitter",             "tip": "Random offset per cell centre."},
        "outline_width": {"default": 0,    "min": 0,    "max": 4,    "step": 1,
                          "label": "Outline width (px)", "tip": "0 = no outline."},
        "transparent_bg":{"default": False, "type": "bool",
                          "label": "Transparent background","tip": "Enable for second-layer use: space between tiles becomes transparent.",
                          "evolvable": False},
        "seed":          {"default": 42,   "min": 0,    "max": 99999, "step": 1,
                          "label": "Random seed",        "tip": "Reproducibility."},
    },

    # ── Collage ───────────────────────────────────────────────────────────────
    "collage": {
        "shape_folder":  {"default": "", "type": "folder",
                          "label": "Shape folder", "tip": "Folder of PNG/JPG files. Empty = built-in procedural shapes."},
        "count":         {"default": 40,   "min": 5,    "max": 300,  "step": 5,
                          "label": "Shape count",        "tip": "Number of placed shapes."},
        "scale_min":     {"default": 0.05, "min": 0.01, "max": 0.5,  "step": 0.01,
                          "label": "Scale min",          "tip": "Smallest shape relative to canvas width."},
        "scale_max":     {"default": 0.25, "min": 0.05, "max": 1.0,  "step": 0.01,
                          "label": "Scale max",          "tip": "Largest shape relative to canvas width."},
        "rotation_range":{"default": 180,  "min": 0,    "max": 180,  "step": 5,
                          "label": "Rotation range (°)", "tip": "±degrees of random rotation."},
        "blend_mode":    {"default": "normal",
                          "options": ["normal", "multiply", "screen", "overlay"],
                          "label": "Blend mode",         "tip": "How shapes blend onto canvas."},
        "tint_strength": {"default": 0.8,  "min": 0.0,  "max": 1.0,  "step": 0.05,
                          "label": "Tint strength",      "tip": "0 = keep original colours; 1 = fully tinted to palette."},
        "transparent_bg":{"default": False, "type": "bool",
                          "label": "Transparent background","tip": "Enable for second-layer use: background between shapes is transparent.",
                          "evolvable": False},
        "seed":          {"default": 42,   "min": 0,    "max": 99999, "step": 1,
                          "label": "Random seed",        "tip": "Reproducibility."},
    },
}

# Preset (feed, kill) pairs for Gray-Scott
RD_PRESETS = {
    "Spots":     (0.035, 0.065),
    "Labyrinth": (0.037, 0.060),
    "Stripes":   (0.060, 0.062),
    "Coral":     (0.062, 0.061),
    "Mitosis":   (0.028, 0.062),
    "Custom":    None,   # use sliders directly
}

EVOLUTION = {
    "population_size":    16,
    "generations":        10,
    "mutation_strength":  0.15,
    "crossover_rate":     0.5,
    "moth_size_min":      80,
    "moth_size_max":      160,
    "fitness_weights": {
        "color":      0.4,
        "texture":    0.35,
        "disruption": 0.25,
    },
    "pso": {
        "c1": 1.5,
        "c2": 1.5,
        "w":  0.7,
    },
}

BLEND_MODES = ["normal", "multiply", "screen", "overlay", "soft_light"]
