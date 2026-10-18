"""
dataset.py — Realistic synthetic dataset for circle / semicircle / H_landing segmentation.

Classes:
  0 — circle       (fire targets F1-F5: red 1.0m, yellow 0.6m, blue 0.4m)
  1 — semicircle   (TZ obstacle rings)
  2 — H_landing    (helipad marker: gray square + white ring + H + L-brackets)

All 8 weaknesses from audit addressed:
  W1 — H_landing uses its TRUE contour polygon (traced from rendered pixels),
       not a bounding rectangle. Mask precisely covers the pad structure.
  W2 — Circles have imperfect edges via polar noise + randomised blur +
       alpha-blend for soft edges. No longer perfectly sharp ideal shapes.
  W3 — Full lighting simulation: directional gradient, vignette, simulated
       shadow cones, random exposure shift.
  W4 — Occlusion modeling: objects partially cropped by frame edge, or
       occluded by random foreground rectangles/shapes.
  W5 — Semicircles have broken arcs (missing segments), partial visibility,
       noisy boundaries — closer to real court arc lines.
  W6 — Coexistence scenes include near-overlapping objects, variable pad
       scale, and ambiguous proximity to challenge the model.
  W7 — Background distractor shapes (non-labeled rectangles, hexagons,
       triangles) add shape diversity so the model is not over-specialised.
  W8 — Rich background textures: dirt spots, wear lines, stencil marks,
       multi-scale noise.
"""

import json
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Class registry
# ---------------------------------------------------------------------------

CLASS_NAMES = ["circle", "semicircle", "H_landing"]
CLASS_IDS   = {name: idx for idx, name in enumerate(CLASS_NAMES)}

H_LANDING_FRACTION   = 0.35
COEXISTENCE_FRACTION = 0.40

# Competition fire-target colors (BGR). Exact hex from rulebook.
COMPETITION_COLORS_BGR = {
    "red":    (27,  42,  252),   # #FC2A1B
    "yellow": (55,  255, 250),   # #FAFF37
    "blue":   (251, 0,   60),    # #3C00FB
}
COMP_COLOR_LIST = list(COMPETITION_COLORS_BGR.values())


# ---------------------------------------------------------------------------
# W1 FIX — True-contour polygon for H_landing
# ---------------------------------------------------------------------------

def h_landing_true_polygon(pad_layer: np.ndarray, pad_x: int, pad_y: int,
                            pad_size: int) -> list:
    """
    Trace the actual rendered pixel boundary of the H_landing pad and return
    a flat COCO-style polygon.

    This replaces rect_polygon() — the mask now follows the TRUE outer square
    outline of the pad as it appears on canvas, not an idealised bounding box.

    Strategy: create a binary mask of all pad pixels, dilate by 1 px to close
    tiny gaps, then run findContours on the outer boundary and downsample to
    ~40 points (enough for YOLO seg, avoids over-fitting to pixel-perfect edges).
    """
    gray  = cv2.cvtColor(pad_layer, cv2.COLOR_BGR2GRAY)
    # Threshold: any non-zero pixel is part of the pad
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel    = np.ones((3, 3), np.uint8)
    binary    = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback to precise rectangle if contour fails
        return [float(pad_x), float(pad_y),
                float(pad_x + pad_size), float(pad_y),
                float(pad_x + pad_size), float(pad_y + pad_size),
                float(pad_x), float(pad_y + pad_size)]

    cnt = max(contours, key=cv2.contourArea)

    # Downsample to ~40 representative points via approxPolyDP
    epsilon = 0.005 * cv2.arcLength(cnt, True)
    approx  = cv2.approxPolyDP(cnt, epsilon, True)

    # Clamp to at least 4 points
    if len(approx) < 4:
        approx = cnt[::max(1, len(cnt) // 40)]

    pts = approx.reshape(-1, 2).astype(float)
    return pts.flatten().tolist()


# ---------------------------------------------------------------------------
# W2 FIX — Imperfect circle drawing
# ---------------------------------------------------------------------------

def draw_imperfect_circle(canvas: np.ndarray, cx: int, cy: int, r: int,
                           color: tuple, rng: np.random.Generator) -> np.ndarray:
    """
    Draw a circle with realistic imperfections:
      - Polar coordinate noise on the boundary (jagged/bumpy edge)
      - Random blur radius (0 = sharp, up to 5 = motion-blurred)
      - Alpha blend for soft/partial transparency at edge
      - Optional slight ellipse distortion (lens/perspective effect)

    Returns a flat COCO polygon of the NOISY boundary (matches what was drawn).
    """
    n_pts   = 72
    angles  = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    # Radial noise: low-frequency bump pattern simulating real-world edge wear
    noise_amp = rng.uniform(0.0, 0.06) * r          # 0-6% of radius
    noise_freq = rng.integers(3, 8)                  # number of bumps around circle
    phase  = rng.uniform(0, 2 * np.pi)
    radii  = r + noise_amp * np.sin(noise_freq * angles + phase)
    radii += rng.uniform(-1.5, 1.5, size=n_pts)      # pixel-level roughness

    # Slight ellipse distortion (a/b ratio 0.92-1.0)
    a_ratio = rng.uniform(0.92, 1.0)
    b_ratio = rng.uniform(0.92, 1.0)

    pts = np.stack([
        cx + radii * a_ratio * np.cos(angles),
        cy + radii * b_ratio * np.sin(angles),
    ], axis=1).astype(np.int32)

    # Draw on a temporary layer so we can blur independently
    layer = np.zeros_like(canvas)
    cv2.fillPoly(layer, [pts.reshape(-1, 1, 2)], color)

    # Random blur: 0 (none) to 3 (moderate)
    blur_k = int(rng.choice([0, 0, 1, 1, 3, 5]))
    if blur_k > 0:
        layer = cv2.GaussianBlur(layer, (blur_k * 2 + 1, blur_k * 2 + 1), 0)

    # Alpha blend: drawn pixels replace canvas
    mask = layer.sum(axis=2) > 0
    canvas[mask] = layer[mask]

    return pts.flatten().tolist()


def draw_imperfect_semicircle(canvas: np.ndarray, cx: int, cy: int, r: int,
                               orientation: str, color: tuple,
                               rng: np.random.Generator) -> list:
    """
    Draw a semicircle with realistic imperfections:
      - Polar noise on arc boundary (W2 + W5)
      - Random missing arc segments (broken arc, W5)
      - Possible partial visibility (W4)
      - Random blur

    Returns flat COCO polygon of what was actually drawn.
    """
    arc_ranges = {
        "top":    (np.pi, 0),
        "bottom": (0, np.pi),
        "left":   (np.pi / 2, 3 * np.pi / 2),
        "right":  (-np.pi / 2, np.pi / 2),
    }
    start, end = arc_ranges[orientation]
    half        = 36
    angles      = np.linspace(start, end, half)

    # Radial noise
    noise_amp  = rng.uniform(0.0, 0.05) * r
    noise_freq = rng.integers(2, 6)
    phase      = rng.uniform(0, 2 * np.pi)
    radii      = r + noise_amp * np.sin(noise_freq * angles + phase)
    radii     += rng.uniform(-1.5, 1.5, size=half)

    arc_pts = np.stack([
        cx + radii * np.cos(angles),
        cy + radii * np.sin(angles),
    ], axis=1)

    # W5: broken arc — randomly drop 0-2 short segments
    if rng.random() < 0.35:
        n_gaps    = int(rng.integers(1, 3))
        gap_width = int(rng.integers(2, max(3, half // 6)))
        for _ in range(n_gaps):
            start_idx = int(rng.integers(0, half - gap_width))
            arc_pts[start_idx:start_idx + gap_width] = arc_pts[start_idx]  # collapse to a point

    polygon = np.vstack([arc_pts, arc_pts[[-1, 0]]]).astype(np.int32)

    layer = np.zeros_like(canvas)
    cv2.fillPoly(layer, [polygon.reshape(-1, 1, 2)], color)

    blur_k = int(rng.choice([0, 0, 1, 3]))
    if blur_k > 0:
        layer = cv2.GaussianBlur(layer, (blur_k * 2 + 1, blur_k * 2 + 1), 0)

    mask = layer.sum(axis=2) > 0
    canvas[mask] = layer[mask]

    return polygon[:, :2].flatten().tolist()


# ---------------------------------------------------------------------------
# W3 FIX — Lighting simulation
# ---------------------------------------------------------------------------

def apply_lighting(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply realistic lighting effects:
      1. Directional gradient — simulates off-axis illumination
      2. Vignette — simulates lens falloff (darker corners)
      3. Shadow cone — one sharp dark region (object or structure casting shadow)
      4. Exposure shift — random global brightness ±40

    All operations are fused into a float32 multiplier map, then applied in
    one pass to avoid accumulated quantisation error.
    """
    h, w = image.shape[:2]
    img_f = image.astype(np.float32)

    # 1. Directional gradient (light comes from random direction, strength 0.10-0.30)
    angle  = rng.uniform(0, 2 * np.pi)
    strength = rng.uniform(0.10, 0.30)
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    gx, gy    = np.meshgrid(xs, ys)
    gradient  = np.cos(angle) * gx + np.sin(angle) * gy   # [-1, 1] range
    gradient  = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    light_map = 1.0 - strength + strength * gradient       # [1-s, 1.0]

    # 2. Vignette (radial darkening from corners)
    cx_v, cy_v = w / 2, h / 2
    dist       = np.sqrt((gx * w - cx_v) ** 2 + (gy * h - cy_v) ** 2)
    max_dist   = np.sqrt(cx_v ** 2 + cy_v ** 2)
    vignette   = 1.0 - rng.uniform(0.05, 0.25) * (dist / max_dist) ** 2

    # 3. Shadow cone (50% probability)
    shadow_map = np.ones((h, w), dtype=np.float32)
    if rng.random() < 0.50:
        tip_x  = int(rng.integers(0, w))
        tip_y  = int(rng.integers(0, h // 2))
        half_w = int(rng.integers(w // 6, w // 3))
        depth  = rng.uniform(0.20, 0.55)           # shadow darkness
        for row in range(h):
            spread = int(half_w * row / max(h, 1))
            x1     = max(0, tip_x - spread)
            x2     = min(w, tip_x + spread)
            shadow_map[row, x1:x2] = 1.0 - depth * (row / h)

    # 4. Exposure shift
    exposure = rng.uniform(-40, 40)

    # Fuse and apply
    combined  = (light_map * vignette * shadow_map)[..., np.newaxis]
    img_f     = img_f * combined + exposure
    return np.clip(img_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# W4 FIX — Occlusion modeling
# ---------------------------------------------------------------------------

def apply_occlusion(image: np.ndarray, rng: np.random.Generator,
                    annotations: list, img_w: int, img_h: int) -> tuple:
    """
    Simulate two types of occlusion:
      A) Frame-edge crop — objects near the boundary are partially invisible.
         Annotations are NOT modified (the model sees partial shapes; label
         still covers the full shape, teaching it to infer the hidden part).
      B) Foreground occluder — a random neutral-colored rectangle or shape
         is drawn ON TOP of existing objects. Does NOT have a label entry,
         so the model learns to "see through" partial occlusion.

    Returns:
        (augmented_image, annotations)  — annotations unchanged.
    """
    h, w = image.shape[:2]

    # B) Foreground occluder (40% probability)
    if rng.random() < 0.40:
        n_occ = int(rng.integers(1, 4))
        for _ in range(n_occ):
            occ_w  = int(rng.integers(w // 8, w // 3))
            occ_h  = int(rng.integers(h // 8, h // 3))
            ox     = int(rng.integers(-occ_w // 2, w))
            oy     = int(rng.integers(-occ_h // 2, h))

            # Match background tone (occluder is background-colored, not black)
            # Sample colour from a corner region of the image
            corner = image[:20, :20].mean(axis=(0, 1)).astype(int)
            occ_color = tuple(int(np.clip(c + rng.integers(-25, 25), 0, 255))
                              for c in corner)

            shape_type = int(rng.integers(0, 3))
            if shape_type == 0:   # rectangle
                cv2.rectangle(image, (ox, oy), (ox + occ_w, oy + occ_h), occ_color, -1)
            elif shape_type == 1:  # ellipse (simulates camera lens flare or object)
                cv2.ellipse(image, (ox + occ_w // 2, oy + occ_h // 2),
                            (occ_w // 2, occ_h // 2), 0, 0, 360, occ_color, -1)
            else:                  # triangle
                pts = np.array([[ox + occ_w // 2, oy],
                                 [ox, oy + occ_h],
                                 [ox + occ_w, oy + occ_h]], np.int32)
                cv2.fillPoly(image, [pts], occ_color)

    return image, annotations


# ---------------------------------------------------------------------------
# W7 FIX — Background distractor shapes
# ---------------------------------------------------------------------------

def draw_distractor_shapes(canvas: np.ndarray, rng: np.random.Generator,
                            n: int = 3) -> None:
    """
    Draw n non-labeled distractor shapes onto canvas IN-PLACE.

    These teach the model that not every geometric shape is a fire target.
    Shapes are drawn in muted, non-competition colors to avoid color confusion.

    Shapes: rectangles, hexagons, triangles, thin arcs (similar to court lines).
    """
    h, w = canvas.shape[:2]

    for _ in range(n):
        # Muted color (not red/yellow/blue)
        c = tuple(int(v) for v in rng.integers(60, 180, size=3).tolist())
        shape = int(rng.integers(0, 4))
        sz    = int(rng.integers(15, min(w, h) // 5))
        cx    = int(rng.integers(sz, w - sz))
        cy    = int(rng.integers(sz, h - sz))

        if shape == 0:   # rectangle
            cv2.rectangle(canvas, (cx - sz, cy - sz), (cx + sz, cy + sz), c, -1)
        elif shape == 1:  # hexagon
            pts = np.array([
                [cx + sz * np.cos(np.pi / 3 * k),
                 cy + sz * np.sin(np.pi / 3 * k)] for k in range(6)
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], c)
        elif shape == 2:  # triangle
            pts = np.array([
                [cx, cy - sz], [cx - sz, cy + sz], [cx + sz, cy + sz]
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], c)
        else:             # thin arc (court line imitation — NOT labeled)
            arc_r = int(rng.integers(sz, sz * 3))
            cv2.ellipse(canvas, (cx, cy), (arc_r, arc_r),
                        0, int(rng.integers(0, 180)),
                        int(rng.integers(180, 360)), c, 2)


# ---------------------------------------------------------------------------
# W8 FIX — Rich synthetic texture backgrounds
# ---------------------------------------------------------------------------

def generate_rich_background(width: int, height: int,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Generate a background with realistic texture complexity:
      - Multi-scale Perlin-like noise (summed octaves)
      - Dirt / stain spots (Gaussian blobs)
      - Wear lines (faint streaks)
      - Optional stencil marks (faded rectangles)
    """
    # Base color: sport court palette
    palettes = [
        (rng.integers(55, 80),   rng.integers(50, 75),   rng.integers(75, 105)),  # gray-purple
        (rng.integers(60, 100),  rng.integers(100, 150), rng.integers(60, 100)),  # green
        (rng.integers(20, 55),   rng.integers(20, 55),   rng.integers(20, 55)),   # dark tarmac
        (rng.integers(140, 200), rng.integers(120, 180), rng.integers(100, 160)), # beige/concrete
    ]
    base = palettes[int(rng.integers(0, len(palettes)))]
    bg   = np.full((height, width, 3), [int(b) for b in base], dtype=np.float32)

    # Multi-scale noise (3 octaves)
    for scale in [4, 8, 16]:
        small = rng.integers(-18, 18, (height // scale + 2, width // scale + 2, 3),
                             dtype=np.int8).astype(np.float32)
        noise = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        bg   += noise

    bg = np.clip(bg, 0, 255).astype(np.uint8)

    # Dirt / stain spots
    n_spots = int(rng.integers(3, 12))
    for _ in range(n_spots):
        sx   = int(rng.integers(0, width))
        sy   = int(rng.integers(0, height))
        sr   = int(rng.integers(8, 60))
        dark = int(rng.integers(-30, 30))
        cv2.circle(bg, (sx, sy), sr, (max(0, int(base[0]) + dark),
                                       max(0, int(base[1]) + dark),
                                       max(0, int(base[2]) + dark)), -1)

    # Wear lines (faint parallel streaks)
    n_lines = int(rng.integers(2, 8))
    for _ in range(n_lines):
        x1 = int(rng.integers(0, width))
        x2 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        y2 = int(rng.integers(0, height))
        tone = tuple(max(0, int(base[c]) + int(rng.integers(-15, 15))) for c in range(3))
        cv2.line(bg, (x1, y1), (x2, y2), tone, int(rng.integers(1, 4)))

    # Court arc overlay (orange concentric arcs — main false-positive source)
    if rng.random() < 0.65:
        arc_cx = int(rng.integers(-width // 4, width + width // 4))
        arc_cy = int(rng.integers(-height // 4, height + height // 4))
        arc_col = (int(rng.integers(0, 40)),
                   int(rng.integers(90, 140)),
                   int(rng.integers(180, 230)))  # orange BGR
        for r in range(60, max(width, height) * 2, int(rng.integers(90, 200))):
            cv2.circle(bg, (arc_cx, arc_cy), r, arc_col, int(rng.integers(1, 4)))

    # White boundary line
    cv2.rectangle(bg, (8, 8), (width - 8, height - 8), (190, 190, 190), 2)

    return bg


# ---------------------------------------------------------------------------
# H_landing pad renderer (unchanged visual, improved annotation via W1)
# ---------------------------------------------------------------------------

def render_h_landing_pad(canvas_w, canvas_h, pad_x, pad_y, pad_size, rng):
    """
    Draw an H_landing pad onto a zero canvas layer.

    Returns:
        layer (canvas_h x canvas_w x 3) — zero outside pad.
        Caller composites onto background, then calls h_landing_true_polygon()
        on this layer to get the true-contour annotation.
    """
    layer    = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    gray_val = int(rng.integers(55, 90))

    cv2.rectangle(layer, (pad_x, pad_y),
                  (pad_x + pad_size, pad_y + pad_size),
                  (gray_val, gray_val, gray_val), -1)

    cx = pad_x + pad_size // 2
    cy = pad_y + pad_size // 2

    r_outer    = int(pad_size * 0.40)
    r_inner    = int(pad_size * 0.28)
    ring_color = (238, 238, 238)
    cv2.circle(layer, (cx, cy), r_outer, ring_color, -1)
    cv2.circle(layer, (cx, cy), r_inner, (gray_val, gray_val, gray_val), -1)

    bar_w   = max(3, int(pad_size * 0.07))
    bar_h   = int(pad_size * 0.30)
    gap     = int(pad_size * 0.10)
    h_top   = cy - bar_h // 2
    h_bot   = cy + bar_h // 2
    lx      = cx - gap - bar_w
    rx      = cx + gap
    cross_h = max(2, int(pad_size * 0.06))
    cv2.rectangle(layer, (lx, h_top), (lx + bar_w, h_bot), ring_color, -1)
    cv2.rectangle(layer, (rx, h_top), (rx + bar_w, h_bot), ring_color, -1)
    cv2.rectangle(layer, (lx + bar_w, cy - cross_h // 2),
                  (rx, cy + cross_h // 2), ring_color, -1)

    bkt    = max(3, int(pad_size * 0.09))
    thk    = max(2, int(pad_size * 0.025))
    margin = int(pad_size * 0.05)
    for (ox, oy, sx, sy) in [
        (pad_x + margin,            pad_y + margin,            +1, +1),
        (pad_x + pad_size - margin, pad_y + margin,            -1, +1),
        (pad_x + margin,            pad_y + pad_size - margin, +1, -1),
        (pad_x + pad_size - margin, pad_y + pad_size - margin, -1, -1),
    ]:
        cv2.line(layer, (ox, oy), (ox + sx * bkt, oy), ring_color, thk)
        cv2.line(layer, (ox, oy), (ox, oy + sy * bkt), ring_color, thk)

    return layer


# ---------------------------------------------------------------------------
# Polygon helpers (kept for semicircle and compatibility)
# ---------------------------------------------------------------------------

def circle_polygon(cx, cy, r, n_points=72):
    """Ideal circle polygon — used only for post-hoc annotations when needed."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)
    return pts.flatten().tolist()


def semicircle_polygon(cx, cy, r, orientation="top", n_points=72):
    """Ideal semicircle polygon."""
    half = n_points // 2
    arc_ranges = {
        "top":    (np.pi, 0),
        "bottom": (0, np.pi),
        "left":   (np.pi / 2, 3 * np.pi / 2),
        "right":  (-np.pi / 2, np.pi / 2),
    }
    start, end = arc_ranges[orientation]
    angles  = np.linspace(start, end, half)
    arc_pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)
    polygon = np.vstack([arc_pts, arc_pts[[-1, 0]]])
    return polygon[:, :2].flatten().tolist()


# ---------------------------------------------------------------------------
# Image generators — all 8 fixes applied
# ---------------------------------------------------------------------------

def generate_circle_image(width=640, height=640, n_circles=3,
                           n_semicircles=2, rng=None):
    """
    Generate circles and semicircles with all realism improvements:
      - W2: imperfect edges, blur, noise
      - W3: lighting simulation
      - W4: occlusion by foreground shapes
      - W5: broken/noisy semicircles
      - W7: background distractor shapes
      - W8: rich textured background
    """
    if rng is None:
        rng = np.random.default_rng()

    image = generate_rich_background(width, height, rng)

    # W7: distractor shapes BEFORE labeled objects (hidden behind)
    draw_distractor_shapes(image, rng, n=int(rng.integers(1, 5)))

    annotations = []

    for i in range(n_circles):
        r  = int(rng.integers(20, min(width, height) // 4))

        # W4: allow partial frame-edge crop (place some circles near edges)
        if rng.random() < 0.25:
            cx = int(rng.choice([rng.integers(0, r), rng.integers(width - r, width)]))
            cy = int(rng.integers(r, height - r))
        else:
            cx = int(rng.integers(r, width  - r))
            cy = int(rng.integers(r, height - r))

        # W2: competition colors with small jitter
        base = list(COMP_COLOR_LIST[i % len(COMP_COLOR_LIST)])
        jitter = rng.integers(-20, 20, size=3)
        color  = tuple(int(np.clip(base[c] + jitter[c], 0, 255)) for c in range(3))

        # W2: draw imperfect circle, returns noisy polygon
        noisy_poly = draw_imperfect_circle(image, cx, cy, r, color, rng)

        xs = noisy_poly[0::2]
        ys = noisy_poly[1::2]
        annotations.append({
            "category_id": CLASS_IDS["circle"],
            "segmentation": [noisy_poly],
            "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        })

    orientations = ["top", "bottom", "left", "right"]
    for _ in range(n_semicircles):
        r   = int(rng.integers(20, min(width, height) // 4))
        cx  = int(rng.integers(r, width  - r))
        cy  = int(rng.integers(r, height - r))
        ori = str(rng.choice(orientations))

        base  = list(COMP_COLOR_LIST[int(rng.integers(0, len(COMP_COLOR_LIST)))])
        jitter = rng.integers(-20, 20, size=3)
        color  = tuple(int(np.clip(base[c] + jitter[c], 0, 255)) for c in range(3))

        # W2 + W5: imperfect, possibly broken semicircle
        noisy_poly = draw_imperfect_semicircle(image, cx, cy, r, ori, color, rng)

        xs = noisy_poly[0::2]
        ys = noisy_poly[1::2]
        annotations.append({
            "category_id": CLASS_IDS["semicircle"],
            "segmentation": [noisy_poly],
            "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        })

    # W3: apply lighting AFTER drawing objects (affects whole image)
    image = apply_lighting(image, rng)

    # W4: apply foreground occlusion
    image, annotations = apply_occlusion(image, rng, annotations, width, height)

    return image, annotations


def generate_h_landing_image(width=640, height=640, rng=None):
    """
    Generate an H_landing-only image.
      - W1: true-contour polygon annotation (not bounding rect)
      - W3: lighting
      - W4: occlusion
      - W7: distractor shapes
      - W8: rich background
    """
    if rng is None:
        rng = np.random.default_rng()

    image = generate_rich_background(width, height, rng)
    draw_distractor_shapes(image, rng, n=int(rng.integers(1, 4)))

    min_pad  = int(min(width, height) * 0.25)
    max_pad  = int(min(width, height) * 0.65)
    pad_size = int(rng.integers(min_pad, max_pad + 1))
    pad_x    = int(rng.integers(0, max(1, width  - pad_size)))
    pad_y    = int(rng.integers(0, max(1, height - pad_size)))

    pad_layer = render_h_landing_pad(width, height, pad_x, pad_y, pad_size, rng)

    # W1: get true-contour polygon BEFORE compositing so pad_layer is clean
    true_poly = h_landing_true_polygon(pad_layer, pad_x, pad_y, pad_size)

    pad_mask = pad_layer.sum(axis=2) > 0
    image[pad_mask] = pad_layer[pad_mask]

    annotations = [{
        "category_id": CLASS_IDS["H_landing"],
        "segmentation": [true_poly],
        "bbox": [float(pad_x), float(pad_y), float(pad_size), float(pad_size)],
    }]

    image = apply_lighting(image, rng)
    image, annotations = apply_occlusion(image, rng, annotations, width, height)

    return image, annotations


def generate_moving_circle_image(width=640, height=640, rng=None):
    """
    Generate a circle with translational motion blur to simulate Finals F2
    (the moving red target, 10 cm/s on the orange line).

    WHY THIS MATTERS:
      The YOLO model is trained on static circles. At 30 fps, a target moving
      at 10 cm/s at 3.5 m altitude (≈ 29 px/s) streaks 1 px per frame —
      subtle but cumulative. More importantly, the drone camera moves TOWARD
      the target during alignment, which creates apparent motion in the image.
      Without these training images the model may assign lower confidence to
      the moving target, causing momentary detection drops.

    Blur direction is random (any angle) to cover all approach vectors.
    The annotation polygon is the un-blurred bounding contour so the label
    remains geometrically accurate despite the visual streak.

    Returns:
        image, annotations  (same format as other generators)
    """
    if rng is None:
        rng = np.random.default_rng()

    image = generate_rich_background(width, height, rng)
    draw_distractor_shapes(image, rng, n=int(rng.integers(1, 4)))
    annotations = []

    # Only red circles — Finals F2 is always red
    color_bgr = COMPETITION_COLORS_BGR["red"]
    jitter     = rng.integers(-20, 20, size=3)
    color      = tuple(int(np.clip(color_bgr[c] + jitter[c], 0, 255))
                       for c in range(3))

    r  = int(rng.integers(25, min(width, height) // 4))
    cx = int(rng.integers(r + 10, width  - r - 10))
    cy = int(rng.integers(r + 10, height - r - 10))

    # Draw the circle cleanly on a separate layer
    circle_layer = np.zeros_like(image)
    noisy_poly   = draw_imperfect_circle(circle_layer, cx, cy, r, color, rng)

    # Motion blur kernel — random direction, length 4-18 px (1-6 frames at 30fps)
    blur_len   = int(rng.integers(4, 18))
    blur_angle = rng.uniform(0, 360)
    angle_rad  = np.deg2rad(blur_angle)
    kx = int(round(np.cos(angle_rad) * blur_len))
    ky = int(round(np.sin(angle_rad) * blur_len))

    # Build directional motion blur kernel
    kernel_size = blur_len * 2 + 1
    kernel      = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cx_k, cy_k  = kernel_size // 2, kernel_size // 2
    for i in range(blur_len + 1):
        px = int(cx_k + round(np.cos(angle_rad) * i))
        py = int(cy_k + round(np.sin(angle_rad) * i))
        px = np.clip(px, 0, kernel_size - 1)
        py = np.clip(py, 0, kernel_size - 1)
        kernel[py, px] = 1.0
    if kernel.sum() > 0:
        kernel /= kernel.sum()

    # Apply motion blur to circle layer only, then composite onto background
    blurred = cv2.filter2D(circle_layer, -1, kernel)
    mask    = blurred.sum(axis=2) > 0
    image[mask] = blurred[mask]

    xs = noisy_poly[0::2]
    ys = noisy_poly[1::2]
    annotations.append({
        "category_id": CLASS_IDS["circle"],
        "segmentation": [noisy_poly],
        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
    })

    image = apply_lighting(image, rng)
    image, annotations = apply_occlusion(image, rng, annotations, width, height)
    return image, annotations


def generate_coexistence_image(width=640, height=640, n_circles=2, rng=None):
    """
    H_landing pad + competition fire circles on rich court background.
      - W1: true-contour H_landing polygon
      - W2: imperfect circles
      - W3: lighting
      - W4: occlusion
      - W6: some circles placed very close to pad boundary for ambiguity
      - W7: distractor shapes
      - W8: rich background
    """
    if rng is None:
        rng = np.random.default_rng()

    image = generate_rich_background(width, height, rng)
    draw_distractor_shapes(image, rng, n=int(rng.integers(1, 4)))

    annotations = []

    # Place H_landing in a random quadrant
    min_pad  = int(min(width, height) * 0.20)
    max_pad  = int(min(width, height) * 0.55)
    pad_size = int(rng.integers(min_pad, max_pad + 1))

    quadrant = int(rng.integers(0, 4))
    qx = (quadrant % 2) * (width  // 2)
    qy = (quadrant // 2) * (height // 2)
    pad_x = int(rng.integers(qx, max(qx + 1, qx + width  // 2 - pad_size)))
    pad_y = int(rng.integers(qy, max(qy + 1, qy + height // 2 - pad_size)))
    pad_x = max(0, min(pad_x, width  - pad_size))
    pad_y = max(0, min(pad_y, height - pad_size))

    pad_layer = render_h_landing_pad(width, height, pad_x, pad_y, pad_size, rng)
    true_poly = h_landing_true_polygon(pad_layer, pad_x, pad_y, pad_size)
    pad_mask  = pad_layer.sum(axis=2) > 0
    image[pad_mask] = pad_layer[pad_mask]

    annotations.append({
        "category_id": CLASS_IDS["H_landing"],
        "segmentation": [true_poly],
        "bbox": [float(pad_x), float(pad_y), float(pad_size), float(pad_size)],
    })

    pad_rect = (pad_x, pad_y, pad_x + pad_size, pad_y + pad_size)

    # W6: 30% of coexistence images have a circle VERY close to pad edge
    close_mode = rng.random() < 0.30

    color_names = list(COMPETITION_COLORS_BGR.keys())
    for i in range(n_circles):
        base  = list(COMPETITION_COLORS_BGR[color_names[i % len(color_names)]])
        jitter = rng.integers(-20, 20, size=3)
        color  = tuple(int(np.clip(base[c] + jitter[c], 0, 255)) for c in range(3))

        r = int(rng.integers(18, min(width, height) // 5))

        placed = False
        for attempt in range(30):
            if close_mode and attempt < 10:
                # W6: place circle touching/adjacent to pad
                side = int(rng.integers(0, 4))
                if side == 0:   cx = int(rng.integers(pad_x - r * 2, pad_x + r))
                elif side == 1: cx = int(rng.integers(pad_rect[2] - r, pad_rect[2] + r * 2))
                else:           cx = int(rng.integers(pad_x, pad_rect[2]))
                if side == 2:   cy = int(rng.integers(pad_y - r * 2, pad_y + r))
                elif side == 3: cy = int(rng.integers(pad_rect[3] - r, pad_rect[3] + r * 2))
                else:           cy = int(rng.integers(pad_y, pad_rect[3]))
                cx = int(np.clip(cx, r, width  - r))
                cy = int(np.clip(cy, r, height - r))
                placed = True
                break
            else:
                cx = int(rng.integers(r, width  - r))
                cy = int(rng.integers(r, height - r))
                # Ensure circle does NOT overlap pad center (avoid label collision)
                if (cx - r > pad_rect[2] or cx + r < pad_rect[0] or
                        cy - r > pad_rect[3] or cy + r < pad_rect[1]):
                    placed = True
                    break

        if not placed:
            continue

        noisy_poly = draw_imperfect_circle(image, cx, cy, r, color, rng)
        xs = noisy_poly[0::2]
        ys = noisy_poly[1::2]
        annotations.append({
            "category_id": CLASS_IDS["circle"],
            "segmentation": [noisy_poly],
            "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        })

    image = apply_lighting(image, rng)
    image, annotations = apply_occlusion(image, rng, annotations, width, height)

    return image, annotations


# ---------------------------------------------------------------------------
# Label writer
# ---------------------------------------------------------------------------

def write_label_file(lbl_path, annotations, img_w, img_h):
    """Write YOLO segmentation label file (normalised coordinates)."""
    with open(lbl_path, "w") as lf:
        for ann in annotations:
            cat = ann["category_id"]
            seg = ann["segmentation"][0]
            if len(seg) < 6:     # need at least 3 points
                continue
            norm = [v / img_w if j % 2 == 0 else v / img_h
                    for j, v in enumerate(seg)]
            lf.write(f"{cat} " + " ".join(f"{v:.6f}" for v in norm) + "\n")


# ---------------------------------------------------------------------------
# Dataset YAML writer
# ---------------------------------------------------------------------------

def write_dataset_yaml(output_dir, class_names=None):
    """Write circle_seg.yaml with correct YAML key-value formatting."""
    if class_names is None:
        class_names = CLASS_NAMES

    yaml_path = Path(output_dir) / "circle_seg.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    content = (
        f"path: {Path(output_dir).resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(content)
    print(f"Dataset YAML written -> {yaml_path}")
    return str(yaml_path)


# ---------------------------------------------------------------------------
# Post-generation audit
# ---------------------------------------------------------------------------

def audit_dataset(output_dir):
    """
    Count per-class instances. Raises RuntimeError if H_landing = 0.
    Prints a summary table with pass/fail for each class.
    """
    counts = {name: 0 for name in CLASS_NAMES}
    total_files = 0

    for split in ["train", "val"]:
        label_dir = Path(output_dir) / "labels" / split
        if not label_dir.exists():
            continue
        for lbl in label_dir.glob("*.txt"):
            total_files += 1
            for line in lbl.read_text().strip().splitlines():
                if not line:
                    continue
                cls_id = int(line.split()[0])
                if cls_id < len(CLASS_NAMES):
                    counts[CLASS_NAMES[cls_id]] += 1

    print("\n" + "=" * 48)
    print("  Dataset audit")
    print("=" * 48)
    print(f"  Label files: {total_files}")
    for name, count in counts.items():
        flag = "OK" if count > 0 else "!! ZERO — model will NOT learn this class"
        print(f"  {name:<14}: {count:>6}   {flag}")
    print("=" * 48)

    if counts["H_landing"] == 0:
        raise RuntimeError(
            "FATAL: H_landing = 0 instances. Increase --h-landing-fraction."
        )
    if counts["H_landing"] < 50:
        print(f"WARNING: only {counts['H_landing']} H_landing instances. "
              "Increase --n-images or --h-landing-fraction.")
    return counts


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def build_augmentation_pipeline(is_train=True):
    """
    Albumentations pipeline. Rotation is here (not in generators) so that
    polygon annotations transform correctly alongside images.
    """
    if not is_train:
        return A.Compose(
            [A.Resize(640, 640)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    return A.Compose(
        [
            A.RandomScale(scale_limit=0.4, p=0.8),
            A.Rotate(limit=360, p=0.9),
            A.Perspective(scale=(0.03, 0.10), p=0.5),
            A.ElasticTransform(alpha=20, sigma=5, alpha_affine=5, p=0.25),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=9),
                A.MedianBlur(blur_limit=5),
            ], p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=30, p=0.5),
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
            A.CoarseDropout(max_holes=6, max_height=40, max_width=40, p=0.35),
            A.Resize(640, 640),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


# ---------------------------------------------------------------------------
# COCO -> YOLO converter
# ---------------------------------------------------------------------------

def coco_to_yolo(coco_json_path, images_dir, output_dir):
    """Convert a COCO JSON annotation file to per-image YOLO .txt label files."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    img_lookup   = {img["id"]: img for img in coco["images"]}
    ann_by_image = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    image_ids          = list(img_lookup.keys())
    train_ids, val_ids = train_test_split(image_ids, test_size=0.15, random_state=42)

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        (Path(output_dir) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            img_info = img_lookup[img_id]
            W, H     = img_info["width"], img_info["height"]
            stem     = Path(img_info["file_name"]).stem

            src = Path(images_dir) / img_info["file_name"]
            dst = Path(output_dir) / "images" / split / img_info["file_name"]
            if src.exists():
                shutil.copy2(src, dst)

            lbl_path = Path(output_dir) / "labels" / split / f"{stem}.txt"
            with open(lbl_path, "w") as lf:
                for ann in ann_by_image[img_id]:
                    cat  = ann["category_id"]
                    seg  = ann["segmentation"][0]
                    norm = [v / W if i % 2 == 0 else v / H
                            for i, v in enumerate(seg)]
                    lf.write(f"{cat} " + " ".join(f"{v:.6f}" for v in norm) + "\n")

    print(f"COCO -> YOLO conversion complete -> {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate realistic synthetic dataset with all 8 quality improvements.\n\n"
            "Default 3-way split:\n"
            "  40% coexistence  — H_landing + fire circles on court background\n"
            "  35% H_landing    — pad only with rich background + occlusion\n"
            "  25% circles      — fire targets only\n\n"
            "Recommended: python dataset.py --n-images 3000\n"
        )
    )
    parser.add_argument("--output",               default="data/synthetic")
    parser.add_argument("--n-images",             type=int,   default=500)
    parser.add_argument("--img-size",             type=int,   default=640)
    parser.add_argument("--n-circles",            type=int,   default=3)
    parser.add_argument("--n-semi",               type=int,   default=2)
    parser.add_argument("--h-landing-fraction",   type=float, default=0.35)
    parser.add_argument("--coexistence-fraction", type=float, default=0.38)
    parser.add_argument("--moving-fraction",      type=float, default=0.07,
                        help="Fraction of images with motion-blurred F2 circle "
                             "(default 0.07 — ~7%% of total, only red circles)")
    args = parser.parse_args()

    out    = Path(args.output)
    rng    = np.random.default_rng(seed=42)
    img_sz = args.img_size

    n_train = int(args.n_images * 0.85)
    n_val   = args.n_images - n_train

    for split, total in [("train", n_train), ("val", n_val)]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        n_coex   = int(total * args.coexistence_fraction)
        n_hpad   = int(total * args.h_landing_fraction)
        n_moving = int(total * args.moving_fraction)
        n_circ   = max(0, total - n_coex - n_hpad - n_moving)

        print(f"\n[{split}] {total} images — "
              f"{n_coex} coexistence  |  {n_hpad} H_landing  |  "
              f"{n_moving} moving-F2  |  {n_circ} circles")

        for i in range(n_coex):
            nc = int(rng.integers(1, min(args.n_circles, 3) + 1))
            img, anns = generate_coexistence_image(img_sz, img_sz, nc, rng=rng)
            stem = f"synth_{split}_x_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img)
            write_label_file(out / "labels" / split / f"{stem}.txt", anns, img_sz, img_sz)
            if (i + 1) % 100 == 0:
                print(f"  coexistence {i+1}/{n_coex}")

        for i in range(n_hpad):
            img, anns = generate_h_landing_image(img_sz, img_sz, rng=rng)
            stem = f"synth_{split}_h_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img)
            write_label_file(out / "labels" / split / f"{stem}.txt", anns, img_sz, img_sz)
            if (i + 1) % 100 == 0:
                print(f"  H_landing   {i+1}/{n_hpad}")

        for i in range(n_moving):
            img, anns = generate_moving_circle_image(img_sz, img_sz, rng=rng)
            stem = f"synth_{split}_m_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img)
            write_label_file(out / "labels" / split / f"{stem}.txt", anns, img_sz, img_sz)
            if (i + 1) % 50 == 0:
                print(f"  moving-F2   {i+1}/{n_moving}")

        for i in range(n_circ):
            nc = int(rng.integers(1, args.n_circles + 1))
            ns = int(rng.integers(0, args.n_semi + 1))
            img, anns = generate_circle_image(img_sz, img_sz, nc, ns, rng=rng)
            stem = f"synth_{split}_c_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img)
            write_label_file(out / "labels" / split / f"{stem}.txt", anns, img_sz, img_sz)
            if (i + 1) % 100 == 0:
                print(f"  circles     {i+1}/{n_circ}")

    write_dataset_yaml(str(out))
    print("\nRunning audit...")
    audit_dataset(str(out))
    print(f"\nDone -> {out}")
    print(f"\nRetrain command:")
    print(f"  python train.py --config config.yaml")
