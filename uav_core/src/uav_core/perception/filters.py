"""Cross-class suppression: circles that are part of the H_landing pad."""

from __future__ import annotations

from collections.abc import Sequence

Box = tuple[int, int, int, int]


def overlaps_h_landing(
    cx: int, cy: int, radius: int, h_boxes: Sequence[Box], threshold: float
) -> bool:
    """True if the circle's bounding square overlaps any H_landing box by >= threshold of
    the square's area (the pad's ring and glyph are often detected as circles)."""
    area = max(1, (2 * radius) ** 2)
    for hx1, hy1, hx2, hy2 in h_boxes:
        ox1, oy1 = max(cx - radius, hx1), max(cy - radius, hy1)
        ox2, oy2 = min(cx + radius, hx2), min(cy + radius, hy2)
        if ox2 > ox1 and oy2 > oy1 and (ox2 - ox1) * (oy2 - oy1) / area >= threshold:
            return True
    return False
