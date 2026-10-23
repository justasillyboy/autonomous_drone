# ADR-0005: Circle geometry comes from contours on the low-resolution mask

- **Status:** Accepted (migrated from legacy decision 4, extended)
- **Date:** 2026-10-24
- **Code:** `uav_core/src/uav_core/perception/geometry.py`,
  `edge/src/uav_edge/runtime/letterbox.py`

## Context

YOLO-seg masks come out at 1/4 of the network input (160 x 160 for 640). Resizing every mask
to the full frame before finding contours is the slow path on the Pi 5. The masks also lie
on the letterboxed canvas, with padding wherever the frame's aspect ratio differs from the
network's.

## Decision

- `extract_geometry` runs `cv2.findContours` on the small mask and scales the contour to the
  frame; the radius comes from a second enclosing-circle fit on the scaled float contour, so
  non-square scaling does not bias it.
- Backends crop the letterbox padding off the masks before they reach uav_core
  (`Letterbox.crop_masks`), so a mask always spans exactly the frame.

## Consequences

- Cost per circle is a contour pass on about 19k pixels instead of 300k.
- The legacy code scaled padded masks straight to the frame. That was harmless at 640 x 480,
  which needs no padding, but would shift every circle toward the centre on a 720p frame.
  Tests cover both cases.
