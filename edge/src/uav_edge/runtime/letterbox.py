"""Letterbox geometry: resize keeping the aspect ratio, pad to the network input.

Segmentation masks come back on the padded canvas. The padding must be cropped off before
uav_core scales a mask to the frame, otherwise every y (or x) coordinate is biased toward
the centre. The legacy code skipped this; it only worked because 640x480 needs no padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114  # Ultralytics' grey


@dataclass(frozen=True, slots=True)
class Letterbox:
    frame_w: int
    frame_h: int
    in_w: int
    in_h: int

    @property
    def gain(self) -> float:
        return min(self.in_w / self.frame_w, self.in_h / self.frame_h)

    @property
    def pad(self) -> tuple[float, float]:
        """(left, top) padding in network-input pixels, before rounding."""
        g = self.gain
        return (self.in_w - round(self.frame_w * g)) / 2, (self.in_h - round(self.frame_h * g)) / 2

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resized and padded network input (BGR, uint8)."""
        g = self.gain
        new_w, new_h = round(self.frame_w * g), round(self.frame_h * g)
        img = frame_bgr
        if (new_w, new_h) != (self.frame_w, self.frame_h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pw, ph = self.pad
        top, bottom = round(ph - 0.1), round(ph + 0.1)
        left, right = round(pw - 0.1), round(pw + 0.1)
        return cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
        )

    def box_to_frame(self, xyxy: np.ndarray) -> np.ndarray:
        """(N, 4) boxes from network-input pixels to frame pixels, clipped to the frame."""
        pw, ph = self.pad
        out = (xyxy - np.array([pw, ph, pw, ph])) / self.gain
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, self.frame_w)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, self.frame_h)
        return out

    def crop_masks(self, masks: np.ndarray) -> np.ndarray:
        """(N, mh, mw) masks on the padded canvas -> the region that covers the frame.

        Works at any mask resolution (network input, or the 1/4-scale prototypes), with the
        same rounding as Ultralytics' scale_masks.
        """
        mh, mw = masks.shape[-2:]
        sx, sy = mw / self.in_w, mh / self.in_h
        pw, ph = self.pad
        top, left = round(ph * sy - 0.1), round(pw * sx - 0.1)
        bottom, right = mh - round(ph * sy + 0.1), mw - round(pw * sx + 0.1)
        return masks[..., top:bottom, left:right]
