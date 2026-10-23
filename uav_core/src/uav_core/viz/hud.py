"""Heads-up display for offline videos and the optional hud_node.

Drawing only: every function takes what the mission already published (StepOutput,
Detections) and returns a new image; the input frame is never modified and nothing here
feeds back into decisions. Ported from the legacy draw_mission_hud / _draw_altitude_bar.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from uav_core.config import MissionConfig, RulebookConfig
from uav_core.types import Color, Detection, Phase, Round, StepOutput, TargetClass

BGR = tuple[int, int, int]

PHASE_BGR: dict[Phase, BGR] = {
    Phase.TAKEOFF: (255, 220, 50),
    Phase.SEARCHING: (180, 180, 180),
    Phase.TARGETING: (30, 200, 255),
    Phase.DROPPING: (0, 80, 255),
    Phase.RECOVERY: (0, 165, 255),
    Phase.FIND_HELIPAD: (255, 180, 60),
    Phase.LANDING: (60, 255, 180),
    Phase.COMPLETE: (0, 255, 80),
}
COLOR_BGR: dict[Color, BGR] = {
    Color.RED: (0, 40, 220),
    Color.YELLOW: (0, 220, 220),
    Color.BLUE: (220, 60, 0),
}
HELIPAD_BGR: BGR = (255, 128, 0)
NEUTRAL_BGR: BGR = (180, 180, 180)
DIM_BGR: BGR = (140, 140, 140)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PANEL_ORIGIN = (10, 10)
_PANEL_SIZE = (310, 158)
_LINE_H = 22
_ALT_BAR_MAX_M = 6.0  # the altitude bar spans 0..6 m
_HELIPAD_PHASES = (Phase.FIND_HELIPAD, Phase.LANDING)


def draw_hud(
    frame_bgr: np.ndarray,
    out: StepOutput,
    detections: Sequence[Detection],
    cfg: MissionConfig,
    rules: RulebookConfig | None = None,
    altitude_m: float | None = None,
) -> np.ndarray:
    """Detections, frame-centre crosshair, status panel, command line and altitude bar."""
    rules = rules if rules is not None else RulebookConfig()
    img = frame_bgr.copy()
    draw_detections(img, detections)
    _draw_crosshair(img)
    _draw_status_panel(img, out, cfg, rules)
    _draw_command_line(img, out)
    if altitude_m is not None:
        _draw_altitude_bar(img, altitude_m, cfg)
    return img


def draw_detections(img: np.ndarray, detections: Sequence[Detection]) -> None:
    """Circles in their perceived colour, helipads as boxes; draws in place on img."""
    for det in detections:
        if det.cls is TargetClass.H_LANDING:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), HELIPAD_BGR, 2)
            cv2.circle(img, (det.cx, det.cy), 6, HELIPAD_BGR, -1)
            continue
        color = COLOR_BGR.get(det.color, NEUTRAL_BGR) if det.color else NEUTRAL_BGR
        cv2.circle(img, (det.cx, det.cy), max(det.radius, 1), color, 2)
        cv2.drawMarker(img, (det.cx, det.cy), color, cv2.MARKER_CROSS, 40, 1)
        edge = " [EDGE]" if det.edge_clipped else ""
        tag = f"{det.color or '?'} fr={det.fill_ratio:.2f}{edge}"
        origin = (max(0, det.cx - 60), max(20, det.cy - det.radius - 10))
        cv2.putText(img, tag, origin, _FONT, 0.45, color, 1, cv2.LINE_AA)


def _draw_crosshair(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 1)


def _status_lines(out: StepOutput, cfg: MissionConfig, rules: RulebookConfig):
    s = out.status
    round_tag = "[FINALS]" if s.round is Round.FINALS else "[QUAL]"
    net = s.score + s.penalties
    lines: list[tuple[str, BGR]] = [
        (f"Phase: {s.phase.value}  {round_tag}  [{s.alt_source.value}]", PHASE_BGR[s.phase]),
        (f"Balls: {s.balls_dropped}/5  |  Score: {s.score} pts", (200, 200, 200)),
        (
            f"Penalties: {s.penalties} pts  Net: {net} pts",
            (200, 80, 80) if s.penalties < 0 else DIM_BGR,
        ),
        (
            f"Drop Z: {cfg.drop.altitude_min_m:.2f}-{cfg.drop.altitude_max_m:.2f} m  "
            f"(align +/-{cfg.drop.align_pixel_thresh_px} px)",
            DIM_BGR,
        ),
    ]
    targets = rules.targets(s.round)
    if s.phase in _HELIPAD_PHASES:
        lines.append(("Target: H_landing pad", HELIPAD_BGR))
    elif s.phase is Phase.COMPLETE:
        lines.append(("MISSION COMPLETE", PHASE_BGR[Phase.COMPLETE]))
    elif s.target_index < len(targets):
        t = targets[s.target_index]
        moving = " moving" if t.moving else ""
        lines.append(
            (
                f"Next: {t.label} {t.color.value}{moving} D={t.diameter_m:.1f} m (+{t.points})",
                COLOR_BGR[t.color],
            )
        )
    remain = max(0.0, rules.match_time_s - s.elapsed_s)
    t_color = (0, 0, 220) if remain < 20 else (0, 165, 255) if remain < 60 else (120, 120, 120)
    lines.append((f"T: {int(s.elapsed_s)} s  left: {int(remain)} s", t_color))
    return lines


def _draw_status_panel(
    img: np.ndarray, out: StepOutput, cfg: MissionConfig, rules: RulebookConfig
) -> None:
    (x, y), (w, h) = _PANEL_ORIGIN, _PANEL_SIZE
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    for i, (text, color) in enumerate(_status_lines(out, cfg, rules)):
        cv2.putText(img, text, (x + 8, y + 20 + i * _LINE_H), _FONT, 0.48, color, 1, cv2.LINE_AA)


def _draw_command_line(img: np.ndarray, out: StepOutput) -> None:
    c = out.command
    drop = "  DROP" if c.drop_ball else ""
    text = (
        f"{c.action}  vx={c.vx:+.2f} vy={c.vy:+.2f} vz={c.vz:+.2f} m/s  "
        f"yaw={c.yaw_rate_dps:+.0f} deg/s{drop}"
    )
    color = (0, 0, 255) if c.drop_ball else PHASE_BGR[c.phase]
    cv2.putText(img, text, (10, img.shape[0] - 60), _FONT, 0.5, color, 1, cv2.LINE_AA)


def _alt_x(z_m: float, x0: int, width: int) -> int:
    return x0 + int(min(max(z_m, 0.0), _ALT_BAR_MAX_M) / _ALT_BAR_MAX_M * width)


def _draw_altitude_bar(img: np.ndarray, z_m: float, cfg: MissionConfig) -> None:
    """Altitude against the drop window: green band = window, marker red when outside."""
    bx, by, bw, bh = 10, img.shape[0] - 36, 200, 14
    lo, hi = cfg.drop.altitude_min_m, cfg.drop.altitude_max_m
    x_lo, x_hi = _alt_x(lo, bx, bw), _alt_x(hi, bx, bw)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    cv2.rectangle(img, (x_lo, by), (x_hi, by + bh), (0, 80, 0), -1)
    marker = (0, 255, 0) if lo <= z_m <= hi else (0, 0, 220)
    x_z = _alt_x(z_m, bx, bw)
    cv2.line(img, (x_z, by - 2), (x_z, by + bh + 2), marker, 2)
    cv2.putText(img, f"Z={z_m:.2f} m", (bx + bw + 6, by + 11), _FONT, 0.4, (200, 200, 200), 1)
