"""Field mapping between uav_core types and the uav_interfaces ROS messages.

Duck-typed on purpose: nothing here imports ROS. The nodes pass generated messages, the tests
pass SimpleNamespace objects. Field names must match ros2_ws/src/uav_interfaces/msg/*.msg.
"""

from __future__ import annotations

from typing import Any

from uav_core.types import Color, Detection, FlightCommand, MissionStatus, Phase, TargetClass


def stamp_to_s(stamp: Any) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def detection_from_msg(m: Any) -> Detection:
    return Detection(
        cls=TargetClass(m.cls),
        color=Color(m.color) if m.color else None,
        cx=int(m.cx),
        cy=int(m.cy),
        radius=int(m.radius),
        bbox=tuple(int(v) for v in m.bbox),
        fill_ratio=float(m.fill_ratio),
        conf=float(m.conf),
        edge_clipped=bool(m.edge_clipped),
    )


def fill_detection_msg(m: Any, d: Detection) -> Any:
    m.cls = d.cls.value
    m.color = d.color.value if d.color else ""
    m.cx, m.cy, m.radius = int(d.cx), int(d.cy), int(d.radius)
    m.bbox = [int(v) for v in d.bbox]
    m.fill_ratio, m.conf = float(d.fill_ratio), float(d.conf)
    m.edge_clipped = bool(d.edge_clipped)
    return m


def fill_command_msg(m: Any, c: FlightCommand) -> Any:
    m.phase, m.action = c.phase.value, c.action
    m.vx, m.vy, m.vz, m.yaw_rate_dps = float(c.vx), float(c.vy), float(c.vz), float(c.yaw_rate_dps)
    m.yaw_lock, m.drop_ball = bool(c.yaw_lock), bool(c.drop_ball)
    return m


def command_from_msg(m: Any) -> FlightCommand:
    return FlightCommand(
        phase=Phase(m.phase),
        action=m.action,
        vx=float(m.vx),
        vy=float(m.vy),
        vz=float(m.vz),
        yaw_rate_dps=float(m.yaw_rate_dps),
        yaw_lock=bool(m.yaw_lock),
        drop_ball=bool(m.drop_ball),
    )


def fill_status_msg(m: Any, s: MissionStatus, e_stopped: bool) -> Any:
    m.phase, m.round = s.phase.value, s.round.value
    m.score, m.penalties = int(s.score), int(s.penalties)
    m.balls_dropped, m.target_index = int(s.balls_dropped), int(s.target_index)
    m.alt_source, m.elapsed_s = s.alt_source.value, float(s.elapsed_s)
    m.e_stopped = bool(e_stopped)
    return m
