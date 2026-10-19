"""Run uav_core (perception + MissionStateMachine) on a scenario and record its commands."""

from __future__ import annotations

from uav_core.config import CameraConfig, MissionConfig, TimingConfig
from uav_core.mission import MissionStateMachine
from uav_core.perception import RawInstance, detect
from uav_core.types import Round, SensorSnapshot, TargetClass

from characterisation.records import Record
from characterisation.scene import F_PX, FRAME_H, FRAME_W, Scenario, iter_steps


def golden_config(rate_hz: int) -> MissionConfig:
    """Defaults, plus the legacy first-step dt (1 / fps_hint) and the scene's focal length."""
    return MissionConfig(
        timing=TimingConfig(first_step_dt_s=1.0 / rate_hz), camera=CameraConfig(f_px=F_PX)
    )


def run_new(scenario: Scenario) -> list[Record]:
    cfg = golden_config(scenario.rate_hz)
    sm = MissionStateMachine(cfg, round_=Round.FINALS if scenario.finals else Round.QUALIFIER)
    records = []
    for step in iter_steps(scenario):
        for event in step.events:
            getattr(sm, event)()
        instances = [
            RawInstance(TargetClass(i.cls_name), i.mask, i.xyxy, i.conf) for i in step.instances
        ]
        detections = detect(instances, step.frame, cfg.perception)
        s = step.sensors
        sensors = SensorSnapshot(
            stamp_s=step.t,
            fc_alt_m=s["fc_alt"],
            tof_alt_m=s["tof"],
            of_vx_ms=s["of_vx"],
            of_vy_ms=s["of_vy"],
            pitch_deg=s["pitch"],
            roll_deg=s["roll"],
            yaw_deg=s["yaw"],
            pitch_rate_dps=s["pitch_rate"],
            roll_rate_dps=s["roll_rate"],
        )
        out = sm.step(step.t, detections, sensors, (FRAME_W, FRAME_H))
        cmd, st = out.command, out.status
        records.append(
            Record(
                i=step.index,
                t=step.t,
                phase=str(cmd.phase),
                action=cmd.action,
                vx=cmd.vx,
                vy=cmd.vy,
                vz=cmd.vz,
                yaw_rate=cmd.yaw_rate_dps,
                yaw_lock=cmd.yaw_lock,
                drop=cmd.drop_ball,
                mission_phase=str(st.phase),
                score=st.score,
                penalties=st.penalties,
                balls=st.balls_dropped,
                target_idx=st.target_index,
                alt_src=str(st.alt_source),
                elapsed=st.elapsed_s,
            )
        )
    return records
