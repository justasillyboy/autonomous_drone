"""Run legacy/vision/mission.py on a scenario and record its commands.

The legacy module reads time.perf_counter() everywhere, so it is loaded from its file path
and its `time` attribute is replaced by a fake clock that returns the scenario time, held
constant within one update(). The YOLO result is faked with just the attributes the legacy
code touches; no torch or ultralytics is needed. This file is deleted together with
legacy/vision/mission.py.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from characterisation.records import Record
from characterisation.scene import CLASS_IDS, F_PX, Instance, Scenario, iter_steps

LEGACY_MISSION = Path(__file__).resolve().parents[3] / "legacy" / "vision" / "mission.py"


class _Value:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _Mask:
    def __init__(self, array):
        self._array = array

    def cpu(self) -> _Mask:
        return self

    def numpy(self):
        return self._array


class _Box:
    def __init__(self, inst: Instance):
        self.cls = [_Value(float(CLASS_IDS[inst.cls_name]))]
        self.xyxy = [[_Value(v) for v in inst.xyxy]]
        self.conf = [_Value(inst.conf)]


def fake_yolo_result(instances: tuple[Instance, ...]) -> SimpleNamespace:
    names = {idx: name for name, idx in CLASS_IDS.items()}
    if not instances:
        return SimpleNamespace(masks=None, boxes=[], names=names)
    return SimpleNamespace(
        masks=SimpleNamespace(data=[_Mask(i.mask) for i in instances]),
        boxes=[_Box(i) for i in instances],
        names=names,
    )


def load_legacy_mission(path: Path = LEGACY_MISSION) -> ModuleType:
    spec = importlib.util.spec_from_file_location("legacy_mission", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it executes
    spec.loader.exec_module(module)
    return module


def run_legacy(scenario: Scenario) -> list[Record]:
    mod = load_legacy_mission()
    clock = SimpleNamespace(t=0.0)
    mod.time = SimpleNamespace(perf_counter=lambda: clock.t)
    with contextlib.redirect_stdout(io.StringIO()):
        sm = mod.MissionStateMachine(f_px=F_PX, is_finals=scenario.finals, takeoff_alt=3.3)
        sm.fps_hint = scenario.rate_hz
        records = []
        for step in iter_steps(scenario):
            clock.t = step.t
            for event in step.events:
                getattr(sm, event)()
            s = step.sensors
            sensor_data = mod.SensorData(
                tof_altitude_m=s["tof"],
                of_vx_ms=s["of_vx"],
                of_vy_ms=s["of_vy"],
                pitch_deg=s["pitch"],
                roll_deg=s["roll"],
                yaw_deg=s["yaw"],
                pitch_rate_dps=s["pitch_rate"],
                roll_rate_dps=s["roll_rate"],
            )
            sm.update(
                step.frame.copy(),
                fake_yolo_result(step.instances),
                current_altitude=s["fc_alt"],
                sensor_data=sensor_data,
            )
            cmd = sm.last_command
            records.append(
                Record(
                    i=step.index,
                    t=step.t,
                    phase=cmd.phase,
                    action=cmd.action,
                    vx=float(cmd.vx),
                    vy=float(cmd.vy),
                    vz=float(cmd.vz),
                    yaw_rate=float(cmd.yaw_rate),
                    yaw_lock=bool(cmd.yaw_lock),
                    drop=bool(cmd.drop_ball),
                    mission_phase=sm.phase.name,
                    score=int(sm.score),
                    penalties=int(sm.penalties),
                    balls=int(sm.balls_dropped),
                    target_idx=int(sm.current_target_idx),
                    alt_src=sm._last_alt_source,
                    elapsed=float(sm.time_elapsed_s),
                )
            )
    return records
