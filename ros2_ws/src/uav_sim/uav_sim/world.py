"""Write a Gazebo Harmonic world (SDF 1.9) for a scenario. No ROS imports.

    ros2 run uav_sim make_world --seed 7 --round qualifier --out /tmp/uav_sim
    -> competition_field.sdf, helipad.png, models/iris_downcam, sim_calibration.npz,
       sitl.parm, scenario.json

The drone model `iris_downcam` (sim/models) is ardupilot_gazebo's iris_with_ardupilot plus a
640x480 downward camera; it is spawned on the helipad facing -x (toward the fires).
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from uav_core.perception.camera import Calibration
from uav_core.types import Round

from uav_sim.field import (
    COLOR_HEX,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    TARGET_PLINTH_M,
    TZ_CENTRE_HEIGHT_M,
    TZ_POST_RADIUS_M,
    TZ_RING_RADIUS_M,
    TZ_TUBE_RADIUS_M,
    Gate,
    Target,
)
from uav_sim.scenario import Scenario, make_scenario, sitl_param_file

WORLD_NAME = "competition_field"
DRONE_MODEL = "iris_downcam"
CAM_W, CAM_H = 640, 480
CAM_HFOV_RAD = 1.8  # wide angle, roughly the IMX219 lens
CAM_DISTORTION = (-0.25, 0.08, 0.0, 0.0, 0.0)  # k1 k2 p1 p2 k3, as OpenCV orders them
CAM_RATE_HZ = 30
RING_SEGMENTS = 24
PLUGINS = (
    ("gz-sim-physics-system", "gz::sim::systems::Physics"),
    ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands"),
    ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster"),
    ("gz-sim-imu-system", "gz::sim::systems::Imu"),
    ("gz-sim-navsat-system", "gz::sim::systems::NavSat"),
)


def _rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return " ".join(f"{int(h[i : i + 2], 16) / 255:.3f}" for i in (0, 2, 4)) + " 1"


def _sub(parent: ET.Element, tag: str, text: str | None = None, **attrib: str) -> ET.Element:
    el = ET.SubElement(parent, tag, attrib)
    if text is not None:
        el.text = text
    return el


def _material(parent: ET.Element, rgba: str, albedo_map: str | None = None) -> None:
    mat = _sub(parent, "material")
    _sub(mat, "ambient", rgba)
    _sub(mat, "diffuse", rgba)
    if albedo_map:
        metal = _sub(_sub(mat, "pbr"), "metal")
        _sub(metal, "albedo_map", albedo_map)


def _shape(
    link: ET.Element,
    name: str,
    geometry: tuple[str, dict[str, str]],
    pose: str,
    rgba: str | None = None,
    collide: bool = True,
    albedo_map: str | None = None,
) -> None:
    kind, dims = geometry
    elements = ("visual", "collision") if collide else ("visual",)
    for tag in elements:
        el = _sub(link, tag, name=f"{name}_{tag}")
        _sub(el, "pose", pose)
        geo = _sub(_sub(el, "geometry"), kind)
        for key, value in dims.items():
            _sub(geo, key, value)
        if tag == "visual" and rgba:
            _material(el, rgba, albedo_map)


def _static_model(world: ET.Element, name: str, pose: str) -> ET.Element:
    model = _sub(world, "model", name=name)
    _sub(model, "static", "true")
    _sub(model, "pose", pose)
    return _sub(model, "link", name="link")


def _target(world: ET.Element, t: Target) -> None:
    link = _static_model(world, f"target_{t.label}", f"{t.x:.3f} {t.y:.3f} 0 0 0 0")
    r = f"{t.radius_m:.3f}"
    h = TARGET_PLINTH_M
    plinth = ("cylinder", {"radius": r, "length": f"{h - 0.005:.3f}"})
    _shape(link, "plinth", plinth, f"0 0 {(h - 0.005) / 2:.4f} 0 0 0", "0.2 0.2 0.2 1")
    top = ("cylinder", {"radius": r, "length": "0.005"})
    _shape(link, "disc", top, f"0 0 {h - 0.0025:.4f} 0 0 0", _rgb(COLOR_HEX[t.color]))


def _gate(world: ET.Element, g: Gate) -> None:
    link = _static_model(world, f"gate_{g.label}", f"{g.x:.3f} {g.y:.3f} 0 0 0 0")
    seg_len = 2 * math.pi * TZ_RING_RADIUS_M / RING_SEGMENTS * 1.05
    tube = 2 * TZ_TUBE_RADIUS_M
    for k in range(RING_SEGMENTS):
        a = 2 * math.pi * k / RING_SEGMENTS
        y, z = TZ_RING_RADIUS_M * math.cos(a), TZ_CENTRE_HEIGHT_M + TZ_RING_RADIUS_M * math.sin(a)
        box = ("box", {"size": f"{tube:.3f} {seg_len:.3f} {tube:.3f}"})
        roll = a + math.pi / 2  # the box's long (y) axis along the ring's tangent
        _shape(link, f"seg{k}", box, f"0 {y:.4f} {z:.4f} {roll:.4f} 0 0", "1 0.45 0 1")
    post_h = TZ_CENTRE_HEIGHT_M - TZ_RING_RADIUS_M
    post = ("cylinder", {"radius": f"{TZ_POST_RADIUS_M}", "length": f"{post_h:.3f}"})
    _shape(link, "post", post, f"0 0 {post_h / 2:.3f} 0 0 0", "0.3 0.3 0.3 1")


def helipad_texture(size_px: int = 512) -> np.ndarray:
    """Grey square, white ring, white H and corner brackets (as on the real pad)."""
    img = np.full((size_px, size_px, 3), 110, np.uint8)
    c, s = size_px // 2, size_px
    cv2.circle(img, (c, c), int(s * 0.36), (240, 240, 240), int(s * 0.05), cv2.LINE_AA)
    bar, gap = int(s * 0.06), int(s * 0.13)
    cv2.rectangle(
        img, (c - gap - bar, c - int(s * 0.2)), (c - gap, c + int(s * 0.2)), (240,) * 3, -1
    )
    cv2.rectangle(
        img, (c + gap, c - int(s * 0.2)), (c + gap + bar, c + int(s * 0.2)), (240,) * 3, -1
    )
    cv2.rectangle(img, (c - gap, c - bar // 2), (c + gap, c + bar // 2), (240,) * 3, -1)
    m, arm, w = int(s * 0.05), int(s * 0.18), int(s * 0.03)
    for sx in (1, -1):
        for sy in (1, -1):
            x0, y0 = (m if sx > 0 else s - m), (m if sy > 0 else s - m)
            cv2.line(img, (x0, y0), (x0 + sx * arm, y0), (240,) * 3, w)
            cv2.line(img, (x0, y0), (x0, y0 + sy * arm), (240,) * 3, w)
    return img


def build_world(s: Scenario, helipad_png: str = "helipad.png") -> ET.ElementTree:
    sdf = ET.Element("sdf", version="1.9")
    world = _sub(sdf, "world", name=WORLD_NAME)
    physics = _sub(world, "physics", name="1ms", type="ignore")
    _sub(physics, "max_step_size", "0.001")
    _sub(physics, "real_time_factor", "1.0")
    for filename, name in PLUGINS:
        _sub(world, "plugin", filename=filename, name=name)
    sensors = _sub(
        world, "plugin", filename="gz-sim-sensors-system", name="gz::sim::systems::Sensors"
    )
    _sub(sensors, "render_engine", "ogre2")
    scene = _sub(world, "scene")
    _sub(scene, "ambient", "0.5 0.5 0.5 1")
    _sub(scene, "background", "0.7 0.75 0.8 1")

    az, el = math.radians(s.sun_azimuth_deg), math.radians(s.sun_elevation_deg)
    sun = _sub(world, "light", type="directional", name="sun")
    _sub(sun, "cast_shadows", "true")
    _sub(sun, "pose", "0 0 20 0 0 0")
    _sub(sun, "diffuse", " ".join([f"{s.sun_intensity:.2f}"] * 3) + " 1")
    _sub(sun, "specular", "0.2 0.2 0.2 1")
    d = (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el))
    _sub(sun, "direction", " ".join(f"{v:.3f}" for v in d))

    ground = _static_model(world, "ground", "0 0 0 0 0 0")
    plane = ("plane", {"normal": "0 0 1", "size": "60 40"})
    _shape(
        ground,
        "outer",
        plane,
        f"{FIELD_LENGTH_M / 2} {FIELD_WIDTH_M / 2} -0.001 0 0 0",
        "0.55 0.55 0.5 1",
    )
    court = ("box", {"size": f"{FIELD_LENGTH_M} {FIELD_WIDTH_M} 0.002"})
    rgb = " ".join(f"{c:.3f}" for c in s.floor_rgb) + " 1"
    _shape(
        ground,
        "court",
        court,
        f"{FIELD_LENGTH_M / 2} {FIELD_WIDTH_M / 2} 0 0 0 0",
        rgb,
        collide=False,
    )

    lay = s.layout
    pad = _static_model(world, "helipad", f"{lay.helipad_x:.3f} {lay.helipad_y:.3f} 0 0 0 0")
    pad_box = ("box", {"size": f"{lay.helipad_size_m} {lay.helipad_size_m} 0.01"})
    _shape(pad, "pad", pad_box, "0 0 0.005 0 0 0", "1 1 1 1", albedo_map=helipad_png)
    for t in lay.targets:
        _target(world, t)
    for g in lay.gates:
        _gate(world, g)

    drone = _sub(world, "include")
    _sub(drone, "uri", f"model://{DRONE_MODEL}")
    _sub(drone, "name", DRONE_MODEL)
    _sub(drone, "pose", f"{lay.helipad_x:.3f} {lay.helipad_y:.3f} 0.2 0 0 {math.pi:.4f}")
    ET.indent(sdf)
    return ET.ElementTree(sdf)


def camera_matrix() -> np.ndarray:
    f = (CAM_W / 2.0) / math.tan(CAM_HFOV_RAD / 2.0)
    return np.array([[f, 0.0, CAM_W / 2.0], [0.0, f, CAM_H / 2.0], [0.0, 0.0, 1.0]])


def drone_model_sdf() -> str:
    """ardupilot_gazebo's iris_with_ardupilot (ArduPilot JSON plugin, IMU) merged with a
    downward camera. The camera link is pitched +90 deg, so the image top is the nose."""
    k1, k2, p1, p2, k3 = CAM_DISTORTION
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{DRONE_MODEL}">
    <include merge="true"><uri>model://iris_with_ardupilot</uri></include>
    <link name="downcam_link">
      <pose relative_to="iris_with_standoffs::base_link">0 0 -0.05 0 1.5708 0</pose>
      <inertial><mass>0.01</mass>
        <inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz></inertia></inertial>
      <sensor name="downcam" type="camera">
        <always_on>true</always_on>
        <update_rate>{CAM_RATE_HZ}</update_rate>
        <topic>downcam</topic>
        <camera>
          <camera_info_topic>downcam/camera_info</camera_info_topic>
          <horizontal_fov>{CAM_HFOV_RAD}</horizontal_fov>
          <image><width>{CAM_W}</width><height>{CAM_H}</height><format>R8G8B8</format></image>
          <clip><near>0.05</near><far>50</far></clip>
          <distortion><k1>{k1}</k1><k2>{k2}</k2><k3>{k3}</k3><p1>{p1}</p1><p2>{p2}</p2>
            <center>0.5 0.5</center></distortion>
        </camera>
      </sensor>
    </link>
    <joint name="downcam_joint" type="fixed">
      <parent>iris_with_standoffs::base_link</parent>
      <child>downcam_link</child>
    </joint>
  </model>
</sdf>
"""


def write_drone_model(models_dir: Path) -> None:
    d = models_dir / DRONE_MODEL
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.sdf").write_text(drone_model_sdf())
    (d / "model.config").write_text(
        f'<?xml version="1.0"?>\n<model><name>{DRONE_MODEL}</name><version>1.0</version>'
        '<sdf version="1.9">model.sdf</sdf>'
        "<description>Iris with ArduPilot and a downward camera</description></model>\n"
    )


def write_scenario(s: Scenario, out_dir: Path) -> Path:
    """World, helipad texture, drone model (out_dir/models, add it to
    GZ_SIM_RESOURCE_PATH), SITL parameters, the sim camera's calibration, scenario.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_drone_model(out_dir / "models")
    Calibration(camera_matrix(), np.array(CAM_DISTORTION), (CAM_W, CAM_H), 0.0).save(
        out_dir / "sim_calibration.npz"
    )
    cv2.imwrite(str(out_dir / "helipad.png"), helipad_texture())
    world_path = out_dir / f"{WORLD_NAME}.sdf"
    build_world(s, str((out_dir / "helipad.png").resolve())).write(
        world_path, encoding="unicode", xml_declaration=True
    )
    (out_dir / "sitl.parm").write_text(sitl_param_file(s))
    meta = asdict(s)
    meta["layout"]["round"] = s.layout.round.value
    for t in meta["layout"]["targets"]:
        t["color"] = t["color"].value
    (out_dir / "scenario.json").write_text(json.dumps(meta, indent=2) + "\n")
    return world_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_world", description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round", choices=[r.value for r in Round], default="qualifier")
    parser.add_argument("--random-layout", action="store_true", help="also for the qualifier")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    scenario = make_scenario(args.seed, Round(args.round), fixed_layout=not args.random_layout)
    print(write_scenario(scenario, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
