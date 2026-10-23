# ADR-0003: One frame convention per layer, conversions only in `frames.py`

- **Status:** Accepted
- **Date:** 2026-10-24
- **Code:** `uav_core/src/uav_core/frames.py`, `uav_core/tests/test_types_frames.py`,
  `ros2_ws/src/uav_hardware/uav_hardware/setpoint.py`, `ros2_ws/src/uav_mission/uav_mission/bridge.py`

## Context

The mission computes velocity commands in the **body FRD** frame: +x forward, +y right, +z down
(so `vz = +0.6` means "descend"), yaw rate in deg/s, positive clockwise seen from above. The
legacy `FlightCommandMsg.msg` documents exactly that.

ROS uses REP 103 instead: body **FLU** (+y left, +z up, yaw rate in rad/s, positive
counter-clockwise) and **ENU** for local frames. The legacy bridge,
[`action_mavros_node.py`](https://github.com/justasillyboy/autonomous_drone/blob/572856f/legacy/ros2/uav_deploy/UAV_node/action_mavros_node.py#L113-L134)
(removed from the tree with the rest of `legacy/ros2`), copied the fields straight across:

```python
twist.linear.x = float(msg.vx)
twist.linear.y = float(msg.vy)
twist.linear.z = float(msg.vz)
twist.angular.z = math.radians(float(msg.yaw_rate))  # unit converted, sign not
self.pub_cmd_vel.publish(twist)  # -> /mavros/setpoint_velocity/cmd_vel_unstamped
```

Only the yaw unit was converted. With the MAVROS `setpoint_velocity` plugin at its default
`mav_frame` (LOCAL_NED, exposed to ROS as ENU), that `Twist` is read as a world-frame velocity:

- `vz = +0.6` ("descend") would make the drone **climb**;
- a clockwise yaw request would turn it **counter-clockwise**;
- "forward" and "right" would become east and north, whatever the heading.

The inputs had the same problem in
[`mission_control_node.py`](https://github.com/justasillyboy/autonomous_drone/blob/572856f/legacy/ros2/uav_deploy/UAV_node/mission_control_node.py#L344-L381):

- `/mavros/local_position/velocity_body` is FLU, but its `linear.y` went into the mission as
  "right", so the dead reckoning and the scored-zone tracking drifted mirrored sideways;
- the IMU quaternion was turned into Euler angles in ROS conventions, so the yaw fed to the
  mission turned counter-clockwise while the mission assumes clockwise, mirroring the
  scored-zone rotation and the geofence return direction. Pitch and roll were used only as
  magnitudes, so the drop gate was unaffected.

The code never flew end to end (defect D1), so none of this was observed. The mission's own
world frame (x along the nose at activation, y to its right, yaw positive clockwise) was also
implicit in the arithmetic.

## Decision

- `uav_core` speaks body FRD and the mission world frame described above. This keeps the
  legacy arithmetic, and therefore the characterisation goldens, unchanged.
- `uav_core.frames` is the **only** place that converts between conventions:
  `body_frd_to_flu` / `flu_to_body_frd`, `flu_rates_to_frd_dps`,
  `enu_quaternion_to_frd_euler_deg`, `wrap_deg`, `rotate` / `rotate_inverse`. Other modules
  must not flip signs or convert angle units inline.
- `action_node` converts each `FlightCommand` with `body_frd_to_flu` and sends it to
  `/mavros/setpoint_raw/local` as a `PositionTarget` with `coordinate_frame = FRAME_BODY_NED`
  and a type mask that uses only velocity and yaw rate. MAVROS takes FLU on its ROS side for
  body frames and converts to FRD itself. Declaring the frame in each message removes the
  dependence on the `setpoint_velocity` plugin's `mav_frame` parameter.
- `mission_node` converts its inputs the same way: body velocity with `flu_to_body_frd`, IMU
  attitude with `enu_quaternion_to_frd_euler_deg`, body rates with `flu_rates_to_frd_dps`.
- The closed-loop simulation (M2) gets a direction test: a commanded +vz descends, +vy moves
  right and a positive yaw rate turns clockwise, checked against SITL ground truth.

## Consequences

- Property tests (hypothesis) cover the round trips: FRD -> FLU -> FRD and rotate -> inverse
  return the input, and `wrap_deg` stays in [-180, 180).
- The legacy sign error is fixed by construction once the bridge uses the helper. The SITL
  direction test is the only proof against a real autopilot, so it is a release gate, not an
  optional check.
- The MAVROS frame behaviour described above comes from the plugin's documented defaults and
  has not yet been checked in SITL; the M2 direction test settles it.
