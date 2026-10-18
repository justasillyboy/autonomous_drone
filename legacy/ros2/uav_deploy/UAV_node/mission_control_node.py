"""mission_control_node — MissionStateMachine driver, per node_split_deploy.md §2 Node 4.

Responsibilities:
  • Load config.yaml via load_mission_config_yaml() BEFORE instantiating
    MissionStateMachine (instance variables like _tof_spike_threshold_m,
    _ff_history_len, _drop_cooldown are frozen at __init__).
  • Subscribe to:
      /perception/targets      (VisionTargetsMsg)     — pre-filtered detections
      /camera/image_raw        (Image)                — frame for HUD + phase logic
      /camera/camera_info      (CameraInfo)           — f_px source of truth
      /bno080/imu              (sensor_msgs/Imu)      — quaternion + angular rates
      /mavros/distance_sensor/rangefinder_sub         — ToF altitude
      /mavros/local_position/velocity_body            — optical-flow body velocity
      /mavros/global_position/rel_alt                 — FC barometer altitude
      /mavros/state            (mavros_msgs/State)    — ARM detect → match timer
      /action/land_complete    (std_msgs/Bool)        — FC LAND_COMPLETE relay
  • Feed MissionStateMachine every frame via mission.update() using a
    _ShimResult adapter that packages VisionTargetsMsg entries into the
    minimum YOLO-Results interface expected by mission.py.
  • Publish /mission/flight_command (FlightCommandMsg, Reliable) every frame.
  • Expose services /mission/reset and /mission/e_stop (std_srvs/Trigger).
  • Watchdog: hover if no VisionTargetsMsg arrives for > 0.5 s.
"""
import math
import os
import time
from collections import deque

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, Imu, Range
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Trigger

try:
    from mavros_msgs.msg import State as MavrosState
except ImportError:  # pragma: no cover — workstation env without mavros
    MavrosState = None

from main import mission as mission_module
from main.mission import (
    DEFAULT_F_PX,
    FIRE_REAL_DIAMETERS,
    FlightCommand,
    MissionStateMachine,
    Phase,
    SensorData,
    load_mission_config_yaml,
)

try:
    from uav_deploy.msg import FlightCommandMsg, VisionTargetsMsg
except ImportError:  # pragma: no cover
    from uav_interfaces.msg import FlightCommandMsg, VisionTargetsMsg


# ---------------------------------------------------------------------------
# Minimal YOLO-Results adapter.  mission.update() internally iterates
# `result.masks.data` and `result.boxes` and calls _extract_geometry / color
# / overlaps_h_landing.  Rebuild just enough of that interface from the
# pre-filtered VisionTargetsMsg so mission.update() runs unchanged.
# ---------------------------------------------------------------------------

class _ShimMask:
    """Binary mask in model-output space (mask_ratio=4) with the same
    .cpu().numpy() interface that mission._extract_geometry consumes.

    Each target's mask is a rasterised disk at the scaled (cx, cy) with the
    scaled radius, so _extract_geometry extracts the exact same geometry
    reported by perception_node.  Cost: one cv2.circle per target, <0.1 ms.
    """

    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _ShimMasks:
    __slots__ = ("data",)

    def __init__(self, masks):
        self.data = masks


class _ShimTensor:
    """Stand-in for torch tensors with .item() access (boxes.cls[0].item())."""

    __slots__ = ("_v",)

    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v

    def __getitem__(self, idx):
        return self._v[idx] if hasattr(self._v, '__getitem__') else self._v


class _ShimBox:
    __slots__ = ("cls", "conf", "xyxy")

    def __init__(self, cls_id: int, conf: float, xyxy: tuple):
        self.cls = [_ShimTensor(cls_id)]
        self.conf = [_ShimTensor(conf)]
        self.xyxy = [[_ShimTensor(v) for v in xyxy]]


class _ShimResult:
    """Duck-types Ultralytics Results enough for mission.update() + helpers."""

    __slots__ = ("masks", "boxes")

    def __init__(self, masks, boxes):
        self.masks = _ShimMasks(masks) if masks else None
        self.boxes = boxes if boxes else None


def _build_shim_result(targets, frame_w: int, frame_h: int,
                       mask_w: int = 160, mask_h: int = 90):
    """Rasterise each VisionTarget into a mask-space binary disk and return
    a _ShimResult matching the full-frame geometry perception_node reported.

    The default 160x90 mask size matches a 640x360 YOLO output with mask_ratio=4.
    """
    masks = []
    boxes = []
    sx = mask_w / float(max(1, frame_w))
    sy = mask_h / float(max(1, frame_h))
    for t in targets:
        cx_m = int(round(t.cx * sx))
        cy_m = int(round(t.cy * sy))
        # Average the two axes for the disk; _extract_geometry will re-derive
        # the correct frame-space radius from the scaled contour.
        r_m = max(1, int(round(t.radius * 0.5 * (sx + sy))))
        mask_arr = np.zeros((mask_h, mask_w), dtype=np.float32)
        import cv2  # local to avoid import cost at module load
        cv2.circle(mask_arr, (cx_m, cy_m), r_m, 1.0, thickness=-1)
        masks.append(_ShimMask(mask_arr))

        # xyxy from the reported circle in FRAME pixels.
        x1 = max(0.0, t.cx - t.radius)
        y1 = max(0.0, t.cy - t.radius)
        x2 = min(float(frame_w), t.cx + t.radius)
        y2 = min(float(frame_h), t.cy + t.radius)
        boxes.append(_ShimBox(int(t.cls_id), float(t.conf), (x1, y1, x2, y2)))
    return _ShimResult(masks, boxes)


# ---------------------------------------------------------------------------
# Quaternion → Euler (deg).  Avoids a tf_transformations dependency.
# ---------------------------------------------------------------------------

def _quat_to_euler_deg(qx: float, qy: float, qz: float, qw: float):
    """Return (roll_deg, pitch_deg, yaw_deg) — XYZ (roll-pitch-yaw) convention."""
    # Roll (X-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class MissionControlNode(Node):
    def __init__(self):
        super().__init__('mission_control_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('takeoff_alt', 3.3)
        self.declare_parameter('is_finals', False)
        self.declare_parameter('f_px_fallback', float(DEFAULT_F_PX))
        self.declare_parameter('config_yaml', '')
        self.declare_parameter('fps_hint', 20.0)
        self.declare_parameter('skip_takeoff', False)
        self.declare_parameter('watchdog_timeout_s', 0.5)

        takeoff_alt = float(self.get_parameter('takeoff_alt').value)
        is_finals = bool(self.get_parameter('is_finals').value)
        f_px_fallback = float(self.get_parameter('f_px_fallback').value)
        config_yaml = self.get_parameter('config_yaml').value
        fps_hint = float(self.get_parameter('fps_hint').value)
        skip_takeoff = bool(self.get_parameter('skip_takeoff').value)
        self._watchdog_timeout_s = float(self.get_parameter('watchdog_timeout_s').value)

        # ── config.yaml MUST load before MissionStateMachine is instantiated ──
        # Several instance variables (_tof_spike_threshold_m, _ff_history_len,
        # _drop_cooldown) read module-global constants at __init__; loading
        # afterwards leaves them at defaults.  CLAUDE.md Danger Zones.
        if config_yaml and os.path.isfile(config_yaml):
            try:
                load_mission_config_yaml(config_yaml)
                self.get_logger().info(f"Loaded mission config from {config_yaml}")
            except Exception as e:
                self.get_logger().error(f"Failed to load config_yaml ({e}) — continuing with defaults")
        elif config_yaml:
            self.get_logger().warn(f"config_yaml not found: {config_yaml}")

        self._f_px = f_px_fallback
        self.mission = MissionStateMachine(
            f_px=self._f_px,
            is_finals=is_finals,
            takeoff_alt=takeoff_alt,
        )
        self.mission.fps_hint = fps_hint
        if skip_takeoff:
            self.mission.skip_takeoff()

        self.bridge = CvBridge()

        # ── Sensor + state caches ─────────────────────────────────────────────
        self._fc_altitude: float | None = None
        self._tof_altitude_m: float | None = None
        self._of_vx_ms: float | None = None
        self._of_vy_ms: float | None = None
        self._pitch_deg: float | None = None
        self._roll_deg: float | None = None
        self._yaw_deg: float | None = None
        self._pitch_rate_dps: float | None = None
        self._roll_rate_dps: float | None = None

        self._armed = False
        self._timer_started = False
        self._e_stopped = False
        self._last_targets_rx_s: float = 0.0

        # Frame buffer — correlate VisionTargetsMsg.header.stamp → image.
        self._frame_buffer: deque = deque(maxlen=20)

        # ── QoS ───────────────────────────────────────────────────────────────
        qos_best = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
        )
        qos_rel = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
        )
        qos_info = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
        )
        qos_imu = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        # /mission/flight_command MUST be Reliable — dropping a drop_ball=True
        # frame silently fails the match (see deploy_guide §7e).
        self.pub_flight_command = self.create_publisher(
            FlightCommandMsg, '/mission/flight_command', qos_rel)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            VisionTargetsMsg, '/perception/targets',
            self.targets_callback, qos_rel)
        self.create_subscription(
            Image, '/camera/image_raw',
            self.image_callback, qos_best)
        self.create_subscription(
            CameraInfo, '/camera/camera_info',
            self.camera_info_callback, qos_info)
        self.create_subscription(
            Imu, '/bno080/imu', self.imu_callback, qos_imu)
        self.create_subscription(
            Range, '/mavros/distance_sensor/rangefinder_sub',
            self.tof_callback, qos_best)
        self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_body',
            self.velocity_callback, qos_best)
        self.create_subscription(
            Float64, '/mavros/global_position/rel_alt',
            self.altitude_callback, qos_best)
        self.create_subscription(
            Bool, '/action/land_complete',
            self.land_complete_callback, qos_rel)
        if MavrosState is not None:
            self.create_subscription(
                MavrosState, '/mavros/state', self.mavros_state_callback, qos_rel)
        else:
            self.get_logger().warn(
                "mavros_msgs not available — match timer falls back to "
                "MissionStateMachine's internal SEARCHING-start rule")

        # ── Services ──────────────────────────────────────────────────────────
        self.create_service(Trigger, '/mission/reset', self.srv_reset)
        self.create_service(Trigger, '/mission/e_stop', self.srv_e_stop)

        # ── Watchdog timer @ 5 Hz ─────────────────────────────────────────────
        self.create_timer(0.2, self._watchdog_tick)

        self.get_logger().info(
            f"mission_control_node started (takeoff_alt={takeoff_alt}, "
            f"finals={is_finals}, fps_hint={fps_hint}, "
            f"skip_takeoff={skip_takeoff})")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def camera_info_callback(self, msg: CameraInfo):
        fx = float(msg.k[0])
        if fx > 0.0 and abs(fx - self._f_px) > 0.5:
            self._f_px = fx
            # Propagate to MissionStateMachine's AltitudeEstimator so vision
            # fallback Z uses the real focal length.
            self.mission.alt_est.f_px = fx
            self.get_logger().info(f"CameraInfo → f_px = {fx:.1f}")

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._frame_buffer.append((stamp_s, frame, msg.header))

    def imu_callback(self, msg: Imu):
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        if qx == 0.0 and qy == 0.0 and qz == 0.0 and qw == 0.0:
            return
        roll_d, pitch_d, yaw_d = _quat_to_euler_deg(qx, qy, qz, qw)
        self._roll_deg = roll_d
        self._pitch_deg = pitch_d
        self._yaw_deg = yaw_d
        # rad/s → deg/s; body frame matches mission.py's convention.
        self._roll_rate_dps = math.degrees(msg.angular_velocity.x)
        self._pitch_rate_dps = math.degrees(msg.angular_velocity.y)

        # Push to MissionStateMachine IMU buffer immediately so
        # _get_synced_imu can pick the sample nearest to frame-grab time.
        sd = SensorData(
            tof_altitude_m=self._tof_altitude_m,
            of_vx_ms=self._of_vx_ms,
            of_vy_ms=self._of_vy_ms,
            pitch_deg=self._pitch_deg,
            roll_deg=self._roll_deg,
            yaw_deg=self._yaw_deg,
            pitch_rate_dps=self._pitch_rate_dps,
            roll_rate_dps=self._roll_rate_dps,
        )
        self.mission.push_sensor_data(sd)

    def tof_callback(self, msg: Range):
        # sensor_msgs/Range units = metres.  Out-of-range sentinels get dropped.
        r = float(msg.range)
        if math.isfinite(r) and msg.min_range <= r <= msg.max_range:
            self._tof_altitude_m = r

    def velocity_callback(self, msg: TwistStamped):
        self._of_vx_ms = float(msg.twist.linear.x)
        self._of_vy_ms = float(msg.twist.linear.y)

    def altitude_callback(self, msg: Float64):
        self._fc_altitude = float(msg.data)

    def land_complete_callback(self, msg: Bool):
        if msg.data and self.mission.phase != Phase.COMPLETE:
            self.mission.signal_landed()

    def mavros_state_callback(self, msg):
        was_armed = self._armed
        self._armed = bool(msg.armed)
        if self._armed and not was_armed and not self._timer_started:
            # Start the match clock.  MissionStateMachine sets _start_time
            # internally on the first non-TAKEOFF frame, but the rulebook
            # clock starts at ARM — override here so the 285 s LAND_EARLY
            # trigger fires at the correct wall-clock time.
            self.mission._start_time = time.perf_counter()
            self._timer_started = True
            self.get_logger().info("ARM detected — match timer started")

    def targets_callback(self, msg: VisionTargetsMsg):
        if self._e_stopped:
            return
        self._last_targets_rx_s = time.perf_counter()

        # Correlate detections → frame by header.stamp.
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        frame = self._lookup_frame(stamp_s)
        if frame is None:
            # No image available yet — skip rather than block.
            return

        frame_h, frame_w = frame.shape[:2]
        shim_result = _build_shim_result(msg.targets, frame_w, frame_h)

        # Use the freshest synchronized SensorData snapshot; mission.update()
        # will pull the timestamp-aligned version from the IMU buffer.
        sd = SensorData(
            tof_altitude_m=self._tof_altitude_m,
            of_vx_ms=self._of_vx_ms,
            of_vy_ms=self._of_vy_ms,
            pitch_deg=self._pitch_deg,
            roll_deg=self._roll_deg,
            yaw_deg=self._yaw_deg,
            pitch_rate_dps=self._pitch_rate_dps,
            roll_rate_dps=self._roll_rate_dps,
        )

        try:
            self.mission.update(
                frame, shim_result,
                current_altitude=self._fc_altitude,
                sensor_data=sd,
                frame_timestamp=stamp_s,
            )
        except Exception as e:
            self.get_logger().error(f"mission.update() failed: {e}")
            return

        self._publish_last_command(msg.header.stamp)

    # ── Watchdog ─────────────────────────────────────────────────────────────

    def _watchdog_tick(self):
        # Publish a hover command when perception goes silent; do NOT mutate
        # mission state — the state machine can resume the instant frames return.
        if self._e_stopped:
            self._publish_hover(reason="E_STOP")
            return
        now = time.perf_counter()
        if (self._last_targets_rx_s > 0.0
                and (now - self._last_targets_rx_s) > self._watchdog_timeout_s):
            self._publish_hover(reason="WATCHDOG")

    # ── Services ─────────────────────────────────────────────────────────────

    def srv_reset(self, request, response):
        self.mission.reset()
        # If drone is already airborne the operator should call skip_takeoff
        # via a separate service or by restarting with skip_takeoff=true.
        self._timer_started = False
        self._e_stopped = False
        response.success = True
        response.message = "mission reset — score, phase, ball count cleared"
        self.get_logger().info("/mission/reset invoked")
        return response

    def srv_e_stop(self, request, response):
        self._e_stopped = True
        self._publish_hover(reason="E_STOP_SRV")
        response.success = True
        response.message = "e-stop engaged — call /mission/reset to resume"
        self.get_logger().warn("/mission/e_stop invoked — hover until reset")
        return response

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _lookup_frame(self, target_stamp_s: float):
        if not self._frame_buffer:
            return None
        # deque scan is O(N) but N<=20.
        best = min(self._frame_buffer, key=lambda f: abs(f[0] - target_stamp_s))
        return best[1]

    def _publish_last_command(self, stamp):
        cmd: FlightCommand = self.mission.last_command
        out = FlightCommandMsg()
        out.header.stamp = stamp
        out.vx = float(cmd.vx)
        out.vy = float(cmd.vy)
        out.vz = float(cmd.vz)
        out.yaw_rate = float(cmd.yaw_rate)
        out.yaw_lock = bool(cmd.yaw_lock)
        out.drop_ball = bool(cmd.drop_ball)
        out.phase = str(cmd.phase or self.mission.phase.name)
        out.action = str(cmd.action or "")
        out.score = int(self.mission.score)
        out.balls_dropped = int(self.mission.balls_dropped)
        self.pub_flight_command.publish(out)

    def _publish_hover(self, reason: str):
        out = FlightCommandMsg()
        out.header.stamp = self.get_clock().now().to_msg()
        out.vx = 0.0
        out.vy = 0.0
        out.vz = 0.0
        out.yaw_rate = 0.0
        out.yaw_lock = True
        out.drop_ball = False
        out.phase = self.mission.phase.name if not self._e_stopped else "E_STOP"
        out.action = reason
        out.score = int(self.mission.score)
        out.balls_dropped = int(self.mission.balls_dropped)
        self.pub_flight_command.publish(out)


def main():
    rclpy.init()
    node = MissionControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
