"""FlightCommand -> MAVROS setpoint, and the drop-servo pulse. No ROS imports.

The setpoint goes to /mavros/setpoint_raw/local as a PositionTarget in FRAME_BODY_NED with a
type_mask that uses only velocity and yaw rate. MAVROS expects ROS conventions on its side
(body FLU, rad/s) and converts to FRD itself for body frames, so the command is converted
once with uav_core.frames.body_frd_to_flu (ADR-0003). Declaring the frame in the message
avoids depending on the setpoint_velocity plugin's mav_frame parameter, whose default
(LOCAL_NED / ENU) is what the legacy node fell into.
"""

from __future__ import annotations

from dataclasses import dataclass

from uav_core.frames import body_frd_to_flu
from uav_core.types import FlightCommand, Phase

FRAME_BODY_NED = 8  # mavros_msgs/PositionTarget.FRAME_BODY_NED
IGNORE_PX, IGNORE_PY, IGNORE_PZ = 1, 2, 4
IGNORE_AFX, IGNORE_AFY, IGNORE_AFZ = 64, 128, 256
IGNORE_YAW = 1024
VELOCITY_AND_YAW_RATE = (
    IGNORE_PX | IGNORE_PY | IGNORE_PZ | IGNORE_AFX | IGNORE_AFY | IGNORE_AFZ | IGNORE_YAW
)
MAV_CMD_DO_SET_SERVO = 183


@dataclass(frozen=True)
class Setpoint:
    """Fields of a mavros_msgs/PositionTarget, in MAVROS (ROS) conventions."""

    coordinate_frame: int
    type_mask: int
    vx: float  # forward, m/s
    vy: float  # LEFT, m/s
    vz: float  # UP, m/s
    yaw_rate: float  # rad/s, counter-clockwise


def to_setpoint(cmd: FlightCommand) -> Setpoint:
    vx, vy, vz, yaw_rate = body_frd_to_flu(cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate_dps)
    return Setpoint(FRAME_BODY_NED, VELOCITY_AND_YAW_RATE, vx, vy, vz, yaw_rate)


def fill_position_target(msg, sp: Setpoint):
    msg.coordinate_frame = sp.coordinate_frame
    msg.type_mask = sp.type_mask
    msg.velocity.x, msg.velocity.y, msg.velocity.z = sp.vx, sp.vy, sp.vz
    msg.yaw_rate = sp.yaw_rate
    return msg


@dataclass(frozen=True)
class Actions:
    """What action_node sends to the FC for one FlightCommand."""

    setpoint: Setpoint | None = None
    servo_pwm: int | None = None
    takeoff_alt_m: float | None = None  # MAV_CMD_NAV_TAKEOFF via /mavros/cmd/takeoff
    set_mode: str | None = None  # via /mavros/set_mode


@dataclass
class ActionGate:
    """Decides what reaches the flight controller.

    - Not in GUIDED: nothing, and critically no drop (a ball released while the pilot flies
      manually lands anywhere). An unknown mode (no /mavros/state yet) blocks too.
    - Takeoff: ArduCopter ignores velocity setpoints while landed, so the first TAKEOFF
      command while armed on the ground also requests MAV_CMD_NAV_TAKEOFF to takeoff_alt_m.
      An unknown landed state (no EXTENDED_SYS_STATE yet) counts as on the ground: arming
      happens on the ground, and ArduCopter refuses NAV_TAKEOFF once it is flying.
    - Landing: once the mission is over the helipad and says LAND, the FC is switched to LAND
      mode, which descends, detects touchdown and disarms (GUIDED velocity descent would sit
      on the pad with motors spinning).
    - A drop is a PWM pulse: drop_pwm now, idle_pwm after pulse_s. The legacy node set the
      drop PWM and never returned to idle, so a release that needs a pulse fired only once.
    """

    require_guided: bool = True
    drop_pwm: int = 1900
    idle_pwm: int = 1100
    pulse_s: float = 0.5
    takeoff_alt_m: float = 3.3
    mode: str | None = None
    armed: bool = False
    on_ground: bool | None = None
    _release_at_s: float | None = None
    _takeoff_sent: bool = False
    _land_sent: bool = False

    def accepts_commands(self) -> bool:
        return not self.require_guided or self.mode == "GUIDED"

    def on_command(self, cmd: FlightCommand, now_s: float) -> Actions:
        if not self.accepts_commands():
            return Actions()
        pwm = None
        if cmd.drop_ball and self._release_at_s is None:
            pwm = self.drop_pwm
            self._release_at_s = now_s + self.pulse_s
        takeoff = None
        airborne = self.on_ground is False
        if cmd.phase is Phase.TAKEOFF and self.armed and not airborne and not self._takeoff_sent:
            takeoff, self._takeoff_sent = self.takeoff_alt_m, True
        mode = None
        if cmd.phase is Phase.LANDING and cmd.action == "LAND" and not self._land_sent:
            mode, self._land_sent = "LAND", True
        return Actions(to_setpoint(cmd), pwm, takeoff, mode)

    def on_disarmed(self) -> None:
        """A new flight (retry) may take off and land again."""
        self._takeoff_sent = self._land_sent = False

    def on_tick(self, now_s: float) -> int | None:
        """Servo PWM to send when the pulse ends, else None."""
        if self._release_at_s is not None and now_s >= self._release_at_s:
            self._release_at_s = None
            return self.idle_pwm
        return None


def servo_command(channel: int, pwm: int) -> dict[str, float | int | bool]:
    """Fields of a mavros_msgs/CommandLong request for MAV_CMD_DO_SET_SERVO."""
    return {
        "broadcast": False,
        "command": MAV_CMD_DO_SET_SERVO,
        "param1": float(channel),
        "param2": float(pwm),
    }
