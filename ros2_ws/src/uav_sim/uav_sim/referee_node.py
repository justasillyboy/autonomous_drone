"""sim_referee: scores the run from Gazebo ground truth and writes results/<scenario>.json.

Poses come from /world/competition_field/dynamic_pose/info (bridged as tf2_msgs/TFMessage);
the match starts on the ARM rising edge. Gate contacts are detected geometrically from the
drone's position (field.gate_clearance), balls when they come to rest (balls.BallTracker).
"""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped
from mavros_msgs.msg import State
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from uav_core.types import Round
from uav_interfaces.msg import MissionStatus

from uav_sim.balls import BallTracker
from uav_sim.field import gate_clearance
from uav_sim.referee import Referee
from uav_sim.scenario import make_scenario
from uav_sim.world import DRONE_MODEL, WORLD_NAME


class RefereeNode(Node):
    def __init__(self) -> None:
        super().__init__("sim_referee")
        seed = int(self.declare_parameter("seed", 0).value)
        round_ = Round(self.declare_parameter("round", "qualifier").value)
        fixed = not bool(self.declare_parameter("random_layout", False).value)
        self.results_dir = Path(self.declare_parameter("results_dir", "results").value)
        self.drone_radius_m = float(self.declare_parameter("drone_radius_m", 0.3).value)
        self.scenario = make_scenario(seed, round_, fixed)
        self.ref = Referee(self.scenario.layout, drone_half_size_m=self.drone_radius_m)
        self.balls = BallTracker()
        self.f2_xy: tuple[float, float] | None = None
        self.armed = False
        self.written = False
        self.create_subscription(
            TFMessage, f"/world/{WORLD_NAME}/dynamic_pose/info", self.on_poses, 10
        )
        self.create_subscription(State, "/mavros/state", self.on_state, 10)
        self.create_subscription(PointStamped, "/sim/f2", self.on_f2, 10)
        self.create_subscription(MissionStatus, "/mission/status", self.on_status, 10)
        self.self_reported: int | None = None
        self.create_timer(1.0, self.on_timer)
        self.get_logger().info(f"referee for {self.scenario.name}")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_state(self, msg: State) -> None:
        if msg.armed and not self.armed:
            self.ref.start(self._now())
            self.get_logger().info("ARM: match started")
        self.armed = msg.armed

    def on_status(self, msg: MissionStatus) -> None:
        self.self_reported = int(msg.score) + int(msg.penalties)

    def on_f2(self, msg: PointStamped) -> None:
        self.f2_xy = (msg.point.x, msg.point.y)

    def on_poses(self, msg: TFMessage) -> None:
        layout = self.scenario.layout
        if self.f2_xy is not None:
            layout = layout.with_target_at("F2", *self.f2_xy)
        for tf in msg.transforms:
            t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            p = tf.transform.translation
            if tf.child_frame_id == DRONE_MODEL:
                self.ref.drone(t, p.x, p.y, p.z)
                for gate in layout.gates:
                    if gate_clearance(gate, p.x, p.y, p.z) < self.drone_radius_m:
                        self.ref.gate_contact(t, gate.label)
            elif tf.child_frame_id.startswith("ball_"):
                rest = self.balls.update(tf.child_frame_id, t, p.x, p.y, p.z)
                if rest is not None:
                    self.ref.ball_at_rest(t, *rest, targets=layout)
                    self.get_logger().info(
                        f"{tf.child_frame_id} at rest at {rest}: {self.ref.score}"
                    )

    def on_timer(self) -> None:
        now = self._now()
        over = self.ref.start_s is not None and self.ref.elapsed(now) > self.ref.rules.match_time_s
        if (self.ref.finished or over) and not self.written:
            self.write(now)

    def write(self, now: float) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "scenario": self.scenario.name,
            **self.ref.result(now),
            "self_reported_score": self.self_reported,
        }
        path = self.results_dir / f"{self.scenario.name}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        self.written = True
        self.get_logger().info(f"score {result['score']} -> {path}; failures {result['failures']}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RefereeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.written:
            node.write(node._now())
        node.destroy_node()
        rclpy.try_shutdown()
