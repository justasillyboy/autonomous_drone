"""The launch files load and start the expected nodes (runs under colcon test only)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("launch_ros")

from launch_ros.actions import Node

LAUNCH_DIR = Path(__file__).resolve().parents[1] / "launch"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LAUNCH_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def test_real_launch_starts_the_five_nodes():
    ld = _load("real.launch.py")
    nodes = [e for e in ld.entities if isinstance(e, Node)]
    executables = sorted(n.node_executable for n in nodes)
    assert executables == [
        "action_node",
        "bno080_node",
        "mavros_node",
        "mission_node",
        "perception_node",
    ]
