import json

import cv2
import numpy as np
import pytest
from uav_core.perception import RawInstance
from uav_core.types import TargetClass
from uav_edge import bench


class _Staged:
    def preprocess(self, frame):
        return None, frame

    def run(self, x):
        return x

    def postprocess(self, raw, lb):
        h, w = raw.shape[:2]
        mask = np.zeros((h // 4, w // 4), np.float32)
        cv2.circle(mask, (w // 8, h // 8), 8, 1.0, -1)
        return [
            RawInstance(
                TargetClass.CIRCLE, mask, (w / 2 - 32, h / 2 - 32, w / 2 + 32, h / 2 + 32), 0.9
            )
        ]


class _Plain:
    def infer(self, frame):
        return []


def _frames(n, h=120, w=160):
    for _ in range(n):
        yield np.full((h, w, 3), (0, 0, 200), np.uint8)


def test_staged_backend_reports_every_stage():
    rep = bench.run(_Staged(), _frames(30), warmup=5)
    assert rep["frames"] == 25 and rep["staged"]
    assert set(rep["latency_ms"]) == set(bench.STAGES)
    lat = rep["latency_ms"]["total"]
    assert lat["p50"] <= lat["p95"] <= lat["p99"] and rep["fps"] > 0
    assert rep["soc"] and "t_s" in rep["soc"][0]


def test_plain_backend_has_no_pre_or_post_stage():
    rep = bench.run(_Plain(), _frames(12), warmup=2, duration_s=60)
    assert rep["frames"] == 10 and not rep["staged"]
    assert "preprocess" not in rep["latency_ms"] and "inference" in rep["latency_ms"]


def test_frames_from_folder_loop_and_from_video(tmp_path):
    for k in range(3):
        cv2.imwrite(str(tmp_path / f"{k}.png"), np.zeros((8, 8, 3), np.uint8))
    assert len(list(bench.frames(tmp_path, 7))) == 7
    assert len(list(bench.frames(tmp_path, None))) == 3
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
    for _ in range(4):
        w.write(np.zeros((16, 16, 3), np.uint8))
    w.release()
    assert len(list(bench.frames(video, 9))) == 9  # loops over the 4 frames
    with pytest.raises(FileNotFoundError):
        next(bench.frames(tmp_path / "none.mp4", 1))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        next(bench.frames(tmp_path / "empty", 1))


def test_cli_writes_a_report(tmp_path, monkeypatch, capsys):
    for k in range(2):
        cv2.imwrite(str(tmp_path / f"{k}.png"), np.zeros((64, 64, 3), np.uint8))
    monkeypatch.setattr(bench, "load_backend", lambda weights: _Plain())
    out = tmp_path / "b.json"
    args = ["--weights", "m.onnx", "--source", str(tmp_path), "--frames", "5", "--warmup", "1"]
    assert bench.main([*args, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["frames"] == 5
    assert '"weights": "m.onnx"' in capsys.readouterr().out
