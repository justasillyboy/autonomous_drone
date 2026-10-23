from pathlib import Path

from uav_core.config import MissionConfig
from uav_core.config_doc import main, render_markdown

REFERENCE = Path(__file__).resolve().parents[2] / "docs" / "config-reference.md"


def test_committed_reference_matches_the_schema():
    assert REFERENCE.read_text() == render_markdown(), (
        "docs/config-reference.md is stale: run `uv run python -m uav_core.config_doc`"
    )


def test_every_yaml_key_is_documented():
    doc = render_markdown()
    for section, info in MissionConfig.model_fields.items():
        for name in info.annotation.model_fields:
            assert f"| `{section}.{name}` |" in doc


def test_table_cells_escape_pipes():
    # "Max |pitch| and |roll| at release" must not split the row.
    row = next(line for line in render_markdown().splitlines() if "drop.max_pitch_deg" in line)
    assert r"\|pitch\|" in row
    assert row.count(" | ") == 5


def test_constraints_and_defaults_are_rendered():
    doc = render_markdown()
    assert "| `drop.align_pixel_thresh_px` | 25 | px | > 0 |" in doc
    assert "| `control.pid_finals` | kp=0.6, ki=0.02, kd=0.2 |" in doc
    assert "red: (0, 120, 80)–(10, 255, 255) ∪ (165, 120, 80)–(180, 255, 255)" in doc


def test_main_writes_the_file(tmp_path):
    out = tmp_path / "ref.md"
    assert main([str(out)]) == 0
    assert out.read_text() == render_markdown()
