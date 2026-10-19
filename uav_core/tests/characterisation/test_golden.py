"""uav_core must reproduce the legacy command sequence on every scripted scenario."""

from __future__ import annotations

import pytest

from characterisation.new_runner import run_new
from characterisation.records import compare, golden_path, read_golden
from characterisation.scenarios import SCENARIOS


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_matches_golden(scenario):
    expected = read_golden(golden_path(scenario.name))
    problems = compare(expected, run_new(scenario))
    assert not problems, "\n".join(problems)
