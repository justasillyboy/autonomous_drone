"""Import the real uav_core before pytest collects uav_core/tests/*: with
--import-mode=importlib pytest names a test module uav_core.tests.test_x and would otherwise
register an empty placeholder package called uav_core that hides the installed one."""

import uav_core  # noqa: F401
