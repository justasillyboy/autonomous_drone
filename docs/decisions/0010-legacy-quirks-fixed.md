# ADR-0010: Legacy mission quirks fixed after the port

- **Status:** Accepted
- **Date:** 2026-10-24
- **Code:** `uav_core/src/uav_core/mission/phases/search.py`, `phases/recovery.py`,
  `mission/state_machine.py`; goldens `qual_vision_only`, `qual_perception_filters`

## Context

The port reproduced `mission.py` bit-exactly, including five behaviours that looked wrong
(design spec, section 6). They were kept so the characterisation goldens could prove the port;
with that proof in git, each is now either fixed or kept on purpose.

## Decision

| Legacy behaviour | Now |
|---|---|
| The vision altitude memory was updated from every circle, before the colour check, with the current target's real diameter: a yellow circle "measured" with the red 1.0 m diameter gave a wrong altitude | Only circles of the target's colour feed it |
| The scored-zone test for a circle used the altitude of the circle examined before it | It uses the circle's own altitude, or the hardware altitude when there is one |
| RECOVERY accepted circles of any size, so an oversized red disc bounced it between RECOVERY and TARGETING | Same size cap as targeting |
| Descending below 3.0 m while landing counted toward the self-tracked altitude penalty | Not counted in FIND_HELIPAD and LANDING (the simulation referee excludes the landing approach too) |
| The geofence override drops `drop_ball` | **Kept**: a ball released while the drone is pulled back from the boundary would miss; the next aligned frame drops it |

## Consequences

- Two goldens were re-recorded from `uav_core` (`record_golden.py`); the other eight are
  unchanged. In `qual_perception_filters` the drone now stays in RECOVERY next to the
  oversized disc instead of oscillating. In `qual_vision_only` it locks the next target from
  t = 6.55 s, where the legacy code coasted because the previous circle's altitude put the
  target inside a scored zone; final score and balls are the same (40 points, 2 balls).
- The goldens no longer track the legacy file for these two scenarios; the legacy recordings
  stay in git history (commit 572856f).
