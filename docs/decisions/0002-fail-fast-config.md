# ADR-0002: Invalid configuration fails at startup

- **Status:** Accepted
- **Date:** 2026-10-24
- **Code:** `uav_core/src/uav_core/config.py`, generated reference
  [`docs/config-reference.md`](../config-reference.md)
- **Supersedes:** legacy decisions 1 ("`mission.py` is the single source of truth for
  constants") and 8 ("HSV calibration must be loaded before any inference frame")

## Context

The legacy constants were module globals in `mission.py`. `load_config_yaml()` overwrote them
from `config.yaml` and was lenient in every failure case:

| Situation | Legacy behaviour |
|---|---|
| File missing | Warning, flies on built-in defaults |
| YAML syntax error | Warning, flies on built-in defaults |
| Unknown or misspelled key | Silently ignored (only keys in a hand-kept map were read) |
| Wrong type | Coerced with `type(old)(value)` |
| `DROP_ALTITUDE_MIN < 3.0` | Silently reset to 3.05 m (REQ-NF-04) |

Because the values were globals, any object built **before** the load kept the old values
(defect D6), and the HSV calibration had to be loaded before the first `update()` call for the
same reason. A config mistake was therefore discovered in flight, if at all: a typo in
`drop_altitude_min` meant flying on the default without any error.

## Decision

- `MissionConfig` is a frozen pydantic v2 model with `extra="forbid"`. Every field carries a
  unit, a description, a rule source and a range; cross-field rules are model validators (drop
  window ordered, inside the rulebook's 3.05-5.0 m band, relaxed tilt >= strict tilt).
- `load_mission_config(path)` raises on a missing file, a YAML error, a non-mapping document, an
  unknown key or an out-of-range value. Nothing is corrected silently: `altitude_min_m < 3.05`
  is an error, not a reset.
- The config is loaded **once at the boundary** (launch file or CLI) and passed explicitly to
  constructors. There are no globals to override, so load order no longer matters.
- Field HSV calibration produces a new config through `with_hsv_ranges()`, validated the same
  way.
- `docs/config-reference.md` is generated from the schema, and a test fails if it drifts.

## Consequences

- A broken config keeps the drone on the ground with an error that names the offending key.
  That is the intent: config mistakes must be caught before take-off, not mid-match.
- Values cannot be tuned live; changing one means restarting the node. Acceptable for a
  five-minute match.
- The rulebook values (scoring tables, field size, match time) live in a separate
  `RulebookConfig` that is not loaded from YAML, so an operator file cannot change the rules.
- This is a deliberate behaviour change, so no characterisation golden depends on the legacy
  correction.
