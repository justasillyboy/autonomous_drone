# ADR-0006: Model classes are matched to TargetClass by name

- **Status:** Accepted (replaces legacy decision 2)
- **Date:** 2026-10-24
- **Code:** `edge/src/uav_edge/runtime/base.py` (`class_map`), `uav_core.types.TargetClass`

## Context

The legacy rule was "derive class ids from `dataset.py` CLASS_NAMES, never hard-code them".
With several backends and exported models, the model file itself is the authority on which
id means what: Ultralytics stores the table in `model.names` and in the ONNX `names`
metadata.

## Decision

Each backend reads the model's own class table and maps it to `TargetClass` by name.
An unknown class name is an error at startup.

## Consequences

- A model trained with another class order still works; a model with a renamed or extra
  class fails loudly instead of mislabelling helipads as circles.
- `TargetClass` values (`circle`, `semicircle`, `H_landing`) are the contract with the
  training data and must match the dataset's `data.yaml`.
