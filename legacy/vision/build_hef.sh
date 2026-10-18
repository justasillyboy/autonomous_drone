#!/usr/bin/env bash
# build_hef.sh — compile best.onnx → best.hef cho Hailo-8L
# CHẠY TRÊN UBUNTU 20.04/22.04 x86_64 với Hailo AI SW Suite đã cài.
# Không chạy được trên macOS / ARM / Windows-native.
set -euo pipefail

ONNX="${ONNX:-best.onnx}"
CALIB_DIR="${CALIB_DIR:-hailo_calib}"
OUT_DIR="${OUT_DIR:-hailo_output}"
HW_ARCH="${HW_ARCH:-hailo8l}"
MODEL_NAME="${MODEL_NAME:-yolo11n_seg}"
CLASSES="${CLASSES:-3}"

[[ -f "$ONNX" ]]      || { echo "Thiếu $ONNX"; exit 1; }
[[ -d "$CALIB_DIR" ]] || { echo "Thiếu $CALIB_DIR"; exit 1; }
command -v hailomz >/dev/null || { echo "hailomz chưa cài — bật Hailo AI SW Suite venv trước"; exit 1; }

mkdir -p "$OUT_DIR"

hailomz compile "$MODEL_NAME" \
    --ckpt "$ONNX" \
    --calib-path "$CALIB_DIR" \
    --hw-arch "$HW_ARCH" \
    --classes "$CLASSES" \
    --output-dir "$OUT_DIR"

HEF="$(find "$OUT_DIR" -name '*.hef' -print -quit)"
[[ -n "$HEF" ]] && cp "$HEF" best.hef && echo "OK → best.hef"
