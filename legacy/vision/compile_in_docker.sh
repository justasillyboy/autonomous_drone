#!/usr/bin/env bash
# compile_in_docker.sh — build best.hef trong Hailo AI SW Suite Docker
#
# YÊU CẦU TRƯỚC:
#   1. Cài Docker Desktop / Docker Engine.
#   2. Đăng ký Hailo Developer Zone (miễn phí): https://hailo.ai/developer-zone/
#   3. Tải "Hailo AI Software Suite — Docker" (.zip), giải nén được file .tar.gz.
#   4. Nạp image:    docker load -i hailo_ai_sw_suite_<ver>.tar.gz
#      Kiểm tra:     docker images | grep hailo_ai_sw_suite
#
# CẢNH BÁO Apple Silicon (arm64):
#   Hailo image là linux/amd64. Docker sẽ emulate qua QEMU → chậm 10–30 lần,
#   có thể crash trên một bước optimize. Nên dùng máy x86_64 (cloud / WSL2).
#
# CHẠY:
#   ./compile_in_docker.sh                       # mặc định: best.onnx + hailo_calib/
#   HAILO_IMAGE=hailo_ai_sw_suite:2024.10 ./compile_in_docker.sh
set -euo pipefail

ONNX="${ONNX:-best.onnx}"
CALIB_DIR="${CALIB_DIR:-hailo_calib}"
OUT_DIR="${OUT_DIR:-hailo_output}"
HW_ARCH="${HW_ARCH:-hailo8l}"
MODEL_NAME="${MODEL_NAME:-yolo11n_seg}"
CLASSES="${CLASSES:-3}"
HAILO_IMAGE="${HAILO_IMAGE:-hailo_ai_sw_suite:latest}"

[[ -f "$ONNX" ]]      || { echo "Thiếu $ONNX trong $(pwd)"; exit 1; }
[[ -d "$CALIB_DIR" ]] || { echo "Thiếu $CALIB_DIR/ trong $(pwd)"; exit 1; }
command -v docker >/dev/null || { echo "Docker chưa cài"; exit 1; }
docker image inspect "$HAILO_IMAGE" >/dev/null 2>&1 || {
    echo "Chưa thấy image '$HAILO_IMAGE'."
    echo "Tải Hailo AI SW Suite Docker từ developer zone, rồi:"
    echo "  docker load -i hailo_ai_sw_suite_<ver>.tar.gz"
    echo "Hoặc set HAILO_IMAGE=<tên-image-bạn-có>"
    exit 1
}

mkdir -p "$OUT_DIR"

# Apple Silicon: ép linux/amd64
PLATFORM_FLAG=""
if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
    PLATFORM_FLAG="--platform linux/amd64"
    echo "[warn] Host là ARM — bật emulation linux/amd64 (sẽ chậm)."
fi

docker run --rm -it $PLATFORM_FLAG \
    -v "$(pwd):/work" -w /work \
    "$HAILO_IMAGE" \
    bash -lc "
        set -e
        # bật venv suite (đường dẫn chuẩn trong image Hailo)
        source /local/shared_with_docker/hailo_venv/bin/activate 2>/dev/null \
          || source /opt/hailo/hailo_venv/bin/activate 2>/dev/null \
          || true

        hailomz compile $MODEL_NAME \
            --ckpt $ONNX \
            --calib-path $CALIB_DIR \
            --hw-arch $HW_ARCH \
            --classes $CLASSES \
            --output-dir $OUT_DIR
    "

HEF="$(find "$OUT_DIR" -name '*.hef' -print -quit)"
[[ -n "$HEF" ]] && cp "$HEF" best.hef && echo "OK → best.hef ($(du -h best.hef | cut -f1))"
