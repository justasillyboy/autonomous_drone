#!/usr/bin/env bash
# setup_ubuntu_and_compile.sh
# Chạy trên Ubuntu 22.04 x86_64 (Cloud Shell / EC2 / WSL2 / VPS).
# Giả định bạn ĐÃ upload sẵn vào thư mục hiện tại:
#   - best.onnx
#   - hailo_calib/         (286 ảnh JPG 640x640)
#   - hailo_ai_sw_suite_<ver>.tar.gz   (tải từ Hailo Developer Zone)
#
# Kết quả: best.hef ngay tại thư mục này.
set -euo pipefail

ONNX="best.onnx"
CALIB_DIR="hailo_calib"
OUT_DIR="hailo_output"
HW_ARCH="hailo8l"
MODEL_NAME="yolo11n_seg"
CLASSES="3"

# ── 1. Sanity checks ───────────────────────────────────────────────────────
[[ "$(uname -m)" == "x86_64" ]] || { echo "Cần host x86_64, đang là $(uname -m)"; exit 1; }
[[ -f "$ONNX" ]]                || { echo "Thiếu $ONNX"; exit 1; }
[[ -d "$CALIB_DIR" ]]           || { echo "Thiếu $CALIB_DIR/"; exit 1; }

HAILO_TGZ="$(ls hailo_ai_sw_suite_*.tar.gz 2>/dev/null | head -n1 || true)"
[[ -n "$HAILO_TGZ" ]] || { echo "Thiếu hailo_ai_sw_suite_<ver>.tar.gz (tải từ developer zone)"; exit 1; }

# ── 2. Cài Docker nếu chưa có ──────────────────────────────────────────────
if ! command -v docker >/dev/null; then
    echo "[setup] Cài Docker..."
    sudo apt-get update -y
    sudo apt-get install -y docker.io
    sudo usermod -aG docker "$USER" || true
    sudo systemctl enable --now docker || true
fi

# ── 3. Nạp image Hailo nếu chưa có ─────────────────────────────────────────
HAILO_IMAGE="$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -i hailo_ai_sw_suite | head -n1 || true)"
if [[ -z "$HAILO_IMAGE" ]]; then
    echo "[setup] Nạp image Hailo từ $HAILO_TGZ (3-5 phút)..."
    docker load -i "$HAILO_TGZ"
    HAILO_IMAGE="$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -i hailo_ai_sw_suite | head -n1)"
fi
echo "[setup] Image: $HAILO_IMAGE"

# ── 4. Compile ─────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"
docker run --rm \
    -v "$(pwd):/work" -w /work \
    "$HAILO_IMAGE" \
    bash -lc "
        set -e
        source /local/shared_with_docker/hailo_venv/bin/activate 2>/dev/null \
          || source /opt/hailo/hailo_venv/bin/activate 2>/dev/null || true

        hailomz compile $MODEL_NAME \
            --ckpt $ONNX \
            --calib-path $CALIB_DIR \
            --hw-arch $HW_ARCH \
            --classes $CLASSES \
            --output-dir $OUT_DIR
    "

HEF="$(find "$OUT_DIR" -name '*.hef' -print -quit)"
[[ -n "$HEF" ]] || { echo "Không thấy .hef trong $OUT_DIR"; exit 1; }
cp "$HEF" best.hef
echo "OK → $(pwd)/best.hef ($(du -h best.hef | cut -f1))"
