#!/usr/bin/env bash
# Download binary assets from the GitHub Release and verify their SHA-256.
#
# Usage:
#   scripts/fetch_assets.sh                 # every asset listed in assets/SHA256SUMS
#   scripts/fetch_assets.sh real-frames-300.zip yolo11n-seg_synth_2026-04-14.pt
#
# Files that are already present with the right checksum are skipped.
# Env: ASSET_TAG (default legacy-assets-v0), ASSET_REPO (default justasillyboy/autonomous_drone)
set -euo pipefail

TAG="${ASSET_TAG:-legacy-assets-v0}"
REPO="${ASSET_REPO:-justasillyboy/autonomous_drone}"
BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
ASSET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/assets"
MANIFEST="${ASSET_DIR}/SHA256SUMS"

if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
else
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

expected_hash() {  # expected_hash <name> -> prints hash or nothing
  awk -v n="$1" '$2 == n { print $1 }' "$MANIFEST"
}

if [[ $# -gt 0 ]]; then
  names=("$@")
else
  names=()
  while read -r _ name; do names+=("$name"); done < "$MANIFEST"
fi

failures=0
for name in "${names[@]}"; do
  want="$(expected_hash "$name")"
  if [[ -z "$want" ]]; then
    echo "unknown asset (not in SHA256SUMS): $name" >&2
    failures=$((failures + 1))
    continue
  fi

  dest="${ASSET_DIR}/${name}"
  if [[ -f "$dest" && "$(sha256 "$dest")" == "$want" ]]; then
    echo "ok (cached)  $name"
    continue
  fi

  echo "download     $name"
  curl -fL --retry 3 --progress-bar -o "${dest}.part" "${BASE_URL}/${name}"
  got="$(sha256 "${dest}.part")"
  if [[ "$got" != "$want" ]]; then
    echo "CHECKSUM MISMATCH $name: expected $want, got $got" >&2
    rm -f "${dest}.part"
    failures=$((failures + 1))
    continue
  fi
  mv "${dest}.part" "$dest"
  echo "ok           $name"
done

if [[ $failures -gt 0 ]]; then
  echo "$failures asset(s) failed" >&2
  exit 1
fi
