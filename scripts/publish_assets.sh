#!/usr/bin/env bash
# Create (or reuse) a GitHub Release and upload every file listed in assets/SHA256SUMS,
# plus the manifest itself. Assets already on the release with the same size are skipped.
#
# Usage:  scripts/publish_assets.sh [tag]          (default tag: legacy-assets-v0)
# Auth:   $GITHUB_TOKEN, else `gh auth token`, else the git credential helper for github.com.
#         The token needs the `repo` scope.
set -euo pipefail

TAG="${1:-legacy-assets-v0}"
REPO="${ASSET_REPO:-justasillyboy/autonomous_drone}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${ROOT}/assets"
MANIFEST="${ASSET_DIR}/SHA256SUMS"
API="https://api.github.com/repos/${REPO}"
UPLOADS="https://uploads.github.com/repos/${REPO}"

die() { echo "error: $*" >&2; exit 1; }

github_token() {
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    printf '%s' "$GITHUB_TOKEN"
  elif command -v gh >/dev/null 2>&1 && gh auth token >/dev/null 2>&1; then
    gh auth token
  else
    printf 'protocol=https\nhost=github.com\n\n' \
      | GIT_TERMINAL_PROMPT=0 git credential fill 2>/dev/null \
      | sed -n 's/^password=//p'
  fi
}

json_field() {  # json_field <python-expression over d> < json
  python3 -c "import json, sys; d = json.load(sys.stdin); print($1)"
}

[[ -f "$MANIFEST" ]] || die "missing manifest: $MANIFEST"
TOKEN="$(github_token)"
[[ -n "$TOKEN" ]] || die "no GitHub token (set GITHUB_TOKEN or run 'gh auth login')"
AUTH=(-H "Authorization: Bearer ${TOKEN}" -H "Accept: application/vnd.github+json"
      -H "X-GitHub-Api-Version: 2022-11-28")

release_json="$(curl -fsS "${AUTH[@]}" "${API}/releases/tags/${TAG}" 2>/dev/null || true)"
if [[ -z "$release_json" ]]; then
  echo "creating release ${TAG}"
  payload="$(TAG="$TAG" python3 -c '
import json, os
tag = os.environ["TAG"]
print(json.dumps({
    "tag_name": tag,
    "target_commitish": "main",
    "name": tag,
    "body": "Binary assets (weights, datasets, videos) imported from the original "
            "competition project. See legacy/README.md for provenance and "
            "scripts/fetch_assets.sh to download and verify them.",
}))')"
  release_json="$(curl -fsS "${AUTH[@]}" -X POST "${API}/releases" -d "$payload")"
fi
release_id="$(json_field 'd["id"]' <<<"$release_json")"
existing="$(json_field '"\n".join(a["name"] + " " + str(a["size"]) for a in d["assets"])' <<<"$release_json")"

upload() {  # upload <path> <name>
  local path="$1" name="$2" size
  size="$(wc -c <"$path" | tr -d ' ')"
  if grep -qx "${name} ${size}" <<<"$existing"; then
    echo "skip (present)  ${name}"
    return
  fi
  echo "upload          ${name} (${size} bytes)"
  curl -fsS "${AUTH[@]}" -H "Content-Type: application/octet-stream" \
    --data-binary @"$path" -o /dev/null \
    "${UPLOADS}/releases/${release_id}/assets?name=${name}"
}

while read -r _ name; do
  [[ -f "${ASSET_DIR}/${name}" ]] || die "listed in manifest but missing locally: ${name}"
  upload "${ASSET_DIR}/${name}" "$name"
done <"$MANIFEST"
upload "$MANIFEST" SHA256SUMS

echo "done: https://github.com/${REPO}/releases/tag/${TAG}"
