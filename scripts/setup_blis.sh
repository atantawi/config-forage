#!/usr/bin/env bash
# Clone and build the pinned BLIS stack (upstream blis-config-search SETUP.md §1–2) into DEST.
# Idempotent: existing clones are kept (not re-cloned) but their commits are still verified;
# the binary is built if missing or older than the checkout's Go sources. Prints the four BLIS_* exports on success.
#
#   scripts/setup_blis.sh ../blis-stack
set -euo pipefail

DEST=${1:?usage: setup_blis.sh DEST}
SIM_TAG=v0.10.0
SIM_SHA=0d092d4ef64f605046de1181adf8e2b6ee564725   # release v0.10.0 (kernel v0.1.0 in go.mod)
CATALOG_TAG=0.2.1
REGISTRY_TAG=v0.1.1                                 # MUST be v0.1.1: it carries cost-model-memory
GH=https://github.com/inference-sim

mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)

clone() {  # clone REPO TAG -- skipped if the directory exists
  local repo=$1 tag=$2
  if [[ -d "$DEST/$repo/.git" ]]; then
    echo "skip clone: $DEST/$repo exists" >&2
  else
    git clone --branch "$tag" --depth 1 "$GH/$repo.git" "$DEST/$repo" >&2
  fi
}

check_tag() {  # check_tag REPO TAG -- HEAD must be the tagged commit
  local repo=$1 tag=$2 head tagged
  head=$(git -C "$DEST/$repo" rev-parse HEAD)
  tagged=$(git -C "$DEST/$repo" rev-parse "refs/tags/$tag^{commit}" 2>/dev/null || true)
  if [[ "$head" != "$tagged" ]]; then
    echo "$repo: HEAD $head is not tag $tag (${tagged:-tag missing}) -- aborting" >&2
    exit 1
  fi
}

clone inference-sim "$SIM_TAG"
# The v0.10.0 tag is not immutable upstream: verify it still resolves to the pinned commit.
HEAD_SHA=$(git -C "$DEST/inference-sim" rev-parse HEAD)
if [[ "$HEAD_SHA" != "$SIM_SHA" ]]; then
  echo "inference-sim HEAD is $HEAD_SHA, expected $SIM_SHA ($SIM_TAG) -- aborting" >&2
  exit 1
fi
BIN="$DEST/inference-sim/blis"
STALE=""
if [[ -x "$BIN" ]]; then  # rebuild if any Go source or go.mod/go.sum is newer than the binary
  STALE=$(find "$DEST/inference-sim" \( -name '*.go' -o -name go.mod -o -name go.sum \) -newer "$BIN" \
          -print -quit)
fi
if [[ -x "$BIN" && -z "$STALE" ]]; then
  echo "skip build: $BIN exists and is newer than the source" >&2
else
  [[ -n "$STALE" ]] && echo "rebuild: $STALE is newer than $BIN" >&2
  (cd "$DEST/inference-sim" && go build -o blis main.go) >&2
fi

clone blis-catalog "$CATALOG_TAG";  check_tag blis-catalog "$CATALOG_TAG"
clone blis-registry "$REGISTRY_TAG"; check_tag blis-registry "$REGISTRY_TAG"

echo "export BLIS_BIN=$DEST/inference-sim/blis"
echo "export BLIS_CATALOG=$DEST/blis-catalog"
echo "export BLIS_REGISTRY=$DEST/blis-registry"
echo "export BLIS_SCENARIOS=\${BLIS_SCENARIOS:-/tmp/cf-scenarios}"
