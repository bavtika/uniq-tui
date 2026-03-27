#!/usr/bin/env bash
set -euo pipefail

# Installs cuda_ed25519_vanity (+ libcuda-crypt.so) into bin/.
# Default CUDA sources: ./cuda-miner relative to repository root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="$REPO_ROOT/bin"
HINT_FILE="$REPO_ROOT/.gpu-binary-path"
DEFAULT_NAME="cuda_ed25519_vanity"
DEFAULT_SOURCE_BIN="cuda_ed25519_vanity"
PARENT_CUDA_MINER="$(cd "$REPO_ROOT/.." && pwd)/cuda-miner"
# Prefer UNIQ_GPU_SOURCE_DIR, then ./cuda-miner, then ../cuda-miner
DEFAULT_SOURCE_DIR="${UNIQ_GPU_SOURCE_DIR:-}"
if [[ -z "$DEFAULT_SOURCE_DIR" ]]; then
  if [[ -d "$REPO_ROOT/cuda-miner/src" && -f "$REPO_ROOT/cuda-miner/src/Makefile" ]]; then
    DEFAULT_SOURCE_DIR="$REPO_ROOT/cuda-miner"
  elif [[ -d "$PARENT_CUDA_MINER/src" && -f "$PARENT_CUDA_MINER/src/Makefile" ]]; then
    DEFAULT_SOURCE_DIR="$PARENT_CUDA_MINER"
  fi
fi

FROM_PATH=""
FROM_URL=""
NAME="$DEFAULT_NAME"
AUTO_MODE=false

usage() {
  cat <<EOF
Usage:
  $0 --from /path/to/gpu_binary [--name cuda_ed25519_vanity]
  $0 --from-url https://example.com/gpu_binary [--name cuda_ed25519_vanity]
  $0 --auto [--name cuda_ed25519_vanity]

Source priority (auto):
  1) Build from \$UNIQ_GPU_SOURCE_DIR, or ./cuda-miner, or ../cuda-miner
  2) Existing $INSTALL_DIR/$DEFAULT_NAME
  3) UNIQ_GPU_BINARY_URL download

After install:
  cd "$REPO_ROOT"
  cargo run --release -- ...
EOF
}

download_to_file() {
  local url="$1"
  local target="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail "$url" -o "$target"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$target" "$url"
  else
    echo "Neither curl nor wget is installed."
    return 1
  fi
}

build_from_source() {
  local out_target="$1"
  local src_dir="$DEFAULT_SOURCE_DIR"
  local src_bin="$DEFAULT_SOURCE_BIN"
  local built=""

  if [[ ! -d "$src_dir" ]]; then
    echo "GPU source not found: $src_dir"
    return 1
  fi
  if [[ ! -f "$src_dir/src/Makefile" ]]; then
    echo "GPU source invalid (no src/Makefile): $src_dir"
    return 1
  fi

  if ! command -v nvcc >/dev/null 2>&1; then
    echo "CUDA not found. Install nvidia-cuda-toolkit or NVIDIA CUDA toolkit."
    return 1
  fi
  if ! command -v make >/dev/null 2>&1; then
    echo "make not installed."
    return 1
  fi

  echo "Building CUDA miner from source: $src_dir"
  if ! make -C "$src_dir/src" -j"$(nproc)" V=release; then
    echo "GPU build failed."
    return 1
  fi

  if [[ -x "$src_dir/src/release/$src_bin" ]]; then
    built="$src_dir/src/release/$src_bin"
  fi

  if [[ -z "$built" || ! -f "$built" ]]; then
    echo "Build succeeded but binary not found."
    return 1
  fi

  cp "$built" "$out_target"
  chmod +x "$out_target"
  local built_dir
  built_dir="$(dirname "$built")"
  if [[ -f "$built_dir/libcuda-crypt.so" ]]; then
    cp "$built_dir/libcuda-crypt.so" "$INSTALL_DIR/libcuda-crypt.so"
  fi
  echo "Built: $(basename "$built")"
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      FROM_PATH="${2:-}"
      shift 2
      ;;
    --from-url)
      FROM_URL="${2:-}"
      shift 2
      ;;
    --name)
      NAME="${2:-}"
      shift 2
      ;;
    --auto)
      AUTO_MODE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$FROM_PATH" && -n "$FROM_URL" ]]; then
  echo "Use only one source: --from or --from-url"
  exit 1
fi

mkdir -p "$INSTALL_DIR"
TARGET="$INSTALL_DIR/$NAME"
TMP_TARGET="${TARGET}.tmp"
RESOLVED_IN_AUTO=false

if [[ "$AUTO_MODE" == true && -z "$FROM_PATH" && -z "$FROM_URL" ]]; then
  if build_from_source "$TARGET"; then
    RESOLVED_IN_AUTO=true
  fi
  if [[ "$RESOLVED_IN_AUTO" != true && -f "$TARGET" ]]; then
    chmod +x "$TARGET"
    ABS_TARGET="$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")"
    printf "%s\n" "$ABS_TARGET" > "$HINT_FILE"
    echo "Using local bundled GPU binary: $ABS_TARGET"
    echo "Wrote hint file: $HINT_FILE"
    exit 0
  fi
  if [[ -n "${UNIQ_GPU_BINARY_URL:-}" ]]; then
    FROM_URL="$UNIQ_GPU_BINARY_URL"
  fi
fi

if [[ "$RESOLVED_IN_AUTO" != true && -z "$FROM_PATH" && -z "$FROM_URL" ]]; then
  echo "Source is required."
  if [[ "$AUTO_MODE" == true ]]; then
    echo "Auto mode could not build or find binary at: $TARGET"
    echo "Set UNIQ_GPU_BINARY_URL or use --from."
  fi
  usage
  exit 1
fi

if [[ "$RESOLVED_IN_AUTO" == true ]]; then
  :
elif [[ -n "$FROM_PATH" ]]; then
  if [[ ! -f "$FROM_PATH" ]]; then
    echo "Source file not found: $FROM_PATH"
    exit 1
  fi
  cp "$FROM_PATH" "$TARGET"
else
  rm -f "$TMP_TARGET"
  if ! download_to_file "$FROM_URL" "$TMP_TARGET"; then
    echo "Failed to download GPU binary from: $FROM_URL"
    exit 1
  fi
  mv "$TMP_TARGET" "$TARGET"
fi

chmod +x "$TARGET"
ABS_TARGET="$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")"
printf "%s\n" "$ABS_TARGET" > "$HINT_FILE"

echo "Installed GPU binary: $ABS_TARGET"
echo "Wrote hint file: $HINT_FILE"
echo
echo "Run:"
echo "  cd \"$REPO_ROOT\""
echo "  cargo run --release -- --prefix YOURS"
