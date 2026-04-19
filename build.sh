#!/bin/bash
set -euo pipefail

OUT_DIR="./build"
mkdir -p "$OUT_DIR"

echo "=== Cross-compiling RAGFile (library) ==="
echo ""

# Targets: linux/amd64, linux/arm64, darwin/amd64, darwin/arm64, linux/arm (armv6), linux/arm (armv7)
TARGETS=(
    "linux/amd64"
    "linux/arm64"
    "darwin/amd64"
    "darwin/arm64"
    "linux/arm/6"
    "linux/arm/7"
)

for target in "${TARGETS[@]}"; do
    IFS='/' read -r goos goarch goarm <<< "$target"
    
    if [ -n "$goarm" ]; then
        suffix="${goos}-${goarch}v${goarm}"
        echo "Building ${goos}/${goarch} (ARMv${goarm})..."
        GOOS=$goos GOARCH=$goarch GOARM=$goarm go build ./...
    else
        suffix="${goos}-${goarch}"
        echo "Building ${goos}/${goarch}..."
        GOOS=$goos GOARCH=$goarch go build ./...
    fi
    
    echo "✓ ${suffix}"
done

echo ""
echo "All targets compiled successfully!"