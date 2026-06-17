#!/usr/bin/env bash
set -euo pipefail

IMAGE="arkiev/phantomkit"

# Derive version from the most recent git tag (matches hatch-vcs behaviour).
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "dev")

echo "Building ${IMAGE}:${VERSION} and ${IMAGE}:latest"

docker buildx build \
    --platform linux/amd64 \
    --file docker/Dockerfile \
    --tag "${IMAGE}:${VERSION}" \
    --tag "${IMAGE}:latest" \
    --push \
    .
