#!/usr/bin/env bash
# Launches the PhantomKit GUI in Docker and opens it in your browser.
#
# One-time setup (macOS / Linux):
#   1. Make sure Docker Desktop is installed and running.
#   2. Double-click this file (or run it from a terminal:
#      `bash phantomkit-gui.sh`).
#
# Your entire home folder is mounted read/write into the container so the
# GUI's built-in file browser can reach your scan data wherever it lives —
# no need to move files into a special folder first.
set -euo pipefail

IMAGE="${PHANTOMKIT_IMAGE:-arkiev/phantomkit:latest}"
PORT="${PHANTOMKIT_PORT:-7878}"
NAME="phantomkit-gui"
URL="http://localhost:${PORT}"

if ! docker info >/dev/null 2>&1; then
    echo "Docker doesn't seem to be running. Please start Docker Desktop and try again."
    read -r -p "Press Enter to close..." _
    exit 1
fi

echo "Checking for a newer image (${IMAGE})…"
docker pull "$IMAGE" || echo "Warning: couldn't reach Docker Hub — using whatever's cached locally."

# Always restart fresh rather than silently reusing whatever's already
# running under this name — otherwise a container started from a stale
# image before an update just keeps serving forever.
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "Starting PhantomKit GUI…"
docker run -d --rm \
    --name "$NAME" \
    --user "$(id -u):$(id -g)" \
    -p "${PORT}:7878" \
    -v "${HOME}:/hostuser" \
    -e PHANTOMKIT_HOME=/hostuser \
    -e HOME=/tmp \
    "$IMAGE" gui >/dev/null

echo "Waiting for the server to come up…"
for _ in $(seq 1 60); do
    if command -v curl >/dev/null 2>&1 && curl -fsS "$URL" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if command -v open >/dev/null 2>&1; then
    open "$URL" 2>/dev/null || echo "Open your browser to: ${URL}"           # macOS
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" 2>/dev/null || echo "Open your browser to: ${URL}"       # Linux
else
    echo "Open your browser to: ${URL}"
fi

echo ""
echo "PhantomKit GUI running at ${URL}"
echo "To stop it later, run: docker stop ${NAME}"
