#!/usr/bin/env bash
# Run a command inside the factory image with the repo mounted (Git Bash / Linux / macOS).
here=$(cd "$(dirname "$0")/.." && (pwd -W 2>/dev/null || pwd))
MSYS_NO_PATHCONV=1 exec docker run --rm -v "$here:/app" -v ege-cache:/cache ${DOCKER_ARGS:-} ege-factory "$@"
