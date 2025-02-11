#!/bin/bash
# nxbench-run.sh
# This wrapper extracts the --config value and uses it to bind mount the host file.
# It supports both relative and absolute paths for the config file.

CONFIG=""
ARGS=()

while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--config" ]]; then
        shift
        CONFIG="$1"
        ARGS+=("--config" "/app/config_override.yaml")
    else
        ARGS+=("$1")
    fi
    shift
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config argument is required."
    exit 1
fi

if [[ "$CONFIG" == ~* ]]; then
    CONFIG="${CONFIG/#\~/$HOME}"
fi

if [[ "$CONFIG" != /* ]]; then
    CONFIG="$(pwd)/$CONFIG"
fi

echo "Using config file: $CONFIG"

docker-compose -f docker/docker-compose.cpu.yaml run --rm \
    -v "$CONFIG":/app/config_override.yaml:ro \
    nxbench nxbench "${ARGS[@]}"
