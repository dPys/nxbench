#!/bin/bash
# nxbench-run.sh
# This wrapper extracts the --config value and uses it to bind mount the host file.
# It supports both relative and absolute paths for the config file.
#
# Additionally, it detects if the user wants to run:
#   - a visualization (viz) command,
#   - export benchmark results (benchmark export), or
#   - any other nxbench subcommand (e.g. data download, data list, benchmark run).
#
# It then selects the appropriate docker-compose service with published ports
# and mounts the host results directory if needed.
#
# You can also pass --gpu to switch between the CPU and GPU docker-compose versions.
#
# Examples:
#   ./docker/nxbench-run.sh --config 'nxbench/configs/example.yaml' benchmark run
#   ./docker/nxbench-run.sh --config 'nxbench/configs/example.yaml' viz serve
#   ./docker/nxbench-run.sh --config 'nxbench/configs/example.yaml' benchmark export 'results/input.json' --output-format csv --output-file 'results/results.csv'
#   ./docker/nxbench-run.sh --config 'nxbench/configs/example.yaml' data download karate
#   ./docker/nxbench-run.sh --config 'nxbench/configs/example.yaml' data list --category social

CONFIG=""
ARGS=()
COMPOSE_FILE="docker/docker-compose.cpu.yaml"
EXPORT_MODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            shift
            CONFIG="$1"
            ARGS+=("--config" "/app/config_override.yaml")
            ;;
        --gpu)
            COMPOSE_FILE="docker/docker-compose.gpu.yaml"
            export NUM_GPU=1
            ;;
        *)
            ARGS+=("$1")
            ;;
    esac
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
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' does not exist."
    exit 1
fi

echo "Using config file: $CONFIG"
echo "Using docker-compose file: $COMPOSE_FILE"

# Build a SUBCOMMAND_ARGS array that excludes the --config flag and its replacement.
SUBCOMMAND_ARGS=()
skip_next=0
for arg in "${ARGS[@]}"; do
    if [ $skip_next -eq 1 ]; then
        skip_next=0
        continue
    fi
    if [ "$arg" == "--config" ]; then
        skip_next=1
        continue
    fi
    if [ "$arg" == "/app/config_override.yaml" ]; then
        continue
    fi
    SUBCOMMAND_ARGS+=("$arg")
done

SERVICE="nxbench"
# Check if we are running "viz" or "benchmark export"
for (( i=0; i<${#SUBCOMMAND_ARGS[@]}; i++ )); do
    if [[ "${SUBCOMMAND_ARGS[$i]}" == "viz" ]]; then
        SERVICE="dashboard"
        break
    fi
    if [[ "${SUBCOMMAND_ARGS[$i]}" == "export" ]]; then
        EXPORT_MODE=1
        # Do not break; we want to process export below.
    fi
done

# For dashboard or export mode we mount the host's results directory.
if [ "$SERVICE" == "dashboard" ] || [ $EXPORT_MODE -eq 1 ]; then
    RESULTS_DIR="$(pwd)/results"
    mkdir -p "$RESULTS_DIR"

    if [ "$SERVICE" == "dashboard" ]; then
        # For viz mode, require that the output CSV already exists.
        RESULTS_FILE="$RESULTS_DIR/results.csv"
        if [ ! -f "$RESULTS_FILE" ]; then
            echo "Error: Results file '$RESULTS_FILE' does not exist. Please run the benchmark first."
            exit 1
        fi
    elif [ $EXPORT_MODE -eq 1 ]; then
        # For export mode, we expect the command to be:
        # nxbench benchmark export <input_json> --output-format ... --output-file <output_csv>
        if [ "${#SUBCOMMAND_ARGS[@]}" -ge 3 ] && [ "${SUBCOMMAND_ARGS[0]}" = "benchmark" ] && [ "${SUBCOMMAND_ARGS[1]}" = "export" ]; then
            INPUT_JSON_FILE="$(pwd)/${SUBCOMMAND_ARGS[2]}"
            if [ ! -f "$INPUT_JSON_FILE" ]; then
                echo "Error: Input results file '$INPUT_JSON_FILE' does not exist."
                exit 1
            fi
        else
            echo "Error: Unable to determine input results file for export command."
            exit 1
        fi
    fi

    docker-compose -f "$COMPOSE_FILE" run --rm --service-ports \
        -v "$CONFIG":/app/config_override.yaml:ro \
        -v "$RESULTS_DIR":/app/results \
        $SERVICE nxbench "${ARGS[@]}"
else
    docker-compose -f "$COMPOSE_FILE" run --rm \
        -v "$CONFIG":/app/config_override.yaml:ro \
        $SERVICE nxbench "${ARGS[@]}"
fi
