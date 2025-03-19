#!/bin/bash

set -euo pipefail

start_primary_process() {
    echo "Starting the primary process"
    echo "Args: [$*]"
    python3 -m vllm.entrypoints.openai.api_server $@ &
}

start_secondary_process() {
    echo "Starting the secondary process"
    uv run python3 -m nilai_models.daemon
}

main() {
    echo "Starting the main process with args: $*"
    if [[ " $* " =~ "--standalone" ]]; then
        echo "Starting the standalone server"
        # Remove --standalone from arguments
        args=("${@/--standalone/}")
        start_primary_process "${args[@]}"
    else
        start_primary_process "$@"
        start_secondary_process
    fi

    # Wait for any process to exit and exit with its status
    wait -n
    exit $?
}

main "$@"