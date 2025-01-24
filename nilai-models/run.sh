#!/bin/bash

echo "Starting the primary process"
# Start the primary process and put it in the background
echo "Args: $*"
python3 -m vllm.entrypoints.openai.api_server $* & #--model $1 --gpu-memory-utilization 0.5 --max-model-len 10000 --tensor-parallel-size 1 &

echo "Starting the secondary process"
# Start the helper process
uv run python3 -m nilai_models.daemon

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
