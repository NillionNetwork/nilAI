# Running vLLM without docker


```shell
# For Llama 8B
uv run bash run.sh \
--model meta-llama/Llama-3.1-8B-Instruct \
--gpu-memory-utilization 0.5 \
--max-model-len 10000 \
--tensor-parallel-size 1
```

```shell
# For Llama 1B
bash run.sh --model meta-llama/Llama-3.2-1B-Instruct \
--gpu-memory-utilization 0.2 \
--max-model-len 10000 \
--tensor-parallel-size 1
```
