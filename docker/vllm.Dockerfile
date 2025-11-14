FROM vllm/vllm-openai:v0.10.1

# Specify model to pre-download during build (optional, for caching)
ARG MODEL_TO_CACHE=""

COPY --link . /daemon/
COPY --link vllm_templates /opt/vllm/templates

WORKDIR /daemon/nilai-models/

RUN apt-get update && \
    apt-get install build-essential -y && \
    pip install uv && \
    uv sync && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Pre-download model if MODEL_TO_CACHE is provided
# This creates a cached layer with the model to avoid re-downloading in CI
RUN --mount=type=secret,id=hf_token \
    if [ -n "$MODEL_TO_CACHE" ]; then \
        echo "Pre-downloading model: $MODEL_TO_CACHE"; \
        if [ -f /run/secrets/hf_token ]; then \
            export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
        fi; \
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_TO_CACHE', cache_dir='/root/.cache/huggingface')" \
        || { echo >&2 "ERROR: Failed to pre-download model '$MODEL_TO_CACHE'. Check your network connection, HF_TOKEN, and model name."; exit 1; }; \
    else \
        echo "No model specified for caching, will download at runtime"; \
    fi

# Expose port 8000 for incoming requests
EXPOSE 8000

ENTRYPOINT ["bash", "run.sh"]

CMD [""]
