FROM vllm/vllm-openai:v0.10.1

# Specify model to pre-download during build (optional, for caching)
ARG MODEL_TO_CACHE=""
ARG HF_TOKEN=""

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
RUN if [ -n "$MODEL_TO_CACHE" ]; then \
        echo "Pre-downloading model: $MODEL_TO_CACHE"; \
        export HF_TOKEN="${HF_TOKEN}"; \
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_TO_CACHE', cache_dir='/root/.cache/huggingface')"; \
    else \
        echo "No model specified for caching, will download at runtime"; \
    fi

# Expose port 8000 for incoming requests
EXPOSE 8000

ENTRYPOINT ["bash", "run.sh"]

CMD [""]
