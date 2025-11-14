FROM vllm/vllm-openai:v0.10.1

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

# Create cache directory structure (will be mounted from host at runtime)
RUN mkdir -p /root/.cache/huggingface

# Expose port 8000 for incoming requests
EXPOSE 8000

ENTRYPOINT ["bash", "run.sh"]

CMD [""]
