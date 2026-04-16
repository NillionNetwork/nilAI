FROM vllm/vllm-openai:v0.19.0

# # Specify model name and path during build
# ARG MODEL_NAME=llama_1b_cpu
# ARG MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

# # Set environment variables
# ENV MODEL_NAME=${MODEL_NAME}
# ENV MODEL_PATH=${MODEL_PATH}
# ENV EXEC_PATH=nilai_models.models.${MODEL_NAME}:app

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COPY --link . /daemon/

WORKDIR /daemon/nilai-models/

# Install daemon dependencies into an isolated venv, separate from vLLM's /opt/venv.
# vLLM base image sets VIRTUAL_ENV=/opt/venv and puts it on PATH.
# UV_PROJECT_ENVIRONMENT ensures uv sync targets the daemon's own .venv.
RUN apt-get update && \
    apt-get install build-essential -y && \
    pip install uv && \
    UV_PROJECT_ENVIRONMENT=/daemon/nilai-models/.venv uv sync && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Upgrade transformers in vLLM's /opt/venv (where pip/python3 resolve via PATH).
# Gemma 4 architecture requires transformers>=5.5.0.
# --no-deps avoids pulling transitive deps that conflict with vLLM's pins.
RUN pip install 'transformers>=5.5.0' --no-deps && pip install 'huggingface-hub>=1.5.0,<2.0'

# Expose port 8000 for incoming requests
EXPOSE 8000

ENTRYPOINT ["bash", "run.sh"]

CMD [""]
