FROM vllm/vllm-openai:v0.11.2

# # Specify model name and path during build
# ARG MODEL_NAME=llama_1b_cpu
# ARG MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

# # Set environment variables
# ENV MODEL_NAME=${MODEL_NAME}
# ENV MODEL_PATH=${MODEL_PATH}
# ENV EXEC_PATH=nilai_models.models.${MODEL_NAME}:app

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

# Expose port 8000 for incoming requests
EXPOSE 8000

ENTRYPOINT ["bash", "run.sh"]

CMD [""]