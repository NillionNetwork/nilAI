FROM vllm/vllm-openai:latest

# # Specify model name and path during build
# ARG MODEL_NAME=llama_1b_cpu
# ARG MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

# # Set environment variables
# ENV MODEL_NAME=${MODEL_NAME}
# ENV MODEL_PATH=${MODEL_PATH}
# ENV EXEC_PATH=nilai_models.models.${MODEL_NAME}:app

COPY --link . /daemon/

WORKDIR /daemon/nilai-models/

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 libgl1 build-essential && \
    pip install uv pillow torchvision torchaudio && \
    uv sync && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies for multimodal models
RUN pip install pillow ftfy regex
RUN pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# Expose port 8000 for incoming requests
EXPOSE 8000

ENTRYPOINT ["bash", "run.sh"]

CMD [""]
