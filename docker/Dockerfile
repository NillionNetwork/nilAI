FROM python:3.12-slim

# Download model from HuggingFace and set path to local dir
ARG MODEL_PATH
ENV MODEL_PATH=${MODEL_PATH}

RUN pip install huggingface-hub[hf_transfer]
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN --mount=type=secret,id=HUGGINGFACE_API_TOKEN \
    HUGGINGFACE_API_TOKEN=$(cat /run/secrets/HUGGINGFACE_API_TOKEN) && \
    huggingface-cli download $MODEL_PATH

# Optionally set model filename
ARG MODEL_FILENAME
ENV MODEL_FILENAME=${MODEL_FILENAME}

COPY --link nilai /app/nilai
COPY pyproject.toml uv.lock .env gunicorn.conf.py /app/

WORKDIR /app

RUN apt-get update && \
apt-get install build-essential certbot -y && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* && \
pip install uv && \
uv sync

EXPOSE 8080 8443

CMD ["uv", "run", "gunicorn", "-c", "gunicorn.conf.py", "nilai.__main__:app"]