FROM python:3.12-slim AS nilai

# Model path can be specified during build
ARG MODEL_NAME=llama_1b_cpu
ENV EXEC_PATH=nilai_models.models.${MODEL_NAME}:app

COPY --link . /app/

WORKDIR /app/nilai-models/

RUN apt-get update && \
    apt-get install build-essential -y && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv && \
    uv sync

EXPOSE 8000

# Use shell form to properly expand the environment variable
CMD uv run gunicorn -c gunicorn.conf.py ${EXEC_PATH}