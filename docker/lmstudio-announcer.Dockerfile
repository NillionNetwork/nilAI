FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY --link packages/nilai-common /app/packages/nilai-common
COPY --link nilai-models /app/nilai-models

RUN pip install --upgrade pip && \
    pip install /app/packages/nilai-common /app/nilai-models

ENTRYPOINT ["python", "-m", "nilai_models.lmstudio_announcer"]
