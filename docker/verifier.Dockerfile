FROM python:3.12-slim AS gpuverifier

COPY --link ./gpuverifier-api /app/

WORKDIR /app
RUN apt-get update && \
apt-get install curl git -y && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* && \
pip install --upgrade uv && \
uv sync

EXPOSE 8000

CMD ["/app/launch.sh"]
