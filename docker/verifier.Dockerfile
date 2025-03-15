FROM python:3.12-slim AS gpuverifier

COPY --link ./gpuverifier-api /app/

ENV PATH="/venv/bin:$PATH"

WORKDIR /app
RUN apt-get update && \
apt-get install curl -y && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* && \
pip install --upgrade .

EXPOSE 8000

CMD ["/app/launch.sh"]
