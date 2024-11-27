FROM python:3.12-slim

COPY --link nilai /app/nilai
COPY pyproject.toml uv.lock .env /app/

WORKDIR /app

RUN sudo apt-get update && \
sudo apt-get install build-essential && \
sudo apt-get clean && \
sudo apt-get autoremove && \
rm -rf /var/lib/apt/lists/* && \
pip install uv && \
uv sync

EXPOSE 12345

# ENTRYPOINT ["uv", "run", "fastapi", "run", "nilai/main.py"]
# CMD ["--host", "0.0.0.0", "--port", "12345"]

CMD ["uv", "run", "fastapi", "run", "nilai/main.py", "--host", "0.0.0.0", "--port", "12345"]