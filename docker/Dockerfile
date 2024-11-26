FROM python:3.12-slim

COPY . /app

WORKDIR /app

RUN pip install uv
RUN uv sync

EXPOSE 12345

ENTRYPOINT ["uv", "run", "fastapi", "run", "nilai/server.py"]
CMD ["--host", "0.0.0.0", "--port", "12345"]