FROM python:3.12-slim

COPY --link nilai /app/nilai
COPY pyproject.toml uv.lock .env /app/

WORKDIR /app

RUN pip install uv
RUN uv sync

EXPOSE 12345

# ENTRYPOINT ["uv", "run", "fastapi", "run", "nilai/main.py"]
# CMD ["--host", "0.0.0.0", "--port", "12345"]

CMD ["uv", "run", "fastapi", "run", "nilai/main.py", "--host", "0.0.0.0", "--port", "12345"]