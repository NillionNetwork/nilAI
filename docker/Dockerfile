FROM python:3.12-slim

COPY . /app

WORKDIR /app

RUN pip install poetry
RUN poetry install 

EXPOSE 12345

ENTRYPOINT ["poetry", "run", "fastapi", "run", "nilai/server.py"]
CMD ["--host", "0.0.0.0", "--port", "12345"]