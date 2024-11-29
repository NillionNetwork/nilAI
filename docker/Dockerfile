FROM python:3.12-slim

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