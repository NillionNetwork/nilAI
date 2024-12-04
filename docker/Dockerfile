FROM golang:1.23.3-bullseye AS sev

COPY --link nilai/sev /app/sev
WORKDIR /app/sev

RUN go build -o libsevguest.so -buildmode=c-shared main.go

FROM python:3.12-slim AS nilai

COPY --link nilai /app/nilai
COPY pyproject.toml uv.lock .env gunicorn.conf.py /app/
COPY --from=sev /app/sev/libsevguest.so /app/nilai/sev/libsevguest.so

WORKDIR /app

RUN apt-get update && \
apt-get install build-essential -y && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* && \
pip install uv && \
uv sync

EXPOSE 8080 8443

CMD ["uv", "run", "gunicorn", "-c", "gunicorn.conf.py", "nilai.__main__:app"]