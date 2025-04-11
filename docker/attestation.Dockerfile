FROM golang:1.23.3-bullseye AS sev

COPY --link nilai-attestation/src/nilai_attestation/attestation/sev /app/sev
WORKDIR /app/sev

RUN go build -o libsevguest.so -buildmode=c-shared main.go

FROM python:3.12-slim AS nilai

COPY --link . /app/
COPY --from=sev /app/sev/libsevguest.so /app/nilai-attestation/src/nilai_attestation/sev/libsevguest.so
COPY --from=sev /app/sev/libsevguest.h /app/nilai-attestation/src/nilai_attestation/sev/libsevguest.h

WORKDIR /app/nilai-attestation/

RUN apt-get update && \
apt-get install build-essential curl -y && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* && \
pip install uv && \
uv sync

EXPOSE 8080

CMD ["uv", "run", "gunicorn", "-c", "gunicorn.conf.py", "nilai_attestation.app:app"]
