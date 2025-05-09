FROM python:3.12-slim AS nilai

COPY --link . /app/

WORKDIR /app/nilai-api/

RUN apt-get update && \
apt-get install build-essential curl git pkg-config automake file -y && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* && \
pip install uv && \
uv sync

EXPOSE 8080

CMD ["./launch.sh"]
