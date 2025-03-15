FROM python:3.12-slim AS gpuverifier

COPY --link ./gpuverifier-api /app/

WORKDIR /app
RUN pip install --upgrade .

EXPOSE 8000

CMD ["/app/launch.sh"]
