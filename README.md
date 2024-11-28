# NilAI

Copy the `.env.sample` to `.env` to and replace the value of the `HUGGINGFACE_API_TOKEN` with the appropriate value. It is required to download Llama3.2 1B.

For development environments:
```shell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build nilai
```

For production environments:
```shell
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d 
```

```
uv run gunicorn -c gunicorn.conf.py nilai.__main__:app
```