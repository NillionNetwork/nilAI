# NilAI

Copy the `.env.sample` to `.env` to and replace the value of the `HUGGINGFACE_API_TOKEN` with the appropriate value. It is required to download Llama3.2 1B.

```shell
docker compose up --build web
```

```
uv run gunicorn -c gunicorn.conf.py nilai.__main__:app
```