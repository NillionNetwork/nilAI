
## Building & Running nilAI API

```shell
docker build -t nillion/nilai-api:latest -f docker/api.Dockerfile .


 docker run -it --rm \
 -p 8080:8080 \
 -v hugging_face_models:/root/.cache/huggingface \
 -v $(pwd)/db/users.sqlite:/app/db/users.sqlite \
 nillion/nilai-api:latest
```

## Building & Running nilAI Model

```shell
docker build -t nillion/nilai-models:latest -f docker/model.Dockerfile --build-arg MODEL_PATH=llama_1b_cpu .


 docker run -it --rm \
 -p 8000:8000 \
 -v hugging_face_models:/root/.cache/huggingface \
 nillion/nilai-models:latest
```

## Running etcd

```sh
docker run -d --name etcd-server \
    --publish 2379:2379 \
    --publish 2380:2380 \
    --env ALLOW_NONE_AUTHENTICATION=yes \
    --env ETCD_ADVERTISE_CLIENT_URLS=http://etcd-server:2379 \
    bitnami/etcd:latest
```

## Announcing LMStudio Models

LMStudio can run on the host at `localhost:1234` while the stack runs inside Docker. Build the announcer image and bring it up alongside the core services:

```sh
docker build -t nilai/lmstudio-announcer:latest -f docker/lmstudio-announcer.Dockerfile .
docker compose -f docker-compose.yml \
  -f docker-compose.dev.yml \
  -f docker/compose/docker-compose.lmstudio.yml \
  up -d lmstudio_announcer
```

The announcer registers every model returned by `http://host.docker.internal:1234/v1/models` in etcd so that `nilai-api` can route chat requests to LMStudio. Override defaults with environment variables such as `LMSTUDIO_MODEL_IDS`, `LMSTUDIO_TOOL_SUPPORT_MODELS`, or `LMSTUDIO_MULTIMODAL_MODELS` inside the compose override.
