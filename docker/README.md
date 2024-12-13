
## Building & Running nilAI API

```shell
docker build -t nillion/nilai-api:latest -f docker/api.Dockerfile .


 docker run -it --rm \
 -p 8080:8080 \
 -p 8443:8443 \
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

```
docker run -d --name etcd-server \
    --publish 2379:2379 \
    --publish 2380:2380 \
    --env ALLOW_NONE_AUTHENTICATION=yes \
    --env ETCD_ADVERTISE_CLIENT_URLS=http://etcd-server:2379 \
    bitnami/etcd:latest
```