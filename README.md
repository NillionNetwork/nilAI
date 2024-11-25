# NilAI


```shell
docker build -t nillion/nilai:latest -f docker/Dockerfile .
```

```shell
docker run -it -p 12345:12345 -v hugging_face_models:/root/.cache/huggingface nillion/nilai:latest
```