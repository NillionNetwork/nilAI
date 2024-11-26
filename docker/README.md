
```shell
docker build -t nillion/nilai:latest -f docker/Dockerfile .


docker run \
  -p 12345:12345 \
  -v hugging_face_models:/root/.cache/huggingface \
  -v $(pwd)/users.sqlite:/app/users.sqlite \
  nillion/nilai:latest
```