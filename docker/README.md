
```shell
docker build -t nillion/nilai:latest -f docker/Dockerfile .


docker run \
  -p 8080:8080 \
  -p 8443:8443 \
  -v hugging_face_models:/root/.cache/huggingface \
  -v $(pwd)/users.sqlite:/app/users.sqlite \
  nillion/nilai:latest
```