
```shell
docker build \
  --build-arg MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct \
  --build-arg MODEL_FILENAME=llama-3.2-1b-instruct.Q4_0.gguf \
  --secret id=HUGGINGFACE_API_TOKEN \
  -t nillion/nilai:latest \
  -f docker/Dockerfile .


docker run \
  -p 8080:8080 \
  -p 8443:8443 \
  -v hugging_face_models:/root/.cache/huggingface \
  -v $(pwd)/users.sqlite:/app/users.sqlite \
  nillion/nilai:latest
```