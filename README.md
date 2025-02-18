# nilAI

Copy the `.env.sample` to `.env` and replace the value of the `HUGGINGFACE_API_TOKEN` with the appropriate value. The `HUGGINGFACE_API_TOKEN` is used to determine whether a user has permission to access certain models. For example, for Llama models, you usually need to have requested access to the model on its [Hugging Face page](https://huggingface.co/meta-llama/Llama-3.2-1B).

There are two ways to deploy **nilAI**. The recommended way is to use docker-compose as it is the easiest and most straightforward.

## Docker

For development environments with
```shell
docker compose -f docker-compose.yml \
-f docker-compose.dev.yml \
-f docker/compose/docker-compose.llama-3b-gpu.yml \
-f docker/compose/docker-compose.llama-8b-gpu.yml \
up --build
```

For production environments:
```shell
docker compose -f docker-compose.yml \
-f docker-compose.prod.yml \
-f docker/compose/docker-compose.llama-3b-gpu.yml \
-f docker/compose/docker-compose.llama-8b-gpu.yml \
up --build
```

## Manual Deployment

**nilAI** consists of the following components:
 - **API Frontend**: Receives user requests and handles them appropriately. For model requests, it forwards them to the appropriate backend model.
 - **Two Databases**:
    - **SQLite**: The main registry of users on the platform. This will be changed as we move to more production-ready environments. It tracks which users are allowed on the platform, their API keys, and their usage.
    - **etcd3**: A distributed key-value database used in Kubernetes. It creates key-value pairs with a lifetime. When a key-value pair's lifetime expires, it is automatically removed. Models register their address information on the etcd3 database with a lifetime and keep this lifetime alive. If a model ever disconnects due to an error, the database removes the entry, and the API Frontend no longer advertises that model.
 - **Models**: There may be zero or more model deployments. Model deployments contain a basic API that responds in the same format to the `/v1/chat/completions` endpoint. The `Model` class defines how models connect to the database and manage their lifecycle.

To deploy the components, first create the `etcd3` instance. The easiest way is to expose it with Docker:

```shell
# This command runs in the background. If it fails, you may already be running etcd-server on ports 2379 and 2380.
docker run -d --name etcd-server -p 2379:2379 -p 2380:2380 -e ALLOW_NONE_AUTHENTICATION=yes bitnami/etcd:latest
```

Run the **nilAI** API server:
```shell
# Shell 1
## For development environment (auto reloads on file changes):
uv run fastapi dev nilai-api/src/nilai_api/__main__.py --port 8080
## For production environment:
uv run fastapi run nilai-api/src/nilai_api/__main__.py --port 8080
```

Run the **nilAI** Llama 3.2 1B model. For different models, adapt the command below:
```shell
# Shell 2
## For development environment (auto reloads on file changes):
uv run fastapi dev nilai-models/src/nilai_models/models/llama_1b_cpu/__init__.py
## For production environment:
uv run fastapi run nilai-models/src/nilai_models/models/llama_1b_cpu/__init__.py
```

## Developer Instructions

If you are developping, you can use `pre-commit` configurations to ensure make the development smoother and not having to wait for CI checks. These are executed before you commit, and perform automatic changes to format your code.

You can install those with:
```shell
uv run pre-commit install
```
