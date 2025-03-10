# nilAI

## Overview
nilAI is a platform designed to run on Confidential VMs with Trusted Execution Environments (TEEs). It ensures secure deployment and management of multiple AI models across different environments, providing a unified API interface for accessing various AI models with robust user management and model lifecycle handling.

## Prerequisites

- Docker
- Docker Compose
- Hugging Face API Token (for accessing certain models)

## Configuration

1. **Environment Setup**
   - Copy the `.env.sample` file to `.env`
   - Replace `HUGGINGFACE_API_TOKEN` with your Hugging Face API token
   - Obtain token by requesting access on the specific model's [Hugging Face page](https://huggingface.co/meta-llama/Llama-3.2-1B)

## Deployment Options

### 1. Docker Compose Deployment (Recommended)

#### Development Environment
```shell
docker compose -f docker-compose.yml \
  -f docker-compose.dev.yml \
  -f docker/compose/docker-compose.llama-3b-gpu.yml \
  -f docker/compose/docker-compose.llama-8b-gpu.yml \
  -f docker/compose/docker-compose.dolphin-8b-gpu.yml \
  -f docker/compose/docker-compose.deepseek-14b-gpu.yml \
  up --build
```

#### Production Environment
```shell
# Build vLLM docker container
docker build -t nillion/nilai-vllm:latest -f docker/vllm.Dockerfile .
# Build nilai_api container
docker build -t nillion/nilai-api:latest -f docker/api.Dockerfile --target nilai .
```
To deploy:
```shell
docker compose -f docker-compose.yml \
-f docker-compose.prod.yml \
-f docker/compose/docker-compose.llama-3b-gpu.yml \
-f docker/compose/docker-compose.llama-8b-gpu.yml \
up -d
```

**Note**: Remove lines for models you do not wish to deploy.

### 2. Manual Deployment

#### Components

- **API Frontend**: Handles user requests and routes model interactions
- **Databases**:
  - **SQLite**: User registry and access management
  - **etcd3**: Distributed key-value store for model lifecycle management

#### Setup Steps

1. **Start etcd3 Instance**
   ```shell
   docker run -d --name etcd-server \
     -p 2379:2379 -p 2380:2380 \
     -e ALLOW_NONE_AUTHENTICATION=yes \
     bitnami/etcd:latest

   docker run -d --name redis \
     -p 6379:6379 \
     redis:latest

   docker run -d --name postgres \
     -e POSTGRES_USER=user \
     -e POSTGRES_PASSWORD=<ASECUREPASSWORD> \
     -e POSTGRES_DB=yourdb \
     -p 5432:5432 \
     postgres:latest
   ```

2. **Run API Server**
   ```shell
   # Development Environment
   uv run fastapi dev nilai-api/src/nilai_api/__main__.py --port 8080

   # Production Environment
   uv run fastapi run nilai-api/src/nilai_api/__main__.py --port 8080
   ```

3. **Run Model Instances**
   ```shell
   # Example: Llama 3.2 1B Model
   # Development Environment
   uv run fastapi dev nilai-models/src/nilai_models/models/llama_1b_cpu/__init__.py

   # Production Environment
   uv run fastapi run nilai-models/src/nilai_models/models/llama_1b_cpu/__init__.py
   ```

## Developer Workflow

### Code Quality and Formatting

Install pre-commit hooks to automatically format code and run checks:

```shell
uv run pre-commit install
```

## Model Lifecycle Management

- Models register themselves in the etcd3 database
- Registration includes address information with an auto-expiring lifetime
- If a model disconnects, it is automatically removed from the available models

## Security

- Hugging Face API token controls model access
- SQLite database manages user permissions
- Distributed architecture allows for flexible security configurations

## Troubleshooting

- Ensure Hugging Face API token is valid
- Check etcd3 and Docker container logs for connection issues
- Verify network ports are not blocked or in use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Install pre-commit hooks
4. Make your changes
5. Submit a pull request

## License

[Add your project's license information here]
