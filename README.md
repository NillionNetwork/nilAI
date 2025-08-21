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
   ```shell
   cp .env.sample .env
   ```
   - Update the environment variables in `.env`:
     - `HUGGINGFACE_API_TOKEN`: Your Hugging Face API token
   - Obtain token by requesting access on the specific model's Hugging Face page. For example, to request access for the Llama 1B model, you can ask [here](https://huggingface.co/meta-llama/Llama-3.2-1B). Note that for the Llama-8B model, you need to make a separate request.

## Deployment Options

### 1. Docker Compose Deployment (Recommended)

#### Development Environment
```shell
# Build nilai_attestation endpoint
docker build -t nillion/nilai-attestation:latest -f docker/attestation.Dockerfile .
# Build vLLM docker container
docker build -t nillion/nilai-vllm:latest -f docker/vllm.Dockerfile .
# Build nilai_api container
docker build -t nillion/nilai-api:latest -f docker/api.Dockerfile --target nilai .
```
Then, to deploy:

```shell

# Deploy with CPU-only configuration
docker compose -f docker-compose.yml \
  -f docker-compose.dev.yml \
  -f docker/compose/docker-compose.llama-1b-gpu.yml \
  up -d

# Monitor logs
docker compose -f docker-compose.yml \
  -f docker-compose.dev.yml \
  -f docker/compose/docker-compose.llama-1b-gpu.yml \
  logs -f
```

#### Production Environment
```shell
# Build nilai_attestation endpoint
docker build -t nillion/nilai-attestation:latest -f docker/attestation.Dockerfile .
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

#### Testing Without GPU

```shell
# Build nilai_attestation endpoint
docker build -t nillion/nilai-attestation:latest -f docker/attestation.Dockerfile .
# Build vLLM docker container
docker build -t nillion/nilai-vllm:latest -f docker/vllm.Dockerfile .
# Build nilai_api container
docker build -t nillion/nilai-api:latest -f docker/api.Dockerfile --target nilai --platform linux/amd64 .
```
To deploy:
```shell

python3 ./scripts/docker-composer.py --dev -f docker/compose/docker-compose.llama-1b-cpu.yml -o development-compose.yml

docker compose -f development-compose.yml up -d
```

### 2. Using the Docker Compose Helper Script

For easier management of multiple compose files and image substitutions, use the `docker-composer.py` script:

#### Basic Usage

```shell
# Generate a composed configuration for development
python3 ./scripts/docker-composer.py --dev -o dev-compose.yml

# Generate a composed configuration for production
python3 ./scripts/docker-composer.py --prod -o prod-compose.yml

# Include specific model configurations
python3 ./scripts/docker-composer.py --prod \
  -f docker-compose.llama-3b-gpu.yml \
  -f docker-compose.llama-8b-gpu.yml \
  -o production-compose.yml
```

#### Image Substitution

Replace default images with custom ones (useful for production deployments with specific image versions):

```shell
# Production example with custom ECR images
python3 ./scripts/docker-composer.py --prod \
  -f docker-compose.llama-3b-gpu.yml \
  --image 'nillion/nilai-api:latest=public.ecr.aws/k5d9x2g2/nilai-api:v0.1.0-rc1' \
  --image 'nillion/nilai-vllm:latest=public.ecr.aws/k5d9x2g2/nilai-vllm:v0.1.0-rc1' \
  --image 'nillion/nilai-attestation:latest=public.ecr.aws/k5d9x2g2/nilai-attestation:v0.1.0-rc1' \
  -o production-compose.yml

# Then deploy with the generated file
docker compose -f production-compose.yml up -d
```

#### Script Options

- `--dev`: Include development-specific configurations
- `--prod`: Include production-specific configurations
- `-f, --file <filename>`: Include additional compose files from `docker/compose/` directory
- `-o, --output <filename>`: Specify output filename (default: `output.yml`)
- `--image <old=new>`: Substitute Docker images (can be used multiple times)
- `-h, --help`: Show help message

#### Production Deployment Example

For a complete production setup with custom images:

```shell
# 1. Generate the production compose file
python3 ./scripts/docker-composer.py --prod \
  -f docker/compose/docker-compose.llama-3b-gpu.yml \
  -f docker/compose/docker-compose.llama-8b-gpu.yml \
  -f docker/compose/docker-compose.deepseek-14b-gpu.yml \
  --image 'nillion/nilai-api:latest=public.ecr.aws/k5d9x2g2/nilai-api:v0.2.0-alpha-0' \
  --image 'nillion/nilai-vllm:latest=public.ecr.aws/k5d9x2g2/nilai-vllm:v0.2.0-alpha-0' \
  --image 'nillion/nilai-attestation:latest=public.ecr.aws/k5d9x2g2/nilai-attestation:v0.2.0-alpha-0' \
  --testnet \
  -o production-compose.yml

# Or:
python3 ./scripts/docker-composer.py --prod \
  -f docker/compose/docker-compose.llama-70b-gpu.yml \
  --image 'nillion/nilai-api:latest=jcabrero/nillion-nilai-api:latest' \
  --image 'nillion/nilai-vllm:latest=public.ecr.aws/k5d9x2g2/nilai-vllm:v0.1.0-rc1' \
  --image 'nillion/nilai-attestation:latest=public.ecr.aws/k5d9x2g2/nilai-attestation:v0.1.0-rc1' \
  -o production-compose.yml


# 2. Deploy using the generated file
docker compose -f production-compose.yml up -d

# 3. Monitor logs
docker compose -f production-compose.yml logs -f
```

### 3. Manual Component Deployment

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
    ```

2. **Start PostgreSQL**
    ```shell
    docker run -d --name postgres \
      -e POSTGRES_USER=${POSTGRES_USER} \
      -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
      -e POSTGRES_DB=${POSTGRES_DB} \
      -p 5432:5432 \
      --network frontend_net \
      --volume postgres_data:/var/lib/postgresql/data \
      postgres:16
    ```

2. **Run API Server**
   ```shell
   # Development Environment
    fastapi dev nilai-api/src/nilai_api/__main__.py --port 8080

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

- Models register themselves in the etcd database
- Registration includes address information with an auto-expiring lifetime
- If a model disconnects, it is automatically removed from the available models

## Security

- Hugging Face API token controls model access
- PostgreSQL database manages user permissions
- Distributed architecture allows for flexible security configurations

## Troubleshooting

Common issues and solutions:

1. **Container Logs**
   ```shell
   # View logs for all services
   docker compose logs -f

   # View logs for specific service
   docker compose logs -f api
   ```

2. **Database Connection**
   ```shell
   # Check PostgreSQL connection
   docker exec -it postgres psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}
   ```

3. **Service Health**
   ```shell
   # Check service health status
   docker compose ps
   ```

### vLLM for Local Execution on macOS
To configure vLLM for **local execution on macOS**, execute the following steps:
```shell
# Clone vLLM repository (root folder)
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.7.3 # We use v0.7.3
# Build vLLM OpenAI (vllm folder)
docker build -f Dockerfile.arm -t vllm/vllm-openai . --shm-size=4g

# Build nilai attestation container (root folder)
docker build -t nillion/nilai-attestation:latest -f docker/attestation.Dockerfile .
# Build vLLM docker container (root folder)
docker build -t nillion/nilai-vllm:latest -f docker/vllm.Dockerfile .
# Build nilai_api container
docker build -t nillion/nilai-api:latest -f docker/api.Dockerfile --target nilai --platform linux/amd64 .
````

## Contributing

1. Fork the repository
2. Create a feature branch
3. Install pre-commit hooks
4. Make your changes
5. Submit a pull request

## License

[Add your project's license information here]
