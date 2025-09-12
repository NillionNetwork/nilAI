# Fast API and serving


from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import Depends, FastAPI
from nilai_api.auth import get_auth_info
from nilai_api.rate_limiting import setup_redis_conn
from nilai_api.routers import private, public
from nilai_api import config
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from nilai_common.config import SETTINGS


@asynccontextmanager
async def lifespan(app: FastAPI):
    client, rate_limit_command = await setup_redis_conn(config.CONFIG.redis.url)

    yield {"redis": client, "redis_rate_limit_command": rate_limit_command}


host = SETTINGS.host
description = f"""
An AI model serving platform powered by secure, confidential computing.

## Easy API Client Generation

Want to use our API in your project? Great news! You can automatically generate a client library in just a few simple steps.

### For Python Developers
```bash
# Install the OpenAPI generator
pip install openapi-generator-cli

# Generate your Python client
openapi-generator-cli generate -i https://{host}/openapi.json -g python -o ./python-client
```

### For JavaScript/TypeScript Developers
```bash
# Install the OpenAPI generator
npm install @openapitools/openapi-generator-cli -g

# Generate your TypeScript client
openapi-generator-cli generate -i https://{host}/openapi.json -o ./typescript-client
```

After generating, you'll have a fully functional client library that makes it easy to interact with our AI services. No more manual API request handling!
"""
app = FastAPI(
    title="NilAI",
    description=description,
    version="0.1.0",
    terms_of_service="https://nillion.com",
    contact={
        "name": "Nillion AI Support",
        "email": "jose.cabrero@nillion.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
    openapi_tags=[
        {
            "name": "Attestation",
            "description": "Retrieve cryptographic attestation information for service verification",
        },
        {
            "name": "Chat",
            "description": "AI-powered chat completion endpoint for generating conversational responses",
        },
        {
            "name": "Health",
            "description": "System health and status monitoring endpoint",
        },
        {
            "name": "Model",
            "description": "Retrieve information about available AI models",
        },
        {
            "name": "Usage",
            "description": "Track and retrieve user token consumption metrics",
        },
    ],
    lifespan=lifespan,
)


app.include_router(public.router)
app.include_router(private.router, dependencies=[Depends(get_auth_info)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app, include_in_schema=False)
