# Fast API and serving

from fastapi import FastAPI
from nilai_attestation.routers import private, public

# Fast API and serving


import logging

logging.getLogger("nv_attestation_sdk").setLevel(logging.WARNING)
logging.getLogger("sdk-logger").setLevel(logging.WARNING)
logging.getLogger("sdk-console").setLevel(logging.WARNING)
logging.getLogger("sdk-file").setLevel(logging.WARNING)
logging.getLogger("gpu-verifier-event").setLevel(logging.WARNING)
logging.getLogger("gpu-verifier-info").setLevel(logging.WARNING)


description = """
An AI model serving platform powered by secure, confidential computing.

## Easy API Client Generation

Want to use our API in your project? Great news! You can automatically generate a client library in just a few simple steps using the OpenAPI specification.
```
After generating, you'll have a fully functional client library that makes it easy to interact with our AI services. No more manual API request handling!
"""
app = FastAPI(
    title="NilAI attestation",
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
        }
    ],
)


app.include_router(private.router)
app.include_router(public.router)
