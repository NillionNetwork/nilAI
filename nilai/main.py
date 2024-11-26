# Fast API and serving
from fastapi import Depends, FastAPI

from nilai.auth import get_user
from nilai.routers import private, public

app = FastAPI(
    title="NilAI",
    description="An AI model serving platform based on TEE",
    version="0.1.0",
    terms_of_service="https://nillion.com",
    contact={
        "name": "Nillion AI Support",
        # "url": "https://nillion.com",
        "email": "jose.cabrero@nillion.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
    openapi_tags=[
        {
            "name": "Attestation",
            "description": "Retrieve attestation information",
        },
        {
            "name": "Chat",
            "description": "Chat completion endpoint",
        },
        {
            "name": "Health",
            "description": "Health check endpoint",
        },
        {
            "name": "Model",
            "description": "Model information",
        },
    ],
)


app.include_router(public.router)
app.include_router(private.router, dependencies=[Depends(get_user)])

if __name__ == "__main__":
    import uvicorn

    # Path to your SSL certificate and key files
    # SSL_CERTFILE = "/path/to/certificate.pem"  # Replace with your certificate file path
    # SSL_KEYFILE = "/path/to/private-key.pem"  # Replace with your private key file path

    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=12345,  # Use port 8443 for HTTPS
        # ssl_certfile=SSL_CERTFILE,
        # ssl_keyfile=SSL_KEYFILE,
    )
