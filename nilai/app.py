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
        {
            "name": "Usage",
            "description": "User token usage",
        },
    ],
)


app.include_router(public.router)
app.include_router(private.router, dependencies=[Depends(get_user)])
