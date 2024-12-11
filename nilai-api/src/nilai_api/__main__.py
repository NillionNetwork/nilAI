import uvicorn

from nilai_api.app import app


def run_uvicorn():
    """
    Function to run the app with Uvicorn for debugging.
    """
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8080,  # Use the desired port
        reload=True,  # Enable auto-reload for development
        # ssl_certfile=SSL_CERTFILE,
        # ssl_keyfile=SSL_KEYFILE,
    )


if __name__ == "__main__":
    run_uvicorn()
