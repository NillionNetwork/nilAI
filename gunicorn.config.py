# gunicorn.config.py

# Bind to address and port
bind = "0.0.0.0:12345"

# Set the number of workers (2)
workers = 2

# Set the number of threads per worker (16)
threads = 16

# Set the timeout (120 seconds)
timeout = 120

# Set the worker class to UvicornWorker for async handling
worker_class = "uvicorn.workers.UvicornWorker"

# SSL settings
# certfile = "/path/to/your/certificate.crt"  # Path to your SSL certificate
# keyfile = "/path/to/your/private.key"       # Path to your SSL private key
