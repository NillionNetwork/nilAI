import os

# from dotenv import load_dotenv

# load_dotenv()  # Only needed locally if using a .env file

SETTINGS = {
    "host": os.getenv("SVC_HOST", "localhost"),
    "port": os.getenv("SVC_PORT", 8000),
    "etcd_host": os.getenv("ETCD_HOST", "localhost"),
    "etcd_port": os.getenv("ETCD_PORT", 2379),
    "tool_support": os.getenv("TOOL_SUPPORT", False),
    "gunicorn_workers": os.getenv("NILAI_GUNICORN_WORKERS", 10),
}
# if environment == "docker":
#     config = "docker_settings.py"
# else:
#     config = "local_settings.py"

# # Import the appropriate config dynamically
# from importlib import import_module
# settings = import_module(config)
