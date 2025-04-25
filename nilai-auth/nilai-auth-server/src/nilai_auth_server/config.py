import os

from dotenv import load_dotenv

load_dotenv()

NILAUTH_TRUSTED_ROOT_ISSUERS = os.getenv("NILAUTH_TRUSTED_ROOT_ISSUERS", "").split(",")

print("With trusted root issuers: ", NILAUTH_TRUSTED_ROOT_ISSUERS)
