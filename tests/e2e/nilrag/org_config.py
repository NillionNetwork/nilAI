"""The SecretVault organization configuration"""

import json

config = None


def load_config():
    global config
    if config is None:
        with open("config.json", "r", encoding="utf-8") as f:
            json_config = json.load(f)
        config = {}
        config["nodes"] = []
        for node in json_config["nodes"]:
            config["nodes"].append(
                {
                    "url": node["url"],
                    "did": node["node_id"],
                }
            )
        config["org_credentials"] = {
            "secret_key": json_config["org_secret_key"],
            "org_did": json_config["org_did"],
        }
    print(config)
    return config
