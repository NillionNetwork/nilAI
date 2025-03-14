from collections import defaultdict
from ecdsa import SigningKey, SECP256k1
import jwt
import logging
import nilql
import os
import requests
import traceback
from jsonschema import validators, Draft7Validator
from jsonschema.exceptions import ValidationError
import uuid
import time

logger = logging.getLogger(__name__)


class SecretVaultHelper:
    def __init__(self, org_did: str, secret_key: str, schema_uuid: str):
        """Initialize config with JWTs signed with ES256K for multiple node_ids; Add cluster key."""
        self.org_did = org_did
        response = requests.post(
            "https://secret-vault-registration.replit.app/api/config",
            headers={
                "Content-Type": "application/json",
            },
            json={"org_did": org_did},
        )
        response.raise_for_status()
        self.nodes = response.json()["nodes"]

        # Convert the secret key from hex to bytes
        private_key = bytes.fromhex(secret_key)
        signer = SigningKey.from_string(private_key, curve=SECP256k1)

        for node in self.nodes:
            # Create payload for each node_id
            payload = {
                "iss": org_did,
                "aud": node["did"],
                "exp": int(time.time()) + 3600,
            }

            # Create and sign the JWT
            node["bearer"] = jwt.encode(payload, signer.to_pem(), algorithm="ES256K")

        self.key = nilql.ClusterKey.generate(
            {"nodes": [{}] * len(self.nodes)}, {"store": True}
        )

        self.schema_list = self.fetch_schemas()
        self.schema_definition = self.find_schema(schema_uuid)
        self.schema_uuid = schema_uuid
        logger.info(
            f"fn:data_upload init complete: {len(self.nodes)} nodes | schema {schema_uuid}"
        )

    def fetch_schemas(self) -> list:
        """Get all my schemas from the first server."""
        headers = {
            "Authorization": f"Bearer {self.nodes[0]['bearer']}",
            "Content-Type": "application/json",
        }

        response = requests.get(
            f"{self.nodes[0]['url']}/api/v1/schemas", headers=headers
        )
        response.raise_for_status()

        assert (
            response.status_code == 200 and response.json().get("errors", []) == []
        ), response.content.decode("utf8")

        schema_list = response.json()["data"]
        assert len(schema_list) > 0, "failed to fetch schemas from nildb"
        return schema_list

    def find_schema(self, schema_uuid: str) -> dict:
        """Filter a list of schemas by single desired schema id."""
        my_schema = None
        for this_schema in self.schema_list:
            if this_schema["_id"] == schema_uuid:
                my_schema = this_schema["schema"]
                break
        assert my_schema is not None, "failed to lookup schema"
        return my_schema

    def _mutate_secret_attributes(self, entry: dict) -> None:
        """Apply encryption or secret sharing to all fields in schema that are indicated w/ $share keyname."""
        keys = list(entry.keys())
        for key in keys:
            value = entry[key]
            if key == "_id":
                entry[key] = str(uuid.uuid4())
            elif key == "$share":
                del entry["$share"]
                entry["$allot"] = nilql.encrypt(self.key, value)
            elif isinstance(value, dict):
                self._mutate_secret_attributes(value)

    def _validator_builder(self):
        """Build a validator to validate the candidate document against loaded schema."""
        return validators.extend(Draft7Validator)

    def data_reveal(self, filter: dict = {}) -> list[dict]:
        """Get filtered data from schema on all nodes then reconstruct secret.

        Args:
            dict: Optional filter paramters

        Returns:
            list[dict]: A list of the downloaded records

        """
        try:
            logger.info(f"fn:data_reveal | {self.schema_uuid} | filter: {filter} ")

            shares = defaultdict(list)
            for node in self.nodes:
                headers = {
                    "Authorization": f"Bearer {node['bearer']}",
                    "Content-Type": "application/json",
                }

                body = {"schema": self.schema_uuid, "filter": filter}

                response = requests.post(
                    f"{node['url']}/api/v1/data/read",
                    headers=headers,
                    json=body,
                )
                assert response.status_code == 200, (
                    "upload failed: " + response.content.decode("utf8")
                )
                data = response.json().get("data")
                for d in data:
                    shares[d["_id"]].append(d)
            decrypted = []
            for k in shares:
                decrypted.append(nilql.unify(self.key, shares[k]))
            return decrypted
        except Exception as e:
            logger.info(f"Error retrieving records in node: {e!r}")
            return []

    def post(self, data_to_store: list) -> list:
        """Create/upload records in the specified node and schema."""
        logger.info(
            f"fn:data_upload {self.schema_uuid} | {type(data_to_store)} | {data_to_store}"
        )
        try:
            builder = self._validator_builder()
            validator = builder(self.schema_definition)

            logger.info(f"fn:data_upload <analysis> | got {len(data_to_store)} records")
            for entry in (
                [data_to_store] if isinstance(data_to_store, dict) else data_to_store
            ):
                self._mutate_secret_attributes(entry)

            record_uuids = [x["_id"] for x in data_to_store]
            payloads = nilql.allot(data_to_store)
            logger.info(f"fn:data_upload <mutated> {payloads}")

            for idx, shard in enumerate(payloads):
                validator.validate(shard)

                node = self.nodes[idx]
                headers = {
                    "Authorization": f"Bearer {node['bearer']}",
                    "Content-Type": "application/json",
                }

                body = {"schema": self.schema_uuid, "data": shard}
                logger.debug(f"fn:data_upload POST{idx} | {body}")

                response = requests.post(
                    f"{node['url']}/api/v1/data/create",
                    headers=headers,
                    json=body,
                )

                assert (
                    response.status_code == 200
                    and response.json().get("errors", []) == []
                ), f"upload (host-{idx}) failed: " + response.content.decode("utf8")
            logger.info(f"fn:data_upload COMPLETED: {record_uuids}")
            return record_uuids
        except Exception as e:
            msg = "".join(traceback.format_tb(e.__traceback__, limit=3))
            logger.info(f"Error creating records in node: {msg}")
            raise ValidationError(f"{e!r}")


if __name__ == "__main__":
    vault = SecretVaultHelper(
        org_did=os.environ["NILLION_ORG_ID"],
        secret_key=os.environ["NILLION_SECRET_KEY"],
        schema_uuid="87cf9ea0-c26c-4776-bf81-3553b8aa3c30",
    )
    vault.post(
        [
            {
                "_id": str(uuid.uuid4()),
                "patient_name": {"$share": "Nick Test"},
                "doctor_name": {"$share": "Dr. Niko Nik"},
                "medical_diagnosis": {"$share": "Dementia or MS"},
                "reasoning_summary": {"$share": "This is just a test."},
            }
        ],
    )
