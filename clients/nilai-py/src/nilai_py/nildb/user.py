import datetime
import json
import os
import glob
from typing import Optional, Tuple, List

from secretvaults.common.blindfold import BlindfoldFactoryConfig, BlindfoldOperation
from secretvaults.common.keypair import Keypair
from secretvaults import SecretVaultUserClient

from nilai_py.nildb.models import UserSetupResult, KeyData, KeypairInfo
from nilai_py.nildb.config import NilDBConfig


def save_keypair_to_json(
    keypair: Keypair, keys_dir: str = "keys"
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Save keypair to separate JSON files for private and public keys"""
    try:
        # Create keys directory if it doesn't exist
        os.makedirs(keys_dir, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # File paths
        private_key_file = os.path.join(keys_dir, f"private_key_{timestamp}.json")
        public_key_file = os.path.join(keys_dir, f"public_key_{timestamp}.json")

        # Private key data using Pydantic model
        private_key_data = KeyData(
            type="private_key",
            key=keypair.private_key_hex(),
            public_key_file=public_key_file,
        )

        # Public key data using Pydantic model
        public_key_data = KeyData(
            type="public_key",
            key=keypair.public_key_hex(),
            did=keypair.to_did_string(),
            private_key_file=private_key_file,
        )

        # Save private key
        with open(private_key_file, "w") as f:
            json.dump(
                private_key_data.model_dump(mode="json"), f, indent=2, default=str
            )

        # Save public key
        with open(public_key_file, "w") as f:
            json.dump(public_key_data.model_dump(mode="json"), f, indent=2, default=str)

        return True, private_key_file, public_key_file

    except Exception as e:
        return False, None, str(e)


def load_keypair_from_json(
    private_key_file: str,
) -> Tuple[bool, Optional[Keypair], Optional[str]]:
    """Load keypair from private key JSON file"""
    try:
        if not os.path.exists(private_key_file):
            return False, None, f"Private key file not found: {private_key_file}"

        with open(private_key_file, "r") as f:
            data = json.load(f)

        # Parse using Pydantic model
        private_key_data = KeyData(**data)

        if private_key_data.type != "private_key":
            return False, None, "Invalid private key file format"

        # Recreate keypair from private key hex
        if not private_key_data.key:
            return False, None, "No private key found in file"

        private_key_hex = private_key_data.key

        keypair = Keypair.from_hex(private_key_hex)
        return True, keypair, None

    except Exception as e:
        return False, None, str(e)


async def setup_user_core(
    config: NilDBConfig, keys_dir: str = "keys"
) -> UserSetupResult:
    """Setup user keypair and client - core functionality without UI concerns"""
    try:
        # Generate a new user keypair
        user_keypair = Keypair.generate()

        # Save keypair to JSON files
        save_success, private_file, public_file = save_keypair_to_json(
            user_keypair, keys_dir
        )
        if not save_success:
            return UserSetupResult(
                success=False, error=f"Failed to save keypair: {public_file}"
            )

        # Create user client
        user_client = await SecretVaultUserClient.from_options(
            keypair=user_keypair,
            base_urls=config.nodes,
            blindfold=BlindfoldFactoryConfig(
                operation=BlindfoldOperation.STORE, use_cluster_key=True
            ),
        )

        if private_file is None or public_file is None:
            return UserSetupResult(
                success=False,
                error=f"Failed to save keypair: {private_file} or {public_file}",
            )

        return UserSetupResult(
            success=True,
            user_client=user_client,
            keypair=user_keypair,
            keys_saved_to={"private_key": private_file, "public_key": public_file},
        )

    except Exception as e:
        return UserSetupResult(success=False, error=e)


def store_keypair(
    keypair: Keypair, keys_dir: str = "keys", name_prefix: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Store keypair to files with optional custom prefix"""
    if name_prefix:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        private_key_file = os.path.join(
            keys_dir, f"{name_prefix}_private_key_{timestamp}.json"
        )
        public_key_file = os.path.join(
            keys_dir, f"{name_prefix}_public_key_{timestamp}.json"
        )

        try:
            os.makedirs(keys_dir, exist_ok=True)

            # Private key data using Pydantic model
            private_key_data = KeyData(
                type="private_key",
                name=name_prefix,
                key=keypair.private_key_hex(),
                public_key_file=public_key_file,
            )

            # Public key data using Pydantic model
            public_key_data = KeyData(
                type="public_key",
                name=name_prefix,
                key=keypair.public_key_hex(),
                did=keypair.to_did_string(),
                private_key_file=private_key_file,
            )

            # Save private key
            with open(private_key_file, "w") as f:
                json.dump(
                    private_key_data.model_dump(mode="json"), f, indent=2, default=str
                )

            # Save public key
            with open(public_key_file, "w") as f:
                json.dump(
                    public_key_data.model_dump(mode="json"), f, indent=2, default=str
                )

            return True, private_key_file, public_key_file

        except Exception as e:
            return False, None, str(e)
    else:
        # Use existing function for default behavior
        return save_keypair_to_json(keypair, keys_dir)


def load_keypair(
    private_key_file: str,
) -> Tuple[bool, Optional[Keypair], Optional[str]]:
    """Load keypair from private key file (alias for load_keypair_from_json)"""
    return load_keypair_from_json(private_key_file)


def load_keypair_by_name(
    name_prefix: str, keys_dir: str = "keys"
) -> Tuple[bool, Optional[Keypair], Optional[str]]:
    """Load keypair by searching for files with given name prefix"""
    try:
        # Search for private key files with the given prefix
        pattern = os.path.join(keys_dir, f"{name_prefix}_private_key_*.json")
        matching_files = glob.glob(pattern)

        if not matching_files:
            return (
                False,
                None,
                f"No private key files found with prefix '{name_prefix}' in {keys_dir}",
            )

        # Sort by modification time, get the most recent
        matching_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = matching_files[0]

        return load_keypair_from_json(latest_file)

    except Exception as e:
        return False, None, str(e)


def list_stored_keypairs(keys_dir: str = "keys") -> List[KeypairInfo]:
    """List all stored keypairs in the directory"""
    try:
        if not os.path.exists(keys_dir):
            return []

        keypairs = []
        pattern = os.path.join(keys_dir, "private_key_*.json")
        private_key_files = glob.glob(pattern)

        for private_key_file in private_key_files:
            try:
                with open(private_key_file, "r") as f:
                    data = json.load(f)

                # Parse using Pydantic model
                private_key_data = KeyData(**data)

                if private_key_data.type == "private_key":
                    # Try to load the keypair to get DID
                    success, keypair, _ = load_keypair_from_json(private_key_file)

                    keypair_info = KeypairInfo(
                        private_key_file=private_key_file,
                        public_key_file=private_key_data.public_key_file,
                        created_at=private_key_data.created_at.isoformat()
                        if private_key_data.created_at
                        else None,
                        name=private_key_data.name or "unnamed",
                        did=keypair.to_did_string()
                        if success and keypair
                        else "unknown",
                    )
                    keypairs.append(keypair_info)
            except Exception:
                continue  # Skip invalid files

        # Sort by creation time, newest first
        keypairs.sort(key=lambda x: x.created_at or "", reverse=True)
        return keypairs

    except Exception:
        return []


def delete_keypair_files(private_key_file: str) -> Tuple[bool, Optional[str]]:
    """Delete both private and public key files"""
    try:
        if not os.path.exists(private_key_file):
            return False, f"Private key file not found: {private_key_file}"

        # Read private key file to find public key file
        with open(private_key_file, "r") as f:
            data = json.load(f)

        # Parse using Pydantic model
        private_key_data = KeyData(**data)
        public_key_file = private_key_data.public_key_file

        # Delete private key file
        os.remove(private_key_file)

        # Delete public key file if it exists
        if public_key_file and os.path.exists(public_key_file):
            os.remove(public_key_file)

        return True, None

    except Exception as e:
        return False, str(e)


async def create_user_if_not_exists(
    config: NilDBConfig, keys_dir: str = "keys"
) -> UserSetupResult:
    """Create a user if no existing keypair exists in the keys directory"""
    try:
        # Check if any keypairs already exist
        existing_keypairs = list_stored_keypairs(keys_dir)

        if existing_keypairs:
            # Load the most recent keypair
            latest_keypair_info = existing_keypairs[
                0
            ]  # Already sorted by creation time
            success, keypair, error = load_keypair_from_json(
                latest_keypair_info.private_key_file
            )
            if not success or keypair is None:
                return UserSetupResult(
                    success=False, error=f"Failed to load existing keypair: {error}"
                )

            # Create user client with existing keypair
            user_client = await SecretVaultUserClient.from_options(
                keypair=keypair,
                base_urls=config.nodes,
                blindfold=BlindfoldFactoryConfig(
                    operation=BlindfoldOperation.STORE, use_cluster_key=True
                ),
            )

            if (
                not latest_keypair_info.private_key_file
                or not latest_keypair_info.public_key_file
            ):
                return UserSetupResult(
                    success=False, error=f"Failed to load existing keypair: {error}"
                )

            return UserSetupResult(
                success=True,
                user_client=user_client,
                keypair=keypair,
                keys_saved_to={
                    "private_key": latest_keypair_info.private_key_file,
                    "public_key": latest_keypair_info.public_key_file,
                },
            )
        else:
            # No existing keypairs, create new user
            return await setup_user_core(config, keys_dir)

    except Exception as e:
        return UserSetupResult(success=False, error=str(e))
