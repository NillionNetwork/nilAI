from typing import Optional, Dict, Any

from secretvaults import SecretVaultUserClient
from secretvaults.dto.users import (
    AclDto,
    UpdateUserDataRequest,
    ReadDataRequestParams,
    DeleteDocumentRequestParams,
)
from secretvaults.dto.data import CreateOwnedDataRequest
from secretvaults.common.types import Uuid, Did

from nilai_py.nildb.models import OperationResult


async def list_data_references_core(
    user_client: SecretVaultUserClient,
) -> OperationResult:
    """List all data references owned by the user - core functionality"""
    try:
        references_response = await user_client.list_data_references()
        if not references_response:
            return OperationResult(
                success=False, message="No data references available"
            )

        if not hasattr(references_response, "data") or not references_response.data:
            return OperationResult(success=False, message="No data references found")

        return OperationResult(success=True, data=references_response.data)

    except Exception:
        return OperationResult(success=False, message="No data references available")


async def read_document_core(
    user_client: SecretVaultUserClient,
    collection_id: str,
    document_id: str,
    relevant_user: Optional[str] = None,
) -> OperationResult:
    """Read a specific document - core functionality"""
    try:
        read_params = ReadDataRequestParams(
            collection=Uuid(collection_id),
            document=Uuid(document_id),
            subject=Uuid(relevant_user) if relevant_user else None,
        )
        document_response = await user_client.read_data(read_params)
        if not document_response:
            return OperationResult(success=False, message="No document data available")

        # Check if response has data attribute (wrapped response)
        if hasattr(document_response, "data") and document_response.data:
            document_data = document_response.data
        else:
            document_data = document_response

        if not document_data:
            return OperationResult(success=False, message="No document data found")

        return OperationResult(success=True, data=document_data)

    except Exception as e:
        return OperationResult(success=False, error=e)


async def delete_document_core(
    user_client: SecretVaultUserClient, collection_id: str, document_id: str
) -> OperationResult:
    """Delete a specific document - core functionality"""
    try:
        delete_params = DeleteDocumentRequestParams(
            collection=Uuid(collection_id), document=Uuid(document_id)
        )
        delete_response = await user_client.delete_data(delete_params)

        if delete_response:
            node_count = (
                len(delete_response) if hasattr(delete_response, "__len__") else 1
            )
            return OperationResult(
                success=True,
                data=delete_response,
                message=f"Deleted from {node_count} node(s)",
            )
        else:
            return OperationResult(
                success=False, message="No response from delete operation"
            )

    except Exception as e:
        return OperationResult(success=False, error=e)


async def update_document_core(
    user_client: SecretVaultUserClient,
    collection_id: str,
    document_id: str,
    update_data: Dict,
) -> OperationResult:
    """Update a specific document - core functionality"""
    try:
        update_request = UpdateUserDataRequest(
            collection=Uuid(collection_id),
            document=Uuid(document_id),
            update=update_data,
        )
        update_response = await user_client.update_data(update_request)

        if update_response and hasattr(update_response, "root"):
            has_errors = False
            for _, response in update_response.root.items():  # pyright: ignore[reportAttributeAccessIssue]
                if hasattr(response, "status") and response.status != 204:
                    has_errors = True
                    break

            if has_errors:
                return OperationResult(
                    success=False, message="Update failed on some nodes"
                )
            else:
                node_count = len(update_response.root)  # pyright: ignore[reportAttributeAccessIssue]
                return OperationResult(
                    success=True,
                    data=update_response,
                    message=f"Updated on {node_count} node(s)",
                )
        else:
            return OperationResult(
                success=False, message="No response from update operation"
            )

    except Exception as e:
        return OperationResult(success=False, error=e)


async def create_document_core(
    user_client: SecretVaultUserClient,
    collection_id: str,
    data: Dict[str, Any],
    delegation_token: str,
    builder_did: str,
) -> OperationResult:
    """Create a document in a collection - core functionality"""
    try:
        # Create delegation token

        create_data_request = CreateOwnedDataRequest(
            collection=Uuid(collection_id),
            owner=Did(user_client.id),
            data=[data],
            acl=AclDto(grantee=Did(builder_did), read=True, write=False, execute=True),
        )

        create_response = await user_client.create_data(
            delegation=delegation_token, body=create_data_request
        )

        # Calculate totals
        total_created = 0
        total_errors = 0

        if hasattr(create_response, "root"):
            for _, response in create_response.root.items():  # pyright: ignore[reportAttributeAccessIssue]
                if hasattr(response, "data"):
                    created_count = (
                        len(response.data.created) if response.data.created else 0
                    )
                    error_count = (
                        len(response.data.errors) if response.data.errors else 0
                    )
                    total_created += created_count
                    total_errors += error_count

            if total_errors > 0:
                return OperationResult(
                    success=False,
                    message=f"Created {total_created} documents but had {total_errors} errors",
                )
            else:
                node_count = len(create_response.root)  # pyright: ignore[reportAttributeAccessIssue]
                return OperationResult(
                    success=True,
                    data=create_response,
                    message=f"Created document in {total_created} instances across {node_count} node(s)",
                )
        else:
            return OperationResult(
                success=False, message="No response from create operation"
            )

    except Exception as e:
        return OperationResult(success=False, error=e)
