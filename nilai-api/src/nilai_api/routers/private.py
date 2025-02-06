# Fast API and serving
import logging
import os
from base64 import b64encode
from typing import AsyncGenerator, Union, List, Tuple
import numpy as np

import nilql
import nilrag
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from nilai_api.auth import get_user
from nilai_api.crypto import sign_message
from nilai_api.db import UserManager, UserModel
from nilai_api.rate_limiting import RateLimit
from nilai_api.state import state
from openai import OpenAI

# Internal libraries
from nilai_common import (
    AttestationResponse,
    ChatRequest,
    SignedChatCompletion,
    Message,
    ModelMetadata,
    Usage,
)
from nilrag.util import (
    decrypt_float_list,
    encrypt_float_list,
    generate_embeddings_huggingface,
    group_shares_by_id,
)


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/usage", tags=["Usage"])
async def get_usage(user: UserModel = Depends(get_user)) -> Usage:
    """
    Retrieve the current token usage for the authenticated user.

    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Usage statistics for the user's token consumption

    ### Example
    ```python
    # Retrieves token usage for the logged-in user
    usage = await get_usage(user)
    ```
    """
    return Usage(
        prompt_tokens=user.prompt_tokens,
        completion_tokens=user.completion_tokens,
        total_tokens=user.prompt_tokens + user.completion_tokens,
        queries=user.queries,  # FIXME this field is not part of Usage
    )


@router.get("/v1/attestation/report", tags=["Attestation"])
async def get_attestation(user: UserModel = Depends(get_user)) -> AttestationResponse:
    """
    Generate a cryptographic attestation report.

    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Attestation details for service verification

    ### Attestation Details
    - `verifying_key`: Public key for cryptographic verification
    - `cpu_attestation`: CPU environment verification
    - `gpu_attestation`: GPU environment verification

    ### Security Note
    Provides cryptographic proof of the service's integrity and environment.
    """
    return AttestationResponse(
        verifying_key=state.verifying_key,
        cpu_attestation=state.cpu_attestation,
        gpu_attestation=state.gpu_attestation,
    )


@router.get("/v1/models", tags=["Model"])
async def get_models(user: UserModel = Depends(get_user)) -> List[ModelMetadata]:
    """
    List all available models in the system.

    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Dictionary of available models

    ### Example
    ```python
    # Retrieves list of available models
    models = await get_models(user)
    ```
    """
    logger.info(f"Retrieving models for user {user.userid} from pid {os.getpid()}")
    return [endpoint.metadata for endpoint in (await state.models).values()]
    # result = [Model(
    #     id = endpoint.metadata.id,
    #     created = 0,
    #     object = "model",
    #     owned_by = endpoint.metadata.author,
    #     data = endpoint.metadata.dict(),
    # ) for endpoint in (await state.models).values()]

    # return result[0]


async def chat_completion_concurrent_rate_limit(request: Request) -> Tuple[int, str]:
    body = await request.json()
    chat_request = ChatRequest(**body)
    key = f"chat:{chat_request.model}"
    match chat_request.model:
        case "meta-llama/Llama-3.2-1B-Instruct":
            return 10, key
        case "meta-llama/Llama-3.2-3B-Instruct":
            return 10, key
        case "meta-llama/Llama-3.1-8B-Instruct":
            return 5, key
        case "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":
            return 3, key
        case _:
            raise HTTPException(status_code=400, detail="Invalid model name")


@router.post("/v1/chat/completions", tags=["Chat"], response_model=None)
async def chat_completion(
    req: ChatRequest = Body(
        ChatRequest(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="What is your name?"),
            ],
        )
    ),
    _=Depends(RateLimit(concurrent_extractor=chat_completion_concurrent_rate_limit)),
    user: UserModel = Depends(get_user),
) -> Union[SignedChatCompletion, StreamingResponse]:
    """
    Generate a chat completion response from the AI model.

    - **req**: Chat completion request containing messages and model specifications
    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Full chat response with model output, usage statistics, and cryptographic signature

    ### Request Requirements
    - Must include non-empty list of messages
    - Must specify a model
    - Supports multiple message formats (system, user, assistant)

    ### Response Components
    - Model-generated text completion
    - Token usage metrics
    - Cryptographically signed response for verification

    ### Processing Steps
    1. Validate input request parameters
    2. Prepare messages for model processing
    3. Generate AI model response
    4. Track and update token usage
    5. Cryptographically sign the response

    ### Potential HTTP Errors
    - **400 Bad Request**:
      - Missing messages list
      - No model specified
    - **500 Internal Server Error**:
      - Model fails to generate a response

    ### Example
    ```python
    # Generate a chat completion
    request = ChatRequest(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, who are you?"}
        ]
    )
    response = await chat_completion(request, user)
    """

    model_name = req.model
    endpoint = await state.get_model(model_name)
    if endpoint is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name {model_name}, check /v1/models for options",
        )

    if not endpoint.metadata.tool_support and req.tools:
        raise HTTPException(
            status_code=400,
            detail="Model does not support tool usage, remove tools from request",
        )
    model_url = endpoint.url + "/v1/"

    logger.info(
        f"Chat completion request for model {model_name} from user {user.userid} on url: {model_url}"
    )

    client = OpenAI(base_url=model_url, api_key="<not-needed>")

    if req.nilrag:
        """
        Endpoint to process a client query.
        1. Initialization: Secret share keys and NilDB instance.
        2. Secret share query and send to NilDB.
        3. Ask NilDB to compute the differences.
        4. Compute distances and sort.
        5. Ask NilDB to return top k chunks.
        6. Append top results to LLM query
        """
        try:
            logger.debug("Rag is starting.")
            # Step 1: Initialization
            # Get NilDB instance from request
            nodes = []
            for node_data in req.nilrag["nodes"]:
                nodes.append(
                    nilrag.Node(
                        url=node_data["url"],
                        node_id=None,
                        org=None,
                        bearer_token=node_data.get("bearer_token"),
                        schema_id=node_data.get("schema_id"),
                        diff_query_id=node_data.get("diff_query_id"),
                    )
                )
            nilDB = nilrag.NilDB(nodes)

            # Initialize secret keys
            num_parties = len(nilDB.nodes)
            additive_key = nilql.secret_key(
                {"nodes": [{}] * num_parties}, {"sum": True}
            )
            xor_key = nilql.secret_key({"nodes": [{}] * num_parties}, {"store": True})

            # Step 2: Secret share query
            logger.debug("Secret sharing query and sending to NilDB...")
            # 2.1 Extract the user query
            for message in req.messages:
                if message.role == "user":
                    query = message.content
                    break
            # 2.2 Generate query embeddings: one string query is assumed.
            query_embedding = generate_embeddings_huggingface([query])[0]
            nilql_query_embedding = encrypt_float_list(additive_key, query_embedding)

            # Step 3: Ask NilDB to compute the differences
            logger.debug("Requesting computation from NilDB...")
            difference_shares = nilDB.diff_query_execute(nilql_query_embedding)

            # Step 4: Compute distances and sort
            logger.debug("Compute distances and sort...")
            # 4.1 Group difference shares by ID
            difference_shares_by_id = group_shares_by_id(
                difference_shares, lambda share: share["difference"]
            )
            # 4.2 Transpose the lists for each _id
            difference_shares_by_id = {
                id: np.array(differences).T.tolist()
                for id, differences in difference_shares_by_id.items()
            }
            # 4.3 Decrypt and compute distances
            reconstructed = [
                {
                    "_id": id,
                    "distances": np.linalg.norm(
                        decrypt_float_list(additive_key, difference_shares)
                    ),
                }
                for id, difference_shares in difference_shares_by_id.items()
            ]
            # 4.4 Sort id list based on the corresponding distances
            sorted_ids = sorted(reconstructed, key=lambda x: x["distances"])

            # Step 5: Query the top k
            logger.debug("Query top k chunks...")
            top_k = 2
            top_k_ids = [item["_id"] for item in sorted_ids[:top_k]]

            # 5.1 Query top k
            chunk_shares = nilDB.chunk_query_execute(top_k_ids)

            # 5.2 Group chunk shares by ID
            chunk_shares_by_id = group_shares_by_id(
                chunk_shares, lambda share: share["chunk"]
            )

            # 5.3 Decrypt chunks
            top_results = [
                {"_id": id, "distances": nilql.decrypt(xor_key, chunk_shares)}
                for id, chunk_shares in chunk_shares_by_id.items()
            ]

            # Step 6: Format top results
            formatted_results = "\n".join(
                f"- {str(result['distances'])}" for result in top_results
            )
            relevant_context = f"\n\nRelevant Context:\n{formatted_results}"

            # Step 7: Update system message
            for message in req.messages:
                if message.role == "system":
                    message.content += (
                        relevant_context  # Append the context to the system message
                    )
                    break
            else:
                # If no system message exists, add one
                req.messages.insert(0, Message(role="system", content=relevant_context))

            logger.debug(
                f"System message updated with relevant context:\n {req.messages}"
            )

        except Exception as e:
            logger.error("An error occurred within nilrag: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))

    if req.stream:
        # Forwarding Streamed Responses
        async def chat_completion_stream_generator() -> AsyncGenerator[str, None]:
            try:
                response = client.chat.completions.create(
                    model=req.model,
                    messages=req.messages,
                    stream=req.stream,
                    top_p=req.top_p,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    tools=req.tools,
                    extra_body={
                        "stream_options": {
                            "include_usage": True,
                            # "continuous_usage_stats": True,
                        }
                    },
                )

                for chunk in response:
                    if chunk.usage is not None:
                        await UserManager.update_token_usage(
                            user.userid,
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        )
                        await UserManager.log_query(
                            user.userid,
                            model=req.model,
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        )
                    else:
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

            except Exception as e:
                logger.error(f"Error streaming response: {e}")
                return

        # Return the streaming response
        return StreamingResponse(
            chat_completion_stream_generator(),
            media_type="text/event-stream",  # Ensure client interprets as Server-Sent Events
        )

    response = client.chat.completions.create(
        model=req.model,
        messages=req.messages,
        stream=req.stream,
        top_p=req.top_p,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        tools=req.tools,
    )

    model_response = SignedChatCompletion(
        **response.model_dump(),
        signature="",
    )
    # Update token usage
    await UserManager.update_token_usage(
        user.userid,
        prompt_tokens=model_response.usage.prompt_tokens,
        completion_tokens=model_response.usage.completion_tokens,
    )

    await UserManager.log_query(
        user.userid,
        model=req.model,
        prompt_tokens=model_response.usage.prompt_tokens,
        completion_tokens=model_response.usage.completion_tokens,
    )

    # Sign the response
    response_json = model_response.model_dump_json()
    signature = sign_message(state.private_key, response_json)
    model_response.signature = b64encode(signature).decode()

    return model_response
