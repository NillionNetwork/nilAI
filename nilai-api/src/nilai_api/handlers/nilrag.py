import logging
import numpy as np

import nilql
import nilrag

from nilai_common import ChatRequest, Message
from fastapi import HTTPException, status
from nilrag.util import (
    decrypt_float_list,
    encrypt_float_list,
    group_shares_by_id,
)
from sentence_transformers import SentenceTransformer
from typing import Union

logger = logging.getLogger(__name__)

embeddings_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
)  # FIXME: Use a GPU model and move to a separate container


def generate_embeddings_huggingface(
    chunks_or_query: Union[str, list],
):
    """
    Generate embeddings for text using a HuggingFace sentence transformer model.

    Args:
        chunks_or_query (str or list): Text string(s) to generate embeddings for

    Returns:
        numpy.ndarray: Array of embeddings for the input text
    """
    embeddings = embeddings_model.encode(chunks_or_query, convert_to_tensor=False)
    return embeddings


async def handle_nilrag(req: ChatRequest):
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
        if not req.nilrag or "nodes" not in req.nilrag:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="nilrag configuration is missing or invalid",
            )
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
        additive_key = nilql.ClusterKey.generate(
            {"nodes": [{}] * num_parties}, {"sum": True}
        )
        xor_key = nilql.ClusterKey.generate(
            {"nodes": [{}] * num_parties}, {"store": True}
        )

        # Step 2: Secret share query
        logger.debug("Secret sharing query and sending to NilDB...")
        # 2.1 Extract the user query
        query = None
        for message in req.messages:
            if message.role == "user":
                query = message.content
                break

        if query is None:
            raise HTTPException(status_code=400, detail="No user query found")

        # 2.2 Generate query embeddings: one string query is assumed.
        query_embedding = generate_embeddings_huggingface([query])[0]
        nilql_query_embedding = encrypt_float_list(additive_key, query_embedding)

        # Step 3: Ask NilDB to compute the differences
        logger.debug("Requesting computation from NilDB...")
        difference_shares = await nilDB.diff_query_execute(nilql_query_embedding)

        # Step 4: Compute distances and sort
        logger.debug("Compute distances and sort...")
        # 4.1 Group difference shares by ID
        difference_shares_by_id = group_shares_by_id(
            difference_shares,  # type: ignore
            lambda share: share["difference"],
        )
        # 4.2 Transpose the lists for each _id
        difference_shares_by_id = {
            id: list(map(list, zip(*differences)))
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
        top_k = req.nilrag.get("num_chunks", 2)
        if not isinstance(top_k, int):
            raise HTTPException(
                status_code=400,
                detail="num_chunks must be an integer as it represents the number of chunks to be retrieved.",
            )
        top_k_ids = [item["_id"] for item in sorted_ids[:top_k]]

        # 5.1 Query top k
        chunk_shares = await nilDB.chunk_query_execute(top_k_ids)

        # 5.2 Group chunk shares by ID
        chunk_shares_by_id = group_shares_by_id(
            chunk_shares,  # type: ignore
            lambda share: share["chunk"],
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
                if message.content is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="system message is empty",
                    )
                message.content += (
                    relevant_context  # Append the context to the system message
                )
                break
        else:
            # If no system message exists, add one
            req.messages.insert(0, Message(role="system", content=relevant_context))

        logger.debug(f"System message updated with relevant context:\n {req.messages}")

    except HTTPException as e:
        raise e

    except Exception as e:
        logger.error("An error occurred within nilrag: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
