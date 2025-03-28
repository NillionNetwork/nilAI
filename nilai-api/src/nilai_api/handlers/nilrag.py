import logging
import numpy as np
import time
import sys

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

def get_size_in_MB(obj):
    return sys.getsizeof(obj) / (1024 * 1024)

def generate_embeddings_huggingface(
    chunks_or_query: Union[str, list],
):
    """
    Generate embeddings for text using a HuggingFace sentence transformer model.

    Args:
        chunks_or_query (str or list): Text string(s) to generate embeddings for
        model_name (str, optional): Name of the HuggingFace model to use.
            Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        numpy.ndarray: Array of embeddings for the input text
    """
    embeddings = embeddings_model.encode(chunks_or_query, convert_to_tensor=False)
    return embeddings


def handle_nilrag(req: ChatRequest):
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
        start_time = time.time()
        num_parties = len(nilDB.nodes)
        additive_key = nilql.secret_key({"nodes": [{}] * num_parties}, {"sum": True})
        xor_key = nilql.secret_key({"nodes": [{}] * num_parties}, {"store": True})
        end_time = time.time()
        secret_keys_initialization_time = end_time - start_time

        # Step 2: Secret share query
        logger.debug("Secret sharing query and sending to NilDB...")
        # 2.1 Extract the user query
        query = None
        start_time = time.time()
        for message in req.messages:
            if message.role == "user":
                query = message.content
                break

        if query is None:
            raise HTTPException(status_code=400, detail="No user query found")
        end_time = time.time()
        extract_user_query_time = end_time - start_time

        # 2.2 Generate query embeddings: one string query is assumed.
        start_time = time.time()
        query_embedding = generate_embeddings_huggingface([query])[0]
        nilql_query_embedding = encrypt_float_list(additive_key, query_embedding)
        end_time = time.time()
        embedding_generation_time = end_time - start_time
        query_size = get_size_in_MB(nilql_query_embedding)

        # Step 3: Ask NilDB to compute the differences
        logger.debug("Requesting computation from NilDB...")
        start_time = time.time()
        difference_shares = nilDB.diff_query_execute(nilql_query_embedding)
        end_time = time.time()
        asking_nilDB_time = end_time - start_time
        difference_shares_size = get_size_in_MB(difference_shares)

        # Step 4: Compute distances and sort
        logger.debug("Compute distances and sort...")
        # 4.1 Group difference shares by ID
        start_time = time.time()
        difference_shares_by_id = group_shares_by_id(
            difference_shares,  # type: ignore
            lambda share: share["difference"],
        )
        end_time = time.time()
        group_shares_by_id_time = end_time - start_time
        # 4.2 Transpose the lists for each _id
        start_time = time.time()
        difference_shares_by_id = {
            id: np.array(differences).T.tolist()
            for id, differences in difference_shares_by_id.items()
        }
        end_time = time.time()
        transpose_lists_time = end_time - start_time
        # 4.3 Decrypt and compute distances
        start_time = time.time()
        reconstructed = [
            {
                "_id": id,
                "distances": np.linalg.norm(
                    decrypt_float_list(additive_key, difference_shares)
                ),
            }
            for id, difference_shares in difference_shares_by_id.items()
        ]
        end_time = time.time()
        decryption_time = end_time - start_time

        # 4.4 Sort id list based on the corresponding distances
        start_time = time.time()
        sorted_ids = sorted(reconstructed, key=lambda x: x["distances"])
        end_time = time.time()
        sort_id_list_time = end_time - start_time

        # Step 5: Query the top k
        logger.debug("Query top k chunks...")
        top_k = 2
        top_k_ids = [item["_id"] for item in sorted_ids[:top_k]]

        # 5.1 Query top k
        start_time = time.time()
        chunk_shares = nilDB.chunk_query_execute(top_k_ids)
        end_time = time.time()
        query_top_chunks_time = end_time - start_time
        chunks_shares_size = get_size_in_MB(chunk_shares)
        # 5.2 Group chunk shares by ID
        start_time = time.time()
        chunk_shares_by_id = group_shares_by_id(
            chunk_shares,  # type: ignore
            lambda share: share["chunk"],
        )
        end_time = time.time()
        group_chunks_time = end_time - start_time

        # 5.3 Decrypt chunks
        start_time = time.time()
        top_results = [
            {"_id": id, "distances": nilql.decrypt(xor_key, chunk_shares)}
            for id, chunk_shares in chunk_shares_by_id.items()
        ]
        end_time = time.time()
        decrypt_chunks_time = end_time - start_time

        # Step 6: Format top results
        start_time = time.time()
        formatted_results = "\n".join(
            f"- {str(result['distances'])}" for result in top_results
        )
        relevant_context = f"\n\nRelevant Context:\n{formatted_results}"
        end_time = time.time()
        format_results_time = end_time - start_time

        # Step 7: Update system message
        start_time = time.time()
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
        end_time = time.time()
        update_system_message_time = end_time - start_time
        logger.debug(f"System message updated with relevant context:\n {req.messages}")
        return {
            "secret_keys_initialization_time": secret_keys_initialization_time,
            "extract_user_query_time": extract_user_query_time,
            "embedding_generation_time": embedding_generation_time,
            "query_size": query_size,
            "asking_nilDB_time": asking_nilDB_time,
            "group_shares_by_id_time": group_shares_by_id_time,
            "transpose_lists_time": transpose_lists_time,
            "decryption_time": decryption_time,
            "sort_id_list_time": sort_id_list_time,
            "query_top_chunks_time": query_top_chunks_time,
            "group_chunks_time": group_chunks_time,
            "decrypt_chunks_time": decrypt_chunks_time,
            "format_results_time": format_results_time,
            "update_system_message_time": update_system_message_time,
            "difference_shares_size": difference_shares_size,
            "chunks_shares_size": chunks_shares_size,
        }

    except Exception as e:
        logger.error("An error occurred within nilrag: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )