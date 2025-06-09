import logging
from typing import Union

import blindrag
from fastapi import HTTPException, status
from nilai_common import ChatRequest, Message
from blindrag.rag_vault import RAGVault
from sentence_transformers import SentenceTransformer

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


async def handle_blindrag(req: ChatRequest):
    """
    Endpoint to process a client query.
    1. Get inputs from request.
    2. Execute blindRAG using blindrag library.
    3. Append top results to LLM query
    """
    try:
        logger.debug("Rag is starting.")

        # Step 1: Get inputs
        # Get nilDB instances
        if not req.blindrag or "nodes" not in req.blindrag:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="blindrag configuration is missing or invalid",
            )
        rag = await RAGVault.create_from_dict(req.blindrag)

        # Get user query
        logger.debug("Extracting user query")
        query = None
        for message in req.messages:
            if message.role == "user":
                query = message.content
                break

        if query is None:
            raise HTTPException(status_code=400, detail="No user query found")
        # Step 2: Execute blindRAG
        relevant_context = await rag.top_num_chunks_execute(query)
        # Step 3: Update system message
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
        logger.error("An error occurred within blindRAG: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
