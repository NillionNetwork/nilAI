import logging

import nilrag

from nilai_common import ChatRequest, Message
from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer
from nilai_api.utils.content_extractor import extract_text_content

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
    1. Get inputs from request.
    2. Execute nilRAG using nilrag library.
    3. & 4. Format and append top results to LLM query
    """
    try:
        logger.debug("Rag is starting.")

        # Step 1: Get inputs
        # Get nilDB instances
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

        # Get user query
        logger.debug("Extracting user query")
        query = None
        for message in req.messages:
            if message.role == "user":
                query = extract_text_content(message.content)
                break

        if not query:
            raise HTTPException(status_code=400, detail="No user query found")

        # Get number of chunks to include
        num_chunks = req.nilrag.get("num_chunks", 2)

        # Step 2: Execute nilRAG
        top_results = await nilDB.top_num_chunks_execute(query, num_chunks)

        # Step 3: Format top results
        formatted_results = "\n".join(
            f"- {str(result['distances'])}" for result in top_results
        )
        relevant_context = f"\n\nRelevant Context:\n{formatted_results}"

        # Step 4: Update system message
        for message in req.messages:
            if message.role == "system":
                if message.content is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="system message is empty",
                    )

                if isinstance(message.content, str):
                    message.content += relevant_context
                elif isinstance(message.content, list):
                    message.content.append({"type": "text", "text": relevant_context})
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
