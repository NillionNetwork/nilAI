import os
import httpx
import json
import traceback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Updated Configuration ---
# Point directly to the specific vLLM service container for summarization
# Service name from docker-compose.llama-1b-gpu.yml
SUMMARIZATION_VLLM_HOST = os.environ.get("SUMMARIZATION_VLLM_HOST", "llama_1b_gpu")
# Internal port the vLLM service listens on
SUMMARIZATION_VLLM_PORT = os.environ.get("SUMMARIZATION_VLLM_PORT", "8000")
VLLM_URL = f"http://{SUMMARIZATION_VLLM_HOST}:{SUMMARIZATION_VLLM_PORT}"
# Standard OpenAI-compatible endpoint provided by vLLM
VLLM_CHAT_ENDPOINT = f"{VLLM_URL}/v1/chat/completions"

# Model name expected by the vLLM payload.
# *** IMPORTANT: This MUST match the model loaded by the target vLLM container ***
# (e.g., "meta-llama/Llama-3.2-1B-Instruct" for the llama_1b_gpu service)
# Ensure this env var is set correctly for the audio_gpu service.
MODEL_NAME = os.environ.get("SUMMARIZATION_MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")

# Timeout for requests directly to the vLLM service
VLLM_TIMEOUT = int(os.environ.get("VLLM_TIMEOUT", "300"))

# --- Constants ---
SUMMARIES_DIR = "summaries"
DEFAULT_MAX_SUMMARY_TOKENS = 2064

def read_transcript(file_path: str):
    """Reads transcript content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Transcript file not found at {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading transcript file {file_path}: {e}", exc_info=True)
        return None

def save_summary(summary_text: str, original_transcript_filename: str, model_name_used: str):
    """Saves the generated summary to a file."""
    if not os.path.exists(SUMMARIES_DIR):
        try:
            os.makedirs(SUMMARIES_DIR)
        except OSError as e:
            logger.error(f"Error creating summaries directory {SUMMARIES_DIR}: {e}")
            return None

    model_short_name = model_name_used.replace('/', '_').replace('\\', '_')
    base_name = os.path.splitext(original_transcript_filename)[0]
    base_name = os.path.basename(base_name)
    summary_filename = f"{base_name}_summary_{model_short_name}.txt"
    summary_path = os.path.join(SUMMARIES_DIR, summary_filename)

    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        logger.info(f"Summary saved to {summary_path}")
        return summary_path
    except Exception as e:
        logger.error(f"Error saving summary to {summary_path}: {e}", exc_info=True)
        return None

async def summarize_transcript(
    system_prompt: str,
    transcript_content: str,
    max_new_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS
) -> str:
    """
    Sends a transcript directly to the specified vLLM service container for summarization.
    """
    if not transcript_content:
        logger.error("Summarization failed: Transcript content is empty.")
        return "Error: Could not generate summary - Transcript content is empty."
    if not system_prompt:
         logger.error("Summarization failed: System prompt is empty.")
         return "Error: Could not generate summary - System prompt is empty."

    logger.info(f"Attempting summarization via vLLM Service at {VLLM_CHAT_ENDPOINT} using model {MODEL_NAME}")
    logger.info(f"System Prompt (first 100 chars): {system_prompt[:100]}...")
    logger.info(f"Transcript Length: {len(transcript_content)}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcript:\n{transcript_content}\n\nSummary:"}
    ]

    # Standard OpenAI-compatible payload for vLLM
    payload = {
        "model": MODEL_NAME, # Must match the model loaded in the target vLLM container
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.05,
        "stream": False
    }

    raw_response_text = ""
    response = None
    try:
        async with httpx.AsyncClient(timeout=float(VLLM_TIMEOUT)) as client:
            logger.info(f"Sending payload to vLLM Service (Payload start): {json.dumps(payload)[:200]}...")
            # No headers needed for direct internal call typically
            response = await client.post(VLLM_CHAT_ENDPOINT, json=payload)
            logger.info(f"vLLM Service Response Status Code: {response.status_code}")

            try:
                raw_response_text = response.text
                logger.info(f"vLLM Service Raw Response Body (first 500 chars): {raw_response_text[:500]}")
            except Exception as log_err:
                logger.warning(f"Could not log raw response body: {log_err}")

            response.raise_for_status()

        result = response.json()

        # Standard OpenAI-compatible response parsing
        if "choices" in result and len(result["choices"]) > 0:
            first_choice = result["choices"][0]
            if "message" in first_choice:
                message = first_choice["message"]
                if "content" in message:
                    summary = message["content"]
                    if summary:
                        logger.info("Summary received successfully from vLLM service.")
                        logger.info(f"Returning summary (length {len(summary)}): '{summary[:100]}...'")
                        return summary.strip()
                    else:
                        logger.warning("vLLM service response choice contained an empty 'content' string.")
                        return "Error: vLLM service returned an empty summary content."
                else:
                     logger.warning("vLLM service response 'message' object missing 'content' key.")
                     return "Error: vLLM service response format unexpected (missing content)."
            else:
                 logger.warning("vLLM service response 'choice' object missing 'message' key.")
                 return "Error: vLLM service response format unexpected (missing message)."
        else:
            logger.warning(f"vLLM service response missing 'choices' key or 'choices' list is empty. Response: {result}")
            return "Error: Unexpected response structure from vLLM service (missing choices)."

    except httpx.TimeoutException as e:
        logger.error(f"Timeout error connecting to vLLM Service at {VLLM_CHAT_ENDPOINT} after {VLLM_TIMEOUT} seconds.", exc_info=True)
        return f"Error: Timeout connecting to vLLM service ({e})"
    except httpx.ConnectError as e:
        logger.error(f"Connection error connecting to vLLM Service at {VLLM_CHAT_ENDPOINT}. Is the '{SUMMARIZATION_VLLM_HOST}' service running and on the same network?", exc_info=True)
        return f"Error: Could not connect to vLLM service ({e})"
    except httpx.RequestError as e:
        logger.error(f"HTTP Request error during vLLM service communication: {e}", exc_info=True)
        return f"Error connecting to vLLM service: {e}"
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Status Error from vLLM service: {e.response.status_code}", exc_info=False)
        logger.error(f"vLLM Service Error Response Body: {e.response.text}")
        return f"Error from vLLM service ({e.response.status_code}): {e.response.text[:500]}"
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from vLLM service. Status: {response.status_code if response else 'N/A'}. Response text: {raw_response_text[:500]}", exc_info=True)
        return f"Error: Could not parse response from vLLM service (Invalid JSON). Status: {response.status_code if response else 'N/A'}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during vLLM service communication: {e}", exc_info=True)
        return f"Error during summary generation via vLLM service: {e}"

# --- Example Usage (Commented out - for reference) ---
# import asyncio
# async def main():
#     transcript_path = "path/to/your/transcript_file.txt"
#     transcript = read_transcript(transcript_path)
#     if transcript:
#         prompt = "Provide a concise summary."
#         summary = await summarize_transcript(prompt, transcript)
#         print(summary)
#         if not summary.startswith("Error:"):
#            save_summary(summary, os.path.basename(transcript_path), MODEL_NAME)
# if __name__ == "__main__":
#      asyncio.run(main())