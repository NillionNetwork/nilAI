import os
import httpx
import json
import traceback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VLLM_HOST = os.environ.get("VLLM_HOST", "vllm")
VLLM_PORT = os.environ.get("VLLM_PORT", "8000")
VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"
VLLM_CHAT_ENDPOINT = f"{VLLM_URL}/v1/chat/completions"
MODEL_NAME = os.environ.get("SUMMARIZATION_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-1M")
VLLM_TIMEOUT = int(os.environ.get("VLLM_TIMEOUT", "300"))
SUMMARIES_DIR = "summaries"
DEFAULT_MAX_SUMMARY_TOKENS = 2064

def read_transcript(file_path: str):
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
):
    if not transcript_content:
        logger.error("Summarization failed: Transcript content is empty.")
        return "Error: Could not generate summary - Transcript content is empty."
    if not system_prompt:
         logger.error("Summarization failed: System prompt is empty.")
         return "Error: Could not generate summary - System prompt is empty."

    logger.info(f"Attempting summarization via vLLM at {VLLM_CHAT_ENDPOINT} using model {MODEL_NAME}")
    logger.info(f"System Prompt (first 100 chars): {system_prompt[:100]}...")
    logger.info(f"Transcript Length: {len(transcript_content)}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcript:\n{transcript_content}\n\nSummary:"}
    ]

    payload = {
        "model": MODEL_NAME,
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
            logger.info(f"Sending payload to vLLM (Payload start): {json.dumps(payload)[:200]}...")
            response = await client.post(VLLM_CHAT_ENDPOINT, json=payload)
            logger.info(f"vLLM Response Status Code: {response.status_code}")

            try:
                raw_response_text = response.text
                logger.info(f"vLLM Raw Response Body (first 500 chars): {raw_response_text[:500]}")
            except Exception as log_err:
                logger.warning(f"Could not log raw response body: {log_err}")

            response.raise_for_status()

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            first_choice = result["choices"][0]
            if "message" in first_choice:
                message = first_choice["message"]
                if "content" in message:
                    summary = message["content"]
                    if summary:
                        logger.info("Summary received successfully from vLLM.")
                        logger.info(f"Returning summary (length {len(summary)}): '{summary[:100]}...'")
                        return summary.strip()
                    else:
                        logger.warning("vLLM response choice contained an empty 'content' string.")
                        return "Error: vLLM returned an empty summary content."
                else:
                     logger.warning("vLLM response 'message' object missing 'content' key.")
                     return "Error: vLLM response format unexpected (missing content)."
            else:
                 logger.warning("vLLM response 'choice' object missing 'message' key.")
                 return "Error: vLLM response format unexpected (missing message)."
        else:
            logger.warning(f"vLLM response missing 'choices' key or 'choices' list is empty. Response: {result}")
            return "Error: Unexpected response structure from vLLM (missing choices)."

    except httpx.TimeoutException as e:
        logger.error(f"Timeout error connecting to vLLM at {VLLM_CHAT_ENDPOINT} after {VLLM_TIMEOUT} seconds.", exc_info=True)
        return f"Error: Timeout connecting to vLLM service ({e})"
    except httpx.ConnectError as e:
        logger.error(f"Connection error connecting to vLLM at {VLLM_CHAT_ENDPOINT}. Is the service running and accessible?", exc_info=True)
        return f"Error: Could not connect to vLLM service ({e})"
    except httpx.RequestError as e:
        logger.error(f"HTTP Request error during vLLM communication: {e}", exc_info=True)
        return f"Error connecting to vLLM service: {e}"
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Status Error from vLLM: {e.response.status_code}", exc_info=False)
        logger.error(f"vLLM Error Response Body: {e.response.text}")
        return f"Error from vLLM service ({e.response.status_code}): {e.response.text[:500]}"
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from vLLM. Status: {response.status_code if response else 'N/A'}. Response text: {raw_response_text[:500]}", exc_info=True)
        return f"Error: Could not parse response from vLLM (Invalid JSON). Status: {response.status_code if response else 'N/A'}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during vLLM communication: {e}", exc_info=True)
        return f"Error during summary generation via vLLM: {e}"