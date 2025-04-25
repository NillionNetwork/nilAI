import os
import shutil
import torch
import whisperx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import datetime
import uuid
from fastapi.responses import JSONResponse
import asyncio
import subprocess
from typing import Optional, Annotated
import traceback
import logging
import tempfile
import uvicorn

from nilai_audio.transcription import transcribe_audio_logic
from nilai_audio.summarization import summarize_transcript, read_transcript, MODEL_NAME as SUMMARIZATION_MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WHISPER_MODEL_NAME = "large-v3"
WHISPER_BATCH_SIZE = 16
WHISPER_COMPUTE_TYPE = "float16"
HF_TOKEN = os.environ.get("HF_TOKEN")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    ml_models["device"] = device

    logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME} using compute type: {WHISPER_COMPUTE_TYPE}...")
    try:
        ml_models["whisper_model"] = whisperx.load_model(
            WHISPER_MODEL_NAME, device, compute_type=WHISPER_COMPUTE_TYPE
        )
        logger.info("Whisper model loaded.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load Whisper model: {e}", exc_info=True)
        ml_models["whisper_model"] = None

    alignment_language_code = "en"
    logger.info(f"Loading alignment model for language: {alignment_language_code}...")
    try:
         ml_models["align_model_tuple"] = whisperx.load_align_model(language_code=alignment_language_code, device=device)
         logger.info("Alignment model loaded.")
    except Exception as e:
         logger.warning(f"Could not load default '{alignment_language_code}' alignment model: {e}. Trying without language code...")
         try:
             ml_models["align_model_tuple"] = whisperx.load_align_model(language_code=None, device=device)
             logger.info("Alignment model loaded (no language code).")
         except Exception as e2:
             logger.error(f"FATAL: Failed to load any alignment model: {e2}", exc_info=True)
             ml_models["align_model_tuple"] = None

    logger.info("Loading diarization pipeline...")
    if not HF_TOKEN:
        logger.warning("Hugging Face token (HF_TOKEN) not found in environment variables. Diarization might fail or require login.")
    try:
        auth_token = HF_TOKEN if HF_TOKEN else None
        ml_models["diarize_pipeline"] = whisperx.DiarizationPipeline(use_auth_token=auth_token, device=device)
        logger.info("Diarization pipeline loaded.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load diarization pipeline: {e}", exc_info=True)
        ml_models["diarize_pipeline"] = None

    logger.info("--- Model loading process complete. ---")

    yield

    logger.info("Cleaning up models...")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete.")

app = FastAPI(lifespan=lifespan)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_model_availability(model_keys: list[str]):
    missing_models = [key for key in model_keys if key not in ml_models or ml_models[key] is None]
    if missing_models:
        detail = f"Required model(s) not available: {', '.join(missing_models)}. Server may not be ready or models failed to load."
        logger.error(f"Model Availability Check Failed: {detail}")
        raise HTTPException(status_code=503, detail=detail)

async def run_transcription_logic_internal(
    audio_path: str,
    output_dir: str,
    num_speakers: Optional[int] = None
) -> tuple[Optional[str], Optional[str]]:
    transcript_path: Optional[str] = None
    transcript_content: Optional[str] = None
    try:
        logger.info(f"Starting transcription for temp file: {audio_path} -> output dir: {output_dir}")
        check_model_availability(["whisper_model", "diarize_pipeline", "device"])

        transcript_path = transcribe_audio_logic(
            audio_file_path=audio_path,
            model=ml_models["whisper_model"],
            align_model_tuple=ml_models.get("align_model_tuple"),
            diarize_pipeline=ml_models["diarize_pipeline"],
            device=ml_models["device"],
            batch_size=WHISPER_BATCH_SIZE,
            output_dir=output_dir,
            num_speakers=num_speakers
        )
        logger.info(f"Transcription processing complete. Potential output path: {transcript_path}")

        if not transcript_path or not os.path.exists(transcript_path):
            files_in_output = os.listdir(output_dir)
            if files_in_output:
                logger.warning(f"Transcription logic returned path '{transcript_path}', but found files: {files_in_output}. Assuming first file.")
                transcript_path = os.path.join(output_dir, files_in_output[0])
            else:
                 raise ValueError("Transcription failed: No output file generated in the temporary directory.")

        if not os.path.exists(transcript_path):
             raise ValueError(f"Transcription failed: Reported output path does not exist: {transcript_path}")

        transcript_content = read_transcript(transcript_path)
        if transcript_content is None:
            raise ValueError(f"Transcription file generated ({transcript_path}) but could not be read")

        logger.info(f"Successfully read transcript content (length {len(transcript_content)}).")
        return transcript_path, transcript_content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An error occurred during transcription logic: {e}", exc_info=True)
        return None, None

@app.post("/transcribe-media/")
async def transcribe_media_endpoint(
    mediaFile: UploadFile = File(...),
    num_speakers: Annotated[Optional[int], Form()] = None
):
    if not mediaFile.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    request_id = uuid.uuid4()
    logger.info(f"[Request {request_id}] Received media file: {mediaFile.filename}, Content-Type: {mediaFile.content_type}")
    if num_speakers is not None:
        logger.info(f"[Request {request_id}] Received num_speakers hint: {num_speakers}")
    else:
        logger.info(f"[Request {request_id}] No num_speakers hint received, using auto-detect.")

    temp_upload_dir: Optional[str] = None
    temp_media_path: Optional[str] = None
    temp_audio_extract_dir: Optional[str] = None
    temp_extracted_audio_path: Optional[str] = None
    temp_transcript_dir: Optional[str] = None
    local_transcript_path: Optional[str] = None
    temp_base_dir: Optional[str] = None

    try:
        temp_base_dir = tempfile.mkdtemp(prefix=f"req_{request_id.hex[:8]}_")
        logger.debug(f"[Request {request_id}] Created base temporary directory: {temp_base_dir}")

        temp_upload_dir = os.path.join(temp_base_dir, "upload")
        os.makedirs(temp_upload_dir)
        sanitized_filename = "".join(c if c.isalnum() or c in ['.', '-', '_'] else '_' for c in mediaFile.filename)
        temp_media_path = os.path.join(temp_upload_dir, sanitized_filename)

        logger.info(f"[Request {request_id}] Saving uploaded file to temporary path: {temp_media_path}")
        try:
            with open(temp_media_path, "wb") as buffer:
                shutil.copyfileobj(mediaFile.file, buffer)
            file_size = os.path.getsize(temp_media_path)
            logger.info(f"[Request {request_id}] Saved {mediaFile.filename} to {temp_media_path}, Size: {file_size / (1024*1024):.2f} MB")
        except Exception as e:
            logger.error(f"[Request {request_id}] Failed to save uploaded file to {temp_media_path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

        file_type = mediaFile.content_type or ""
        audio_path_for_transcription: Optional[str] = None

        if file_type.startswith('audio/'):
            logger.info(f"[Request {request_id}] Processing as audio file.")
            audio_path_for_transcription = temp_media_path
        elif file_type.startswith('video/'):
            logger.info(f"[Request {request_id}] Processing as video file, extracting audio...")
            temp_audio_extract_dir = os.path.join(temp_base_dir, "extracted_audio")
            os.makedirs(temp_audio_extract_dir)
            base_name, _ = os.path.splitext(sanitized_filename)
            temp_audio_filename = f"{base_name}.mp3"
            temp_extracted_audio_path = os.path.join(temp_audio_extract_dir, temp_audio_filename)

            logger.info(f"[Request {request_id}] Extracting audio to: {temp_extracted_audio_path}")
            try:
                command = [
                    "ffmpeg", "-i", temp_media_path, "-vn",
                    "-acodec", "libmp3lame", "-q:a", "2",
                    "-ar", "16000", "-ac", "1",
                    "-y",
                    temp_extracted_audio_path
                ]
                logger.info(f"[Request {request_id}] Running FFmpeg command: {' '.join(command)}")
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600.0)

                if process.returncode != 0:
                    stderr_decoded = stderr.decode('utf-8', errors='ignore') if stderr else 'No stderr output'
                    logger.error(f"[Request {request_id}] FFmpeg failed (Code: {process.returncode}). Stderr: {stderr_decoded}")
                    raise RuntimeError(f"Failed to extract audio. FFmpeg error: {stderr_decoded}")
                else:
                    stdout_decoded = stdout.decode('utf-8', errors='ignore') if stdout else 'No stdout output'
                    logger.info(f"[Request {request_id}] Audio extracted successfully. FFmpeg stdout: {stdout_decoded[:200]}...")
                    logger.info(f"[Request {request_id}] FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')[:200]}...")

                if not os.path.exists(temp_extracted_audio_path) or os.path.getsize(temp_extracted_audio_path) == 0:
                     logger.error(f"[Request {request_id}] FFmpeg reported success but output file is missing or empty: {temp_extracted_audio_path}")
                     raise RuntimeError("Audio extraction resulted in an empty or missing file.")

                audio_path_for_transcription = temp_extracted_audio_path

            except FileNotFoundError:
                logger.error(f"[Request {request_id}] Error: ffmpeg command not found.")
                raise HTTPException(status_code=500, detail="Audio extraction tool (ffmpeg) not found on server.")
            except asyncio.TimeoutError:
                logger.error(f"[Request {request_id}] Error: FFmpeg process timed out after 600 seconds.")
                if process.returncode is None:
                    try:
                        process.kill()
                        await process.wait()
                        logger.warning(f"[Request {request_id}] Killed timed-out FFmpeg process.")
                    except ProcessLookupError:
                         logger.warning(f"[Request {request_id}] Timed-out FFmpeg process already finished.")
                    except Exception as kill_err:
                        logger.error(f"[Request {request_id}] Error trying to kill timed-out FFmpeg process: {kill_err}")
                raise HTTPException(status_code=500, detail="Audio extraction timed out.")
            except Exception as e:
                 logger.error(f"[Request {request_id}] An unexpected error occurred during audio extraction: {e}", exc_info=True)
                 raise RuntimeError(f"An unexpected error occurred during audio extraction: {str(e)}")
        else:
            logger.warning(f"[Request {request_id}] Unsupported file type received: {file_type}")
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {file_type}. Please upload audio or video.")

        if not audio_path_for_transcription:
              logger.error(f"[Request {request_id}] Could not determine audio path for transcription after processing.")
              raise HTTPException(status_code=500, detail="Could not determine audio path for transcription.")

        temp_transcript_dir = os.path.join(temp_base_dir, "transcript")
        os.makedirs(temp_transcript_dir)
        logger.info(f"[Request {request_id}] Starting transcription for: {audio_path_for_transcription} -> {temp_transcript_dir}")

        local_transcript_path, transcript_content = await run_transcription_logic_internal(
            audio_path=audio_path_for_transcription,
            output_dir=temp_transcript_dir,
            num_speakers=num_speakers
        )

        if local_transcript_path is None or transcript_content is None:
            logger.error(f"[Request {request_id}] Transcription process failed or returned None.")
            raise HTTPException(status_code=500, detail="Transcription process failed.")

        logger.info(f"[Request {request_id}] Transcription successful. Transcript path: {local_transcript_path}")

        base_name, _ = os.path.splitext(sanitized_filename)
        unique_suffix = request_id.hex[:8]
        job_id = f"{base_name}_{unique_suffix}"

        logger.info(f"[Request {request_id}] Successfully transcribed media file '{mediaFile.filename}'. Returning transcript content.")
        response_content = {
            "transcript_content": transcript_content,
            "original_filename": mediaFile.filename,
            "job_id": job_id,
            "transcript_filename": os.path.basename(local_transcript_path)
        }
        return JSONResponse(content=response_content)

    except HTTPException as http_exc:
        logger.error(f"[Request {request_id}] HTTP Exception during processing file {mediaFile.filename}: {http_exc.status_code} - {http_exc.detail}", exc_info=False)
        raise http_exc
    except Exception as e:
        logger.error(f"[Request {request_id}] Unexpected error processing file {mediaFile.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred processing the file.")
    finally:
        logger.info(f"[Request {request_id}] Starting final cleanup for temporary resources.")
        await mediaFile.close()
        logger.debug(f"[Request {request_id}] Closed uploaded file handle for {mediaFile.filename}.")

        if temp_base_dir and os.path.exists(temp_base_dir):
            try:
                shutil.rmtree(temp_base_dir)
                logger.info(f"[Request {request_id}] Successfully removed temporary directory: {temp_base_dir}")
            except OSError as e:
                logger.error(f"[Request {request_id}] Error removing temporary directory {temp_base_dir}: {e}", exc_info=True)
        else:
             logger.debug(f"[Request {request_id}] No base temporary directory to clean up or it was already removed.")

        logger.info(f"[Request {request_id}] Final cleanup complete.")


@app.post("/generate-from-transcript/")
async def generate_from_transcript_endpoint(
    transcript_content: Annotated[str, Form()],
    customPromptTemplate: Annotated[Optional[str], Form()] = None,
    original_filename: Annotated[Optional[str], Form()] = None,
    job_id: Annotated[Optional[str], Form()] = None
):
    request_id = uuid.uuid4()
    log_prefix = f"[Request {request_id}, Job {job_id or 'Unknown'}]"
    logger.info(f"{log_prefix} Received request to generate from transcript (length: {len(transcript_content)}). Original file: {original_filename or 'Unknown'}")

    if not transcript_content:
         logger.error(f"{log_prefix} Error: Received empty transcript content.")
         raise HTTPException(status_code=400, detail="Transcript content cannot be empty.")

    try:
        check_model_availability(["device"])

        logger.info(f"{log_prefix} Starting generation step using model '{SUMMARIZATION_MODEL_NAME}'...")
        system_base = "You are a helpful assistant that summarizes meeting transcripts. You must only write in english for the summary (no chinese)."
        instructions_base = "Base your response *only* on the provided transcript. Structure the output clearly. Do not add any information not present in the transcript."
        structure_instruction = "Provide a concise summary covering key decisions, main topics, action items, and critical follow-up points based *only* on the provided transcript. Structure the output clearly with headings for each section (e.g., ### Key Decisions)."

        system_message = system_base
        if customPromptTemplate:
            logger.info(f"{log_prefix} Using custom prompt instruction.")
            system_message += f"\n\nFollow these specific instructions: {customPromptTemplate}"
        else:
            logger.info(f"{log_prefix} Using default structure instruction.")
            system_message += f"\n\n{structure_instruction}"
        system_message += f"\n\n{instructions_base}"

        generated_text = await summarize_transcript(
            system_prompt=system_message,
            transcript_content=transcript_content
        )

        logger.info(f"{log_prefix} Received result from summarize_transcript function.")
        if isinstance(generated_text, str):
              logger.info(f"{log_prefix} Generation result type: str, Length: {len(generated_text)}, Starts with: '{generated_text[:100]}...'")
        elif generated_text is None:
             logger.error(f"{log_prefix} Generation failed: summarize_transcript returned None.")
             raise HTTPException(status_code=500, detail="Summary generation failed (None returned).")
        else:
              logger.warning(f"{log_prefix} Unexpected result type from summarize_transcript: {type(generated_text)}, Value: {generated_text}")
              raise HTTPException(status_code=500, detail=f"Summary generation returned unexpected type: {type(generated_text)}.")


        if generated_text.startswith("Error:"):
            error_detail = generated_text
            logger.error(f"{log_prefix} Generation failed: {error_detail}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {error_detail}")

        logger.info(f"{log_prefix} Generation successful.")

        summary_filename_metadata = None
        if original_filename:
              base_name, _ = os.path.splitext(os.path.basename(original_filename))
              if job_id and job_id.startswith(base_name) and len(job_id) > len(base_name) + 1:
                  unique_suffix = job_id[len(base_name)+1:]
                  base_name_with_suffix = f"{base_name}_{unique_suffix}"
              else:
                  base_name_with_suffix = job_id or base_name
              safe_model_name = "".join(c if c.isalnum() else '_' for c in SUMMARIZATION_MODEL_NAME)
              summary_filename_metadata = f"{base_name_with_suffix}_summary_{safe_model_name}.txt"
              logger.info(f"{log_prefix} Constructed summary filename for response metadata: {summary_filename_metadata}")
        else:
             safe_model_name = "".join(c if c.isalnum() else '_' for c in SUMMARIZATION_MODEL_NAME)
             fallback_id = job_id or request_id.hex[:8]
             summary_filename_metadata = f"summary_{fallback_id}_{safe_model_name}.txt"
             logger.info(f"{log_prefix} Constructed fallback summary filename for response metadata: {summary_filename_metadata}")


        logger.info(f"{log_prefix} Preparing final JSON response...")
        response_content = {
            "summary": {
                 "summary": generated_text,
                 "summary_filename": summary_filename_metadata
             }
        }

        logger.info(f"{log_prefix} Successfully generated from transcript. Returning response.")
        return JSONResponse(content=response_content)

    except HTTPException as http_exc:
        logger.error(f"{log_prefix} HTTP Exception during generation: {http_exc.status_code} - {http_exc.detail}", exc_info=False)
        raise http_exc
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error during generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during generation.")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"HF_TOKEN available: {'Yes' if HF_TOKEN else 'No'}")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)