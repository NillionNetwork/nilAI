FROM python:3.10-slim

WORKDIR /app

COPY --link . /app/
WORKDIR /app/nilai-audio/

# Install system dependencies
RUN apt-get update && \
apt-get install build-essential curl ffmpeg -y && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* && \
pip install uv && \
uv sync

ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/:${LD_LIBRARY_PATH}

# Create necessary directories
RUN mkdir -p uploads transcript_results_diarized summaries extracted_audio

# Expose port for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "uvicorn", "nilai_audio.main:app", "--host", "0.0.0.0", "--port", "8000"]
