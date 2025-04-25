import whisperx
import os
import torch
import datetime
import traceback
from typing import Optional

def format_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def transcribe_audio_logic(
    audio_file_path: str,
    model,
    align_model_tuple,
    diarize_pipeline,
    device: str,
    batch_size: int,
    output_dir: str,
    num_speakers: Optional[int] = None
):
    print(f"--- Starting Transcription Logic for: {audio_file_path} ---")
    print(f"Using device: {device}, Batch Size: {batch_size}")
    if num_speakers is not None and num_speakers > 0:
         print(f"Speaker number hint provided: {num_speakers}")
    else:
         print("No speaker number hint, using auto-detection.")
    os.makedirs(output_dir, exist_ok=True)
    transcript_path = None

    try:
        print("Loading audio...")
        audio = whisperx.load_audio(audio_file_path)
        print("Audio loaded.")

        print("Starting transcription...")
        result = model.transcribe(audio, batch_size=batch_size)
        detected_language = result.get("language", "unknown")
        print(f"Transcription complete. Detected language: {detected_language}")

        if align_model_tuple:
            model_a, metadata = align_model_tuple
            print(f"Aligning transcript using language: {metadata.get('language_code', 'default')}...")
            try:
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
                print("Alignment complete.")
            except Exception as align_error:
                print(f"Warning: Alignment failed: {align_error}. Proceeding without alignment.")
                traceback.print_exc()
        else:
            print("Skipping alignment as alignment model was not loaded.")
            if "segments" not in result:
                print("Warning: 'segments' key missing from transcription result even without alignment.")
                return None

        print("Performing diarization...")
        try:
            diarization_args = {}
            if num_speakers is not None and num_speakers > 0:
                print(f"--> Attempting diarization with fixed number of speakers: {num_speakers}")
                diarization_args["min_speakers"] = num_speakers
                diarization_args["max_speakers"] = num_speakers
            else:
                print("--> Attempting diarization with automatic speaker number detection.")

            diarize_segments = diarize_pipeline(audio, **diarization_args)
            print("Diarization complete.")

            print("Assigning speakers to words...")
            if align_model_tuple and 'segments' in result and result['segments'] and 'words' in result['segments'][0]:
                 result = whisperx.assign_word_speakers(diarize_segments, result)
                 print("Speaker assignment complete (using word timings).")
            elif 'segments' in result:
                 try:
                     print("Warning: Word-level alignment missing or failed. Speaker assignment might be less accurate or skipped.")
                 except AttributeError:
                      print("Warning: Could not find assign_speakers fallback method. Proceeding without explicit segment speaker assignment.")
                 print("Speaker assignment might be incomplete (used segment timings - potential fallback).")
            else:
                 print("Warning: Cannot assign speakers - 'segments' key is missing.")

        except Exception as diarize_error:
            print(f"Warning: Diarization or speaker assignment failed: {diarize_error}")
            traceback.print_exc()

        base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
        transcript_filename = f"{base_filename}_transcript_diarized.txt"
        transcript_path = os.path.join(output_dir, transcript_filename)

        print(f"Saving transcript to: {transcript_path}")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(f"Transcript for: {os.path.basename(audio_file_path)}\n")
            f.write(f"Detected Language: {result.get('language', 'N/A')}\n")
            f.write("-" * 30 + "\n\n")

            for segment in result.get("segments", []):
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment.get('end', segment['start'] + 0.1))

                speaker = "SPEAKER_UNKNOWN"
                if 'speaker' in segment:
                    speaker = segment['speaker']
                elif 'words' in segment and segment['words'] and 'speaker' in segment['words'][0]:
                    speaker = segment['words'][0].get('speaker', 'SPEAKER_UNKNOWN')

                text = segment.get('text', '').strip()
                f.write(f"[{start_time} --> {end_time}] {speaker}: {text}\n")

        print("Transcript saved successfully.")
        return transcript_path

    except Exception as e:
        print(f"\n--- ERROR during transcription process for {audio_file_path} ---")
        print(f"An error occurred: {e}")
        traceback.print_exc()
        if transcript_path and os.path.exists(transcript_path):
            try:
                os.remove(transcript_path)
                print(f"Removed incomplete transcript file: {transcript_path}")
            except OSError as rm_err:
                print(f"Error removing incomplete transcript file {transcript_path}: {rm_err}")
        return None