"""
Speech-to-Text Service
Supports both OpenAI Whisper API (online) and Local Whisper (offline)
Automatically falls back to local Whisper if OpenAI key is not available
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger("STT_Service")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_LOCAL_WHISPER = not OPENAI_API_KEY  # Auto-switch to local if no API key

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("Using OpenAI Whisper API for STT")
else:
    client = None
    logger.info("OpenAI API key not found. Using LOCAL Whisper for STT (offline)")


def transcribe_audio(
    audio_file_path: str,
    model: str = "whisper-1",
    response_format: str = "text"
) -> Optional[str]:
    """
    Transcribe audio file to text
    Uses OpenAI Whisper API if available, otherwise uses local Whisper

    Args:
        audio_file_path: Path to the audio file
        model: Whisper model to use (default: whisper-1 for OpenAI, base for local)
        response_format: Response format (text, json, verbose_json)

    Returns:
        Transcribed text or None if error
    """
    # Use local Whisper if OpenAI client not available
    if USE_LOCAL_WHISPER:
        try:
            from app.services.local_whisper_service import transcribe_audio_from_bytes
            logger.info(f"Using LOCAL Whisper to transcribe: {audio_file_path}")

            # Load audio file into memory to avoid file locking issues
            try:
                with open(audio_file_path, 'rb') as f:
                    audio_bytes = f.read()
                logger.info(f"Loaded audio into memory ({len(audio_bytes)} bytes), transcribing...")
                return transcribe_audio_from_bytes(audio_bytes, model_name="base")
            except FileNotFoundError:
                logger.error(f"Audio file not found: {audio_file_path}")
                return None

        except Exception as e:
            logger.error(f"Error with local Whisper: {e}")
            return None

    # Use OpenAI Whisper API
    try:
        with open(audio_file_path, "rb") as audio_file:
            logger.info(f"Transcribing audio file with OpenAI: {audio_file_path}")

            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format=response_format
            )

            # Extract text from response
            if hasattr(transcription, 'text'):
                text = transcription.text
            else:
                text = str(transcription)

            logger.info(f"Transcription successful: {text[:50]}...")
            return text

    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_file_path}")
        return None
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None


async def transcribe_audio_bytes(
    audio_bytes: bytes,
    filename: str = "temp_audio.wav",
    model: str = "whisper-1"
) -> Optional[str]:
    """
    Transcribe audio from bytes (useful for uploaded files)
    Automatically uses local Whisper if OpenAI API key not available

    Args:
        audio_bytes: Audio file bytes
        filename: Temporary filename to use
        model: Whisper model to use (whisper-1 for OpenAI, base for local)

    Returns:
        Transcribed text or None if error
    """
    import tempfile
    import time
    import uuid

    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)

    # Use a unique filename in our own temp directory (not system temp)
    temp_filename = f"voice_{uuid.uuid4().hex[:8]}.wav"
    temp_path = os.path.join(temp_dir, temp_filename)

    text = None
    try:
        # Write audio bytes to file
        with open(temp_path, "wb") as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Small delay to ensure file is fully closed (Windows issue)
        time.sleep(0.2)

        # Verify file exists and has data
        if not os.path.exists(temp_path):
            logger.error(f"Temp file disappeared: {temp_path}")
            return None

        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            logger.error(f"Temp file is empty: {temp_path}")
            return None

        logger.info(f"Temp file ready for transcription: {temp_path} (size: {file_size} bytes)")

        # Transcribe (will auto-use local Whisper if no OpenAI key)
        text = transcribe_audio(temp_path, model=model)

        logger.info(f"Transcription complete, result: {text[:50] if text else 'None'}...")

    except Exception as e:
        logger.error(f"Error transcribing audio bytes: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Clean up temp file AFTER transcription is complete
        if temp_path and os.path.exists(temp_path):
            try:
                time.sleep(0.1)  # Small delay before cleanup
                os.unlink(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")

    return text
