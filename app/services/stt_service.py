"""
Speech-to-Text Service using OpenAI Whisper API
Converts audio files to text
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger("STT_Service")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment. STT service will not work.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def transcribe_audio(
    audio_file_path: str,
    model: str = "whisper-1",
    response_format: str = "text"
) -> Optional[str]:
    """
    Transcribe audio file to text using OpenAI Whisper API

    Args:
        audio_file_path: Path to the audio file
        model: Whisper model to use (default: whisper-1)
        response_format: Response format (text, json, verbose_json)

    Returns:
        Transcribed text or None if error
    """
    if not client:
        logger.error("OpenAI client not initialized. Check OPENAI_API_KEY.")
        return None

    try:
        with open(audio_file_path, "rb") as audio_file:
            logger.info(f"Transcribing audio file: {audio_file_path}")

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

    Args:
        audio_bytes: Audio file bytes
        filename: Temporary filename to use
        model: Whisper model to use

    Returns:
        Transcribed text or None if error
    """
    import tempfile
    import os

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        # Transcribe
        text = transcribe_audio(temp_path, model=model)

        # Clean up
        os.unlink(temp_path)

        return text

    except Exception as e:
        logger.error(f"Error transcribing audio bytes: {e}")
        return None
