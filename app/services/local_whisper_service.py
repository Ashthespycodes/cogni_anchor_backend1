"""
Local Whisper STT Service (Offline)
Uses OpenAI's Whisper model running locally for speech-to-text
"""

import os
import logging
import tempfile
from typing import Optional

logger = logging.getLogger("LocalWhisperService")

# Global Whisper model instance (lazy loaded)
_whisper_model = None
_model_loaded = False


def load_whisper_model(model_name: str = "base"):
    """
    Load Whisper model (lazy loading)

    Available models (by size and accuracy):
    - tiny: Fastest, least accurate (~1GB RAM)
    - base: Fast, good for most cases (~1GB RAM) - DEFAULT
    - small: More accurate (~2GB RAM)
    - medium: High accuracy (~5GB RAM)
    - large: Best accuracy (~10GB RAM)

    Args:
        model_name: Name of the Whisper model to load

    Returns:
        Loaded Whisper model
    """
    global _whisper_model, _model_loaded

    if _model_loaded and _whisper_model is not None:
        return _whisper_model

    try:
        import whisper
        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model = whisper.load_model(model_name)
        _model_loaded = True
        logger.info(f"Whisper model '{model_name}' loaded successfully!")
        return _whisper_model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        logger.error("Make sure you installed whisper: pip install openai-whisper")
        _model_loaded = False
        return None


def transcribe_audio_local(
    audio_file_path: str,
    model_name: str = "base",
    language: str = "en"
) -> Optional[str]:
    """
    Transcribe audio file using local Whisper model

    Args:
        audio_file_path: Path to audio file (mp3, wav, m4a, etc.)
        model_name: Whisper model to use (tiny/base/small/medium/large)
        language: Language code (en, es, fr, etc.) - None for auto-detect

    Returns:
        Transcribed text or None if error
    """
    model = load_whisper_model(model_name)

    if model is None:
        logger.error("Whisper model not available")
        return None

    try:
        logger.info(f"Transcribing audio file: {audio_file_path}")

        # Transcribe with Whisper
        result = model.transcribe(
            audio_file_path,
            language=language,
            fp16=False  # Use FP32 for CPU (FP16 for GPU)
        )

        text = result["text"].strip()
        logger.info(f"Transcription successful: {text[:50]}...")
        return text

    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_file_path}")
        return None
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None


def transcribe_audio_from_bytes(
    audio_bytes: bytes,
    model_name: str = "base",
    language: str = "en"
) -> Optional[str]:
    """
    Transcribe audio directly from bytes (no temp file needed!)

    Args:
        audio_bytes: Audio file bytes (WAV format)
        model_name: Whisper model to use
        language: Language code

    Returns:
        Transcribed text or None if error
    """
    model = load_whisper_model(model_name)

    if model is None:
        logger.error("Whisper model not available")
        return None

    import whisper
    import time
    import uuid

    # Whisper's load_audio() needs a file path, not BytesIO
    # So we save to a persistent temp file, load into memory, then delete
    temp_path = None
    try:
        # Create dedicated temp directory
        temp_dir = os.path.join(os.getcwd(), "temp_whisper")
        os.makedirs(temp_dir, exist_ok=True)

        temp_filename = f"whisper_{uuid.uuid4().hex[:8]}.wav"
        temp_path = os.path.join(temp_dir, temp_filename)

        logger.info(f"Saving audio to {temp_path} ({len(audio_bytes)} bytes)")

        # Write bytes to file
        with open(temp_path, 'wb') as f:
            f.write(audio_bytes)
            f.flush()
            os.fsync(f.fileno())

        time.sleep(0.1)  # Let Windows release file handle

        if not os.path.exists(temp_path):
            logger.error(f"Temp file disappeared: {temp_path}")
            return None

        logger.info(f"Loading audio into memory using Whisper...")

        # Load audio into numpy array (Whisper does this internally via ffmpeg)
        audio_array = whisper.load_audio(temp_path)

        logger.info(f"Audio loaded, starting transcription...")

        # Transcribe from the in-memory audio array
        result = model.transcribe(
            audio_array,
            language=language,
            fp16=False
        )

        text = result["text"].strip()

        # Check if transcription is empty
        if not text or len(text) == 0:
            logger.warning(f"⚠️ Transcription returned EMPTY text!")
            logger.warning(f"Audio array shape: {audio_array.shape}")
            logger.warning(f"Audio duration: {len(audio_array) / 16000:.2f} seconds")
            logger.warning(f"This usually means: silence, very short audio, or no speech detected")
            logger.warning(f"Possible causes: emulator microphone not working, too quiet, background noise")
            return None  # Return None for empty transcriptions

        logger.info(f"✓ Transcription successful: '{text[:100]}'")
        return text

    except Exception as e:
        logger.error(f"Error transcribing from bytes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                time.sleep(0.1)
                os.unlink(temp_path)
                logger.info(f"Cleaned up: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {cleanup_error}")


def _transcribe_with_temp_file(
    audio_bytes: bytes,
    model_name: str = "base",
    language: str = "en"
) -> Optional[str]:
    """Fallback method using temp file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode='wb') as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_path = temp_file.name

        # Keep file open longer
        import time
        time.sleep(0.5)

        text = transcribe_audio_local(temp_path, model_name=model_name, language=language)

        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass

        return text

    except Exception as e:
        logger.error(f"Error in fallback transcription: {e}")
        return None


async def transcribe_audio_bytes_local(
    audio_bytes: bytes,
    filename: str = "temp_audio.wav",
    model_name: str = "base"
) -> Optional[str]:
    """
    Transcribe audio from bytes using local Whisper (async wrapper)

    Args:
        audio_bytes: Audio file bytes
        filename: Temporary filename extension (ignored, kept for compatibility)
        model_name: Whisper model to use

    Returns:
        Transcribed text or None if error
    """
    # Use the new direct bytes method
    return transcribe_audio_from_bytes(audio_bytes, model_name=model_name)


# Preload model on module import (optional - comment out to lazy load)
# load_whisper_model("base")
