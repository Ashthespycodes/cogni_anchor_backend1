"""
Text-to-Speech Service
Converts text to speech using pyttsx3 (offline) or OpenAI TTS API (online)
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
import pyttsx3

load_dotenv()

logger = logging.getLogger("TTS_Service")


class TTSService:
    """Text-to-Speech service with both offline and online options"""

    def __init__(self, use_online: bool = False):
        """
        Initialize TTS service

        Args:
            use_online: If True, use OpenAI TTS API. If False, use pyttsx3 (offline)
        """
        self.use_online = use_online

        if not use_online:
            # Initialize pyttsx3 for offline TTS
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)  # Speech speed
                self.engine.setProperty('volume', 1.0)  # Max volume
                logger.info("Initialized offline TTS (pyttsx3)")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
                self.engine = None
        else:
            # For online TTS using OpenAI
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("Initialized online TTS (OpenAI)")
            else:
                logger.warning("OPENAI_API_KEY not found. Online TTS not available.")
                self.openai_client = None

    def speak_offline(self, text: str) -> bool:
        """
        Speak text using offline TTS (pyttsx3)

        Args:
            text: Text to speak

        Returns:
            True if successful, False otherwise
        """
        if not self.engine:
            logger.error("pyttsx3 engine not initialized")
            return False

        try:
            logger.info(f"Speaking (offline): {text[:50]}...")
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
            return False

    def generate_audio_file(
        self,
        text: str,
        output_path: str = "output.mp3",
        voice: str = "alloy"
    ) -> Optional[str]:
        """
        Generate audio file from text using OpenAI TTS API

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)

        Returns:
            Path to generated audio file or None if error
        """
        if not self.use_online or not self.openai_client:
            logger.error("Online TTS not available")
            return None

        try:
            logger.info(f"Generating audio file for: {text[:50]}...")

            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )

            # Save to file
            with open(output_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Audio file generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating audio file: {e}")
            return None

    def text_to_speech(
        self,
        text: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Convert text to speech (chooses online or offline based on initialization)

        Args:
            text: Text to convert
            output_path: If provided, save to file (online mode only)

        Returns:
            Path to audio file if online mode, None if offline mode
        """
        if self.use_online and output_path:
            return self.generate_audio_file(text, output_path)
        else:
            self.speak_offline(text)
            return None


# Global TTS instance (offline by default)
tts_service = TTSService(use_online=False)


def speak(text: str) -> bool:
    """
    Quick helper function to speak text using offline TTS

    Args:
        text: Text to speak

    Returns:
        True if successful
    """
    return tts_service.speak_offline(text)


def generate_speech_file(
    text: str,
    output_path: str = "speech.mp3",
    voice: str = "alloy"
) -> Optional[str]:
    """
    Quick helper to generate speech file using online TTS

    Args:
        text: Text to convert
        output_path: Output file path
        voice: Voice to use

    Returns:
        Path to generated file or None
    """
    online_tts = TTSService(use_online=True)
    return online_tts.generate_audio_file(text, output_path, voice)
