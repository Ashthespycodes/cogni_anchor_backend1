"""
Test script to debug voice transcription
"""
import os
import sys
import asyncio

# Add the app directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.stt_service import transcribe_audio_bytes

async def test_voice_transcription():
    """Test voice transcription with a sample audio file"""

    # Read a test audio file (you'll need to provide one)
    test_audio_path = input("Enter path to test audio file (WAV/MP3): ").strip()

    if not os.path.exists(test_audio_path):
        print(f"❌ File not found: {test_audio_path}")
        return

    print(f"✅ Reading audio file: {test_audio_path}")

    with open(test_audio_path, "rb") as f:
        audio_bytes = f.read()

    print(f"✅ Audio file size: {len(audio_bytes)} bytes")
    print("🔄 Transcribing...")

    # Test transcription
    transcription = await transcribe_audio_bytes(audio_bytes)

    if transcription:
        print(f"✅ Transcription successful!")
        print(f"📝 Text: {transcription}")
    else:
        print("❌ Transcription failed")

if __name__ == "__main__":
    asyncio.run(test_voice_transcription())
