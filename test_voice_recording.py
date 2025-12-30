"""
Test voice recording and transcription with a real microphone
"""
import os
import sys
import time
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("=" * 60)
print("MICROPHONE TEST - Record and Transcribe")
print("=" * 60)

# Test 1: Check available audio devices
print("\n[Test 1] Available audio devices:")
print(sd.query_devices())

# Test 2: Record from microphone
print("\n[Test 2] Recording from microphone...")
print("Speak now for 5 seconds!")

duration = 5  # seconds
sample_rate = 16000  # Whisper expects 16kHz

print(f"Recording for {duration} seconds...")
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("🎤 RECORDING NOW - Speak clearly!")

try:
    # Record audio
    audio_data = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='int16')
    sd.wait()  # Wait until recording is finished

    print("✓ Recording complete!")

    # Save to file
    os.makedirs("test_audio_files", exist_ok=True)
    test_file = "test_audio_files/microphone_test.wav"
    wavfile.write(test_file, sample_rate, audio_data)

    print(f"✓ Saved to: {test_file}")
    print(f"  File size: {os.path.getsize(test_file)} bytes")

    # Check if audio is not silent
    max_amplitude = np.max(np.abs(audio_data))
    print(f"  Max amplitude: {max_amplitude}")

    if max_amplitude < 100:
        print("⚠️  WARNING: Audio is very quiet or silent!")
        print("   Check your microphone settings or speak louder")
    else:
        print("✓ Audio has good volume")

except Exception as e:
    print(f"✗ Error recording: {e}")
    print("Make sure you have a working microphone!")
    sys.exit(1)

# Test 3: Transcribe with Whisper
print("\n[Test 3] Transcribing with Whisper...")

try:
    import whisper

    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("✓ Model loaded")

    print(f"Transcribing {test_file}...")
    result = model.transcribe(test_file, language="en", fp16=False)

    transcription = result["text"].strip()

    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT:")
    print("=" * 60)
    if transcription:
        print(f"✓ SUCCESS: '{transcription}'")
    else:
        print("✗ EMPTY - No speech detected")
        print("Possible causes:")
        print("  - Microphone not working")
        print("  - Volume too low")
        print("  - Background noise")
        print("  - Didn't speak during recording")
    print("=" * 60)

except Exception as e:
    print(f"✗ Error during transcription: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Test complete!")
print(f"Audio file saved at: {os.path.abspath(test_file)}")
print("You can manually check this file to verify the recording worked.")
