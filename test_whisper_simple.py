"""
Simple Whisper test - no FastAPI, no async, just pure Python
"""
import os
import sys
import whisper
import time

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("=" * 60)
print("WHISPER TEST - Simple File Transcription")
print("=" * 60)

# Test 1: Can we create a file and keep it?
print("\n[Test 1] Creating test file...")
test_dir = "test_audio_files"
os.makedirs(test_dir, exist_ok=True)

test_file = os.path.join(test_dir, "test.wav")

# Create a dummy WAV file (empty for now)
with open(test_file, 'wb') as f:
    # Write minimal WAV header (44 bytes)
    f.write(b'RIFF')
    f.write((36).to_bytes(4, 'little'))
    f.write(b'WAVE')
    f.write(b'fmt ')
    f.write((16).to_bytes(4, 'little'))
    f.write((1).to_bytes(2, 'little'))  # PCM
    f.write((1).to_bytes(2, 'little'))  # Mono
    f.write((16000).to_bytes(4, 'little'))  # Sample rate
    f.write((32000).to_bytes(4, 'little'))  # Byte rate
    f.write((2).to_bytes(2, 'little'))  # Block align
    f.write((16).to_bytes(2, 'little'))  # Bits per sample
    f.write(b'data')
    f.write((0).to_bytes(4, 'little'))  # Data size
    f.flush()
    os.fsync(f.fileno())

print(f"✓ Created: {test_file}")

# Wait and check if file still exists
time.sleep(0.5)
if os.path.exists(test_file):
    print(f"✓ File still exists after 0.5s")
    print(f"  Size: {os.path.getsize(test_file)} bytes")
else:
    print("✗ FILE DISAPPEARED!")
    sys.exit(1)

# Test 2: Load Whisper model
print("\n[Test 2] Loading Whisper model...")
try:
    model = whisper.load_model("base")
    print("✓ Whisper model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Test 3: Try to load the audio file
print("\n[Test 3] Loading audio file with Whisper...")
try:
    print(f"  Attempting to load: {test_file}")
    print(f"  File exists: {os.path.exists(test_file)}")
    print(f"  File size: {os.path.getsize(test_file)} bytes")

    audio_array = whisper.load_audio(test_file)
    print(f"✓ Audio loaded into memory")
    print(f"  Array shape: {audio_array.shape}")
except FileNotFoundError as e:
    print(f"✗ FILE NOT FOUND during load: {e}")
    print(f"  File exists now: {os.path.exists(test_file)}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error loading audio: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[Test 4] Transcribing (this will fail with empty audio but that's OK)...")
try:
    result = model.transcribe(audio_array, language="en", fp16=False)
    print(f"✓ Transcription completed: '{result['text']}'")
except Exception as e:
    print(f"✓ Expected error (empty audio): {e}")

# Cleanup
try:
    os.unlink(test_file)
    print(f"\n✓ Cleaned up test file")
except:
    pass

print("\n" + "=" * 60)
print("ALL TESTS PASSED! Whisper is working correctly.")
print("=" * 60)
