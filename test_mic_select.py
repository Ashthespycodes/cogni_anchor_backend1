"""
Select and test microphone
"""
import os
import sys
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import time

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("Available INPUT devices:")
print("-" * 60)
devices = sd.query_devices()
input_devices = []
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"{i}: {device['name']} ({device['max_input_channels']} channels)")
        input_devices.append(i)

print("-" * 60)
device_id = int(input("Enter device number to test (try 2 for Realtek): "))

print(f"\nUsing device: {devices[device_id]['name']}")
print("Recording for 3 seconds in 3... 2... 1...")
time.sleep(3)
print("🎤 SPEAK NOW!")

duration = 3
sample_rate = 16000

audio_data = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype='int16',
                   device=device_id)
sd.wait()

os.makedirs("test_audio_files", exist_ok=True)
test_file = "test_audio_files/selected_mic_test.wav"
wavfile.write(test_file, sample_rate, audio_data)

max_amp = np.max(np.abs(audio_data))
print(f"\n✓ Recording complete!")
print(f"  Max amplitude: {max_amp} (should be > 1000 for good audio)")
print(f"  File: {test_file}")

if max_amp > 1000:
    print("✓ Good volume! Transcribing...")
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(test_file, language="en", fp16=False)
    print(f"\nTranscription: '{result['text']}'")
else:
    print("⚠️  Volume too low! Increase microphone volume in Windows settings")
