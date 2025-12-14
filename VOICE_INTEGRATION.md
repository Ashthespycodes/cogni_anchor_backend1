# Voice Integration (STT/TTS) - Documentation

## Overview

I've integrated Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities into your Cogni Anchor backend by incorporating modules from the speech-agent repository.

---

## What Was Added

### 1. **Services Folder Structure**
```
app/services/
├── __init__.py
├── stt_service.py      # Speech-to-Text using OpenAI Whisper
├── tts_service.py      # Text-to-Speech using pyttsx3 & OpenAI TTS
├── stt_whisper.py      # Original from speech-agent (reference)
└── tts_local.py        # Original from speech-agent (reference)
```

### 2. **New API Endpoint**

#### **POST /api/v1/chat/voice** - Voice Chat with AI

**Purpose**: Accept audio input, transcribe it, get AI response, and return both text and audio response.

**How it works**:
1. Receives audio file from Flutter app
2. Transcribes audio to text using OpenAI Whisper (STT)
3. Sends transcribed text to Grok AI for response
4. Converts AI response to speech using OpenAI TTS
5. Returns both text response and audio file URL

**Request**:
```
POST /api/v1/chat/voice
Content-Type: multipart/form-data

Form Data:
- patient_id: "patient123"
- audio: [audio file - WAV, MP3, etc.]
```

**Response**:
```json
{
  "patient_id": "patient123",
  "transcription": "Hello, how are you?",
  "response": "I'm doing well! How can I help you today?",
  "audio_url": "/temp/response_abc123.mp3",
  "mode": "audio"
}
```

---

## Services Documentation

### **STT Service** (`app/services/stt_service.py`)

**Functions**:

1. **`transcribe_audio(audio_file_path, model="whisper-1")`**
   - Transcribes audio file to text
   - Uses OpenAI Whisper API
   - Returns transcribed text string

2. **`transcribe_audio_bytes(audio_bytes, filename="temp_audio.wav")`**
   - Transcribes audio from bytes (for uploaded files)
   - Creates temporary file
   - Returns transcribed text
   - Cleans up temp file after

**Requirements**:
- OpenAI API key in `.env` file
- Audio file in supported format (WAV, MP3, M4A, etc.)

**Example Usage**:
```python
from app.services.stt_service import transcribe_audio

text = transcribe_audio("recording.wav")
print(f"Transcribed: {text}")
```

---

### **TTS Service** (`app/services/tts_service.py`)

**Two Modes**:

1. **Offline TTS** (using pyttsx3)
   - Works without internet
   - Lower quality, robotic voice
   - Instant generation
   - No API costs

2. **Online TTS** (using OpenAI TTS API)
   - Requires internet and OpenAI API key
   - High quality, natural voice
   - Generates MP3 files
   - API costs apply

**Functions**:

1. **`speak(text)`** - Quick offline speech
   ```python
   from app.services.tts_service import speak
   speak("Hello, how are you?")
   ```

2. **`generate_speech_file(text, output_path, voice="alloy")`** - Generate audio file
   ```python
   from app.services.tts_service import generate_speech_file

   audio_path = generate_speech_file(
       text="Hello there!",
       output_path="greeting.mp3",
       voice="nova"  # alloy, echo, fable, onyx, nova, shimmer
   )
   ```

3. **`TTSService` class** - Full control
   ```python
   from app.services.tts_service import TTSService

   # Offline TTS
   offline_tts = TTSService(use_online=False)
   offline_tts.speak_offline("Hello!")

   # Online TTS
   online_tts = TTSService(use_online=True)
   online_tts.generate_audio_file("Hello!", "output.mp3")
   ```

---

## Updated Chatbot Features

The chatbot (`app/chatbot.py`) now has:

✅ **Text chat** - `/api/v1/chat/message`
✅ **Voice chat** - `/api/v1/chat/voice` (NEW)
✅ **Conversation history** - `/api/v1/chat/history/{patient_id}`
✅ **Health check** - `/api/v1/chat/health`

**Health check now reports**:
```json
{
  "status": "healthy",
  "service": "chatbot",
  "api": "grok",
  "features": ["text_chat", "voice_chat", "stt", "tts"]
}
```

---

## Environment Variables Needed

Update your `.env` file:

```env
# Existing
GROK_API_KEY=your-grok-api-key-here

# NEW - For STT/TTS (OpenAI)
OPENAI_API_KEY=your-openai-api-key-here

# Database (existing)
DATABASE_URL=postgresql://username:password@localhost:5432/facedb
```

**Get OpenAI API Key**:
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Add to `.env` file

---

## Dependencies Added

Updated `req_chatbot_only.txt`:
```
pyttsx3       # Offline text-to-speech
sounddevice   # Audio recording
scipy         # Audio file handling
```

**Install**:
```bash
cd cogni_anchor_backend
venv/Scripts/activate
pip install -r req_chatbot_only.txt
```

---

## Testing the Voice Endpoint

### **Option 1: Using cURL**

```bash
# Record audio first (use speech-agent/record_audio.py or any tool)
# Then send to API:

curl -X POST "http://localhost:8000/api/v1/chat/voice" \
  -F "patient_id=test123" \
  -F "audio=@recording.wav"
```

### **Option 2: Using Python**

```python
import requests

# Read audio file
with open("recording.wav", "rb") as f:
    audio_data = f.read()

# Send to API
response = requests.post(
    "http://localhost:8000/api/v1/chat/voice",
    data={"patient_id": "test123"},
    files={"audio": ("recording.wav", audio_data, "audio/wav")}
)

result = response.json()
print(f"You said: {result['transcription']}")
print(f"AI response: {result['response']}")
print(f"Audio URL: {result['audio_url']}")
```

### **Option 3: Using Swagger UI**

1. Go to http://localhost:8000/docs
2. Find **POST /api/v1/chat/voice**
3. Click "Try it out"
4. Enter patient_id
5. Upload audio file
6. Click "Execute"
7. See transcription, response, and audio URL

---

## Flutter Integration Example

```dart
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> sendVoiceMessage(
  String patientId,
  File audioFile
) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('http://10.0.2.2:8000/api/v1/chat/voice'),
  );

  request.fields['patient_id'] = patientId;
  request.files.add(
    await http.MultipartFile.fromPath('audio', audioFile.path)
  );

  var response = await request.send();
  var responseData = await response.stream.bytesToString();

  return jsonDecode(responseData);
}
```

---

## Voice Workflow

```
User speaks → Flutter records audio → Upload to /voice endpoint
    ↓
Backend receives audio → STT (Whisper) → Extract text
    ↓
Text → Grok AI → Generate response
    ↓
Response text → TTS (OpenAI) → Generate audio file
    ↓
Return: {transcription, response, audio_url}
    ↓
Flutter displays text + plays audio response
```

---

## Cost Considerations

### **OpenAI Pricing**:
- **Whisper STT**: $0.006 per minute of audio
- **TTS**: $0.015 per 1,000 characters (standard voice)

**Example**:
- 5-second voice message = ~$0.0005 STT
- 50-character response = ~$0.00075 TTS
- **Total per voice interaction: ~$0.00125**

**For 1,000 voice messages**: ~$1.25

---

## Offline Alternative

To avoid API costs, use offline mode:

1. **STT**: Use local Whisper model (from speech-agent)
   - Install `whisper` package
   - Use `stt_whisper_local.py` approach
   - Runs on your machine, slower but free

2. **TTS**: Use pyttsx3 (already integrated)
   - Works offline
   - Lower quality but free
   - Instant generation

---

## Speech-Agent Repository

Cloned to: `cogni_anchor_backend/speech-agent/`

**Contains**:
- `record_audio.py` - Record audio from microphone
- `speech_agent.py` - Complete speech loop
- `stt_whisper.py` - STT implementation
- `tts_local.py` - TTS implementation
- Sample audio files for testing

**Can be used for**:
- Testing voice functionality
- Recording test audio files
- Reference implementation

---

## Next Steps

1. **Add OpenAI API key** to `.env`
2. **Install new dependencies**: `pip install -r req_chatbot_only.txt`
3. **Restart server**: `uvicorn app.main_chatbot:app --reload`
4. **Test voice endpoint** using Swagger UI
5. **Integrate with Flutter** app

---

## Summary

✅ **STT Service** - Converts speech to text (OpenAI Whisper)
✅ **TTS Service** - Converts text to speech (OpenAI TTS + pyttsx3)
✅ **Voice Endpoint** - Complete voice chat API
✅ **Both online and offline** options available
✅ **Ready for Flutter integration**

Your backend now supports full voice interaction! 🎤🔊
