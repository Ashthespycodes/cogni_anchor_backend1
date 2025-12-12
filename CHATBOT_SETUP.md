# Chatbot Setup Guide

## What Was Created

I've created a basic chatbot API for your Cogni Anchor app with the following files:

### 1. `app/chatbot.py` - Main Chatbot Module
- **POST /api/v1/chat/message** - Send messages and get responses
- **GET /api/v1/chat/history/{patient_id}** - Get conversation history
- **DELETE /api/v1/chat/history/{patient_id}** - Clear conversation history
- **GET /api/v1/chat/health** - Health check endpoint

### 2. `app/main.py` - Updated
- Integrated chatbot router into the main FastAPI app

### 3. `req.txt` - Updated
- Added `openai` package for Grok API
- Added `python-dotenv` for environment variables

### 4. `.env.example` - Environment Variables Template
- Shows what environment variables you need

### 5. `test_chatbot.py` - Test Script
- Simple script to test all chatbot endpoints

### 6. `README.md` - Documentation
- Complete setup and usage instructions

---

## How to Run

### Step 1: Navigate to Backend Folder
```bash
cd cogni_anchor_backend
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies
```bash
pip install -r req.txt
```

This will install:
- FastAPI
- Uvicorn
- OpenAI SDK (for Grok)
- SQLAlchemy
- All other dependencies

### Step 5: Create .env File
Create a file named `.env` in the `cogni_anchor_backend` folder:

```
GROK_API_KEY=your-actual-grok-api-key-here
DATABASE_URL=postgresql://username:password@localhost:5432/facedb
```

**Get your Grok API key from:** https://console.x.ai/

### Step 6: Run the Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 7: Test the Chatbot

**Option A: Use the test script**
Open a new terminal (keep server running):
```bash
cd cogni_anchor_backend
python test_chatbot.py
```

**Option B: Test via Browser**
1. Open browser and go to: http://localhost:8000/docs
2. This opens Swagger UI (interactive API documentation)
3. Click on **POST /api/v1/chat/message**
4. Click "Try it out"
5. Enter test data:
```json
{
  "patient_id": "test123",
  "message": "Hello, how are you?",
  "mode": "text"
}
```
6. Click "Execute"
7. See the response!

**Option C: Use cURL**
```bash
curl -X POST "http://localhost:8000/api/v1/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "test123",
    "message": "Hello!",
    "mode": "text"
  }'
```

---

## API Endpoints Summary

### 1. Send Chat Message
```
POST /api/v1/chat/message
```
**Request:**
```json
{
  "patient_id": "patient123",
  "message": "Hello!",
  "mode": "text"
}
```
**Response:**
```json
{
  "response": "Hi there! How can I help you today?",
  "patient_id": "patient123",
  "mode": "text"
}
```

### 2. Get Chat History
```
GET /api/v1/chat/history/patient123
```

### 3. Clear Chat History
```
DELETE /api/v1/chat/history/patient123
```

### 4. Health Check
```
GET /api/v1/chat/health
```

---

## Features Implemented

✅ **Conversational AI** using Grok API
✅ **Patient-specific conversations** (separate history per patient)
✅ **Memory** (remembers last 10 messages)
✅ **Empathetic system prompt** (designed for cognitive health)
✅ **Short, clear responses** (max 150 tokens, 2 sentences)
✅ **RESTful API** (ready for Flutter integration)

---

## What the Chatbot Can Do

The chatbot is configured with a compassionate system prompt for cognitive health patients:
- Speaks with warmth and patience
- Uses simple, short sentences
- Provides emotional support
- Never corrects harshly
- Validates feelings

---

## Next Steps

### For Testing:
1. Get your Grok API key from https://console.x.ai/
2. Add it to `.env` file
3. Run the server
4. Test using Swagger UI at http://localhost:8000/docs

### For Flutter Integration:
1. Keep the Python server running
2. In Flutter, use `http` package to call the API
3. Example Flutter code:
```dart
final response = await http.post(
  Uri.parse('http://localhost:8000/api/v1/chat/message'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'patient_id': 'patient123',
    'message': userMessage,
    'mode': 'text'
  }),
);
final data = jsonDecode(response.body);
print(data['response']);  // AI response
```

---

## Troubleshooting

### Server won't start
- Make sure virtual environment is activated: `venv\Scripts\activate`
- Install dependencies: `pip install -r req.txt`

### "Module not found" error
- Activate virtual environment
- Reinstall dependencies

### Chatbot returns error
- Check your Grok API key in `.env` file
- Make sure you have internet connection
- Check server logs for detailed error

### Can't connect from Flutter
- Make sure server is running: `uvicorn app.main:app --reload`
- Use `http://10.0.2.2:8000` for Android emulator
- Use `http://localhost:8000` for web/desktop
- Use your computer's IP (e.g., `http://192.168.1.5:8000`) for physical device

---

## File Structure

```
cogni_anchor_backend/
├── app/
│   ├── main.py              # FastAPI app (face recognition + chatbot)
│   └── chatbot.py           # Chatbot module (NEW)
├── venv/                    # Virtual environment (created by you)
├── .env                     # Your API keys (create this, NOT in git)
├── .env.example             # Environment template (NEW)
├── req.txt                  # Python dependencies (UPDATED)
├── test_chatbot.py          # Test script (NEW)
├── CHATBOT_SETUP.md         # This file (NEW)
└── README.md                # Main documentation (UPDATED)
```

---

## Important Notes

1. **Conversation history is in memory** - It will reset when you restart the server. For production, you'd save it to a database.

2. **API key security** - Never commit `.env` file to git. It's already in `.gitignore`.

3. **Message limit** - Only keeps last 10 messages per patient to avoid token limits.

4. **Grok API** - Uses OpenAI SDK format, so we use the `openai` Python package.

---

Ready to test! 🚀
