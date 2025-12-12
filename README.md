# Cogni Anchor Backend

Backend API for the Cogni Anchor cognitive health companion app.

## Features

- **Face Recognition API**: Enroll and recognize faces
- **Chatbot API**: Conversational AI using Grok API

## Setup

### 1. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r req.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` and add your Grok API key:

```
GROK_API_KEY=your-actual-grok-api-key
DATABASE_URL=postgresql://username:password@localhost:5432/facedb
```

### 3. Run the Server

```bash
# From cogni_anchor_backend directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Chatbot Endpoints

#### 1. Send Chat Message
```
POST /api/v1/chat/message
```

**Request Body:**
```json
{
  "patient_id": "patient123",
  "message": "Hello, how are you?",
  "mode": "text"
}
```

**Response:**
```json
{
  "response": "Hello! I'm doing well. How can I help you today?",
  "patient_id": "patient123",
  "mode": "text"
}
```

#### 2. Get Conversation History
```
GET /api/v1/chat/history/{patient_id}
```

**Response:**
```json
{
  "patient_id": "patient123",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

#### 3. Clear Conversation History
```
DELETE /api/v1/chat/history/{patient_id}
```

#### 4. Health Check
```
GET /api/v1/chat/health
```

### Face Recognition Endpoints

#### 1. Enroll a Person
```
POST /api/v1/faces/enroll
```

#### 2. Recognize a Face
```
POST /api/v1/faces/recognize
```

## Testing the Chatbot

### Using cURL

```bash
# Send a message
curl -X POST "http://localhost:8000/api/v1/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "test123",
    "message": "Hello, what time is it?",
    "mode": "text"
  }'

# Get conversation history
curl "http://localhost:8000/api/v1/chat/history/test123"

# Health check
curl "http://localhost:8000/api/v1/chat/health"
```

### Using Python

```python
import requests

# Send chat message
response = requests.post(
    "http://localhost:8000/api/v1/chat/message",
    json={
        "patient_id": "test123",
        "message": "Hello!",
        "mode": "text"
    }
)

print(response.json())
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
cogni_anchor_backend/
├── app/
│   ├── main.py          # FastAPI app & face recognition
│   └── chatbot.py       # Chatbot API endpoints
├── req.txt              # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Notes

- Conversation history is stored in memory (resets when server restarts)
- For production, use a database to persist conversation history
- The chatbot keeps the last 10 messages per patient to manage token limits
- Responses are limited to 150 tokens to keep them short and clear

## Grok API

This chatbot uses the Grok API from x.ai. Get your API key from:
https://console.x.ai/

The Grok API uses the OpenAI SDK format, so we use the `openai` Python package.
