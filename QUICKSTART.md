# Quick Start Guide

## Your chatbot is ready! Here's how to run it:

### Step 1: Install Dependencies

Open terminal in `cogni_anchor_backend` folder and run:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install packages
pip install -r req.txt
```

### Step 2: Start the Server

```bash
uvicorn app.main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Step 3: Test It!

**Option 1: Open Swagger UI in browser**
Go to: http://localhost:8000/docs

Click on **POST /api/v1/chat/message** → Try it out → Enter:
```json
{
  "patient_id": "test123",
  "message": "Hello!",
  "mode": "text"
}
```
Click Execute and see the response!

**Option 2: Run the test script**
Open a NEW terminal (keep server running):
```bash
cd cogni_anchor_backend
python test_chatbot.py
```

---

## What's Configured

✅ Grok API key is set in `.env` file
✅ Chatbot endpoint: `/api/v1/chat/message`
✅ Face recognition endpoints still work
✅ All ready to integrate with Flutter!

---

## Files Created

- `app/chatbot.py` - Chatbot logic
- `.env` - Your API key (configured)
- `test_chatbot.py` - Test script
- `README.md` - Full documentation
- `CHATBOT_SETUP.md` - Detailed setup guide

---

## Next Steps

1. **Test the chatbot** using Swagger UI or test script
2. **Integrate with Flutter** - See CHATBOT_SETUP.md for Flutter code examples
3. **Add more features** - Reminders, voice, etc.

Happy coding! 🚀
