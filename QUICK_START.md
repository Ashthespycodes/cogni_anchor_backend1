# CogniAnchor Backend - Quick Start Guide

## 🚀 What's New?

Your backend now has **ALL features** from the cloned repository integrated:
- ✅ Face Recognition (DeepFace AI)
- ✅ Reminder Management
- ✅ User Authentication & Pairing
- ✅ PLUS your existing chatbot features

## 📋 Before You Start

### 1. Setup Supabase Database (REQUIRED)

1. Go to your Supabase project: https://joayctkupytsedmpfyng.supabase.co
2. Click "SQL Editor" in the left sidebar
3. Open the file `database_setup.sql` and copy all contents
4. Paste into SQL Editor and click "Run"
5. Create storage bucket:
   - Go to "Storage" in left sidebar
   - Click "New bucket"
   - Name: `face-images`
   - Set to **Public**

### 2. Verify Environment Variables

Your `.env` file is already configured with:
```
✅ GEMINI_API_KEY (for chatbot)
✅ SUPABASE_URL (for database)
✅ SUPABASE_ANON_KEY (for auth)
```

## 🎮 Running the Server

```bash
cd cogni_anchor_backend
venv/Scripts/python -m uvicorn app.main_chatbot:app --host 0.0.0.0 --port 8000
```

**Server is already running!** ✅
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## 🧪 Quick Tests

### Test 1: Chatbot (Already Working)
```bash
curl -X POST "http://localhost:8000/api/v1/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "test",
    "message": "Hello!",
    "mode": "text"
  }'
```

### Test 2: Create Reminder (After Database Setup)
```bash
curl -X POST "http://localhost:8000/api/v1/reminders/" \
  -H "Content-Type: application/json" \
  -d '{
    "pair_id": "test-pair-123",
    "title": "Take medicine",
    "date": "25 Dec 2024",
    "time": "02:30 PM"
  }'
```

### Test 3: Face Recognition (After Database Setup)
Open http://localhost:8000/docs and try:
- POST `/api/v1/face/addPerson` - Upload a face photo
- GET `/api/v1/face/getPeople?pair_id=test-pair-123` - See enrolled people

## 📱 Connecting Your Flutter App

Update your Flutter app's API base URL to:
```dart
static const String baseUrl = "http://YOUR_IP:8000";
```

All endpoints are ready:
- Chatbot: `/api/v1/chat/*`
- Face Recognition: `/api/v1/face/*`
- Reminders: `/api/v1/reminders/*`
- Users: `/api/v1/users/*`
- Pairs: `/api/v1/pairs/*`

## 📚 Full API Documentation

Visit http://localhost:8000/docs for interactive API documentation (Swagger UI)

All endpoints are documented with:
- Request/response schemas
- Example values
- "Try it out" functionality

## 🎯 Key Features

| Feature | Endpoint | Status |
|---------|----------|--------|
| Text Chatbot | `/api/v1/chat/message` | ✅ Working |
| Voice Chat | `/api/v1/chat/voice` | ✅ Working |
| Chat History | `/api/v1/chat/history/{patient_id}` | ✅ Working |
| Add Person (Face) | `/api/v1/face/addPerson` | ✅ Ready |
| Scan Face | `/api/v1/face/scan` | ✅ Ready |
| List People | `/api/v1/face/getPeople` | ✅ Ready |
| Create Reminder | `/api/v1/reminders/` | ✅ Ready |
| List Reminders | `/api/v1/reminders/{pair_id}` | ✅ Ready |
| User Signup | `/api/v1/users/signup` | ✅ Ready |
| User Login | `/api/v1/users/login` | ✅ Ready |

## 🔥 What Changed?

### New Files Added:
1. `app/models/database_models.py` - All data models
2. `app/routes/reminders.py` - Reminder endpoints
3. `app/routes/users_pairs.py` - User/pair endpoints
4. `app/routes/face_recognition.py` - Face recognition endpoints
5. `app/services/supabase_client.py` - Database connection
6. `app/services/face_recognition_service.py` - Face AI
7. `database_setup.sql` - Database schema
8. `.env` - Updated with Supabase credentials

### Updated Files:
1. `app/main_chatbot.py` - Added all new routers

### Packages Installed:
```
supabase, deepface, tensorflow, opencv-python, pillow, aiofiles
```

## ⚠️ Important Notes

1. **Database Setup Required**: Run `database_setup.sql` in Supabase before testing reminders/face recognition
2. **Storage Bucket**: Create `face-images` bucket in Supabase Storage
3. **Face Photos**: Upload clear, well-lit face photos for best recognition
4. **Pair ID**: Each patient gets a unique pair_id when signing up

## 🐛 Troubleshooting

### "No face detected in image"
- Ensure photo has a clear, visible face
- Face should be at least 80x80 pixels
- Use well-lit photos

### "Supabase connection error"
- Check SUPABASE_URL in `.env`
- Verify Supabase project is active
- Run `database_setup.sql` if tables don't exist

### "CORS error from Flutter"
- Update Flutter app's base URL to your IP address
- Don't use `localhost` from mobile device

## 📞 Support

- API Documentation: http://localhost:8000/docs
- Integration Details: See `INTEGRATION_COMPLETE.md`
- Database Schema: See `database_setup.sql`

---

## ✅ Next Steps

1. [ ] Run `database_setup.sql` in Supabase
2. [ ] Create `face-images` storage bucket
3. [ ] Test endpoints in Swagger UI
4. [ ] Update Flutter app API URL
5. [ ] Start testing features!

---

**All features integrated successfully!** 🎉

Your backend now has everything from the cloned repository plus your existing chatbot features.
