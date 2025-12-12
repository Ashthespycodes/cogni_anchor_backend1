"""
Simple FastAPI app for chatbot only (no face recognition)
Use this to test the chatbot without needing face recognition dependencies
"""

import logging
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CogniAnchorAPI")

# --- FastAPI Application ---
app = FastAPI(
    title="Cogni Anchor Chatbot API",
    description="Backend API for cognitive health companion app - Chatbot only"
)

# Import and include chatbot router
from app.chatbot import router as chatbot_router
app.include_router(chatbot_router)

@app.on_event("startup")
def startup_event():
    logger.info("Chatbot API startup complete!")
    logger.info("API documentation available at: http://localhost:8000/docs")

@app.get("/")
def read_root():
    return {
        "message": "Cogni Anchor Chatbot API",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "chat": "/api/v1/chat/message",
            "history": "/api/v1/chat/history/{patient_id}",
            "health": "/api/v1/chat/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
