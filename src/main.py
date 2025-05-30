"""
Main FastAPI application for Voice AI Agent.
"""
import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from loguru import logger
from dotenv import load_dotenv

from .database import db_manager
from .audio_processor import audio_processor
from .ai_agent import ai_agent

# Load environment variables
load_dotenv()

# Configure logging
logger.add(
    "logs/app.log",
    rotation="100 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)

# Create FastAPI app
app = FastAPI(
    title="Voice AI Agent",
    description="AI Agent with voice recognition and intelligent responses",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create necessary directories
for directory in ["uploads", "data", "logs", "models", "static"]:
    Path(directory).mkdir(exist_ok=True)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    include_context: bool = True

class ConfigUpdate(BaseModel):
    whisper_model: Optional[str] = None
    ai_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class ContextData(BaseModel):
    key: str
    value: str
    category: str = "general"

# Global variables for tracking
startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    logger.info("Starting Voice AI Agent...")
    
    try:
        # Initialize database
        await db_manager.init_database()
        logger.info("Database initialized")
        
        # Initialize audio processor
        await audio_processor.initialize_model()
        logger.info("Audio processor initialized")
        
        logger.info("Voice AI Agent started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Voice AI Agent...")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "version": "1.0.0",
        "services": {
            "database": "online",
            "audio_processor": "online" if audio_processor.model else "offline",
            "ai_agent": "online"
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - Redirect to web interface"""
    return FileResponse('static/index.html')

# Audio upload and processing endpoint
@app.post("/upload-audio")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Language code (e.g., 'ja', 'en')")
):
    """Upload and process audio file."""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in audio_processor.supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported: {list(audio_processor.supported_formats)}"
        )
    
    # Check file size (limit to 100MB)
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 100MB)")
    
    try:
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        temp_file = upload_dir / f"{int(time.time())}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process audio in background
        background_tasks.add_task(
            process_audio_file,
            temp_file,
            file.filename,
            language
        )
        
        return {
            "message": "Audio file uploaded successfully",
            "filename": file.filename,
            "status": "processing",
            "file_path": str(temp_file)
        }
        
    except Exception as e:
        logger.error(f"Audio upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_audio_file(file_path: Path, original_filename: str, language: Optional[str]):
    """Background task to process uploaded audio file."""
    try:
        logger.info(f"Processing audio file: {file_path.name}")
        
        # Transcribe audio
        transcription_result = await audio_processor.transcribe_audio(
            file_path, 
            language=language
        )
        
        # Get conversation context
        recent_conversations = await db_manager.get_recent_conversations(limit=5)
        
        # Generate AI response
        ai_response_result = await ai_agent.generate_response(
            transcription_result['text'],
            conversation_context=recent_conversations
        )
        
        # Save conversation to database
        await db_manager.save_conversation(
            transcribed_text=transcription_result['text'],
            ai_response=ai_response_result['response'],
            audio_file_path=str(file_path),
            processing_time=transcription_result['processing_time'],
            model_used=transcription_result['model_used'],
            confidence_score=transcription_result.get('language_probability'),
            metadata={
                'original_filename': original_filename,
                'language': transcription_result['language'],
                'ai_model': ai_response_result.get('model_used'),
                'tokens_used': ai_response_result.get('tokens_used', 0)
            }
        )
        
        logger.info(f"Successfully processed audio file: {file_path.name}")
        
    except Exception as e:
        logger.error(f"Failed to process audio file {file_path}: {e}")
    
    finally:
        # Clean up uploaded file after processing
        if file_path.exists():
            file_path.unlink()

# Text chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    """Process text-based chat message."""
    try:
        # Get conversation context if requested
        conversation_context = None
        if request.include_context:
            conversation_context = await db_manager.get_recent_conversations(limit=5)
        
        # Generate AI response
        ai_response_result = await ai_agent.generate_response(
            request.message,
            conversation_context=conversation_context
        )
        
        # Save conversation to database
        await db_manager.save_conversation(
            transcribed_text=request.message,
            ai_response=ai_response_result['response'],
            model_used=ai_response_result.get('model_used', 'text_input'),
            metadata={
                'input_type': 'text',
                'tokens_used': ai_response_result.get('tokens_used', 0)
            }
        )
        
        return {
            "response": ai_response_result['response'],
            "metadata": {
                "model_used": ai_response_result.get('model_used'),
                "tokens_used": ai_response_result.get('tokens_used', 0),
                "timestamp": ai_response_result.get('timestamp')
            }
        }
        
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Get conversations endpoint
@app.get("/conversations")
async def get_conversations(
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search in conversations")
):
    """Get conversation history."""
    try:
        if search:
            conversations = await db_manager.search_conversations(search, limit=limit)
        else:
            conversations = await db_manager.get_recent_conversations(limit=limit)
        
        return {
            "conversations": conversations,
            "count": len(conversations)
        }
        
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")

# Get specific conversation
@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    """Get specific conversation by ID."""
    try:
        conversation = await db_manager.get_conversation_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")

# Statistics endpoint
@app.get("/stats")
async def get_stats():
    """Get application statistics."""
    try:
        db_stats = await db_manager.get_conversation_stats()
        audio_info = audio_processor.get_model_info()
        agent_info = ai_agent.get_agent_info()
        
        return {
            "database": db_stats,
            "audio_processor": audio_info,
            "ai_agent": agent_info,
            "uptime_seconds": round(time.time() - startup_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )