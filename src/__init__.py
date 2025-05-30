"""
Voice AI Agent - A Python package combining speech recognition with AI responses.

This package provides:
- FastAPI web application with REST API
- faster-whisper integration for speech recognition
- OpenAI API integration for AI responses
- SQLite database for conversation history
- Web interface for easy interaction
"""

__version__ = "1.0.0"
__author__ = "Voice AI Agent Team"
__email__ = "contact@voice-ai-agent.com"

# Package metadata
__all__ = [
    "main",
    "database", 
    "audio_processor",
    "ai_agent"
]