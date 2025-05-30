# Voice AI Agent

Real-time AI voice recognition and conversation system

[????README](README_JP.md) | [English README](README.md)

---

## Overview

Voice AI Agent is a real-time voice conversation system that combines faster-whisper for high-speed speech recognition with OpenAI API for intelligent AI dialogue. You can engage in natural conversations with AI agents through voice file uploads and text chat.

## Features

- ? **High-speed speech recognition**: High-accuracy speech recognition using faster-whisper
- ? **AI dialogue function**: Intelligent response generation using OpenAI API
- ? **Conversation history storage**: Persistence of conversation data using SQLite
- ? **REST API**: Web interface based on FastAPI
- ? **Web dashboard**: Intuitive HTML/JavaScript UI
- ? **Docker support**: Easy deployment through containerization
- ? **Statistics**: System usage monitoring
- ? **Configuration management**: Flexible system settings

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/nullpox7/voice-ai-agent.git
cd voice-ai-agent
```

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env file and set OpenAI API key
```

### 3. Start Docker

```bash
# Start containers
docker-compose up -d
```

### 4. Access Verification

- **Web UI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## System Requirements

- Docker & Docker Compose
- Python 3.11+ (for local development)
- OpenAI API Key (required)
- 4GB+ RAM recommended

## Tech Stack

- **Backend**: Python 3.11, FastAPI
- **Speech Recognition**: faster-whisper
- **AI**: OpenAI API
- **Database**: SQLite
- **Infrastructure**: Docker, Docker Compose
- **Frontend**: HTML, CSS, JavaScript

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3)
- M4A (.m4a)
- OGG (.ogg)
- FLAC (.flac)
- AAC (.aac)

## Configuration

### Whisper Model Selection

| Model | Parameters | Required VRAM | Processing Speed | Accuracy |
|-------|------------|---------------|------------------|----------|
| tiny | 39M | ~1GB | ~32x | Low |
| base | 74M | ~1GB | ~16x | Medium |
| small | 244M | ~2GB | ~6x | Medium |
| medium | 769M | ~5GB | ~2x | High |
| large | 1550M | ~10GB | 1x | Highest |

### Environment Variables

```bash
# OpenAI Settings
OPENAI_API_KEY=your_api_key_here
AI_MODEL=gpt-3.5-turbo

# Whisper Settings
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu

# Server Settings
PORT=8000
DEBUG_MODE=false
```

## Local Development

```bash
# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start application
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Audio Processing
- `POST /upload-audio` - Upload and process audio files
- `GET /conversations` - Get conversation history
- `GET /conversations/{id}` - Get specific conversation details

### Chat
- `POST /chat` - Text-based dialogue

### System
- `GET /health` - Health check
- `GET /stats` - Statistics
- `POST /config` - Configuration changes

## Testing

```bash
# Run tests
pytest

# Coverage report
pytest --cov=src --cov-report=html
```

## License

MIT License

## Contributing

1. Create a fork
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Create pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/nullpox7/voice-ai-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nullpox7/voice-ai-agent/discussions)

---

**Voice AI Agent** - Powering new conversation experiences with voice and AI ?

## Note

Due to encoding issues with automated updates, the Japanese README (README_JP.md) may display garbled characters. This will be manually fixed in a future update to ensure proper UTF-8 encoding.