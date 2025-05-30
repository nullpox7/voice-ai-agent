#!/bin/bash

# Voice AI Agent setup script

echo "? Voice AI Agent setup started"

# Create necessary directories
echo "? Creating directories..."
mkdir -p data logs uploads models src static

# Copy .env.example to .env if not exists
if [ ! -f .env ]; then
    echo "?? Creating .env file..."
    cp .env.example .env
    echo "??  Please edit .env file to configure API keys"
else
    echo "? .env file already exists"
fi

# Setup Python virtual environment (optional)
if command -v python3 &> /dev/null; then
    echo "? Creating Python virtual environment..."
    python3 -m venv venv
    
    echo "? Installing dependencies..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        echo "? Python dependencies installed successfully"
    else
        echo "?  Failed to create virtual environment"
    fi
fi

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "? Docker is available"
    
    # Build with Docker Compose
    if command -v docker-compose &> /dev/null; then
        echo "? Building Docker image..."
        docker-compose build
        echo "? Docker build completed"
    else
        echo "?  docker-compose not found"
    fi
else
    echo "?  Docker not found. Please install manually."
fi

echo ""
echo "? Setup completed!"
echo ""
echo "Usage:"
echo "  Docker: docker-compose up -d"
echo "  Python: source venv/bin/activate && uvicorn src.main:app --reload"
echo ""
echo "API URL: http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo ""
echo "??  Important: Configure OPENAI_API_KEY in .env file"