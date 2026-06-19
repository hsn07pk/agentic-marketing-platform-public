#!/bin/bash
# scripts/setup_ollama.sh

echo "Setting up Ollama for Agentic AI Platform..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Start Ollama service
ollama serve &
sleep 5

# Pull recommended models based on available resources
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

if [ -z "$VRAM" ]; then
    echo "No GPU detected, using CPU models"
    ollama pull mistral:7b
    ollama pull tinyllama
else
    echo "GPU detected with ${VRAM}MB VRAM"
    
    if [ "$VRAM" -gt 40000 ]; then
        echo "Pulling large models..."
        ollama pull llama3:70b
        ollama pull mixtral:8x7b
    elif [ "$VRAM" -gt 26000 ]; then
        echo "Pulling medium models..."
        ollama pull mixtral:8x7b
        ollama pull llama3:8b
    elif [ "$VRAM" -gt 8000 ]; then
        echo "Pulling small models..."
        ollama pull llama3:8b
        ollama pull mistral:7b
    else
        echo "Limited VRAM, pulling lightweight models..."
        ollama pull mistral:7b
        ollama pull tinyllama
    fi
fi

echo "Ollama setup complete!"
ollama list