import requests
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    """Ollama API client for local LLM inference"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.model = None
        
    def set_model(self, model_name: str):
        """Set the model to use"""
        self.model = model_name
        
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """Generate text using Ollama"""
        
        model_to_use = model or self.model
        
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                stream=stream
            )
            
            if stream:
                return self._handle_stream(response)
            else:
                result = response.json()
                if 'error' in result:
                    raise RuntimeError(f"Ollama error: {result['error']}")
                return result.get('response', '')
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    def _handle_stream(self, response):
        """Handle streaming response"""
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'response' in data:
                    full_response += data['response']
        return full_response
    
    def list_models(self) -> list:
        """List available models"""
        try:
            response = requests.get(f"{self.host}/api/tags")
            return response.json().get('models', [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model_name: str):
        """Pull a model from Ollama library"""
        payload = {"name": model_name}
        response = requests.post(
            f"{self.host}/api/pull",
            json=payload,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'status' in data:
                    logger.info(f"Pull status: {data['status']}")
                    
    def select_best_model(task_type: str, available_vram: int) -> str:
        """Select best model based on task and resources"""
        
        model_requirements = {
            'mixtral:8x7b': 26000,  # MB
            'llama3:70b': 40000,
            'llama3:8b': 8000,
            'mistral:7b': 4000,
            'tinyllama': 1000
        }
        
        task_preferences = {
            'content_generation': ['mixtral:8x7b', 'llama3:8b', 'mistral:7b'],
            'safety_validation': ['llama3:70b', 'mixtral:8x7b', 'mistral:7b'],
            'quick_draft': ['mistral:7b', 'tinyllama']
        }
        
        preferred_models = task_preferences.get(task_type, ['mistral:7b'])
        
        for model in preferred_models:
            if model_requirements.get(model, 0) <= available_vram:
                return model
        
        return 'tinyllama'  # Fallback to smallest model