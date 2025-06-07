"""API client implementations for llm-chatifier."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

from .utils import extract_error_message, is_auth_error, build_base_url


logger = logging.getLogger(__name__)


class BaseClient(ABC):
    """Base class for all API clients."""
    
    def __init__(self, base_url: str, token: Optional[str] = None, model: Optional[str] = None):
        if httpx is None:
            raise ImportError("httpx is required for API clients")
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.model = model
        self.history: List[Dict[str, str]] = []
        self.client = httpx.Client(timeout=30.0, verify=False)
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
    
    @abstractmethod
    def send_message(self, text: str) -> str:
        """Send a message and return the response."""
        pass
    
    @abstractmethod
    def test_connection(self) -> None:
        """Test connection to the API. Raises exception on failure."""
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        """Get list of available models. Returns empty list if not supported."""
        pass
    
    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
    
    def get_headers(self) -> Dict[str, str]:
        """Get common headers for requests."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


class OpenAIClient(BaseClient):
    """Client for OpenAI-compatible APIs (OpenAI, llama.cpp, vLLM, etc.)."""
    
    def test_connection(self) -> None:
        """Test connection by checking models endpoint."""
        url = f"{self.base_url}/v1/models"
        response = self.client.get(url, headers=self.get_headers())
        
        if is_auth_error(response):
            raise Exception("Authentication required or invalid token")
        
        if response.status_code >= 400:
            raise Exception(extract_error_message(response))
    
    def get_models(self) -> List[str]:
        """Get available models from OpenAI-compatible API."""
        url = f"{self.base_url}/v1/models"
        response = self.client.get(url, headers=self.get_headers())
        
        if response.status_code >= 400:
            logger.warning(f"Failed to get models: {extract_error_message(response)}")
            return ["gpt-3.5-turbo", "gpt-4"]  # Fallback models
        
        try:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return sorted(models) if models else ["gpt-3.5-turbo", "gpt-4"]
        except Exception as e:
            logger.warning(f"Failed to parse models response: {e}")
            return ["gpt-3.5-turbo", "gpt-4"]
    
    def send_message(self, text: str) -> str:
        """Send message using OpenAI chat completions format."""
        # Add user message to history
        self.history.append({"role": "user", "content": text})
        
        # Prepare request
        model_to_use = self.model or "gpt-3.5-turbo"  # Use specified model or default
        payload = {
            "model": model_to_use,
            "messages": self.history,
            "stream": False
        }
        
        url = f"{self.base_url}/v1/chat/completions"
        response = self.client.post(url, headers=self.get_headers(), json=payload)
        
        if response.status_code >= 400:
            error_msg = extract_error_message(response)
            if is_auth_error(response):
                raise Exception(f"Authentication failed: {error_msg}")
            raise Exception(f"API error: {error_msg}")
        
        try:
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            
            # Add assistant response to history
            self.history.append({"role": "assistant", "content": reply})
            
            return reply
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise Exception(f"Invalid response format: {e}")


class OllamaClient(BaseClient):
    """Client for Ollama API."""
    
    def test_connection(self) -> None:
        """Test connection by checking tags endpoint."""
        url = f"{self.base_url}/api/tags"
        response = self.client.get(url)
        
        if response.status_code >= 400:
            raise Exception(extract_error_message(response))
    
    def get_models(self) -> List[str]:
        """Get available models from Ollama API."""
        url = f"{self.base_url}/api/tags"
        response = self.client.get(url)
        
        if response.status_code >= 400:
            logger.warning(f"Failed to get models: {extract_error_message(response)}")
            return ["llama2"]  # Fallback model
        
        try:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return sorted(models) if models else ["llama2"]
        except Exception as e:
            logger.warning(f"Failed to parse models response: {e}")
            return ["llama2"]
    
    def send_message(self, text: str) -> str:
        """Send message using Ollama generate format."""
        # Build conversation context
        context = ""
        for msg in self.history:
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            else:
                context += f"Assistant: {msg['content']}\n"
        
        prompt = context + f"User: {text}\nAssistant: "
        
        model_to_use = self.model or "llama2"  # Use specified model or default
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": False
        }
        
        url = f"{self.base_url}/api/generate"
        response = self.client.post(url, json=payload)
        
        if response.status_code >= 400:
            raise Exception(extract_error_message(response))
        
        try:
            data = response.json()
            reply = data["response"]
            
            # Add to history
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            
            return reply
        except (KeyError, json.JSONDecodeError) as e:
            raise Exception(f"Invalid response format: {e}")


class AnthropicClient(BaseClient):
    """Client for Anthropic Claude API."""
    
    def get_headers(self) -> Dict[str, str]:
        """Get Anthropic-specific headers."""
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        if self.token:
            headers["x-api-key"] = self.token
        return headers
    
    def test_connection(self) -> None:
        """Test connection by making a simple request."""
        # Anthropic doesn't have a models endpoint, so we'll just try a minimal message
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "Hi"}]
        }
        
        url = f"{self.base_url}/v1/messages"
        response = self.client.post(url, headers=self.get_headers(), json=payload)
        
        if is_auth_error(response):
            raise Exception("Authentication required or invalid API key")
        
        if response.status_code >= 400:
            raise Exception(extract_error_message(response))
    
    def get_models(self) -> List[str]:
        """Get available Anthropic models (hardcoded list)."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
    
    def send_message(self, text: str) -> str:
        """Send message using Anthropic messages format."""
        # Convert history to Anthropic format (no system messages in history)
        messages = []
        for msg in self.history:
            if msg["role"] != "system":
                messages.append(msg)
        
        # Add current message
        messages.append({"role": "user", "content": text})
        
        model_to_use = self.model or "claude-3-sonnet-20240229"  # Use specified model or default
        payload = {
            "model": model_to_use,
            "max_tokens": 4000,
            "messages": messages
        }
        
        url = f"{self.base_url}/v1/messages"
        response = self.client.post(url, headers=self.get_headers(), json=payload)
        
        if response.status_code >= 400:
            error_msg = extract_error_message(response)
            if is_auth_error(response):
                raise Exception(f"Authentication failed: {error_msg}")
            raise Exception(f"API error: {error_msg}")
        
        try:
            data = response.json()
            reply = data["content"][0]["text"]
            
            # Add to history
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            
            return reply
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise Exception(f"Invalid response format: {e}")


class GenericClient(BaseClient):
    """Generic client that tries common chat API patterns."""
    
    def __init__(self, base_url: str, token: Optional[str] = None):
        super().__init__(base_url, token)
        self.chat_endpoint = None
        self._discover_endpoints()
    
    def _discover_endpoints(self):
        """Try to discover working endpoints."""
        test_endpoints = ['/chat', '/api/chat', '/message', '/api/message', '/completion']
        
        for endpoint in test_endpoints:
            url = f"{self.base_url}{endpoint}"
            # Try a simple GET first
            response = self.client.get(url, headers=self.get_headers())
            if response.status_code < 500:  # Any response means endpoint exists
                self.chat_endpoint = endpoint
                logger.debug(f"Found chat endpoint: {endpoint}")
                break
        
        if not self.chat_endpoint:
            self.chat_endpoint = '/chat'  # Default fallback
    
    def test_connection(self) -> None:
        """Test connection to generic endpoint."""
        url = f"{self.base_url}{self.chat_endpoint}"
        response = self.client.get(url, headers=self.get_headers())
        
        if response.status_code >= 500:
            raise Exception(f"Server error: {response.status_code}")
    
    def get_models(self) -> List[str]:
        """Get available models (no standard for generic APIs)."""
        return []  # Generic APIs don't have a standard models endpoint
    
    def send_message(self, text: str) -> str:
        """Send message using generic patterns."""
        # Try multiple common payload formats
        payloads = [
            # OpenAI-like
            {
                "messages": [{"role": "user", "content": text}],
                "model": "default"
            },
            # Simple text
            {
                "message": text,
                "user": "user"
            },
            # Direct text
            {
                "text": text
            },
            # Query format
            {
                "query": text
            }
        ]
        
        url = f"{self.base_url}{self.chat_endpoint}"
        
        for payload in payloads:
            try:
                response = self.client.post(url, headers=self.get_headers(), json=payload)
                
                if response.status_code < 400:
                    data = response.json()
                    
                    # Try common response patterns
                    reply = None
                    for key in ['response', 'message', 'text', 'reply', 'answer']:
                        if key in data:
                            reply = data[key]
                            if isinstance(reply, dict) and 'content' in reply:
                                reply = reply['content']
                            break
                    
                    if reply:
                        # Add to history
                        self.history.append({"role": "user", "content": text})
                        self.history.append({"role": "assistant", "content": str(reply)})
                        return str(reply)
                
            except Exception as e:
                logger.debug(f"Payload {payload} failed: {e}")
                continue
        
        raise Exception("Unable to get response from generic API")


class GeminiClient(BaseClient):
    """Client for Google Gemini API."""
    
    def test_connection(self) -> None:
        """Test connection by checking models endpoint."""
        if not self.token:
            raise Exception("API key is required for Gemini")
        
        url = f"{self.base_url}/v1beta/models?key={self.token}"
        response = self.client.get(url)
        
        if response.status_code >= 400:
            raise Exception(extract_error_message(response))
    
    def get_models(self) -> List[str]:
        """Get available models from Gemini API."""
        if not self.token:
            return ["gemini-pro"]  # Fallback
            
        url = f"{self.base_url}/v1beta/models?key={self.token}"
        response = self.client.get(url)
        
        if response.status_code >= 400:
            logger.warning(f"Failed to get models: {extract_error_message(response)}")
            return ["gemini-pro", "gemini-pro-vision"]  # Fallback models
        
        try:
            data = response.json()
            models = []
            for model in data.get("models", []):
                name = model.get("name", "")
                if name.startswith("models/"):
                    name = name[7:]  # Remove "models/" prefix
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    models.append(name)
            return sorted(models) if models else ["gemini-pro"]
        except Exception as e:
            logger.warning(f"Failed to parse models response: {e}")
            return ["gemini-pro"]
    
    def send_message(self, text: str) -> str:
        """Send message using Gemini generateContent format."""
        if not self.token:
            raise Exception("API key is required for Gemini")
        
        model_to_use = self.model or "gemini-pro"
        
        # Build conversation context in Gemini format
        contents = []
        for msg in self.history:
            role = "user" if msg["role"] == "user" else "model" 
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        # Add current message
        contents.append({
            "role": "user",
            "parts": [{"text": text}]
        })
        
        payload = {
            "contents": contents
        }
        
        url = f"{self.base_url}/v1beta/models/{model_to_use}:generateContent?key={self.token}"
        response = self.client.post(url, json=payload)
        
        if response.status_code >= 400:
            error_msg = extract_error_message(response)
            raise Exception(f"Gemini API error: {error_msg}")
        
        try:
            data = response.json()
            reply = data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Add to history
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            
            return reply
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise Exception(f"Invalid Gemini response format: {e}")


class CohereClient(BaseClient):
    """Client for Cohere API."""
    
    def test_connection(self) -> None:
        """Test connection by making a simple chat request."""
        if not self.token:
            raise Exception("API token is required for Cohere")
        
        # Test with minimal payload
        url = f"{self.base_url}/v1/chat"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        payload = {"message": "Hello", "model": "command"}
        
        response = self.client.post(url, headers=headers, json=payload)
        
        if response.status_code >= 400:
            raise Exception(extract_error_message(response))
    
    def get_models(self) -> List[str]:
        """Get available Cohere models (hardcoded list)."""
        return [
            "command",
            "command-light", 
            "command-nightly",
            "command-r",
            "command-r-plus"
        ]
    
    def send_message(self, text: str) -> str:
        """Send message using Cohere chat format."""
        if not self.token:
            raise Exception("API token is required for Cohere")
        
        model_to_use = self.model or "command"
        
        # Build conversation history for Cohere
        chat_history = []
        for msg in self.history:
            role = "USER" if msg["role"] == "user" else "CHATBOT"
            chat_history.append({
                "role": role,
                "message": msg["content"]
            })
        
        payload = {
            "message": text,
            "model": model_to_use,
            "chat_history": chat_history
        }
        
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        url = f"{self.base_url}/v1/chat"
        response = self.client.post(url, headers=headers, json=payload)
        
        if response.status_code >= 400:
            error_msg = extract_error_message(response)
            raise Exception(f"Cohere API error: {error_msg}")
        
        try:
            data = response.json()
            reply = data["text"]
            
            # Add to history
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            
            return reply
        except (KeyError, json.JSONDecodeError) as e:
            raise Exception(f"Invalid Cohere response format: {e}")


def create_client(api_type: str, base_url: Optional[str] = None, token: Optional[str] = None, model: Optional[str] = None) -> BaseClient:
    """Factory function to create appropriate client.
    
    Args:
        api_type: Type of API ('openai', 'ollama', 'anthropic', 'gemini', 'cohere', 'generic')
        base_url: Base URL for the API
        token: API token/key
        model: Model name to use
    
    Returns:
        Appropriate client instance
    """
    if not base_url:
        # Construct base URL based on API type and defaults
        if api_type == 'openai':
            base_url = "https://api.openai.com"
        elif api_type == 'anthropic':
            base_url = "https://api.anthropic.com"
        elif api_type == 'gemini':
            base_url = "https://generativelanguage.googleapis.com"
        elif api_type == 'cohere':
            base_url = "https://api.cohere.ai"
        elif api_type == 'ollama':
            base_url = "http://localhost:11434"
        else:
            base_url = "http://localhost:8080"
    
    if api_type == 'openai':
        return OpenAIClient(base_url, token, model)
    elif api_type == 'ollama':
        return OllamaClient(base_url, token, model)
    elif api_type == 'anthropic':
        return AnthropicClient(base_url, token, model)
    elif api_type == 'gemini':
        return GeminiClient(base_url, token, model)
    elif api_type == 'cohere':
        return CohereClient(base_url, token, model)
    elif api_type == 'generic':
        return GenericClient(base_url, token, model)
    else:
        raise ValueError(f"Unknown API type: {api_type}")
