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
    
    def __init__(self, base_url: str, token: Optional[str] = None, model: Optional[str] = None, verbose: bool = False):
        if httpx is None:
            raise ImportError("httpx is required for API clients")
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.model = model
        self.verbose = verbose
        self.history: List[Dict[str, str]] = []
        self.client = httpx.Client(timeout=30.0, verify=False, follow_redirects=True)
    
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
    
    def test_with_simple_prompt(self) -> str:
        """Test API with a simple prompt to check if it works.
        
        This should be a lightweight test that checks if the API is accessible
        and whether authentication is working.
        
        Returns:
            Response from the API
            
        Raises:
            Exception: If the API call fails (auth errors, network errors, etc.)
        """
        # Default implementation: try to get models list as a test
        # Individual clients can override this for more appropriate tests
        try:
            models = self.get_models()
            if models:
                # Cache the models so we don't need to fetch again
                self._cached_models = models
                return f"API accessible. Found {len(models)} models."
            else:
                # Cache empty models list too
                self._cached_models = []
                return "API accessible but no models endpoint available."
        except Exception as e:
            # If getting models fails, that's our test result
            raise e
    
    def get_headers(self) -> Dict[str, str]:
        """Get common headers for requests."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def _log_http_details(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, 
                         payload: Optional[Dict] = None, response = None):
        """Log detailed HTTP request/response info in verbose mode."""
        if not self.verbose:
            return
            
        # Log request details
        logger.info(f"=== HTTP {method} Request ===")
        logger.info(f"URL: {url}")
        
        if headers:
            logger.info("Request Headers:")
            for key, value in headers.items():
                # Redact sensitive headers
                if key.lower() in ('authorization', 'x-api-key', 'anthropic-version'):
                    value = f"{value[:10]}..." if len(value) > 10 else "***"
                logger.info(f"  {key}: {value}")
        
        if payload:
            import json
            logger.info("Request Body:")
            logger.info(f"  {json.dumps(payload, indent=2)}")
        
        # Log response details
        if response:
            logger.info(f"=== HTTP Response ===")
            logger.info(f"Status: {response.status_code} {response.reason_phrase}")
            logger.info("Response Headers:")
            for key, value in response.headers.items():
                logger.info(f"  {key}: {value}")
            
            # Log response body (truncated for safety)
            try:
                response_text = response.text
                if len(response_text) > 1000:
                    response_text = response_text[:1000] + "... (truncated)"
                logger.info(f"Response Body:\n{response_text}")
            except Exception:
                logger.info("Response Body: <unable to decode>")
        
        logger.info("=" * 30)


class OpenAIClient(BaseClient):
    """Client for OpenAI-compatible APIs (OpenAI, llama.cpp, vLLM, etc.)."""
    
    def test_with_simple_prompt(self) -> str:
        """Test OpenAI API with a real API call if model is specified."""
        # If user specified a model, test it directly
        if self.model:
            try:
                # Test the specific model with a minimal request
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                }
                
                url = f"{self.base_url}/v1/chat/completions"
                headers = self.get_headers()
                
                self._log_http_details("POST", url, headers, payload)
                response = self.client.post(url, headers=headers, json=payload)
                self._log_http_details("POST", url, response=response)
                
                if response.status_code >= 400:
                    error_msg = extract_error_message(response)
                    if 'auth' in error_msg.lower() or 'token' in error_msg.lower():
                        raise Exception("Authentication required or invalid token")
                    elif 'model' in error_msg.lower():
                        raise Exception(f"Invalid model '{self.model}' - {error_msg}")
                    else:
                        raise Exception(f"API test failed: {error_msg}")
                
                return f"API test successful with model {self.model}"
            except Exception as e:
                raise e
        else:
            # Fall back to default behavior (get models)
            return super().test_with_simple_prompt()
    
    def test_connection(self) -> None:
        """Test connection by checking models endpoint."""
        url = f"{self.base_url}/v1/models"
        headers = self.get_headers()
        
        self._log_http_details("GET", url, headers)
        response = self.client.get(url, headers=headers)
        self._log_http_details("GET", url, response=response)
        
        if is_auth_error(response):
            raise Exception("Authentication required or invalid token")
        
        if response.status_code >= 400:
            raise Exception(extract_error_message(response))
    
    def get_models(self) -> List[str]:
        """Get available models from OpenAI-compatible API."""
        url = f"{self.base_url}/v1/models"
        headers = self.get_headers()
        
        self._log_http_details("GET", url, headers)
        response = self.client.get(url, headers=headers)
        self._log_http_details("GET", url, response=response)
        
        if response.status_code >= 400:
            logger.warning(f"Failed to get models: {extract_error_message(response)}")
            raise Exception(f"Unable to fetch models list: {extract_error_message(response)}")
        
        try:
            # Check if response is empty or not JSON
            if not response.text.strip():
                logger.warning("Empty response from models endpoint")
                raise Exception("Models endpoint returned empty response")
            
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            if not models:
                raise Exception("No models found in API response")
            return sorted(models)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse models response: {e}")

            raise Exception("Models endpoint returned invalid JSON")
    
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


class OpenRouterClient(BaseClient):
    """Client for OpenRouter API."""
    
    def test_with_simple_prompt(self) -> str:
        """Test OpenRouter API with a real API call if model is specified."""
        # If user specified a model, test it directly
        if self.model:
            try:
                # Test the specific model with a minimal request
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                }
                
                url = f"{self.base_url}/api/v1/chat/completions"
                headers = self.get_headers()
                
                self._log_http_details("POST", url, headers, payload)
                response = self.client.post(url, headers=headers, json=payload)
                self._log_http_details("POST", url, response=response)
                
                if response.status_code >= 400:
                    error_msg = extract_error_message(response)
                    if 'auth' in error_msg.lower() or 'token' in error_msg.lower():
                        raise Exception("Authentication required or invalid token")
                    elif 'model' in error_msg.lower():
                        raise Exception(f"Invalid model '{self.model}' - {error_msg}")
                    else:
                        raise Exception(f"API test failed: {error_msg}")
                
                return f"API test successful with model {self.model}"
            except Exception as e:
                raise e
        else:
            # Fall back to default behavior (get models)
            return super().test_with_simple_prompt()
    
    def test_connection(self) -> None:
        """Test connection by checking models endpoint."""
        url = f"{self.base_url}/api/v1/models"
        headers = self.get_headers()
        
        self._log_http_details("GET", url, headers)
        response = self.client.get(url, headers=headers)
        self._log_http_details("GET", url, response=response)
        
        if response.status_code >= 400:
            raise Exception(extract_error_message(response))
        
        # OpenRouter's models endpoint is public, but we need auth for actual usage
        # If we don't have a token, warn that auth is required
        if not self.token:
            raise Exception("Authentication required or invalid token")
    
    def get_models(self) -> List[str]:
        """Get available models from OpenRouter API."""
        url = f"{self.base_url}/api/v1/models"
        headers = self.get_headers()
        
        self._log_http_details("GET", url, headers)
        response = self.client.get(url, headers=headers)
        self._log_http_details("GET", url, response=response)
        
        if response.status_code >= 400:
            logger.warning(f"Failed to get models: {extract_error_message(response)}")
            raise Exception(f"Unable to fetch models list: {extract_error_message(response)}")
        
        try:
            # Check if response is empty or not JSON
            response_text = response.text.strip()
            if not response_text:
                logger.warning("Empty response from models endpoint")
                raise Exception("No models found in API response")
            
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            if not models:
                logger.warning("No models found in API response")
                raise Exception("No models found in API response")
            return sorted(models)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse models response: {e}")
            raise Exception("Models endpoint returned invalid JSON")
    
    def send_message(self, text: str) -> str:
        """Send message using OpenRouter chat completions format."""
        self.history.append({"role": "user", "content": text})
        
        payload = {
            "model": self.model,
            "messages": self.history,
            "stream": False
        }
        
        url = f"{self.base_url}/api/v1/chat/completions"
        headers = self.get_headers()
        
        self._log_http_details("POST", url, headers, payload)
        response = self.client.post(url, headers=headers, json=payload)
        self._log_http_details("POST", url, response=response)
        
        if response.status_code >= 400:
            raise Exception(extract_error_message(response))
        
        try:
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except (KeyError, json.JSONDecodeError) as e:
            raise Exception(f"Invalid OpenRouter response format: {e}")


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
        
        self._log_http_details("GET", url)
        response = self.client.get(url)
        self._log_http_details("GET", url, response=response)
        
        if response.status_code >= 400:
            logger.warning(f"Failed to get models: {extract_error_message(response)}")
            raise Exception(f"Unable to fetch models from Ollama: {extract_error_message(response)}")
        
        try:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            if not models:
                raise Exception("No models found in Ollama. Please install a model first with 'ollama pull <model-name>'")
            return sorted(models)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse models response: {e}")
            raise Exception("Ollama returned invalid response format")
    
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
    
    def test_with_simple_prompt(self) -> str:
        """Test Anthropic API with a real API call to check authentication."""
        if not self.token:
            raise Exception("Authentication required - x-api-key header is required")
        
        # Use user-specified model or fallback to a known working model
        test_model = self.model or "claude-3-sonnet-20240229"
        
        # Make a minimal test call to verify authentication and model work
        payload = {
            "model": test_model,
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Hi"}]
        }
        
        url = f"{self.base_url}/v1/messages"
        headers = self.get_headers()
        
        self._log_http_details("POST", url, headers, payload)
        response = self.client.post(url, headers=headers, json=payload)
        self._log_http_details("POST", url, response=response)
        
        if response.status_code >= 400:
            error_msg = extract_error_message(response)
            if 'auth' in error_msg.lower() or 'api key' in error_msg.lower():
                raise Exception("Authentication required - invalid or missing x-api-key")
            elif 'model' in error_msg.lower() and self.model:
                raise Exception(f"Invalid model '{self.model}' - {error_msg}")
            else:
                raise Exception(f"API test failed: {error_msg}")
        
        try:
            data = response.json()
            reply = data["content"][0]["text"]
            return f"API test successful: {reply}"
        except (KeyError, json.JSONDecodeError) as e:
            return "API test successful: Authentication verified"
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for Anthropic API requests."""
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        if self.token:
            headers["x-api-key"] = self.token
        return headers
    
    def test_connection(self) -> None:
        """Test connection by checking authentication."""
        # For Anthropic, we just check if we have the required x-api-key
        # The actual connection will be tested when sending the first message
        if not self.token:
            raise Exception("Authentication required - x-api-key header is required")
    
    def get_models(self) -> List[str]:
        """Anthropic doesn't have a public models endpoint."""
        # Anthropic doesn't provide a models list endpoint
        # Users must specify a model manually
        return []
    
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
    
    def __init__(self, base_url: str, token: Optional[str] = None, model: Optional[str] = None, verbose: bool = False):
        super().__init__(base_url, token, model, verbose)
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
            raise Exception("API key is required to fetch Gemini models")
            
        url = f"{self.base_url}/v1beta/models?key={self.token}"
        
        self._log_http_details("GET", url)
        response = self.client.get(url)
        self._log_http_details("GET", url, response=response)
        
        if response.status_code >= 400:
            logger.warning(f"Failed to get models: {extract_error_message(response)}")
            raise Exception(f"Unable to fetch Gemini models: {extract_error_message(response)}")
        
        try:
            data = response.json()
            models = []
            for model in data.get("models", []):
                name = model.get("name", "")
                if name.startswith("models/"):
                    name = name[7:]  # Remove "models/" prefix
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    models.append(name)
            if not models:
                raise Exception("No compatible Gemini models found")
            return sorted(models)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse models response: {e}")
            raise Exception("Gemini returned invalid response format")
    
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
        """Try to get models from Cohere API."""
        # Cohere might have a models endpoint, let's try it
        if not self.token:
            raise Exception("Authentication required for Cohere")
        
        url = f"{self.base_url}/v1/models"
        headers = self.get_headers()
        
        self._log_http_details("GET", url, headers)
        response = self.client.get(url, headers=headers)
        self._log_http_details("GET", url, response=response)
        
        if response.status_code >= 400:
            # If models endpoint doesn't exist, return empty list
            if response.status_code == 404:
                return []
            else:
                error_msg = extract_error_message(response)
                raise Exception(f"Failed to get models: {error_msg}")
        
        try:
            all_models = []
            next_page_token = None
            
            while True:
                data = response.json()
                
                # Extract models from response
                models_data = []
                if isinstance(data, list):
                    models_data = data
                elif "models" in data:
                    models_data = data["models"]
                elif "data" in data:
                    models_data = data["data"]
                
                # Extract model names
                for model in models_data:
                    if isinstance(model, dict):
                        name = model.get("name", model.get("id", str(model)))
                        if name:
                            all_models.append(name)
                
                # Check for pagination
                next_page_token = data.get("next_page_token")
                if not next_page_token:
                    break
                
                # Fetch next page
                next_url = f"{url}?page_token={next_page_token}"
                self._log_http_details("GET", next_url, headers)
                response = self.client.get(next_url, headers=headers)
                self._log_http_details("GET", next_url, response=response)
                
                if response.status_code >= 400:
                    break
            
            return sorted(all_models) if all_models else []
        except json.JSONDecodeError:
            return []
    
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


def create_client(api_type: str, base_url: Optional[str] = None, token: Optional[str] = None, model: Optional[str] = None, verbose: bool = False) -> BaseClient:
    """Factory function to create appropriate client.
    
    Args:
        api_type: Type of API ('openai', 'openrouter', 'ollama', 'anthropic', 'gemini', 'cohere', 'generic')
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
        elif api_type == 'openrouter':
            base_url = "https://openrouter.ai"
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
        return OpenAIClient(base_url, token, model, verbose)
    elif api_type == 'openrouter':
        return OpenRouterClient(base_url, token, model, verbose)
    elif api_type == 'ollama':
        return OllamaClient(base_url, token, model, verbose)
    elif api_type == 'anthropic':
        return AnthropicClient(base_url, token, model, verbose)
    elif api_type == 'gemini':
        return GeminiClient(base_url, token, model, verbose)
    elif api_type == 'cohere':
        return CohereClient(base_url, token, model, verbose)
    elif api_type == 'generic':
        return GenericClient(base_url, token, model, verbose)
    else:
        raise ValueError(f"Unknown API type: {api_type}")
