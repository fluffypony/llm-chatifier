"""API detection module for llm-chatifier."""

import logging
from typing import Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    httpx = None

from .utils import try_connection, build_base_url


logger = logging.getLogger(__name__)

# Common API endpoints for detection
API_ENDPOINTS = {
    'openai': ['/v1/models', '/v1/chat/completions', '/chat/completions'],
    'openrouter': ['/api/v1/models', '/api/v1/chat/completions'],
    'anthropic': ['/v1/messages', '/v1/models'],
    'ollama': ['/api/tags', '/api/generate'],
    'gemini': ['/v1beta/models', '/v1beta/models/gemini-pro:generateContent'],
    'cohere': ['/v1/chat'],
    'generic': ['/chat', '/api/chat', '/message', '/api/message']
}

# Common ports to try if none specified
DEFAULT_PORTS = [8080, 8000, 3000, 5000, 11434, 80, 443]

# Ports for domain names (standard web ports first)
DOMAIN_PORTS = [443, 80, 8080, 8000, 3000, 5000]


def detect_api(host: str, port: Optional[int] = None, prefer_https: bool = False, token: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Detect API type by testing common endpoints.
    
    Args:
        host: Hostname or IP address
        port: Port number (if None, tries common ports)
        prefer_https: Whether to prefer HTTPS (if URL was provided with https://)
        token: Optional API token for authentication during detection
    
    Returns:
        Dict with 'type', 'host', 'port', 'base_url' if detected, None otherwise
    """
    # Domain-specific API type hints
    domain_hints = {
        'openrouter.ai': 'openrouter',
        'api.anthropic.com': 'anthropic', 
        'api.openai.com': 'openai',
        'api.perplexity.ai': 'openai',
        'generativelanguage.googleapis.com': 'gemini',
        'api.cohere.ai': 'cohere'
    }
    
    # If we have a domain hint, try that API type first
    priority_apis = []
    if host.lower() in domain_hints:
        priority_apis = [domain_hints[host.lower()]]
    if port:
        ports_to_try = [port]
    else:
        # Use standard web ports for domain names, otherwise use default ports
        if '.' in host and not host.replace('.', '').replace('-', '').isdigit():
            ports_to_try = DOMAIN_PORTS
        else:
            ports_to_try = DEFAULT_PORTS
    
    for test_port in ports_to_try:
        logger.debug(f"Trying port {test_port}")
        
        # Try protocols in order of preference
        protocols = [True, False] if prefer_https else [True, False]
        tls_failed_on_port = False
        
        for use_https in protocols:
            # Skip HTTPS if we already had a TLS failure on this port
            if use_https and tls_failed_on_port:
                logger.debug(f"Skipping HTTPS on port {test_port} due to previous TLS failure")
                continue
                
            base_url = build_base_url(host, test_port, use_https)
            logger.debug(f"Testing {base_url}")
            
            # Test each API type (priority APIs first, then others)
            apis_to_test = priority_apis + [api for api in API_ENDPOINTS.keys() if api not in priority_apis]
            for api_type in apis_to_test:
                endpoints = API_ENDPOINTS[api_type]
                for endpoint in endpoints:
                    try:
                        url = f"{base_url}{endpoint}"
                        
                        # Prepare headers based on API type and token
                        headers = None
                        if token:
                            if api_type == 'anthropic':
                                headers = {
                                    "x-api-key": token,
                                    "anthropic-version": "2023-06-01"
                                }
                            elif api_type == 'gemini':
                                url = f"{url}?key={token}"  # Gemini uses query param
                            elif api_type in ['openai', 'openrouter', 'cohere']:
                                headers = {"Authorization": f"Bearer {token}"}
                        
                        success, response, is_tls_failure = try_connection(url, headers)
                        
                        if success:
                            logger.debug(f"Found {api_type} API at {base_url}")
                            return {
                                'type': api_type,
                                'host': host,
                                'port': test_port,
                                'base_url': base_url,
                                'use_https': use_https,
                                'detected_endpoint': endpoint
                            }
                        
                        # If we got a TLS failure on HTTPS, mark it and stop trying HTTPS for this port
                        if is_tls_failure and use_https:
                            logger.debug(f"TLS handshake failed on port {test_port}, will skip HTTPS for remaining endpoints")
                            tls_failed_on_port = True
                            break  # Break out of endpoint loop to try next protocol
                        
                        logger.debug(f"Failed to connect to {url}")
                    except Exception as e:
                        logger.debug(f"Error testing {url}: {e}")
                        continue  # Move on to next endpoint
                
                # If TLS failed, break out of API type loop too
                if tls_failed_on_port and use_https:
                    break
    
    logger.debug(f"No API detected on {host}")
    return None


def detect_specific_api(base_url: str, api_type: str) -> bool:
    """Test if a specific API type is available at the given URL.
    
    Args:
        base_url: Base URL to test
        api_type: API type to test for
    
    Returns:
        True if API type is detected, False otherwise
    """
    if api_type not in API_ENDPOINTS:
        return False
    
    for endpoint in API_ENDPOINTS[api_type]:
        url = f"{base_url}{endpoint}"
        success, _, _ = try_connection(url)
        if success:
            return True
    
    return False


def get_api_info(api_type: str, base_url: str, detected_endpoint: Optional[str] = None) -> Dict[str, str]:
    """Get API-specific information for client creation.
    
    Args:
        api_type: Type of API
        base_url: Base URL
        detected_endpoint: The specific endpoint that was detected during discovery
    
    Returns:
        Dict with API configuration info
    """
    # This would be expanded based on specific API requirements
    info = {
        'type': api_type,
        'base_url': base_url,
    }
    
    # If we have a detected endpoint that looks like a chat endpoint, use it
    if detected_endpoint and ('chat' in detected_endpoint or 'completions' in detected_endpoint):
        info['chat_endpoint'] = detected_endpoint
        # Derive models endpoint from chat endpoint
        if detected_endpoint == '/chat/completions':
            info['models_endpoint'] = '/models'
        elif detected_endpoint == '/v1/chat/completions':
            info['models_endpoint'] = '/v1/models'
        elif detected_endpoint.startswith('/api/v1/'):
            info['models_endpoint'] = '/api/v1/models'
        else:
            info['models_endpoint'] = '/models'
    elif detected_endpoint == '/v1/models' and 'perplexity.ai' in base_url.lower():
        # Special case: Perplexity detected via /v1/models but uses /chat/completions
        info['chat_endpoint'] = '/chat/completions'
        info['models_endpoint'] = '/models'
    else:
        # Add API-specific defaults
        if api_type == 'openai':
            info['chat_endpoint'] = '/v1/chat/completions'
            info['models_endpoint'] = '/v1/models'
        elif api_type == 'ollama':
            info['chat_endpoint'] = '/api/generate'
            info['models_endpoint'] = '/api/tags'
        elif api_type == 'anthropic':
            info['chat_endpoint'] = '/v1/messages'
            info['models_endpoint'] = '/v1/models'
        else:  # generic
            info['chat_endpoint'] = '/chat'
            info['models_endpoint'] = '/models'
    
    return info
