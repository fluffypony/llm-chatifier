"""Utility functions for llm-chatifier."""

import getpass
import logging
from urllib.parse import urlparse
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

try:
    import httpx
except ImportError:
    httpx = None


logger = logging.getLogger(__name__)


def try_connection(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> Tuple[bool, Optional['httpx.Response']]:
    """Try to connect to a URL and return success status.
    
    Args:
        url: URL to test
        headers: Optional headers to send
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (success, response) where success is bool and response is httpx.Response or None
    """
    if httpx is None:
        raise ImportError("httpx is required for try_connection")
    
    try:
        with httpx.Client(timeout=timeout, verify=False, follow_redirects=True) as client:
            # Try HEAD first (lighter), but immediately fall back to GET if 405 (Method Not Allowed)
            for method in ['HEAD', 'GET']:
                try:
                    response = client.request(method, url, headers=headers)
                    # Consider 2xx, 401, 403 as "success" (API exists)
                    # Also consider 405 on HEAD as success (means GET might work)
                    if response.status_code < 500:
                        if response.status_code == 405 and method == 'HEAD':
                            continue  # Try GET instead
                        logger.debug(f"{method} {url} -> {response.status_code}")
                        return True, response
                except httpx.RequestError:
                    continue
        
        return False, None
    
    except Exception as e:
        logger.debug(f"Connection to {url} failed: {e}")
        return False, None


def prompt_for_token() -> str:
    """Securely prompt user for API token.
    
    Returns:
        API token string
    """
    return getpass.getpass("Enter API token: ")


def build_base_url(host: str, port: int, use_https: bool = True) -> str:
    """Build a proper base URL from components.
    
    Args:
        host: Hostname or IP
        port: Port number
        use_https: Whether to use HTTPS
    
    Returns:
        Formatted base URL
    """
    scheme = "https" if use_https else "http"
    
    # Don't add port for standard ports
    if (use_https and port == 443) or (not use_https and port == 80):
        return f"{scheme}://{host}"
    else:
        return f"{scheme}://{host}:{port}"


def extract_error_message(response: 'httpx.Response') -> str:
    """Extract a meaningful error message from an HTTP response.
    
    Args:
        response: HTTP response object
    
    Returns:
        Error message string
    """
    try:
        # Try to parse JSON error
        data = response.json()
        if 'error' in data:
            if isinstance(data['error'], dict):
                return data['error'].get('message', str(data['error']))
            return str(data['error'])
        elif 'message' in data:
            return data['message']
        else:
            return f"HTTP {response.status_code}: {response.reason_phrase}"
    except Exception:
        # Fall back to status text
        return f"HTTP {response.status_code}: {response.reason_phrase}"


def is_auth_error(response: 'httpx.Response') -> bool:
    """Check if response indicates an authentication error.
    
    Args:
        response: HTTP response object
    
    Returns:
        True if auth error, False otherwise
    """
    if response.status_code in [401, 403]:
        return True
    
    try:
        text = response.text.lower()
        auth_keywords = ['unauthorized', 'forbidden', 'authentication', 'token', 'api key']
        return any(keyword in text for keyword in auth_keywords)
    except Exception:
        return False


def format_model_name(model: str) -> str:
    """Format model name for display.
    
    Args:
        model: Raw model name
    
    Returns:
        Formatted model name
    """
    # Remove common prefixes/suffixes for cleaner display
    prefixes_to_remove = ['text-', 'chat-', 'gpt-']
    suffixes_to_remove = ['-latest', '-preview']
    
    formatted = model
    for prefix in prefixes_to_remove:
        if formatted.startswith(prefix):
            formatted = formatted[len(prefix):]
    
    for suffix in suffixes_to_remove:
        if formatted.endswith(suffix):
            formatted = formatted[:-len(suffix)]
    
    return formatted


def parse_host_input(host_input: str) -> Tuple[str, Optional[int], bool]:
    """Parse user input that could be IP, hostname, or full URL.
    
    Args:
        host_input: User input (IP, hostname, or URL)
    
    Returns:
        Tuple of (hostname, port, use_https)
    """
    # If it looks like a URL, parse it
    if '://' in host_input:
        parsed = urlparse(host_input)
        hostname = parsed.hostname or parsed.netloc
        port = parsed.port
        use_https = parsed.scheme == 'https'
        return hostname, port, use_https
    
    # If it contains a port, split it
    if ':' in host_input and not host_input.count(':') > 1:  # Not IPv6
        try:
            hostname, port_str = host_input.rsplit(':', 1)
            port = int(port_str)
            return hostname, port, False  # Default to HTTP for IP:port format
        except ValueError:
            pass
    
    # Otherwise, treat as hostname/IP with no port specified
    # For domain names like "api.anthropic.com", default to HTTPS
    if '.' in host_input and not host_input.replace('.', '').replace('-', '').isdigit():
        return host_input, None, True  # Default to HTTPS for domain names
    else:
        return host_input, None, False  # Default to HTTP for IPs
