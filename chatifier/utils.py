"""Utility functions for llm-chatifier."""

import getpass
import logging
import os
import re
from urllib.parse import urlparse
from typing import Dict, Optional, Tuple, TYPE_CHECKING, List

if TYPE_CHECKING:
    import httpx

try:
    import httpx
except ImportError:
    httpx = None


logger = logging.getLogger(__name__)


def try_connection(url: str, headers: Optional[Dict[str, str]] = None,
                   timeout: float = 5.0) -> Tuple[bool, Optional['httpx.Response'], bool]:
    """Try to connect to a URL and return success status.

    Args:
        url: URL to test
        headers: Optional headers to send
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, response, is_tls_failure) where:
        - success is bool
        - response is httpx.Response or None
        - is_tls_failure is bool indicating if this was a TLS handshake failure
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
                        return True, response, False
                except httpx.RequestError as req_err:
                    # Check if this is a TLS-related error
                    error_str = str(req_err).lower()
                    is_tls_error = (any(tls_term in error_str for tls_term in [
                        'ssl', 'tls', 'handshake', 'certificate', 'cert', 'timeout'
                    ]) and 'https' in url.lower())

                    if is_tls_error:
                        logger.debug(f"TLS error on {method} {url}: {req_err}")
                        return False, None, True
                    continue

        return False, None, False

    except Exception as e:
        error_str = str(e).lower()
        is_tls_error = any(tls_term in error_str for tls_term in [
            'ssl', 'tls', 'handshake', 'certificate', 'cert', 'timeout'
        ]) and 'https' in url.lower()
        logger.debug(f"Connection to {url} failed: {e}")
        return False, None, is_tls_error


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


def find_api_key_in_env(api_type: str, base_url: str = "") -> Optional[str]:
    """Smart environment variable detection for API keys.

    Searches environment variables for API keys that match the detected API type.
    Works across Linux, Windows, and macOS.

    Args:
        api_type: Detected API type (openai, anthropic, etc.)
        base_url: Base URL to help with matching (optional)

    Returns:
        Best matching API key or None if not found
    """
    keys = find_all_api_keys_in_env(api_type, base_url)
    return keys[0] if keys else None


def find_all_api_keys_in_env(api_type: str, base_url: str = "") -> List[str]:
    """Smart environment variable detection for API keys - returns all matches in priority order.

    Searches environment variables for API keys that match the detected API type.
    Works across Linux, Windows, and macOS.

    Args:
        api_type: Detected API type (openai, anthropic, etc.)
        base_url: Base URL to help with matching (optional)

    Returns:
        List of API keys in priority order (best match first)
    """
    # Get all environment variables
    env_vars = dict(os.environ)

    # Filter for variables that look like API keys
    api_key_vars = {}
    for name, value in env_vars.items():
        name_upper = name.upper()
        if ('API_KEY' in name_upper or
            'APIKEY' in name_upper or
            'API_TOKEN' in name_upper or
            ('TOKEN' in name_upper and len(value) > 10)):  # Basic sanity check
            api_key_vars[name] = value

    if not api_key_vars:
        return []

    # Score each API key variable based on how well it matches
    scored_vars = []

    for var_name, var_value in api_key_vars.items():
        score = 0
        var_name_lower = var_name.lower()

        # Primary matching: API type name in variable name
        if api_type.lower() in var_name_lower:
            score += 100

        # Secondary matching: related terms
        api_terms = {
            'openai': ['openai', 'gpt'],
            'anthropic': ['anthropic', 'claude'],
            'openrouter': ['openrouter', 'router'],
            'gemini': ['gemini', 'google', 'gcp', 'vertex'],
            'cohere': ['cohere', 'co'],
            'ollama': ['ollama']
        }

        if api_type in api_terms:
            for term in api_terms[api_type]:
                if term in var_name_lower:
                    score += 50

        # Tertiary matching: URL-based hints
        if base_url:
            url_lower = base_url.lower()
            if 'openai' in url_lower and any(term in var_name_lower for term in ['openai', 'gpt']):
                score += 30
            elif 'anthropic' in url_lower and any(term in var_name_lower for term in ['anthropic', 'claude']):
                score += 30
            elif 'openrouter' in url_lower and 'router' in var_name_lower:
                score += 30
            elif 'google' in url_lower and any(term in var_name_lower for term in ['google', 'gemini', 'gcp']):
                score += 30
            elif 'cohere' in url_lower and 'cohere' in var_name_lower:
                score += 30

        # Bonus for common naming patterns
        if re.match(r'.*API_?KEY$', var_name.upper()):
            score += 10

        # Penalty for very generic names
        generic_names = ['API_KEY', 'APIKEY', 'TOKEN', 'API_TOKEN']
        if var_name.upper() in generic_names:
            score -= 20

        scored_vars.append((score, var_name, var_value))

    # Sort by score (highest first) and return all matches
    scored_vars.sort(reverse=True, key=lambda x: x[0])

    # Return all keys that scored above 0, or fallback to all found keys
    good_matches = [var_value for score, var_name, var_value in scored_vars if score > 0]
    if good_matches:
        return good_matches

    # If no good matches, return all API keys found as fallback
    return list(api_key_vars.values()) if api_key_vars else []
