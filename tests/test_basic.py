"""Basic tests that don't require external dependencies."""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatifier.utils import build_base_url, format_model_name


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality without external dependencies."""
    
    def test_build_base_url_https(self):
        """Test HTTPS URL building."""
        url = build_base_url("example.com", 8080, True)
        self.assertEqual(url, "https://example.com:8080")
    
    def test_build_base_url_http(self):
        """Test HTTP URL building."""
        url = build_base_url("localhost", 3000, False)
        self.assertEqual(url, "http://localhost:3000")
    
    def test_build_base_url_standard_ports(self):
        """Test URL building with standard ports."""
        # HTTPS standard port
        url = build_base_url("example.com", 443, True)
        self.assertEqual(url, "https://example.com")
        
        # HTTP standard port
        url = build_base_url("example.com", 80, False)
        self.assertEqual(url, "http://example.com")
    
    def test_format_model_name(self):
        """Test model name formatting."""
        # Test removing prefixes
        self.assertEqual(format_model_name("gpt-3.5-turbo"), "3.5-turbo")
        self.assertEqual(format_model_name("text-davinci-003"), "davinci-003")
        
        # Test removing suffixes
        self.assertEqual(format_model_name("claude-3-latest"), "claude-3")
        self.assertEqual(format_model_name("llama-2-preview"), "llama-2")
        
        # Test no changes needed
        self.assertEqual(format_model_name("claude-3"), "claude-3")


if __name__ == '__main__':
    unittest.main()
