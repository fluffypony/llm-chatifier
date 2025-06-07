"""Tests for API detector module."""

import unittest
from unittest.mock import Mock, patch

from chatifier.detector import detect_api, detect_specific_api, get_api_info


class TestDetector(unittest.TestCase):
    """Test API detection functionality."""
    
    @patch('chatifier.detector.try_connection')
    def test_detect_openai_api(self, mock_try_connection):
        """Test detection of OpenAI API."""
        # Mock successful connection to OpenAI endpoint
        mock_try_connection.side_effect = lambda url: (
            (True, Mock()) if '/v1/models' in url else (False, None)
        )
        
        result = detect_api('localhost', 8080)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['type'], 'openai')
        self.assertEqual(result['host'], 'localhost')
        self.assertEqual(result['port'], 8080)
    
    @patch('chatifier.detector.try_connection')
    def test_detect_ollama_api(self, mock_try_connection):
        """Test detection of Ollama API."""
        # Mock successful connection to Ollama endpoint
        mock_try_connection.side_effect = lambda url: (
            (True, Mock()) if '/api/tags' in url else (False, None)
        )
        
        result = detect_api('localhost', 11434)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['type'], 'ollama')
    
    @patch('chatifier.detector.try_connection')
    def test_no_api_detected(self, mock_try_connection):
        """Test when no API is detected."""
        # Mock all connections failing
        mock_try_connection.return_value = (False, None)
        
        result = detect_api('localhost', 9999)
        
        self.assertIsNone(result)
    
    def test_get_api_info_openai(self):
        """Test getting OpenAI API info."""
        info = get_api_info('openai', 'http://localhost:8080')
        
        self.assertEqual(info['type'], 'openai')
        self.assertEqual(info['base_url'], 'http://localhost:8080')
        self.assertEqual(info['chat_endpoint'], '/v1/chat/completions')
        self.assertEqual(info['models_endpoint'], '/v1/models')
    
    def test_get_api_info_ollama(self):
        """Test getting Ollama API info."""
        info = get_api_info('ollama', 'http://localhost:11434')
        
        self.assertEqual(info['type'], 'ollama')
        self.assertEqual(info['chat_endpoint'], '/api/generate')
        self.assertEqual(info['models_endpoint'], '/api/tags')
    
    @patch('chatifier.detector.try_connection')
    def test_detect_specific_api_success(self, mock_try_connection):
        """Test specific API detection success."""
        mock_try_connection.return_value = (True, Mock())
        
        result = detect_specific_api('http://localhost:8080', 'openai')
        
        self.assertTrue(result)
    
    @patch('chatifier.detector.try_connection')
    def test_detect_specific_api_failure(self, mock_try_connection):
        """Test specific API detection failure."""
        mock_try_connection.return_value = (False, None)
        
        result = detect_specific_api('http://localhost:8080', 'openai')
        
        self.assertFalse(result)
    
    def test_detect_specific_api_unknown_type(self):
        """Test specific API detection with unknown type."""
        result = detect_specific_api('http://localhost:8080', 'unknown')
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
