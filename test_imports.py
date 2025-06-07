#!/usr/bin/env python3
"""Test script to verify imports work without external dependencies."""

import sys
import os

# Add current directory to path so we can import chatifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test that all modules can be imported."""
    try:
        print("Testing basic imports...")
        
        # Test utilities (no external deps)
        from chatifier.utils import build_base_url, format_model_name
        print("✓ utils imported")
        
        # Test that build_base_url works
        url = build_base_url("localhost", 8080, True)
        assert url == "https://localhost:8080"
        print("✓ build_base_url works")
        
        # Test format_model_name
        formatted = format_model_name("gpt-3.5-turbo-latest")
        assert "3.5-turbo" in formatted
        print("✓ format_model_name works")
        
        print("✓ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Import/basic test failed: {e}")
        return False

def test_module_structure():
    """Test that all modules exist and have expected structure."""
    try:
        print("\nTesting module structure...")
        
        import chatifier
        print("✓ chatifier package imported")
        
        # Check version
        assert hasattr(chatifier, '__version__')
        print(f"✓ Version: {chatifier.__version__}")
        
        # Check that all modules exist (will fail if dependencies missing)
        module_files = [
            'cli.py', 'detector.py', 'clients.py', 'ui.py', 'utils.py', '__main__.py'
        ]
        
        for module in module_files:
            path = os.path.join('chatifier', module)
            assert os.path.exists(path), f"Missing {module}"
            print(f"✓ {module} exists")
        
        print("✓ All modules present!")
        return True
        
    except Exception as e:
        print(f"✗ Module structure test failed: {e}")
        return False

if __name__ == '__main__':
    print("=== Testing llm-chatifier without external dependencies ===\n")
    
    success = True
    success &= test_basic_imports()
    success &= test_module_structure()
    
    if success:
        print("\n🎉 All tests passed! The basic structure is correct.")
        print("\nNote: Full functionality requires installing dependencies:")
        print("  pip install click httpx rich prompt-toolkit")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
