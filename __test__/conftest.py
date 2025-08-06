"""Test configuration and fixtures for all tests."""

import sys
import types
import pytest


# Heavy external libraries that we don't need for domain logic testing
HEAVY_MODULES = [
    "outlines",
    "outlines.inputs", 
    "lmstudio",
    "torch",
    "transformers",
    "huggingface_hub",
]


@pytest.fixture(autouse=True, scope="session") 
def patch_heavy_modules():
    """
    Replace heavy external libraries with smart mocks for the entire test session.
    This allows imports to work without actually loading heavy dependencies.
    """
    from unittest.mock import MagicMock
    
    original = {}
    
    for name in HEAVY_MODULES:
        original[name] = sys.modules.get(name)
        
        # Create a smart mock that can handle attribute access
        mock_module = MagicMock()
        mock_module.__spec__ = MagicMock()
        mock_module.__name__ = name
        
        # For transformers, add specific attributes that are commonly imported
        if name == "transformers":
            mock_module.AutoTokenizer = MagicMock()
            mock_module.AutoModelForCausalLM = MagicMock()
            
        sys.modules[name] = mock_module
    
    yield
    
    # Restore original modules after tests
    for name, mod in original.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod