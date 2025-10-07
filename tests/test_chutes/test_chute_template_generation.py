import os
import sys
import tempfile
import pytest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

from babelbit.utils.chutes_helpers import render_chute_template
from babelbit.chute_template.schemas import BBPredictedUtterance, BBPredictOutput


def test_render_chute_template_syntax():
    """Test that the rendered chute template is valid Python syntax"""
    # Use a known revision for testing
    test_revision = "4d40cc920ef20b8c570a4300d39e7cc31efb51b7"
    
    # Render the template
    rendered_template = render_chute_template(revision=test_revision)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(str(rendered_template))
        tmp_file_path = tmp_file.name
    
    try:
        # Test that the file compiles without syntax errors
        with open(tmp_file_path, 'r') as f:
            source_code = f.read()
        
        # Compile the source code to check for syntax errors
        compile(source_code, tmp_file_path, 'exec')
        
        print(f"✅ Generated chute template compiled successfully")
        print(f"Generated file length: {len(source_code)} characters")
        
    finally:
        # Clean up
        os.unlink(tmp_file_path)


def test_rendered_chute_template_structure():
    """Test that the rendered template contains expected structure"""
    test_revision = "4d40cc920ef20b8c570a4300d39e7cc31efb51b7"
    rendered_template = render_chute_template(revision=test_revision)
    content = str(rendered_template)
    
    # Check for required components
    required_elements = [
        "class BBPredictedUtterance",
        "class BBPredictOutput", 
        "def _load_model",
        "def _predict",
        "def _health",
        "def init_chute",
        "@chute.on_startup",
        "@chute.cord",
        "from chutes.chute import Chute",
        "from pydantic import BaseModel"
    ]
    
    for element in required_elements:
        assert element in content, f"Missing required element: {element}"
    
    # Check that there are no external babelbit imports in the generated file
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'from babelbit' in line and not line.strip().startswith('#'):
            pytest.fail(f"Found external babelbit import on line {i+1}: {line.strip()}")
    
    print("✅ Generated chute template has correct structure and no external imports")


def test_rendered_chute_template_imports():
    """Test that all imports in the rendered template can be resolved"""
    test_revision = "4d40cc920ef20b8c570a4300d39e7cc31efb51b7"
    rendered_template = render_chute_template(revision=test_revision)
    content = str(rendered_template)
    
    # Write to temporary file and try to import it
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Add the temp directory to sys.path temporarily
        temp_dir = os.path.dirname(tmp_file_path)
        if temp_dir not in sys.path:
            sys.path.insert(0, temp_dir)
        
        # Try to load the module
        module_name = os.path.basename(tmp_file_path)[:-3]  # Remove .py extension
        spec = spec_from_file_location(module_name, tmp_file_path)
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")
        module = module_from_spec(spec)
        
        # This will fail if there are import errors
        spec.loader.exec_module(module)
        
        # Check that key functions exist
        assert hasattr(module, '_load_model'), "Missing _load_model function"
        assert hasattr(module, '_predict'), "Missing _predict function" 
        assert hasattr(module, '_health'), "Missing _health function"
        assert hasattr(module, 'init_chute'), "Missing init_chute function"
        assert hasattr(module, 'BBPredictedUtterance'), "Missing BBPredictedUtterance class"
        assert hasattr(module, 'BBPredictOutput'), "Missing BBPredictOutput class"
        
        print("✅ Generated chute template imports and loads successfully")
        
    except ImportError as e:
        pytest.fail(f"Import error in generated chute template: {e}")
    except Exception as e:
        pytest.fail(f"Error loading generated chute template: {e}")
    finally:
        # Clean up
        if temp_dir in sys.path:
            sys.path.remove(temp_dir)
        os.unlink(tmp_file_path)


def test_rendered_chute_predict_function():
    """Test that the _predict function in the rendered template works"""
    test_revision = "4d40cc920ef20b8c570a4300d39e7cc31efb51b7"
    rendered_template = render_chute_template(revision=test_revision)
    content = str(rendered_template)
    
    # Write to temporary file and load the module
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Load the module
        temp_dir = os.path.dirname(tmp_file_path)
        if temp_dir not in sys.path:
            sys.path.insert(0, temp_dir)
        
        module_name = os.path.basename(tmp_file_path)[:-3]
        spec = spec_from_file_location(module_name, tmp_file_path)
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Test the _predict function with mock data
        test_utterance = module.BBPredictedUtterance(
            index="test-123",
            step=1,
            prefix="Hello",
            prediction="",
            context="This is a test context"
        )
        
        # Test with None model (should handle gracefully)
        result = module._predict(model=None, data=test_utterance, model_name="test-model")
        
        assert isinstance(result, module.BBPredictOutput), "Result should be BBPredictOutput"
        assert result.success is False, "Should fail with None model"
        assert "Model not loaded" in result.error, "Should have appropriate error message"
        
        print("✅ Generated chute _predict function works correctly with None model")
        
    finally:
        # Clean up
        if temp_dir in sys.path:
            sys.path.remove(temp_dir)
        os.unlink(tmp_file_path)


if __name__ == "__main__":
    # Run tests directly
    test_render_chute_template_syntax()
    test_rendered_chute_template_structure() 
    test_rendered_chute_template_imports()
    test_rendered_chute_predict_function()
    print("🎉 All chute template tests passed!")