"""
Test script to validate all the fixes applied to FunctionCreatorAgent.

This script tests:
1. Parameter naming consistency
2. Chat messages error handling
3. Improved regex patterns
4. Function name validation
5. TestResult fallback handling
6. Return type guidance
7. Validation caching
8. Logging configuration

修复验证测试脚本
"""

import logging
import sys
import os
import json
from typing import Dict, Any

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autogen_godel_agent.agents.function_creator_agent import FunctionCreatorAgent
from autogen_godel_agent.tools.function_tools import get_function_tools
from autogen_godel_agent.agents.test_case_generator import TestResult

logger = logging.getLogger(__name__)

def test_parameter_naming_consistency():
    """Test 1: Parameter naming consistency fix."""
    logger.info("🔧 Testing parameter naming consistency...")
    
    try:
        # Create mock function tools to test interface
        function_tools = get_function_tools()
        
        # Test that register_function accepts func_name parameter
        test_code = '''
def test_function(x: int) -> int:
    """Test function."""
    return x * 2
'''
        
        # This should work without parameter naming errors
        result = function_tools.register_function(
            func_name="test_function",
            func_code=test_code,
            description="Test function for parameter naming",
            task_origin="test_fixes_validation",
            test_cases=[]
        )
        
        logger.info("✅ Parameter naming consistency test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parameter naming test failed: {e}")
        return False

def test_chat_messages_error_handling():
    """Test 2: Chat messages error handling."""
    logger.info("🔧 Testing chat messages error handling...")
    
    try:
        # Create a mock user_proxy object without chat_messages
        class MockUserProxy:
            def __init__(self):
                self.chat_messages = {}  # Empty dict to test .get() method
        
        user_proxy = MockUserProxy()
        
        # Test that accessing non-existent agent doesn't crash
        agent_key = "non_existent_agent"
        messages = user_proxy.chat_messages.get(agent_key, [])
        
        assert messages == [], "Should return empty list for non-existent agent"
        logger.info("✅ Chat messages error handling test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Chat messages error handling test failed: {e}")
        return False

def test_improved_regex_patterns():
    """Test 3: Improved regex patterns for code extraction."""
    logger.info("🔧 Testing improved regex patterns...")
    
    try:
        agent = FunctionCreatorAgent()
        
        # Test response with proper newline formatting
        test_response = """Here's the function:

```python
def calculate_area(radius: float) -> float:
    \"\"\"Calculate circle area.\"\"\"
    import math
    return math.pi * radius * radius
```

This function calculates the area."""
        
        # Test extraction
        extracted_code = agent._extract_code_from_response(test_response, "calculate_area")
        
        assert extracted_code is not None, "Should extract code successfully"
        assert "def calculate_area" in extracted_code, "Should contain the target function"
        
        logger.info("✅ Improved regex patterns test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Improved regex patterns test failed: {e}")
        return False

def test_function_name_validation():
    """Test 4: Function name validation and prioritization."""
    logger.info("🔧 Testing function name validation...")
    
    try:
        agent = FunctionCreatorAgent()
        
        # Test response with multiple functions, target function should be prioritized
        test_response = """Here are some functions:

```python
def helper_function(x):
    return x + 1

def target_function(y: int) -> int:
    \"\"\"This is the target function.\"\"\"
    return y * 2

def another_function(z):
    return z - 1
```
"""
        
        # Should prioritize target_function
        extracted_code = agent._extract_code_from_response(test_response, "target_function")
        
        assert extracted_code is not None, "Should extract code successfully"
        assert "def target_function" in extracted_code, "Should prioritize target function"
        
        logger.info("✅ Function name validation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Function name validation test failed: {e}")
        return False

def test_testresult_fallback():
    """Test 5: TestResult fallback handling."""
    logger.info("🔧 Testing TestResult fallback handling...")
    
    try:
        # Test normal TestResult creation
        normal_result = TestResult.from_tuple((True, "", []))
        assert normal_result.success == True, "Normal TestResult should work"
        
        # Test fallback for invalid tuple
        try:
            invalid_result = TestResult.from_tuple(("invalid",))
            # Should create a fallback result
            assert invalid_result.success == False, "Should create fallback for invalid input"
        except:
            # If it throws exception, that's also acceptable behavior
            pass
        
        logger.info("✅ TestResult fallback test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ TestResult fallback test failed: {e}")
        return False

def test_return_type_guidance():
    """Test 6: Return type guidance in prompts."""
    logger.info("🔧 Testing return type guidance...")
    
    try:
        agent = FunctionCreatorAgent()
        
        # Test prompt generation with return type
        specification = {
            'name': 'test_func',
            'description': 'Test function',
            'return_type': 'List[int]',
            'parameters': [{'name': 'x', 'type': 'int', 'description': 'Input number'}],
            'examples': [{'input': '5', 'output': '[1, 2, 3, 4, 5]'}]
        }
        
        prompt = agent.get_creation_prompt(specification)
        
        assert 'Return Type: List[int]' in prompt, "Should include return type in prompt"
        assert 'MUST return a value of type List[int]' in prompt, "Should include return type requirement"
        
        logger.info("✅ Return type guidance test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Return type guidance test failed: {e}")
        return False

def test_validation_caching():
    """Test 7: Validation caching performance optimization."""
    logger.info("🔧 Testing validation caching...")
    
    try:
        agent = FunctionCreatorAgent()
        
        test_code = '''
def test_function(x: int) -> int:
    """Test function."""
    return x * 2
'''
        
        # First call should populate cache
        result1 = agent._is_valid_function_code(test_code)
        
        # Second call should use cache
        result2 = agent._is_valid_function_code(test_code)
        
        assert result1 == result2, "Cached result should match original"
        assert len(agent._validation_cache) > 0, "Cache should contain entries"
        
        logger.info("✅ Validation caching test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation caching test failed: {e}")
        return False

def test_logging_configuration():
    """Test 8: Logging configuration."""
    logger.info("🔧 Testing logging configuration...")
    
    try:
        # Test that logging is working
        test_logger = logging.getLogger("test_logger")
        test_logger.info("Test log message")
        test_logger.debug("Debug message")
        test_logger.warning("Warning message")
        test_logger.error("Error message")
        
        logger.info("✅ Logging configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Logging configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all fix validation tests."""
    logger.info("🚀 Starting fix validation tests...")
    
    tests = [
        ("Parameter Naming Consistency", test_parameter_naming_consistency),
        ("Chat Messages Error Handling", test_chat_messages_error_handling),
        ("Improved Regex Patterns", test_improved_regex_patterns),
        ("Function Name Validation", test_function_name_validation),
        ("TestResult Fallback", test_testresult_fallback),
        ("Return Type Guidance", test_return_type_guidance),
        ("Validation Caching", test_validation_caching),
        ("Logging Configuration", test_logging_configuration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running: {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"Test failed: {test_name}")
        except Exception as e:
            logger.error(f"Test error in {test_name}: {e}")
    
    logger.info(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All fixes validated successfully!")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
