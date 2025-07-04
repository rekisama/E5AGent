#!/usr/bin/env python3
"""
Test script to verify the modular refactoring of test_case_generator.py functionality.

This script tests that all the components from the original test_case_generator.py
have been successfully integrated into the tools/ directory structure.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'autogen_godel_agent'))

def test_imports():
    """Test that all modular components can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Test secure_executor imports
        from autogen_godel_agent.tools.secure_executor import (
            SecurityValidator, 
            validate_function_code, 
            execute_code_safely,
            FunctionSignatureParser
        )
        print("✓ secure_executor imports successful")
        
        # Test test_runner imports
        from autogen_godel_agent.tools.test_runner import (
            TestCaseGenerator,
            TestResult,
            TestGenerationConfig,
            TestCaseComplexity,
            InputFormat,
            TestCaseStandardizer,
            TestResponseParser
        )
        print("✓ test_runner imports successful")
        
        # Test function_tools imports
        from autogen_godel_agent.tools.function_tools import (
            FunctionCreationResult,
            FunctionToolsInterface,
            create_function_tools
        )
        print("✓ function_tools imports successful")
        
        # Test function_registry imports
        from autogen_godel_agent.tools.function_registry import (
            FunctionRegistry,
            get_registry
        )
        print("✓ function_registry imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_function_signature_parser():
    """Test FunctionSignatureParser functionality."""
    print("\nTesting FunctionSignatureParser...")
    
    try:
        from autogen_godel_agent.tools.secure_executor import FunctionSignatureParser
        
        # Test signature parsing
        signature = "def test_func(a: int, b: str = 'default', c: float = 3.14) -> bool"
        parser = FunctionSignatureParser()
        params = parser.parse_signature(signature)
        
        print(f"Parsed {len(params)} parameters from signature")
        for param in params:
            print(f"  - {param['name']}: {param['type']} (default: {param.get('default', 'None')})")
        
        assert len(params) == 3
        assert params[0]['name'] == 'a'
        assert params[0]['type'] == 'int'
        assert not params[0]['has_default']
        
        assert params[1]['name'] == 'b'
        assert params[1]['type'] == 'str'
        assert params[1]['has_default']
        assert params[1]['default'] == "'default'"
        
        print("✓ FunctionSignatureParser test passed")
        return True
        
    except Exception as e:
        print(f"✗ FunctionSignatureParser test failed: {e}")
        return False

def test_test_case_standardizer():
    """Test TestCaseStandardizer functionality."""
    print("\nTesting TestCaseStandardizer...")
    
    try:
        from autogen_godel_agent.tools.test_runner import TestCaseStandardizer, InputFormat
        
        standardizer = TestCaseStandardizer(InputFormat.DICT)
        
        # Test case data
        test_cases = [
            {
                'description': 'Test addition',
                'input': {'a': 1, 'b': 2},
                'expected_output': 3
            },
            {
                'description': 'Test string input',
                'input': 'hello',
                'expected_value': 'HELLO'
            }
        ]
        
        standardized = standardizer.standardize_test_cases(test_cases)
        
        print(f"Standardized {len(standardized)} test cases")
        for i, case in enumerate(standardized):
            print(f"  Case {i+1}: {case['description']}")
            print(f"    Input: {case['input']}")
            print(f"    Expected: {case['expected_output']}")
        
        assert len(standardized) == 2
        assert 'input' in standardized[0]
        assert 'expected_output' in standardized[0]
        assert 'description' in standardized[0]
        
        print("✓ TestCaseStandardizer test passed")
        return True
        
    except Exception as e:
        print(f"✗ TestCaseStandardizer test failed: {e}")
        return False

def test_test_response_parser():
    """Test TestResponseParser functionality."""
    print("\nTesting TestResponseParser...")
    
    try:
        from autogen_godel_agent.tools.test_runner import TestResponseParser
        
        parser = TestResponseParser()
        
        # Test JSON response parsing
        json_response = '''
        {
            "test_cases": [
                {
                    "description": "Test basic functionality",
                    "input": {"value": 42},
                    "expected_output": "success"
                }
            ]
        }
        '''
        
        parsed_cases = parser.parse_test_response(json_response)
        
        print(f"Parsed {len(parsed_cases)} test cases from JSON response")
        for case in parsed_cases:
            print(f"  - {case.get('description', 'No description')}")
        
        assert len(parsed_cases) >= 1
        assert 'description' in parsed_cases[0]
        
        print("✓ TestResponseParser test passed")
        return True
        
    except Exception as e:
        print(f"✗ TestResponseParser test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Modular Refactoring of test_case_generator.py")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_function_signature_parser,
        test_test_case_standardizer,
        test_test_response_parser
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modular refactoring successful!")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
