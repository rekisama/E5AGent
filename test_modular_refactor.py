#!/usr/bin/env python3
"""
Test script to verify the modular refactoring of TaskPlannerAgent.

This script tests that all the modular components work correctly together
and that the TaskPlannerAgent can still function properly after refactoring.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modular components can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Test session manager
        from autogen_godel_agent.tools.session_manager import get_session_manager
        session_manager = get_session_manager()
        print("✓ SessionManager imported and instantiated successfully")
        
        # Test response parser
        from autogen_godel_agent.tools.response_parser import get_response_parser
        response_parser = get_response_parser()
        print("✓ ResponseParser imported and instantiated successfully")
        
        # Test agent pool
        from autogen_godel_agent.tools.agent_pool import get_user_proxy_pool
        agent_pool = get_user_proxy_pool()
        print("✓ UserProxyPool imported and instantiated successfully")
        
        # Test secure executor with new validation functions
        from autogen_godel_agent.tools.secure_executor import validate_function_signature, extract_function_signature_from_code
        print("✓ Function signature validation imported successfully")
        
        # Test TaskPlannerAgent with new modular architecture
        from autogen_godel_agent.agents.planner_agent import TaskPlannerAgent
        print("✓ TaskPlannerAgent imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        return False


def test_session_manager():
    """Test session manager functionality."""
    print("\nTesting SessionManager...")
    
    try:
        from autogen_godel_agent.tools.session_manager import get_session_manager
        
        session_manager = get_session_manager()
        
        # Test session creation
        session_id = session_manager.create_session()
        print(f"✓ Created session: {session_id}")
        
        # Test session retrieval
        session = session_manager.get_session(session_id)
        assert session is not None, "Session should exist"
        print("✓ Retrieved session successfully")
        
        # Test token usage tracking
        session_manager.add_token_usage(session_id, 100, 50)
        stats = session_manager.get_session_stats(session_id)
        assert stats['token_usage']['total_tokens'] == 150, "Token usage should be tracked"
        print("✓ Token usage tracking works")
        
        return True
        
    except Exception as e:
        print(f"✗ SessionManager test failed: {e}")
        return False


def test_response_parser():
    """Test response parser functionality."""
    print("\nTesting ResponseParser...")
    
    try:
        from autogen_godel_agent.tools.response_parser import get_response_parser
        
        response_parser = get_response_parser()
        
        # Test JSON parsing
        test_response = '''
        Here's my analysis:
        ```json
        {
            "function_found": true,
            "matched_functions": [{"name": "test_func", "description": "Test function", "signature": "def test_func() -> str"}],
            "needs_new_function": false,
            "reasoning": "Found existing function"
        }
        ```
        '''
        
        result = response_parser.parse_llm_analysis(test_response, "test task")
        assert result['function_found'] == True, "Should parse function_found correctly"
        assert len(result['matched_functions']) == 1, "Should parse matched functions"
        print("✓ JSON parsing works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ ResponseParser test failed: {e}")
        return False


def test_agent_pool():
    """Test agent pool functionality."""
    print("\nTesting UserProxyPool...")
    
    try:
        from autogen_godel_agent.tools.agent_pool import get_user_proxy_pool
        
        agent_pool = get_user_proxy_pool()
        
        # Test agent retrieval with context manager
        with agent_pool.get_user_proxy() as proxy:
            assert proxy is not None, "Should get a proxy agent"
            assert hasattr(proxy, 'chat_messages'), "Proxy should have chat_messages"
            print("✓ Agent pool context manager works")
        
        # Test pool statistics
        stats = agent_pool.get_pool_stats()
        assert 'created_count' in stats, "Should have creation statistics"
        print("✓ Pool statistics available")
        
        return True
        
    except Exception as e:
        print(f"✗ UserProxyPool test failed: {e}")
        return False


def test_function_signature_validation():
    """Test function signature validation."""
    print("\nTesting function signature validation...")
    
    try:
        from autogen_godel_agent.tools.secure_executor import validate_function_signature, extract_function_signature_from_code
        
        # Test valid signature
        valid_sig = "def test_func(x: int, y: str) -> bool"
        assert validate_function_signature(valid_sig) == True, "Should validate correct signature"
        print("✓ Valid signature validation works")
        
        # Test signature extraction
        test_code = '''
def example_function(a: int, b: str) -> float:
    """Example function."""
    return float(a)
'''
        extracted = extract_function_signature_from_code(test_code)
        assert extracted is not None, "Should extract signature from code"
        assert "example_function" in extracted, "Should contain function name"
        print("✓ Signature extraction works")
        
        return True
        
    except Exception as e:
        print(f"✗ Function signature validation test failed: {e}")
        return False


def test_planner_agent_integration():
    """Test that TaskPlannerAgent works with modular components."""
    print("\nTesting TaskPlannerAgent integration...")
    
    try:
        from autogen_godel_agent.agents.planner_agent import TaskPlannerAgent
        
        # Mock LLM config for testing
        llm_config = {
            "model": "gpt-3.5-turbo",
            "api_key": "test-key",
            "temperature": 0.1
        }
        
        # Create planner agent
        planner = TaskPlannerAgent(llm_config)
        
        # Test that all modular components are initialized
        assert hasattr(planner, 'session_manager'), "Should have session manager"
        assert hasattr(planner, 'response_parser'), "Should have response parser"
        assert hasattr(planner, 'user_proxy_pool'), "Should have user proxy pool"
        print("✓ TaskPlannerAgent has all modular components")
        
        # Test session management methods
        stats = planner.get_session_stats("nonexistent")
        assert stats is None, "Should return None for nonexistent session"
        print("✓ Session management methods work")
        
        return True
        
    except Exception as e:
        print(f"✗ TaskPlannerAgent integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MODULAR REFACTORING VERIFICATION TESTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_session_manager,
        test_response_parser,
        test_agent_pool,
        test_function_signature_validation,
        test_planner_agent_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modular refactoring successful!")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
