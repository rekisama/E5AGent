"""
Demo script for AutoGen Self-Expanding Agent System.

This script demonstrates the system's capabilities without requiring API keys.
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.function_registry import get_registry
from tools.function_tools import get_function_tools


def demo_function_creation():
    """Demonstrate creating and testing a new function."""
    print("🎯 Demo: Creating a New Function")
    print("=" * 40)
    
    tools = get_function_tools()
    
    # Example: Create a URL validator function
    func_name = "validate_url"
    func_code = '''
def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL format is valid, False otherwise
    """
    import re
    
    # Basic URL pattern
    pattern = r'^https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}(/.*)?$'
    return bool(re.match(pattern, url))
'''
    
    print(f"📝 Creating function: {func_name}")
    
    # Step 1: Validate code
    print("\n1️⃣ Validating function code...")
    is_valid, error_msg, extracted_name = tools.validate_function_code(func_code)
    
    if is_valid:
        print(f"✅ Code validation passed: {extracted_name}")
    else:
        print(f"❌ Code validation failed: {error_msg}")
        return False
    
    # Step 2: Create test cases
    print("\n2️⃣ Creating test cases...")
    test_cases = [
        {
            'input': {'url': 'https://www.example.com'},
            'expected_type': 'bool',
            'expected_value': True,
            'description': 'Valid HTTPS URL'
        },
        {
            'input': {'url': 'http://test.org/path'},
            'expected_type': 'bool',
            'expected_value': True,
            'description': 'Valid HTTP URL with path'
        },
        {
            'input': {'url': 'not-a-url'},
            'expected_type': 'bool',
            'expected_value': False,
            'description': 'Invalid URL'
        }
    ]
    
    print(f"📋 Created {len(test_cases)} test cases")
    
    # Step 3: Test function
    print("\n3️⃣ Testing function...")
    success, error_msg, test_results = tools.test_function(func_code, func_name, test_cases)
    
    if success:
        print("✅ All tests passed!")
        for result in test_results:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {result['description']}: {result.get('result', 'N/A')}")
    else:
        print(f"❌ Tests failed: {error_msg}")
        for result in test_results:
            if not result['success']:
                print(f"  ❌ {result['description']}: {result['error']}")
        return False
    
    # Step 4: Register function
    print("\n4️⃣ Registering function...")
    success = tools.register_function(
        func_name=func_name,
        func_code=func_code,
        description="Validates URL format using regex pattern",
        task_origin="Demo: URL validation task",
        test_cases=test_cases
    )
    
    if success:
        print(f"✅ Function '{func_name}' registered successfully!")
    else:
        print(f"❌ Failed to register function '{func_name}'")
        return False
    
    # Step 5: Test registered function
    print("\n5️⃣ Testing registered function...")
    registry = get_registry()
    func = registry.get_function(func_name)
    
    if func:
        test_urls = [
            "https://www.google.com",
            "http://example.org/test",
            "invalid-url",
            "ftp://not-http.com"
        ]
        
        print("Testing with various URLs:")
        for url in test_urls:
            try:
                result = func(url)
                status = "✅" if result else "❌"
                print(f"  {status} {url}: {result}")
            except Exception as e:
                print(f"  ❌ {url}: Error - {e}")
    
    return True


def demo_function_search():
    """Demonstrate function search capabilities."""
    print("\n🔍 Demo: Function Search")
    print("=" * 30)
    
    tools = get_function_tools()
    
    # Search for validation functions
    search_queries = ["validate", "email", "url", "password"]
    
    for query in search_queries:
        print(f"\n🔎 Searching for '{query}':")
        results = tools.search_functions(query)
        
        if results:
            for func in results:
                print(f"  📋 {func['name']}: {func['description']}")
                print(f"     Signature: {func['signature']}")
        else:
            print(f"  ❌ No functions found matching '{query}'")


def demo_system_stats():
    """Show system statistics."""
    print("\n📊 Demo: System Statistics")
    print("=" * 35)
    
    tools = get_function_tools()
    registry = get_registry()
    
    # Get all functions
    functions = tools.list_all_functions()
    
    print(f"📈 Total Functions: {len(functions)}")
    print(f"📅 Registry File: {registry.registry_file}")
    
    if functions:
        print("\n📋 Available Functions:")
        for i, func in enumerate(functions, 1):
            print(f"  {i}. {func['name']}")
            print(f"     Description: {func['description']}")
            print(f"     Created: {func['created_at']}")
            if func['task_origin']:
                print(f"     Origin: {func['task_origin']}")
            print()
    else:
        print("❌ No functions registered yet")


def demo_function_execution():
    """Demonstrate executing registered functions."""
    print("\n⚡ Demo: Function Execution")
    print("=" * 35)
    
    registry = get_registry()
    
    # Get available functions
    functions = registry.list_functions()
    
    if not functions:
        print("❌ No functions available for execution")
        return
    
    print("🎯 Testing registered functions:")
    
    # Test email validation if available
    if registry.has_function("validate_email"):
        print("\n📧 Testing email validation:")
        email_func = registry.get_function("validate_email")
        test_emails = [
            "user@example.com",
            "test.email+tag@domain.co.uk",
            "invalid-email",
            "@domain.com",
            "user@"
        ]
        
        for email in test_emails:
            try:
                result = email_func(email)
                status = "✅" if result else "❌"
                print(f"  {status} {email}: {result}")
            except Exception as e:
                print(f"  ❌ {email}: Error - {e}")
    
    # Test URL validation if available
    if registry.has_function("validate_url"):
        print("\n🌐 Testing URL validation:")
        url_func = registry.get_function("validate_url")
        test_urls = [
            "https://www.example.com",
            "http://test.org/path?param=value",
            "https://subdomain.domain.com/deep/path",
            "not-a-url",
            "ftp://file.server.com"
        ]
        
        for url in test_urls:
            try:
                result = url_func(url)
                status = "✅" if result else "❌"
                print(f"  {status} {url}: {result}")
            except Exception as e:
                print(f"  ❌ {url}: Error - {e}")
    
    # Test BMI calculation if available
    if registry.has_function("calculate_bmi"):
        print("\n🏃 Testing BMI calculation:")
        bmi_func = registry.get_function("calculate_bmi")
        test_cases = [
            (70.0, 1.75),  # Normal weight
            (85.0, 1.80),  # Overweight
            (55.0, 1.65),  # Underweight
        ]
        
        for weight, height in test_cases:
            try:
                result = bmi_func(weight, height)
                print(f"  📊 Weight: {weight}kg, Height: {height}m → BMI: {result}")
            except Exception as e:
                print(f"  ❌ Weight: {weight}kg, Height: {height}m → Error: {e}")


def main():
    """Run the demo."""
    print("🚀 AutoGen Self-Expanding Agent System - Demo")
    print("=" * 50)
    print("This demo shows the system's core capabilities:")
    print("• Creating and testing new functions")
    print("• Searching for existing functions")
    print("• Executing registered functions")
    print("• System statistics and management")
    print()
    
    try:
        # Demo 1: Create a new function
        if demo_function_creation():
            print("\n" + "="*50)
            
            # Demo 2: Search functions
            demo_function_search()
            print("\n" + "="*50)
            
            # Demo 3: Show system stats
            demo_system_stats()
            print("\n" + "="*50)
            
            # Demo 4: Execute functions
            demo_function_execution()
            
            print("\n" + "="*50)
            print("✅ Demo completed successfully!")
            print("\n💡 Next steps:")
            print("1. Set up your OpenAI API key in .env file")
            print("2. Run 'python main.py' to use the full system with LLM agents")
            print("3. Try asking the system to create more complex functions")
            
        else:
            print("❌ Demo failed during function creation")
            return 1
            
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
