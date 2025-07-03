"""
Interactive script for AutoGen Self-Expanding Agent System.

Allows users to manually create, test, and manage functions.
"""

import sys
import os
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.function_registry import get_registry
from tools.function_tools import get_function_tools


def print_menu():
    """Print the main menu."""
    print("\n" + "="*50)
    print("🚀 AutoGen Self-Expanding Agent System")
    print("="*50)
    print("1. 📋 List all functions")
    print("2. 🔍 Search functions")
    print("3. ⚡ Execute a function")
    print("4. 🔧 Create a new function")
    print("5. 📊 Show system statistics")
    print("6. 🧪 Test a function")
    print("7. ❌ Exit")
    print("="*50)


def list_functions():
    """List all registered functions."""
    tools = get_function_tools()
    functions = tools.list_all_functions()
    
    if not functions:
        print("❌ No functions registered yet.")
        return
    
    print(f"\n📋 Found {len(functions)} function(s):")
    print("-" * 60)
    
    for i, func in enumerate(functions, 1):
        print(f"{i}. {func['name']}")
        print(f"   Description: {func['description']}")
        print(f"   Created: {func['created_at']}")
        if func['task_origin']:
            print(f"   Origin: {func['task_origin']}")
        print()


def search_functions():
    """Search for functions."""
    query = input("\n🔍 Enter search query: ").strip()
    
    if not query:
        print("❌ Please enter a search query.")
        return
    
    tools = get_function_tools()
    results = tools.search_functions(query)
    
    if not results:
        print(f"❌ No functions found matching '{query}'")
        return
    
    print(f"\n✅ Found {len(results)} function(s) matching '{query}':")
    print("-" * 60)
    
    for func in results:
        print(f"📋 {func['name']}: {func['description']}")
        print(f"   Signature: {func['signature']}")
        if func['docstring']:
            print(f"   Documentation: {func['docstring']}")
        print()


def execute_function():
    """Execute a registered function."""
    registry = get_registry()
    functions = registry.list_functions()
    
    if not functions:
        print("❌ No functions available for execution.")
        return
    
    print("\n⚡ Available functions:")
    for i, func in enumerate(functions, 1):
        print(f"{i}. {func['name']}: {func['description']}")
    
    try:
        choice = int(input("\nSelect function number: ")) - 1
        if choice < 0 or choice >= len(functions):
            print("❌ Invalid selection.")
            return
        
        func_name = functions[choice]['name']
        func = registry.get_function(func_name)
        
        if not func:
            print(f"❌ Function '{func_name}' not found.")
            return
        
        # Get function info for parameter guidance
        info = get_function_tools().get_function_info(func_name)
        print(f"\n📋 Function: {func_name}")
        print(f"Signature: {info['signature']}")
        
        # Simple parameter input (for demonstration)
        if func_name == "validate_email":
            email = input("Enter email to validate: ")
            result = func(email)
            print(f"✅ Result: {result}")
        
        elif func_name == "validate_url":
            url = input("Enter URL to validate: ")
            result = func(url)
            print(f"✅ Result: {result}")
        
        elif func_name == "calculate_bmi":
            try:
                weight = float(input("Enter weight (kg): "))
                height = float(input("Enter height (m): "))
                result = func(weight, height)
                print(f"✅ BMI: {result}")
            except ValueError:
                print("❌ Please enter valid numbers.")
        
        else:
            print("❌ Interactive execution not implemented for this function.")
            print("💡 You can view the function code and call it manually.")
    
    except ValueError:
        print("❌ Please enter a valid number.")
    except Exception as e:
        print(f"❌ Error executing function: {e}")


def create_function():
    """Guide user through creating a new function."""
    print("\n🔧 Create New Function")
    print("-" * 30)
    
    func_name = input("Function name: ").strip()
    if not func_name:
        print("❌ Function name is required.")
        return
    
    tools = get_function_tools()
    
    if tools.has_function(func_name):
        print(f"❌ Function '{func_name}' already exists.")
        return
    
    description = input("Function description: ").strip()
    if not description:
        print("❌ Function description is required.")
        return
    
    print("\n📝 Enter function code (end with empty line):")
    print("Example:")
    print("def my_function(param: str) -> bool:")
    print("    \"\"\"Function description.\"\"\"")
    print("    return True")
    print()
    
    code_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        code_lines.append(line)
    
    if not code_lines:
        print("❌ No code provided.")
        return
    
    func_code = "\n".join(code_lines)
    
    # Validate code
    print("\n1️⃣ Validating code...")
    is_valid, error_msg, extracted_name = tools.validate_function_code(func_code)
    
    if not is_valid:
        print(f"❌ Code validation failed: {error_msg}")
        return
    
    if extracted_name != func_name:
        print(f"❌ Function name mismatch: expected '{func_name}', found '{extracted_name}'")
        return
    
    print("✅ Code validation passed!")
    
    # Create test cases
    print("\n2️⃣ Creating test cases...")
    test_cases = []
    
    while True:
        add_test = input("Add a test case? (y/n): ").strip().lower()
        if add_test != 'y':
            break
        
        print("Enter test case parameters as JSON (e.g., {\"param1\": \"value1\"}):")
        try:
            params_str = input("Parameters: ")
            params = json.loads(params_str) if params_str.strip() else {}
            
            expected_type = input("Expected return type (bool/str/int/float/list/dict): ").strip()
            description = input("Test description: ").strip()
            
            test_case = {
                'input': params,
                'expected_type': expected_type,
                'description': description or f"Test case {len(test_cases) + 1}"
            }
            
            test_cases.append(test_case)
            print(f"✅ Added test case: {test_case['description']}")
            
        except json.JSONDecodeError:
            print("❌ Invalid JSON format for parameters.")
        except Exception as e:
            print(f"❌ Error adding test case: {e}")
    
    # Test function
    if test_cases:
        print("\n3️⃣ Testing function...")
        success, error_msg, test_results = tools.test_function(func_code, func_name, test_cases)
        
        if not success:
            print(f"❌ Tests failed: {error_msg}")
            for result in test_results:
                if not result['success']:
                    print(f"  ❌ {result['description']}: {result['error']}")
            return
        
        print("✅ All tests passed!")
        for result in test_results:
            print(f"  ✅ {result['description']}: {result.get('result', 'N/A')}")
    
    # Register function
    print("\n4️⃣ Registering function...")
    success = tools.register_function(
        func_name=func_name,
        func_code=func_code,
        description=description,
        task_origin="Interactive creation",
        test_cases=test_cases
    )
    
    if success:
        print(f"✅ Function '{func_name}' registered successfully!")
    else:
        print(f"❌ Failed to register function '{func_name}'")


def show_statistics():
    """Show system statistics."""
    tools = get_function_tools()
    registry = get_registry()
    
    functions = tools.list_all_functions()
    
    print("\n📊 System Statistics")
    print("-" * 30)
    print(f"📈 Total Functions: {len(functions)}")
    print(f"📅 Registry File: {registry.registry_file}")
    
    if functions:
        # Group by creation date
        dates = {}
        for func in functions:
            date = func['created_at'][:10]  # Get date part
            dates[date] = dates.get(date, 0) + 1
        
        print(f"\n📅 Functions by date:")
        for date, count in sorted(dates.items()):
            print(f"  {date}: {count} function(s)")


def test_function():
    """Test a specific function with custom inputs."""
    registry = get_registry()
    functions = registry.list_functions()
    
    if not functions:
        print("❌ No functions available for testing.")
        return
    
    print("\n🧪 Available functions:")
    for i, func in enumerate(functions, 1):
        print(f"{i}. {func['name']}: {func['description']}")
    
    try:
        choice = int(input("\nSelect function number: ")) - 1
        if choice < 0 or choice >= len(functions):
            print("❌ Invalid selection.")
            return
        
        func_name = functions[choice]['name']
        info = get_function_tools().get_function_info(func_name)
        
        print(f"\n📋 Testing function: {func_name}")
        print(f"Signature: {info['signature']}")
        print(f"Description: {info['description']}")
        
        if info['test_cases']:
            print(f"\n🧪 Existing test cases ({len(info['test_cases'])}):")
            for i, test_case in enumerate(info['test_cases'], 1):
                print(f"  {i}. {test_case['description']}")
                print(f"     Input: {test_case['input']}")
        
        # Run existing tests
        if info['test_cases']:
            run_existing = input("\nRun existing test cases? (y/n): ").strip().lower()
            if run_existing == 'y':
                tools = get_function_tools()
                success, error_msg, test_results = tools.test_function(
                    info['code'], func_name, info['test_cases']
                )
                
                print(f"\n📊 Test Results:")
                for result in test_results:
                    status = "✅" if result['success'] else "❌"
                    print(f"  {status} {result['description']}: {result.get('result', result.get('error', 'N/A'))}")
    
    except ValueError:
        print("❌ Please enter a valid number.")
    except Exception as e:
        print(f"❌ Error testing function: {e}")


def main():
    """Main interactive loop."""
    print("🚀 Welcome to AutoGen Self-Expanding Agent System!")
    print("This interactive tool lets you manage and test functions.")
    
    while True:
        print_menu()
        
        try:
            choice = input("Select option (1-7): ").strip()
            
            if choice == '1':
                list_functions()
            elif choice == '2':
                search_functions()
            elif choice == '3':
                execute_function()
            elif choice == '4':
                create_function()
            elif choice == '5':
                show_statistics()
            elif choice == '6':
                test_function()
            elif choice == '7':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-7.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
