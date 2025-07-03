"""
Enhanced Function management tools for the Self-Expanding Agent System.

Provides secure utilities for checking, generating, testing, and registering functions
with comprehensive security validation and robust error handling.
"""

import re
import ast
import inspect
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, get_type_hints
from .function_registry import get_registry
from .secure_executor import get_secure_executor
from .test_runner import TestRunner

"""
潜在问题与改进建议
⚠️ 1. secure_executor.execute_code 假设返回 result.get(func_name) 为 callable
如果用户写了多个函数，或者函数是嵌套在类里的，这里会取不到 func，造成误判。

建议改进：

python
复制
编辑
func = result.get(func_name)
if not callable(func):
    logger.error(f"Function {func_name} is not callable or missing in result")
    return []
⚠️ 2. 测试生成中 expected_type: auto_generated 含义不清
多处测试用例中返回 expected_type: "auto_generated"，但在测试执行逻辑中是否能处理不清晰。

建议：

统一文档说明或替换为明确类型，例如：number, str, bool, NoneType。

或在测试执行器中加逻辑：若expected_output缺失且只有 expected_type，则验证返回值类型是否匹配。

⚠️ 3. _generate_boundary_tests 仅处理 int 和 str，float/bool/list 等未覆盖
目前边界测试中只对 int 和 str 做了处理。

建议添加：

python
复制
编辑
elif param_type == float or 'float' in str(param_type).lower():
    test_cases.extend([
        {'input': {param_name: 0.0}, 'expected_type': 'float', 'description': f'Zero float test'},
        {'input': {param_name: -1.0}, 'expected_type': 'float', 'description': f'Negative float test'},
    ])
elif param_type == bool or 'bool' in str(param_type).lower():
    test_cases.extend([
        {'input': {param_name: True}, 'expected_type': 'bool', 'description': f'True test for {param_name}'},
        {'input': {param_name: False}, 'expected_type': 'bool', 'description': f'False test for {param_name}'}
    ])
⚠️ 4. FunctionTools = EnhancedFunctionTools 写法会让旧调用者认为还是旧类
如果外部系统大量使用 FunctionTools，这样做能保证兼容。

但同时也可能隐藏真实变化（如是否必须使用 secure_executor 等）。

建议在文档/注释中注明升级点，或者通过版本号控制明确标注行为差异。

⚠️ 5. test_function 里 test_cases or [] 可能隐藏测试失败信息
如果 test_cases=[] 传入空列表，或 generate_test_cases 返回空，仍会调用 test_runner.run_tests()，可能造成误解。

建议加判断：

python
复制
编辑
if not test_cases:
    return False, "No test cases available for testing", []
⚠️ 6. 全局 _function_tools 是单例但线程不安全
如果这个模块要支持并发 Agent 调用，可能存在线程安全问题。

建议：将单例创建放在线程锁内，或使用某种线程安全的 singleton 工厂。

⚠️ 7. _format_test_case 中没有处理 invalid expected_output 或 expected_type 组合
有些 test case 只提供了 description 和 input，如果用户误用，最终测试器可能报错。

建议加校验，例如如果缺少 expected_output 和 expected_type 都报错或日志警告。

✅ 推荐补充的功能点（高级）
函数依赖分析与提示

可在注册函数时分析其依赖（如是否引用某些常见辅助函数），并提示是否需要同时注册。

代码版本控制

注册函数支持 version 增量和修改记录。

函数执行性能监控

可记录每次执行时间，添加超时保护、资源上限。

支持异步函数（async def）

当前版本只适用于同步函数，如后续引入异步 Agent 架构，可扩展对 async 的处理。


"""

# Configure logger
logger = logging.getLogger(__name__)


class EnhancedFunctionTools:
    """Enhanced tools for managing dynamic function generation and registration with security."""

    def __init__(self):
        self.registry = get_registry()
        self.secure_executor = get_secure_executor()
        self.test_runner = TestRunner()
        logger.info("Enhanced Function Tools initialized")
    
    def has_function(self, func_name: str) -> bool:
        """Check if a function exists in the registry."""
        return self.registry.has_function(func_name)
    
    def search_functions(self, query: str, tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for functions that might match the query.

        Args:
            query: Search query string
            tags: Optional list of tags to filter by

        Returns:
            List of function information dictionaries
        """
        try:
            matching_names = self.registry.search_functions(query, tags)
            functions_info = []

            for name in matching_names:
                meta = self.registry.get_function_metadata(name)
                if meta:
                    functions_info.append({
                        'name': name,
                        'description': meta.description,
                        'signature': meta.signature,
                        'docstring': meta.docstring,
                        'tags': meta.tags,
                        'author': meta.author,
                        'version': meta.version,
                        'usage_count': meta.usage_count
                    })
                else:
                    logger.warning(f"Metadata not found for function: {name}")

            return functions_info

        except Exception as e:
            logger.error(f"Error searching functions: {e}")
            return []
    
    def validate_function_code(self, code: str, allow_multiple_functions: bool = False) -> Tuple[bool, str, Optional[str]]:
        """
        Validate function code syntax and security using enhanced security executor.

        Args:
            code: Python function code to validate
            allow_multiple_functions: Whether to allow multiple function definitions (default: False)

        Returns:
            Tuple[bool, str, Optional[str]]: A 3-tuple containing:
                - is_valid (bool): True if code is valid and safe
                - error_message (str): Error description if validation fails, empty string if success
                - function_name (Optional[str]): Extracted function name if valid, None if invalid

        Note:
            By default, only single function definitions are allowed to maintain
            simplicity and security. Set allow_multiple_functions=True to allow
            helper functions within the code block.
        """
        try:
            # Parse the code to check syntax and extract function info
            tree = ast.parse(code)

            # Find function definitions
            func_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if not func_defs:
                return False, "No function definition found in code", None

            if not allow_multiple_functions and len(func_defs) > 1:
                return False, "Multiple function definitions found. Please provide only one function, or set allow_multiple_functions=True.", None

            # Get the main function name (first one found)
            main_func_name = func_defs[0].name

            # Use secure executor for comprehensive security validation
            is_valid, violations = self.secure_executor.validate_code(code)
            if not is_valid:
                error_msg = f"Security violations detected: {'; '.join(violations)}"
                logger.warning(f"Code validation failed for {main_func_name}: {error_msg}")
                return False, error_msg, main_func_name

            logger.debug(f"Code validation passed for function: {main_func_name}")
            return True, "", main_func_name

        except SyntaxError as e:
            error_msg = f"Syntax error: {e}"
            logger.error(f"Syntax error in code validation: {error_msg}")
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Validation error: {e}"
            logger.error(f"Unexpected error in code validation: {error_msg}")
            return False, error_msg, None



    def generate_test_cases(self, func_name: str, func_code: str,
                          task_description: str, max_cases: int = 5) -> List[Dict[str, Any]]:
        """
        Generate comprehensive test cases for a function based on its signature and task.

        Args:
            func_name: Name of the function
            func_code: Function source code
            task_description: Description of what the function should do
            max_cases: Maximum number of test cases to generate

        Returns:
            List of test case dictionaries with consistent format
        """
        try:
            # Use secure executor to safely get function signature
            success, result = self.secure_executor.execute_code(func_code, timeout=5)
            if not success:
                logger.error(f"Failed to execute code for test generation: {result}")
                return []

            func = result.get(func_name)
            if not func:
                logger.error(f"Function {func_name} not found in executed code")
                return []

            # Get function signature with enhanced type information
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            # Try to get type hints for better type inference
            try:
                type_hints = get_type_hints(func)
            except (NameError, AttributeError):
                type_hints = {}

            test_cases = []

            # Enhanced test case generation based on function characteristics
            test_cases.extend(self._generate_pattern_based_tests(
                func_name, task_description, sig, type_hints, params
            ))

            # Generate boundary and edge case tests
            test_cases.extend(self._generate_boundary_tests(
                func_name, sig, type_hints, params
            ))

            # Limit the number of test cases
            if len(test_cases) > max_cases:
                test_cases = test_cases[:max_cases]
                logger.info(f"Limited test cases to {max_cases} for function {func_name}")

            # Ensure consistent format
            formatted_cases = []
            for case in test_cases:
                formatted_case = self._format_test_case(case)
                if formatted_case:
                    formatted_cases.append(formatted_case)

            logger.info(f"Generated {len(formatted_cases)} test cases for {func_name}")
            return formatted_cases

        except Exception as e:
            logger.error(f"Failed to generate test cases for {func_name}: {e}")
            return []

    def _generate_pattern_based_tests(self, func_name: str, task_description: str,
                                    sig: inspect.Signature, type_hints: Dict,
                                    params: List[str]) -> List[Dict[str, Any]]:
        """Generate test cases based on function name and description patterns."""
        test_cases = []
        func_name_lower = func_name.lower()
        task_lower = task_description.lower()

        # Email validation pattern
        if 'email' in func_name_lower or 'email' in task_lower:
            test_cases.extend([
                {
                    'input': {'email': 'test@example.com'},
                    'expected_output': True,
                    'description': 'Valid email test'
                },
                {
                    'input': {'email': 'invalid-email'},
                    'expected_output': False,
                    'description': 'Invalid email test'
                },
                {
                    'input': {'email': 'user@domain.co.uk'},
                    'expected_output': True,
                    'description': 'Valid email with subdomain'
                }
            ])

        # Phone validation pattern
        elif 'phone' in func_name_lower or 'phone' in task_lower:
            test_cases.extend([
                {
                    'input': {'phone': '+1234567890'},
                    'expected_output': True,
                    'description': 'Valid phone with country code'
                },
                {
                    'input': {'phone': '1234567890'},
                    'expected_output': True,
                    'description': 'Valid phone without country code'
                },
                {
                    'input': {'phone': '123'},
                    'expected_output': False,
                    'description': 'Invalid short phone'
                }
            ])

        # Mathematical functions
        elif any(keyword in func_name_lower for keyword in ['calculate', 'compute', 'math', 'fibonacci', 'factorial']):
            if len(params) >= 1:
                param_name = params[0]
                test_cases.extend([
                    {
                        'input': {param_name: 5},
                        'expected_type': 'number',
                        'description': 'Basic calculation test'
                    },
                    {
                        'input': {param_name: 0},
                        'expected_type': 'number',
                        'description': 'Zero input test'
                    }
                ])

        # String processing functions
        elif any(keyword in func_name_lower for keyword in ['process', 'format', 'clean', 'parse']):
            if len(params) >= 1:
                param_name = params[0]
                test_cases.extend([
                    {
                        'input': {param_name: 'test string'},
                        'expected_type': 'str',
                        'description': 'Basic string processing'
                    },
                    {
                        'input': {param_name: ''},
                        'expected_type': 'str',
                        'description': 'Empty string test'
                    }
                ])

        return test_cases

    def _generate_boundary_tests(self, func_name: str, sig: inspect.Signature,
                               type_hints: Dict, params: List[str]) -> List[Dict[str, Any]]:
        """Generate boundary and edge case tests."""
        test_cases = []

        for param_name in params:
            param = sig.parameters[param_name]
            param_type = type_hints.get(param_name, param.annotation)

            # Generate tests based on parameter type
            if param_type == int or 'int' in str(param_type).lower():
                test_cases.extend([
                    {
                        'input': {param_name: -1},
                        'expected_type': 'auto_generated',
                        'description': f'Negative integer test for {param_name}'
                    },
                    {
                        'input': {param_name: 0},
                        'expected_type': 'auto_generated',
                        'description': f'Zero test for {param_name}'
                    },
                    {
                        'input': {param_name: 1},
                        'expected_type': 'auto_generated',
                        'description': f'Positive integer test for {param_name}'
                    }
                ])

            elif param_type == str or 'str' in str(param_type).lower():
                test_cases.extend([
                    {
                        'input': {param_name: ''},
                        'expected_type': 'auto_generated',
                        'description': f'Empty string test for {param_name}'
                    },
                    {
                        'input': {param_name: 'a'},
                        'expected_type': 'auto_generated',
                        'description': f'Single character test for {param_name}'
                    }
                ])

        return test_cases

    def _format_test_case(self, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format test case to ensure consistent structure."""
        try:
            formatted_case = {
                'input': case.get('input', {}),
                'description': case.get('description', 'Auto-generated test case')
            }

            # Handle expected output vs expected type
            if 'expected_output' in case:
                formatted_case['expected_output'] = case['expected_output']
            elif 'expected_type' in case:
                formatted_case['expected_type'] = case['expected_type']
            else:
                formatted_case['expected_type'] = 'auto_generated'

            # Ensure input is properly formatted
            if isinstance(formatted_case['input'], dict):
                # Convert input values to proper string representation for execution
                formatted_input = {}
                for key, value in formatted_case['input'].items():
                    formatted_input[key] = value
                formatted_case['input'] = formatted_input

            return formatted_case

        except Exception as e:
            logger.error(f"Error formatting test case: {e}")
            return None
    
    def test_function(self, func_code: str, func_name: str,
                     test_cases: List[Dict[str, Any]] = None,
                     auto_generate_tests: bool = True,
                     task_description: str = "") -> Tuple[bool, str, List[Dict]]:
        """
        Test a function with provided or generated test cases using secure execution.

        Args:
            func_code: Function source code
            func_name: Name of the function to test
            test_cases: Optional list of test cases
            auto_generate_tests: Whether to auto-generate tests if none provided
            task_description: Description for auto-generating relevant tests

        Returns:
            Tuple[bool, str, List[Dict]]: (success, error_message, test_results)

        Note:
            Test results include detailed information about each test execution,
            including input, expected output, actual output, and pass/fail status.
        """
        try:
            # Validate code first using secure validation
            is_valid, error_msg, extracted_name = self.validate_function_code(func_code)
            if not is_valid:
                logger.error(f"Function validation failed: {error_msg}")
                return False, error_msg, []

            if extracted_name != func_name:
                error_msg = f"Function name mismatch: expected {func_name}, found {extracted_name}"
                logger.error(error_msg)
                return False, error_msg, []

            # Auto-generate test cases if none provided and requested
            if not test_cases and auto_generate_tests:
                logger.info(f"Auto-generating test cases for {func_name}")
                test_cases = self.generate_test_cases(func_name, func_code, task_description)
                if not test_cases:
                    logger.warning(f"No test cases generated for {func_name}")

            # Use secure test runner with enhanced error handling
            success, error_msg, results = self.test_runner.run_tests(func_code, func_name, test_cases or [])

            if success:
                logger.info(f"Function {func_name} passed all tests ({len(results)} test cases)")
            else:
                logger.warning(f"Function {func_name} failed tests: {error_msg}")

            return success, error_msg, results

        except Exception as e:
            error_msg = f"Test execution failed: {e}"
            logger.error(f"Unexpected error in test_function: {error_msg}")
            return False, error_msg, []
    
    def register_function(self, func_name: str, func_code: str,
                         description: str, task_origin: str = "",
                         test_cases: List[Dict] = None,
                         require_tests: bool = True,
                         auto_generate_tests: bool = True) -> bool:
        """
        Register a function to the registry with optional pre-registration testing.

        Args:
            func_name: Name of the function
            func_code: Function source code
            description: Description of what the function does
            task_origin: Origin/source of the task
            test_cases: Optional test cases
            require_tests: Whether to require passing tests before registration
            auto_generate_tests: Whether to auto-generate tests if none provided

        Returns:
            bool: True if registration successful, False otherwise

        Note:
            If require_tests is True, the function must pass all tests before
            being registered. This helps ensure code quality and correctness.
        """
        try:
            # Pre-registration testing if required
            if require_tests:
                logger.info(f"Running pre-registration tests for {func_name}")
                test_success, test_error, test_results = self.test_function(
                    func_code, func_name, test_cases, auto_generate_tests, description
                )

                if not test_success:
                    logger.error(f"Pre-registration testing failed for {func_name}: {test_error}")
                    return False

                # Use the test results for registration if no test cases were provided
                if not test_cases and test_results:
                    test_cases = test_results
                    logger.info(f"Using {len(test_cases)} auto-generated test cases for registration")

            # Register the function
            success = self.registry.register_function(
                func_name, func_code, description, task_origin, test_cases
            )

            if success:
                logger.info(f"Successfully registered function: {func_name}")
            else:
                logger.error(f"Failed to register function: {func_name}")

            return success

        except Exception as e:
            logger.error(f"Error registering function {func_name}: {e}")
            return False
    
    def get_function_info(self, func_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a registered function.

        Args:
            func_name: Name of the function

        Returns:
            Dictionary with function information or None if not found
        """
        try:
            if not self.has_function(func_name):
                logger.warning(f"Function {func_name} not found in registry")
                return None

            # Get metadata using the proper method
            meta = self.registry.get_function_metadata(func_name)
            func = self.registry.get_function(func_name)

            if not meta:
                logger.error(f"Metadata not found for function: {func_name}")
                return None

            return {
                'name': func_name,
                'function': func,
                'description': meta.description,
                'signature': meta.signature,
                'docstring': meta.docstring,
                'created_at': meta.created_at,
                'updated_at': meta.updated_at,
                'version': meta.version,
                'task_origin': meta.task_origin,
                'test_cases': meta.test_cases,
                'tags': meta.tags,
                'author': meta.author,
                'usage_count': meta.usage_count,
                'last_used': meta.last_used,
                'dependencies': meta.dependencies
            }

        except Exception as e:
            logger.error(f"Error getting function info for {func_name}: {e}")
            return None
    
    def list_all_functions(self) -> List[Dict[str, Any]]:
        """
        List all registered functions with their metadata.

        Returns:
            List of dictionaries containing function information
        """
        try:
            return self.registry.list_functions()
        except Exception as e:
            logger.error(f"Error listing functions: {e}")
            return []


# Global instance
_function_tools = None

def get_function_tools() -> EnhancedFunctionTools:
    """Get the global enhanced function tools instance."""
    global _function_tools
    if _function_tools is None:
        _function_tools = EnhancedFunctionTools()
    return _function_tools

# Backward compatibility alias
FunctionTools = EnhancedFunctionTools
