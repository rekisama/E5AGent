"""
Enhanced Test runner for dynamically generated functions.

Provides secure, cross-platform execution environment for testing generated code
with comprehensive timeout handling and flexible result comparison.
"""

import sys
import io
import traceback
import signal
import time
import math
import logging
from contextlib import contextmanager
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from multiprocessing import Process, Queue
import platform

"""
1. ✅ 增加支持自定义比较函数
某些测试用例可能需要定制比较逻辑：

python
复制
编辑
test_case = {
    "input": {"x": 2},
    "expected_value": None,
    "custom_compare": lambda r: isinstance(r, int) and r % 2 == 0
}
修改 run_tests 支持：

python
复制
编辑
custom_compare = test_case.get('custom_compare')
if custom_compare:
    try:
        passed = custom_compare(result)
        if not passed:
            test_result['success'] = False
            test_result['error'] = "Custom compare failed"
            all_passed = False
    except Exception as e:
        test_result['success'] = False
        test_result['error'] = f"Custom compare error: {e}"
        all_passed = False
2. ✅ 增加输出捕获功能（可选）
有些函数可能通过 print() 输出关键中间值。可选支持将标准输出一并捕获：

python
复制
编辑
from contextlib import redirect_stdout
f = io.StringIO()
with redirect_stdout(f):
    result = eval(eval_expr or f"{func_name}(**inputs)", global_scope, local_scope)
stdout = f.getvalue()
queue.put((True, result, "", stdout))
3. 🟡 函数未定义或语法错误时，success == False 但 error 不明确
在 safe_execute 中，如果：

python
复制
编辑
if func_name not in local_scope:
    queue.put((False, None, f"Function {func_name} not found"))
但如果代码根本就语法错、SyntaxError 之类，local_scope 也不会生成，建议统一 exec() 报错时直接退出。

4. 🟡 类型验证范围可扩展
当前支持基本类型：

python
复制
编辑
'bool', 'str', 'int', 'float', 'number', 'list', 'dict'
你可以加入：

'set'

'tuple'

'None'

'callable'

5. 🟡 支持测试报告导出（如 JSON）
可在 run_tests 最后返回时附带结构化字段：

python
复制
编辑
return {
    "summary": {
        "passed": passed_count,
        "failed": failed_count,
        "total": len(test_results),
        "message": error_msg,
        "success": all_passed,
    },
    "details": test_results
}
"""

# Configure logger
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Raised when function execution times out."""
    pass


def _safe_import(name: str):
    """
    Safe import function that only allows whitelisted modules.

    Args:
        name: Module name to import

    Returns:
        Imported module if safe, raises ImportError otherwise
    """
    # Whitelist of safe modules
    safe_modules = {
        're', 'math', 'datetime', 'json', 'random', 'string',
        'itertools', 'functools', 'collections', 'decimal',
        'fractions', 'statistics', 'uuid', 'hashlib', 'base64'
    }

    if name in safe_modules:
        return __import__(name)
    else:
        raise ImportError(f"Module '{name}' is not allowed for security reasons")


def _run_func_in_process(code: str, func_name: str, inputs: Dict[str, Any],
                        eval_expr: Optional[str], queue: Queue):
    """
    Execute function in a separate process for cross-platform timeout support.

    Args:
        code: Function source code
        func_name: Name of the function to execute
        inputs: Input parameters for the function
        eval_expr: Optional expression to evaluate (for class methods, etc.)
        queue: Queue to return results
    """
    try:
        # Create secure global scope with essential functions for basic Python functionality
        global_scope = {
            '__builtins__': {
                # Safe built-in functions
                'len': len, 'str': str, 'int': int, 'float': float,
                'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
                'set': set, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'round': round, 'isinstance': isinstance,
                'print': print,
                # Essential for class definitions and exceptions
                '__build_class__': __build_class__,
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'AttributeError': AttributeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'ZeroDivisionError': ZeroDivisionError,
                'ImportError': ImportError,
                'NameError': NameError,
                # Safe import mechanism (restricted to whitelisted modules)
                '__import__': lambda name, *args, **kwargs: _safe_import(name),
                # Removed dangerous functions: hasattr, getattr, globals, locals, vars, dir, open, eval, exec
            },
            # Essential module variables
            '__name__': '__main__',
            '__file__': '<string>',
            # Pre-import safe modules only
            're': __import__('re'),
            'math': __import__('math'),
            'datetime': __import__('datetime'),
            'json': __import__('json'),
        }

        local_scope = {}
        exec(code, global_scope, local_scope)

        if eval_expr:
            # Support for class methods and complex expressions
            result = eval(eval_expr, global_scope, local_scope)
        else:
            # Standard function execution
            if func_name not in local_scope:
                queue.put((False, None, f"Function {func_name} not found"))
                return
            result = local_scope[func_name](**inputs)

        queue.put((True, result, ""))

    except Exception as e:
        queue.put((False, None, f"{type(e).__name__}: {str(e)}"))


class EnhancedTestRunner:
    """
    Enhanced secure test runner with cross-platform timeout support.

    Features:
    - Cross-platform timeout using multiprocessing
    - Secure execution environment without dangerous built-ins
    - Flexible result comparison (float tolerance, deep comparison)
    - Support for class methods and complex expressions
    - Comprehensive error handling and logging
    """

    def __init__(self, timeout_seconds: int = 30, float_tolerance: float = 1e-9):
        """
        Initialize enhanced test runner.

        Args:
            timeout_seconds: Maximum execution time per test
            float_tolerance: Tolerance for floating point comparisons
        """
        self.timeout_seconds = timeout_seconds
        self.float_tolerance = float_tolerance
        logger.info(f"Enhanced test runner initialized with {timeout_seconds}s timeout")

    def safe_execute(self, code: str, func_name: str,
                    test_input: Dict[str, Any],
                    eval_expr: Optional[str] = None) -> Tuple[bool, Any, str]:
        """
        Safely execute a function with cross-platform timeout support.

        Args:
            code: Function source code
            func_name: Name of the function to execute
            test_input: Input parameters
            eval_expr: Optional expression for class methods (e.g., "MyClass().method")

        Returns:
            Tuple[bool, Any, str]: (success, result, error_message)
        """
        try:
            # Use multiprocessing for true cross-platform timeout
            queue = Queue()
            process = Process(
                target=_run_func_in_process,
                args=(code, func_name, test_input, eval_expr, queue)
            )

            process.start()
            process.join(timeout=self.timeout_seconds)

            if process.is_alive():
                # Process timed out
                process.terminate()
                process.join()  # Wait for cleanup
                error_msg = f"Execution timed out after {self.timeout_seconds} seconds"
                logger.warning(f"Function {func_name} timed out")
                return False, None, error_msg

            # Get result from queue
            if not queue.empty():
                success, result, error_msg = queue.get()
                if success:
                    logger.debug(f"Function {func_name} executed successfully")
                else:
                    logger.warning(f"Function {func_name} failed: {error_msg}")
                return success, result, error_msg
            else:
                return False, None, "No result returned from process"

        except Exception as e:
            error_msg = f"Process execution error: {e}"
            logger.error(f"Error executing {func_name}: {error_msg}")
            return False, None, error_msg
    
    def flexible_compare(self, actual: Any, expected: Any) -> Tuple[bool, str]:
        """
        Flexible comparison supporting different data types.

        Args:
            actual: Actual result from function
            expected: Expected result

        Returns:
            Tuple[bool, str]: (is_equal, error_message)
        """
        try:
            # Handle None values
            if actual is None and expected is None:
                return True, ""
            if actual is None or expected is None:
                return False, f"Expected {expected}, got {actual}"

            # Float comparison with tolerance
            if isinstance(actual, float) and isinstance(expected, float):
                if math.isclose(actual, expected, rel_tol=self.float_tolerance, abs_tol=self.float_tolerance):
                    return True, ""
                else:
                    return False, f"Expected {expected}, got {actual} (difference: {abs(actual - expected)})"

            # Mixed number comparison
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                if math.isclose(float(actual), float(expected), rel_tol=self.float_tolerance, abs_tol=self.float_tolerance):
                    return True, ""
                else:
                    return False, f"Expected {expected}, got {actual}"

            # String comparison (case-insensitive option could be added)
            if isinstance(actual, str) and isinstance(expected, str):
                if actual == expected:
                    return True, ""
                else:
                    return False, f"Expected '{expected}', got '{actual}'"

            # List comparison (order matters)
            if isinstance(actual, list) and isinstance(expected, list):
                if len(actual) != len(expected):
                    return False, f"Expected list of length {len(expected)}, got {len(actual)}"

                for i, (a, e) in enumerate(zip(actual, expected)):
                    is_equal, error = self.flexible_compare(a, e)
                    if not is_equal:
                        return False, f"List element {i}: {error}"
                return True, ""

            # Dictionary comparison (order doesn't matter)
            if isinstance(actual, dict) and isinstance(expected, dict):
                if set(actual.keys()) != set(expected.keys()):
                    return False, f"Expected keys {set(expected.keys())}, got {set(actual.keys())}"

                for key in expected:
                    is_equal, error = self.flexible_compare(actual[key], expected[key])
                    if not is_equal:
                        return False, f"Dict key '{key}': {error}"
                return True, ""

            # Set comparison (order doesn't matter)
            if isinstance(actual, set) and isinstance(expected, set):
                if actual == expected:
                    return True, ""
                else:
                    return False, f"Expected set {expected}, got {actual}"

            # Default comparison
            if actual == expected:
                return True, ""
            else:
                return False, f"Expected {expected}, got {actual}"

        except Exception as e:
            return False, f"Comparison error: {e}"
    
    def validate_result(self, result: Any, expected_type: str) -> Tuple[bool, str]:
        """
        Validate function result against expected type.
        
        Args:
            result: The actual result from function execution
            expected_type: Expected type ('bool', 'str', 'int', 'float', 'list', 'dict', 'number')
        
        Returns:
            (is_valid, error_message)
        """
        try:
            if expected_type == 'bool':
                if not isinstance(result, bool):
                    return False, f"Expected bool, got {type(result).__name__}"
            elif expected_type == 'str':
                if not isinstance(result, str):
                    return False, f"Expected str, got {type(result).__name__}"
            elif expected_type == 'int':
                if not isinstance(result, int):
                    return False, f"Expected int, got {type(result).__name__}"
            elif expected_type == 'float':
                if not isinstance(result, float):
                    return False, f"Expected float, got {type(result).__name__}"
            elif expected_type == 'number':
                if not isinstance(result, (int, float)):
                    return False, f"Expected number, got {type(result).__name__}"
            elif expected_type == 'list':
                if not isinstance(result, list):
                    return False, f"Expected list, got {type(result).__name__}"
            elif expected_type == 'dict':
                if not isinstance(result, dict):
                    return False, f"Expected dict, got {type(result).__name__}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def run_tests(self, func_code: str, func_name: str,
                  test_cases: List[Dict[str, Any]],
                  eval_expr: Optional[str] = None) -> Tuple[bool, str, List[Dict]]:
        """
        Run all test cases for a function with enhanced comparison.

        Args:
            func_code: Function source code
            func_name: Name of the function to test
            test_cases: List of test cases
            eval_expr: Optional expression for class methods

        Returns:
            Tuple[bool, str, List[Dict]]: (all_passed, error_message, test_results)
        """
        if not test_cases:
            # If no test cases provided, just try to compile the function
            try:
                success, result = self.safe_execute(func_code, func_name, {}, eval_expr)
                if success or "not found" not in str(result):
                    logger.info(f"Function {func_name} compiled successfully")
                    return True, "No test cases provided, but function compiled successfully", []
                else:
                    return False, f"Function {func_name} compilation failed", []
            except Exception as e:
                return False, f"Function compilation failed: {e}", []

        test_results = []
        all_passed = True

        for i, test_case in enumerate(test_cases):
            test_input = test_case.get('input', {})
            expected_type = test_case.get('expected_type', 'any')
            expected_value = test_case.get('expected_value')
            expected_output = test_case.get('expected_output')  # Alternative key
            description = test_case.get('description', f'Test case {i+1}')

            # Use expected_output if expected_value not provided
            if expected_value is None and expected_output is not None:
                expected_value = expected_output

            logger.debug(f"Running test case {i+1}: {description}")

            # Execute the test with enhanced security
            success, result, error_msg = self.safe_execute(
                func_code, func_name, test_input, eval_expr
            )

            test_result = {
                'description': description,
                'input': test_input,
                'success': success,
                'result': result,
                'error': error_msg,
                'passed': success  # For backward compatibility
            }

            if success:
                # Validate result type if specified
                if expected_type != 'any':
                    type_valid, type_error = self.validate_result(result, expected_type)
                    if not type_valid:
                        test_result['success'] = False
                        test_result['passed'] = False
                        test_result['error'] = type_error
                        all_passed = False
                        logger.warning(f"Type validation failed for {func_name}: {type_error}")

                # Enhanced value comparison
                if expected_value is not None:
                    is_equal, compare_error = self.flexible_compare(result, expected_value)
                    if not is_equal:
                        test_result['success'] = False
                        test_result['passed'] = False
                        test_result['error'] = compare_error
                        all_passed = False
                        logger.warning(f"Value comparison failed for {func_name}: {compare_error}")
                    else:
                        logger.debug(f"Test case {i+1} passed: {description}")
            else:
                all_passed = False
                logger.warning(f"Test case {i+1} failed: {error_msg}")

            test_results.append(test_result)

        if all_passed:
            logger.info(f"All {len(test_results)} tests passed for {func_name}")
            return True, "All tests passed", test_results
        else:
            failed_count = sum(1 for r in test_results if not r.get('success', False))
            error_msg = f"{failed_count} out of {len(test_results)} tests failed"
            logger.warning(f"Function {func_name}: {error_msg}")
            return False, error_msg, test_results


# Backward compatibility
TestRunner = EnhancedTestRunner
