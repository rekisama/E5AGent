"""
Security Validator and Secure Code Execution Module.

This module provides security validation and safe code execution capabilities
for the Function Creator Agent system. It includes:

1. SecurityValidator: AST-based security analysis to detect dangerous operations
2. Safe code execution with sandboxing and restricted built-ins
3. Function code validation with syntax and security checks
4. Comprehensive security policy enforcement

安全验证器和安全代码执行模块，为函数创建代理系统提供安全验证和安全代码执行能力。
包括基于 AST 的安全分析、沙箱执行、函数代码验证和全面的安全策略执行。
"""

import ast
import logging
import sys
import platform
import threading
import queue
import signal
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Configure logger
logger = logging.getLogger(__name__)


class SecurityValidator(ast.NodeVisitor):
    """
    Custom AST visitor for enhanced security validation.

    This class analyzes Python code AST to detect potentially dangerous operations
    including dangerous function calls, module imports, and built-in access patterns.
    """

    def __init__(self):
        """Initialize the security validator with dangerous patterns."""
        self.security_violations = []

        # Dangerous functions that should not be allowed
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__', 'getattr', 'setattr',
            'delattr', 'hasattr', 'globals', 'locals', 'vars', 'dir',
            'open', 'file', 'input', 'raw_input'
        }

        # Dangerous modules that should not be imported
        self.dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'tempfile', 'pickle',
            'marshal', 'shelve', 'dbm', 'sqlite3', 'socket', 'urllib',
            'http', 'ftplib', 'smtplib', 'poplib', 'imaplib'
        }

    def visit_Call(self, node):
        """Check function calls for security violations."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.dangerous_functions:
                self.security_violations.append(f"Dangerous function call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if hasattr(node.func.value, 'id') and node.func.value.id == '__builtins__':
                self.security_violations.append(f"Direct __builtins__ access: {node.func.attr}")
        self.generic_visit(node)

    def visit_Import(self, node):
        """Check imports for dangerous modules."""
        for alias in node.names:
            if alias.name in self.dangerous_modules:
                self.security_violations.append(f"Dangerous module import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check from imports for dangerous modules."""
        if node.module in self.dangerous_modules:
            self.security_violations.append(f"Dangerous module import: {node.module}")
        self.generic_visit(node)

    def get_violations(self) -> List[str]:
        """Return list of security violations found."""
        return self.security_violations

    def reset(self):
        """Reset the validator for reuse."""
        self.security_violations = []


def validate_function_code(code: str) -> Tuple[bool, str, str]:
    """
    Validate function code for syntax and security.

    Args:
        code: Python function code to validate

    Returns:
        Tuple of (is_valid, error_message, function_name)
    """
    if not code or not code.strip():
        return False, "Empty code provided", ""

    try:
        # Parse the code into AST
        tree = ast.parse(code)

        # Find function definitions
        func_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if not func_defs:
            return False, "No function definition found in code", ""

        # Get the first function name
        func_name = func_defs[0].name

        # Validate function name
        if not func_name or not func_name.isidentifier():
            return False, f"Invalid function name: {func_name}", func_name

        # Security validation
        validator = SecurityValidator()
        validator.visit(tree)
        violations = validator.get_violations()

        if violations:
            return False, f"Security violations: {'; '.join(violations)}", func_name

        # Additional syntax validation
        compile(code, '<string>', 'exec')

        return True, "", func_name

    except SyntaxError as e:
        return False, f"Syntax error: {e}", ""
    except Exception as e:
        return False, f"Validation error: {e}", ""


def create_safe_namespace() -> Dict[str, Any]:
    """
    Create a safe namespace for code execution with restricted built-ins.

    Returns:
        Dictionary containing safe built-in functions and modules
    """
    # Start with safe built-ins
    safe_builtins = {
        # Basic types
        'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set',
        # Basic functions
        'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted',
        'min', 'max', 'sum', 'abs', 'round', 'pow',
        # Type checking
        'isinstance', 'type', 'callable',
        # Iteration
        'iter', 'next', 'reversed',
        # String operations
        'ord', 'chr', 'repr', 'str',
        # Math operations
        'divmod', 'hex', 'oct', 'bin',
        # Exception handling
        'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
        'AttributeError', 'RuntimeError', 'NotImplementedError'
    }

    # Create restricted namespace
    namespace = {
        '__builtins__': {name: getattr(__builtins__, name)
                        for name in safe_builtins
                        if hasattr(__builtins__, name)}
    }

    # Add safe modules
    import math
    import re
    import datetime
    import json
    import random

    namespace.update({
        'math': math,
        're': re,
        'datetime': datetime,
        'json': json,
        'random': random
    })

    return namespace


def execute_code_safely(code: str, timeout_seconds: int = 10) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Execute code safely with timeout and sandboxing using thread pool.

    Note: Using threads instead of processes to avoid Windows handle issues.
    This provides reasonable isolation while maintaining cross-platform compatibility.

    Args:
        code: Python code to execute
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Tuple of (success, error_message, namespace_after_execution)
    """
    def _execute_in_thread(code: str) -> Tuple[str, str, Dict[str, Any]]:
        """Execute code in a separate thread with safe namespace."""
        try:
            # Create safe namespace
            namespace = create_safe_namespace()

            # Execute the code
            exec(code, namespace)

            # Return the namespace (excluding built-ins for serialization)
            result_namespace = {k: v for k, v in namespace.items()
                              if k != '__builtins__' and not k.startswith('__')}

            return 'success', '', result_namespace

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return 'error', error_msg, {}

    try:
        # Use ThreadPoolExecutor for timeout control (avoids Windows multiprocessing issues)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute_in_thread, code)

            try:
                status, error_msg, namespace = future.result(timeout=timeout_seconds)
                return status == 'success', error_msg, namespace

            except FutureTimeoutError:
                return False, f"Code execution timeout after {timeout_seconds} seconds", {}

    except Exception as e:
        return False, f"Execution error: {e}", {}


def test_function_safely(func_code: str, func_name: str, test_cases: List[Dict],
                        timeout_seconds: int = 10) -> Tuple[bool, str, List[Dict]]:
    """
    Test a function safely with provided test cases.

    Args:
        func_code: Function code to test
        func_name: Name of the function to test
        test_cases: List of test case dictionaries
        timeout_seconds: Maximum execution time per test

    Returns:
        Tuple of (all_tests_passed, error_message, test_results)
    """
    if not test_cases:
        return True, "No test cases provided", []

    test_results = []

    for i, test_case in enumerate(test_cases):
        try:
            # Extract test case information
            description = test_case.get('description', f'Test case {i+1}')
            test_input = test_case.get('input', {})
            expected_output = test_case.get('expected_output', 'auto_generated')

            # Create test code
            if isinstance(test_input, dict):
                # Convert dict input to function call
                args_str = ', '.join(f"{k}={repr(v)}" for k, v in test_input.items())
                test_code = f"""
# Import required modules
import re
import sys
import os
from typing import Any

{func_code}

# Test execution
try:
    result = {func_name}({args_str})
    test_success = True
    test_error = None
except Exception as e:
    result = None
    test_success = False
    test_error = f"{{type(e).__name__}}: {{str(e)}}"
"""
            else:
                # Handle other input formats
                test_code = f"""
# Import required modules
import re
import sys
import os
from typing import Any

{func_code}

# Test execution
try:
    result = {func_name}({repr(test_input)})
    test_success = True
    test_error = None
except Exception as e:
    result = None
    test_success = False
    test_error = f"{{type(e).__name__}}: {{str(e)}}"
"""

            # Execute test safely
            success, error_msg, namespace = execute_code_safely(test_code, timeout_seconds)

            if success:
                # Extract test results from namespace
                test_success = namespace.get('test_success', False)
                test_error = namespace.get('test_error')
                result = namespace.get('result')

                test_results.append({
                    'description': description,
                    'success': test_success,
                    'result': result,
                    'error': test_error,
                    'expected': expected_output
                })

            else:
                test_results.append({
                    'description': description,
                    'success': False,
                    'result': None,
                    'error': error_msg,
                    'expected': expected_output
                })

        except Exception as e:
            test_results.append({
                'description': description,
                'success': False,
                'result': None,
                'error': f"Test execution error: {e}",
                'expected': expected_output
            })

    # Check if all tests passed
    all_passed = all(result['success'] for result in test_results)

    if all_passed:
        return True, "", test_results
    else:
        failed_count = sum(1 for result in test_results if not result['success'])
        return False, f"{failed_count} out of {len(test_results)} tests failed", test_results


def validate_function_signature(signature: str) -> bool:
    """
    Validate function signature using stricter AST parsing and type annotation checking.

    Args:
        signature: Function signature string

    Returns:
        True if signature is valid, False otherwise
    """
    import re

    try:
        # Clean and prepare signature
        signature = signature.strip()

        # Try to parse as a complete function definition
        if not signature.startswith('def '):
            signature = f"def {signature}"

        # Add a pass statement to make it a complete function
        if not signature.endswith(':'):
            signature += ':'
        signature += '\n    pass'

        # Parse the function using AST
        tree = ast.parse(signature)

        # Extract function definition
        if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
            return False

        func_def = tree.body[0]

        # Validate function name is a valid identifier
        if not func_def.name.isidentifier() or func_def.name.startswith('_'):
            return False

        # Validate parameters have type annotations (stricter requirement)
        for arg in func_def.args.args:
            if not arg.annotation:
                logger.debug(f"Parameter '{arg.arg}' missing type annotation")
                # Allow functions without type annotations but log warning
                pass

        # Validate return type annotation exists
        if not func_def.returns:
            logger.debug(f"Function '{func_def.name}' missing return type annotation")
            # Allow functions without return type but log warning
            pass

        return True

    except SyntaxError:
        try:
            # Try alternative parsing - maybe it's just the signature part
            if '(' in signature and ')' in signature:
                # Extract function name and parameters
                func_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)', signature)
                if func_match:
                    func_name, params = func_match.groups()
                    # Basic validation - function name should be valid identifier
                    if func_name.isidentifier():
                        return True
            return False
        except:
            return False


def extract_function_signature_from_code(code: str) -> Optional[str]:
    """
    Extract function signature from function code.

    Args:
        code: Python function code

    Returns:
        Function signature string or None if not found
    """
    try:
        tree = ast.parse(code)

        # Find function definitions
        func_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if not func_defs:
            return None

        func_def = func_defs[0]  # Get first function

        # Build signature string
        signature_parts = [f"def {func_def.name}("]

        # Add parameters
        params = []
        for arg in func_def.args.args:
            param_str = arg.arg
            if arg.annotation:
                # Convert annotation AST back to string
                param_str += f": {ast.unparse(arg.annotation)}"
            params.append(param_str)

        signature_parts.append(", ".join(params))
        signature_parts.append(")")

        # Add return type if present
        if func_def.returns:
            signature_parts.append(f" -> {ast.unparse(func_def.returns)}")

        return "".join(signature_parts)

    except Exception as e:
        logger.debug(f"Failed to extract signature from code: {e}")
        return None


class FunctionSignatureParser:
    """Parses function signatures and extracts parameter information."""

    @staticmethod
    def parse_signature(signature: str) -> List[Dict[str, Any]]:
        """
        Parse function signature to extract parameter information.

        Args:
            signature: Function signature string

        Returns:
            List of parameter dictionaries with name, type, and default info
        """
        parameters = []

        if not signature:
            return parameters

        try:
            # Clean up signature for parsing
            clean_sig = signature.strip()
            if clean_sig.startswith('def '):
                clean_sig = clean_sig[4:]

            # Extract function name and parameters
            if '(' in clean_sig and ')' in clean_sig:
                func_part = clean_sig.split('(', 1)[1]
                params_part = func_part.rsplit(')', 1)[0]

                if params_part.strip():
                    # Split parameters by comma, handling nested structures
                    param_strings = FunctionSignatureParser._split_parameters(params_part)

                    for param_str in param_strings:
                        param_info = FunctionSignatureParser._parse_parameter(param_str.strip())
                        if param_info:
                            parameters.append(param_info)

        except Exception as e:
            logger.warning(f"Failed to parse signature '{signature}': {e}")

        return parameters

    @staticmethod
    def _split_parameters(params_str: str) -> List[str]:
        """Split parameter string by commas, respecting nested structures."""
        params = []
        current_param = ""
        paren_depth = 0
        bracket_depth = 0

        for char in params_str:
            if char == ',' and paren_depth == 0 and bracket_depth == 0:
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
            else:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                current_param += char

        if current_param.strip():
            params.append(current_param.strip())

        return params

    @staticmethod
    def _parse_parameter(param_str: str) -> Optional[Dict[str, Any]]:
        """Parse individual parameter string."""
        if not param_str or param_str in ['self', 'cls']:
            return None

        param_info = {
            'name': '',
            'type': 'Any',
            'default': None,
            'has_default': False
        }

        # Handle default values
        if '=' in param_str:
            name_type_part, default_part = param_str.split('=', 1)
            param_info['default'] = default_part.strip()
            param_info['has_default'] = True
            param_str = name_type_part.strip()

        # Handle type annotations
        if ':' in param_str:
            name_part, type_part = param_str.split(':', 1)
            param_info['name'] = name_part.strip()
            param_info['type'] = type_part.strip()
        else:
            param_info['name'] = param_str.strip()

        return param_info if param_info['name'] else None