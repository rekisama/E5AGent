"""
Function Tools - Unified Interface Module.

This module provides a unified interface for all function-related operations
in the Function Creator Agent system. It integrates:

1. Security validation and safe code execution
2. Test case generation and execution
3. Function registration and management
4. Unified API using factory/proxy pattern

函数工具统一接口模块，为函数创建代理系统提供所有函数相关操作的统一接口。
整合了安全验证和安全代码执行、测试用例生成和执行、函数注册和管理等功能。
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

# Import modular components
from .secure_executor import SecurityValidator, validate_function_code, execute_code_safely, FunctionSignatureParser
from .test_runner import (
    TestCaseGenerator, TestResult, TestGenerationConfig,
    TestCaseComplexity, InputFormat, TestCaseStandardizer,
    TestResponseParser
)
from .function_registry import FunctionRegistry, get_registry

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class FunctionCreationResult:
    """Result of function creation process."""
    success: bool
    function_name: str
    function_code: str
    error_message: str
    test_results: List[Dict]
    validation_results: Dict[str, Any]


class FunctionToolsInterface:
    """
    Abstract interface for function tools.

    This interface defines the contract for function creation, validation,
    testing, and registration operations.
    """

    def validate_function_code(self, code: str) -> Tuple[bool, str, str]:
        """Validate function code for syntax and security."""
        raise NotImplementedError

    def execute_code_safely(self, code: str, timeout_seconds: int = 10) -> Tuple[bool, str, Dict[str, Any]]:
        """Execute code safely in sandboxed environment."""
        raise NotImplementedError

    def generate_test_cases(self, func_name: str, func_code: str, task_description: str) -> List[Dict]:
        """Generate test cases for a function."""
        raise NotImplementedError

    def run_tests(self, func_code: str, test_cases: List[Dict]) -> TestResult:
        """Run test cases against function code."""
        raise NotImplementedError

    def has_function(self, func_name: str) -> bool:
        """Check if function exists in registry."""
        raise NotImplementedError

    def register_function(self, func_name: str, func_code: str, description: str,
                         task_origin: str = "", test_cases: List[Dict] = None) -> bool:
        """Register function in registry."""
        raise NotImplementedError

    def create_function_complete(self, func_name: str, task_description: str,
                               func_code: str) -> FunctionCreationResult:
        """Complete function creation workflow."""
        raise NotImplementedError


class FunctionTools(FunctionToolsInterface):
    """
    Unified function tools implementation using factory/proxy pattern.

    This class integrates all function-related operations through a single
    interface, providing security validation, test generation, and registration.
    """

    def __init__(self, llm_config: Dict[str, Any] = None,
                 test_config: TestGenerationConfig = None,
                 registry: FunctionRegistry = None):
        """
        Initialize function tools with configuration.

        Args:
            llm_config: LLM configuration for test generation
            test_config: Test generation configuration
            registry: Function registry instance
        """
        self.llm_config = llm_config
        self.test_config = test_config or TestGenerationConfig()
        self.registry = registry or get_registry()

        # Initialize components
        self.security_validator = SecurityValidator()
        self.test_generator = TestCaseGenerator(self.test_config, self.llm_config)

        logger.info("FunctionTools initialized with modular components")

    def validate_function_code(self, code: str) -> Tuple[bool, str, str]:
        """
        Validate function code for syntax and security.

        Args:
            code: Function source code to validate

        Returns:
            Tuple of (is_valid, status_message, details)
        """
        return validate_function_code(code)

    def execute_code_safely(self, code: str, timeout_seconds: int = 10) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute code safely in sandboxed environment.

        Args:
            code: Code to execute
            timeout_seconds: Execution timeout

        Returns:
            Tuple of (success, output, namespace)
        """
        return execute_code_safely(code, timeout_seconds)

    def generate_test_cases(self, func_name: str, func_code: str, task_description: str) -> List[Dict]:
        """
        Generate test cases for a function.

        Args:
            func_name: Name of the function
            func_code: Function source code
            task_description: Description of what the function should do

        Returns:
            List of test case dictionaries
        """
        return self.test_generator.generate_test_cases(func_name, func_code, task_description)

    def test_function_with_cases(self, func_code: str, func_name: str, test_cases: List[Dict]) -> Tuple[bool, str, str]:
        """
        Test function with provided test cases (backward compatibility method).

        Args:
            func_code: Function source code
            func_name: Name of the function
            test_cases: List of test cases

        Returns:
            Tuple of (success, status_message, details)
        """
        test_result = self.run_tests(func_code, test_cases)
        return (test_result.success, test_result.error_msg, str(test_result.test_results))

    def run_tests(self, func_code: str, test_cases: List[Dict]) -> TestResult:
        """
        Run test cases against function code.

        Args:
            func_code: Function source code
            test_cases: List of test cases to run

        Returns:
            TestResult with success status and results
        """
        try:
            # Validate code first
            is_valid, status_msg, _ = self.validate_function_code(func_code)
            if not is_valid:
                return TestResult(
                    success=False,
                    error_msg=f"Code validation failed: {status_msg}",
                    test_results=[]
                )

            # Execute tests safely
            test_results = []
            all_passed = True

            for i, test_case in enumerate(test_cases):
                try:
                    # Prepare test execution code
                    test_input = test_case.get('input', {})
                    expected_output = test_case.get('expected_output', 'auto_generated')

                    # Create test execution code
                    test_code = self._create_test_execution_code(func_code, test_input, expected_output)

                    # Execute test
                    success, output, namespace = self.execute_code_safely(test_code, timeout_seconds=5)

                    test_result = {
                        'test_index': i,
                        'description': test_case.get('description', f'Test {i+1}'),
                        'input': test_input,
                        'expected_output': expected_output,
                        'actual_output': namespace.get('actual_result', 'No result'),
                        'passed': success and namespace.get('test_passed', False),
                        'error': output if not success else None
                    }

                    test_results.append(test_result)

                    if not test_result['passed']:
                        all_passed = False

                except Exception as e:
                    test_results.append({
                        'test_index': i,
                        'description': test_case.get('description', f'Test {i+1}'),
                        'input': test_input,
                        'expected_output': expected_output,
                        'actual_output': None,
                        'passed': False,
                        'error': str(e)
                    })
                    all_passed = False

            return TestResult(
                success=all_passed,
                error_msg="" if all_passed else "Some tests failed",
                test_results=test_results
            )

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return TestResult(
                success=False,
                error_msg=f"Test execution error: {str(e)}",
                test_results=[]
            )

    def _create_test_execution_code(self, func_code: str, test_input: Dict[str, Any],
                                  expected_output: Any) -> str:
        """Create code for test execution."""
        # Extract function name from code
        import ast
        try:
            tree = ast.parse(func_code)
            func_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    break

            if not func_name:
                raise ValueError("No function definition found in code")

            # Create test execution code
            test_code = f"""
{func_code}

# Test execution
try:
    # Prepare arguments
    test_input = {repr(test_input)}
    expected_output = {repr(expected_output)}

    # Call function with test input
    if isinstance(test_input, dict):
        actual_result = {func_name}(**test_input)
    else:
        actual_result = {func_name}(test_input)

    # Check result
    if expected_output == 'auto_generated':
        test_passed = True  # Accept any result for auto-generated tests
    else:
        test_passed = actual_result == expected_output

except Exception as e:
    actual_result = f"Error: {{str(e)}}"
    test_passed = False
"""
            return test_code

        except Exception as e:
            logger.error(f"Failed to create test execution code: {e}")
            return f"""
{func_code}

# Simple test execution
actual_result = "Test creation failed: {str(e)}"
test_passed = False
"""

    def has_function(self, func_name: str) -> bool:
        """
        Check if function exists in registry.

        Args:
            func_name: Name of the function

        Returns:
            True if function exists, False otherwise
        """
        return self.registry.has_function(func_name)

    def register_function(self, func_name: str, func_code: str, description: str,
                         task_origin: str = "", test_cases: List[Dict] = None) -> bool:
        """
        Register function in registry.

        Args:
            func_name: Name of the function
            func_code: Function source code
            description: Function description
            task_origin: Origin task or context
            test_cases: List of test cases

        Returns:
            True if registration successful, False otherwise
        """
        return self.registry.register_function(func_name, func_code, description, task_origin, test_cases)

    def create_function_complete(self, func_name: str, task_description: str,
                               func_code: str) -> FunctionCreationResult:
        """
        Complete function creation workflow.

        This method performs the full function creation process:
        1. Validate function code
        2. Generate test cases
        3. Run tests
        4. Register function if successful

        Args:
            func_name: Name of the function
            task_description: Description of what the function should do
            func_code: Function source code

        Returns:
            FunctionCreationResult with complete workflow results
        """
        logger.info(f"Starting complete function creation for: {func_name}")

        # Step 1: Validate function code
        is_valid, status_msg, details = self.validate_function_code(func_code)
        validation_results = {
            'is_valid': is_valid,
            'status_message': status_msg,
            'details': details
        }

        if not is_valid:
            return FunctionCreationResult(
                success=False,
                function_name=func_name,
                function_code=func_code,
                error_message=f"Code validation failed: {status_msg}",
                test_results=[],
                validation_results=validation_results
            )

        # Step 2: Generate test cases
        try:
            test_cases = self.generate_test_cases(func_name, func_code, task_description)
            logger.info(f"Generated {len(test_cases)} test cases")
        except Exception as e:
            logger.error(f"Test case generation failed: {e}")
            test_cases = []

        # Step 3: Run tests
        test_result = self.run_tests(func_code, test_cases)

        # Step 4: Register function if tests pass
        registration_success = False
        if test_result.success:
            try:
                registration_success = self.register_function(
                    func_name, func_code, task_description,
                    task_origin="FunctionTools.create_function_complete",
                    test_cases=test_cases
                )
                if registration_success:
                    logger.info(f"Function '{func_name}' successfully registered")
                else:
                    logger.warning(f"Function '{func_name}' tests passed but registration failed")
            except Exception as e:
                logger.error(f"Function registration failed: {e}")

        # Create result
        overall_success = is_valid and test_result.success and registration_success
        error_message = ""
        if not overall_success:
            if not is_valid:
                error_message = f"Validation failed: {status_msg}"
            elif not test_result.success:
                error_message = f"Tests failed: {test_result.error_msg}"
            elif not registration_success:
                error_message = "Registration failed"

        return FunctionCreationResult(
            success=overall_success,
            function_name=func_name,
            function_code=func_code,
            error_message=error_message,
            test_results=test_result.test_results,
            validation_results=validation_results
        )

    def get_function_info(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Get function information from registry."""
        return self.registry.get_function_info(func_name)

    def list_functions(self) -> List[str]:
        """Get list of all registered functions."""
        return self.registry.list_functions()

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return self.registry.get_registry_stats()


# Factory function for creating FunctionTools instances
def create_function_tools(llm_config: Dict[str, Any] = None,
                         test_config: TestGenerationConfig = None,
                         registry: FunctionRegistry = None) -> FunctionTools:
    """
    Factory function to create FunctionTools instance.

    Args:
        llm_config: LLM configuration
        test_config: Test generation configuration
        registry: Function registry instance

    Returns:
        Configured FunctionTools instance
    """
    return FunctionTools(llm_config, test_config, registry)


# Global instance for backward compatibility
_global_function_tools = None


def get_function_tools() -> FunctionTools:
    """Get global FunctionTools instance."""
    global _global_function_tools
    if _global_function_tools is None:
        _global_function_tools = FunctionTools()
    return _global_function_tools