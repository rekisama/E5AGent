"""
Function Creator Agent for Self-Expanding Agent System.

This agent generates new Python functions based on specifications,
tests them, and registers them if they pass validation.
提供自动生成、验证、测试、注册 Python 函数的能力。
"""

import autogen
from typing import Dict, List, Any, Optional, Tuple, Union
import sys
import os
import json
import ast
import logging
import re
import textwrap
import traceback
import inspect

# Configure logger
logger = logging.getLogger(__name__)

# Add parent directory to path for imports (use insert for priority)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..tools.function_tools import get_function_tools
from ..tools.test_runner import TestResult
from ..config import Config




# SecurityValidator is now imported from tools.secure_executor


# PromptBuilder functionality is now integrated into the main agent


class FunctionCreatorAgent:
    """Agent responsible for creating, testing, and registering new functions."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None, function_tools=None):
        """
        Initialize FunctionCreatorAgent.

        Args:
            llm_config: Optional LLM configuration dictionary.
                       If None, will use Config.get_llm_config() from config.py
            function_tools: Optional function tools instance for dependency injection.
                          If None, will use default get_function_tools()
        """
        # Get LLM configuration
        if llm_config is None:
            try:
                # Validate configuration first
                Config.validate_config()
                llm_config = Config.get_llm_config()
                logger.info("Using LLM configuration from config.py")
            except (ValueError, RuntimeError) as e:
                logger.error(f"Failed to get LLM config: {e}")
                raise

        self.llm_config = llm_config
        self.function_tools = function_tools or get_function_tools()
        self.max_attempts = 3

        # Validate function_tools interface
        self._validate_function_tools_interface()

        # Define the system message for the function creator agent
        system_message = """You are a Function Creator Agent in a self-expanding AI system.

Your responsibilities:
1. Generate Python functions based on detailed specifications
2. Create comprehensive test cases for the functions
3. Ensure functions are safe, efficient, and well-documented
4. Handle function testing and registration

When asked to create a function:
1. Write clean, well-documented Python code
2. Include proper type hints where possible
3. Add comprehensive docstrings
4. Avoid dangerous operations (file I/O, system calls, imports of unsafe modules)
5. Focus on pure functions when possible
6. Generate appropriate test cases

Code generation guidelines:
- Use only safe built-in functions and standard library modules (re, math, datetime, etc.)
- Avoid: os, sys, subprocess, eval, exec, __import__, file operations
- Include error handling where appropriate
- Write clear, readable code with good variable names
- Add docstrings explaining the function's purpose, parameters, and return value

Available tools:
- validate_function_code(code): Validate function syntax and safety
- generate_test_cases(func_name, code, description): Generate test cases
- test_function(code, func_name, test_cases): Test the function
- register_function(func_name, func_code, description, task_origin, test_cases):
  Register the function

Always validate and test functions before attempting to register them."""

        self.agent = autogen.AssistantAgent(
            name="FunctionCreator",
            system_message=system_message,
            llm_config=self.llm_config,
            function_map={
                "validate_function_code": self._validate_function_code,
                "generate_test_cases": self._generate_test_cases,
                "test_function": self._test_function,
                "register_function": self._register_function,
            }
        )
    
    def _validate_function_code(self, code: str) -> str:
        """Validate function code for syntax and safety."""
        try:
            is_valid, error_msg, func_name = self.function_tools.validate_function_code(code)
            
            if is_valid:
                return f"✅ Code validation passed. Function name: {func_name}"
            else:
                return f"❌ Code validation failed: {error_msg}"
        except Exception as e:
            logger.error(f"Validation error: {e}\n{traceback.format_exc()}")
            return f"❌ Validation error: {e}"
    
    def _generate_test_cases(self, func_name: str, code: str, description: str) -> str:
        """Generate test cases for a function."""
        try:
            test_cases = self.function_tools.generate_test_cases(func_name, code, description)
            
            if test_cases:
                response = f"Generated {len(test_cases)} test case(s) for {func_name}:\n\n"
                for i, test_case in enumerate(test_cases, 1):
                    response += f"Test {i}: {test_case['description']}\n"
                    response += f"  Input: {test_case['input']}\n"
                    response += f"  Expected type: {test_case['expected_type']}\n\n"
                return response
            else:
                return f"No test cases generated for {func_name}. You may need to create custom test cases."
        except Exception as e:
            logger.error(f"Error generating test cases: {e}\n{traceback.format_exc()}")
            return f"❌ Error generating test cases: {e}"
    
    def _test_function(self, code: str, func_name: str, test_cases: Union[List[Dict], str, Dict] = None) -> str:
        """Test a function with provided test cases."""
        try:
            # Enhanced input handling for multiple formats
            # Use function_tools for test case normalization
            if isinstance(test_cases, str):
                try:
                    import json
                    normalized_test_cases = json.loads(test_cases) if test_cases != "[]" else []
                except json.JSONDecodeError:
                    normalized_test_cases = [{'description': test_cases, 'input': {}, 'expected_output': 'auto_generated'}]
            elif isinstance(test_cases, dict):
                normalized_test_cases = [test_cases]
            elif isinstance(test_cases, list):
                normalized_test_cases = test_cases
            else:
                normalized_test_cases = []

            # Use TestResult structure for better error handling
            result_tuple = self.function_tools.test_function_with_cases(
                code, func_name, normalized_test_cases
            )
            test_result = TestResult.from_tuple(result_tuple)

            if test_result.success:
                response = f"✅ All tests passed for {func_name}!\n\n"
                if test_result.test_results:
                    response += "Test Results:\n"
                    for result in test_result.test_results:
                        status = "✅" if result['success'] else "❌"
                        response += f"{status} {result['description']}\n"
                        if result['success']:
                            response += f"   Result: {result['result']}\n"
                        else:
                            response += f"   Error: {result['error']}\n"
                return response
            else:
                response = f"❌ Tests failed for {func_name}: {test_result.error_msg}\n\n"
                if test_result.test_results:
                    response += "Test Results:\n"
                    for result in test_result.test_results:
                        status = "✅" if result['success'] else "❌"
                        response += f"{status} {result['description']}\n"
                        if not result['success']:
                            response += f"   Error: {result['error']}\n"
                return response
        except Exception as e:
            logger.error(f"Error testing function {func_name}: {e}\n{traceback.format_exc()}")
            return f"❌ Error testing function: {e}"

    def _register_function(self, name: str, code: str, description: str,
                          origin: str = "", test_cases_json: str = "[]") -> str:
        """Register a tested function to the registry."""
        try:
            # Parse test cases from JSON string
            test_cases = json.loads(test_cases_json) if test_cases_json != "[]" else []

            success = self.function_tools.register_function(
                func_name=name,
                func_code=code,
                description=description,
                task_origin=origin,
                test_cases=test_cases
            )

            if success:
                return f"✅ Function '{name}' successfully registered to the system!"
            else:
                return f"❌ Failed to register function '{name}'"
        except Exception as e:
            logger.error(f"Error registering function: {e}\n{traceback.format_exc()}")
            return f"❌ Error registering function: {e}"

    def create_function(self, specification: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        """
        Create a function based on specification.

        Args:
            specification: Dict containing:
                - name: Function name
                - description: What the function should do
                - parameters: Expected parameters
                - return_type: Expected return type
                - examples: Usage examples

        Returns:
            (success, message, function_code)
        """
        func_name = specification.get('name', '')
        description = specification.get('description', '')

        if not func_name or not description:
            return False, "Function name and description are required", None

        # Check if function already exists
        if self.function_tools.has_function(func_name):
            return False, f"Function '{func_name}' already exists in registry", None

        # Use LLM to generate the actual function code
        try:
            logger.info(f"Starting function creation for: {func_name}")
            logger.debug(f"Function specification: {specification}")

            prompt = self.get_creation_prompt(specification)

            # Create a temporary user proxy for this interaction
            user_proxy = autogen.UserProxyAgent(
                name="temp_user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )

            # Start conversation to generate function
            user_proxy.initiate_chat(
                self.agent,
                message=prompt
            )

            # Extract the generated code from the conversation
            messages = user_proxy.chat_messages[self.agent]
            if messages:
                # Look for the first message with function code (not the last one)
                code = None
                for message in messages:
                    content = message['content']
                    extracted_code = self._extract_code_from_response(content)
                    if extracted_code and 'def ' in extracted_code:
                        code = extracted_code
                        break

                if not code:
                    # Fallback to last message
                    last_message = messages[-1]['content']
                    code = self._extract_code_from_response(last_message)

                if code:
                    logger.debug(f"Code extracted successfully for {func_name}")

                    # Validate and test the generated code
                    is_valid, error_msg, extracted_name = self.function_tools.validate_function_code(code)

                    if is_valid:
                        logger.info(f"Code validation passed for {extracted_name}")

                        # Generate test cases using function_tools
                        test_cases = self.function_tools.generate_test_cases(extracted_name, code, description)
                        logger.debug(f"Generated {len(test_cases)} test cases for {extracted_name}")

                        # Test the function using TestResult structure
                        result_tuple = self.function_tools.test_function_with_cases(code, extracted_name, test_cases)
                        test_result = TestResult.from_tuple(result_tuple)

                        if test_result.success:
                            logger.info(f"Function testing passed for {extracted_name}")

                            # Register the function with unified parameter naming
                            register_success = self.function_tools.register_function(
                                name=extracted_name,  # Unified parameter naming
                                code=code,
                                description=description,
                                origin=f"Auto-generated for: {description}",
                                test_cases=test_cases
                            )

                            if register_success:
                                logger.info(f"Function {extracted_name} successfully registered")
                                return True, f"Successfully created and registered function '{extracted_name}'", code
                            else:
                                logger.error(f"Registration failed for {extracted_name}")
                                return False, f"Function created but failed to register: {extracted_name}", code
                        else:
                            logger.warning(f"Function testing failed for {extracted_name}: {test_result.error_msg}")
                            return False, f"Function created but failed tests: {test_result.error_msg}", code
                    else:
                        logger.warning(f"Code validation failed for {func_name}: {error_msg}")
                        return False, f"Generated code validation failed: {error_msg}", code
                else:
                    logger.error(f"No valid code generated for {func_name}")
                    return False, "No valid Python code found in LLM response", None
            else:
                return False, "No response from LLM", None

        except Exception as e:
            logger.error(f"Error during function creation: {e}\n{traceback.format_exc()}")
            return False, f"Error during function creation: {e}", None



    def _validate_function_tools_interface(self):
        """Validate that function_tools has all required methods with correct signatures."""
        required_methods = {
            'validate_function_code': ['code'],
            'test_function_with_cases': ['func_code', 'func_name', 'test_cases'],
            'register_function': ['func_name', 'func_code', 'description'],
            'has_function': ['func_name'],
            'generate_test_cases': ['func_name', 'func_code', 'task_description']
        }

        for method_name, expected_params in required_methods.items():
            if not hasattr(self.function_tools, method_name):
                raise AttributeError(f"FunctionTools missing required method: {method_name}")

            method = getattr(self.function_tools, method_name)
            if not callable(method):
                raise TypeError(f"FunctionTools.{method_name} is not callable")

            # Use inspect to check method signature
            try:
                sig = inspect.signature(method)
                actual_params = list(sig.parameters.keys())

                # Remove 'self' parameter if present
                if actual_params and actual_params[0] == 'self':
                    actual_params = actual_params[1:]

                # Check if all expected parameters are present (allowing for additional optional params)
                for expected_param in expected_params:
                    if expected_param not in actual_params:
                        logger.warning(f"Method {method_name} missing expected parameter: {expected_param}")
                        logger.info(f"  Expected: {expected_params}")
                        logger.info(f"  Actual: {actual_params}")

            except Exception as e:
                logger.warning(f"Could not inspect signature for {method_name}: {e}")

        logger.info("Function tools interface validation completed")

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response with performance optimization."""
        # Performance optimization: Pre-filter with regex before AST parsing
        if not re.search(r'\bdef\s+\w+\s*\(', response):
            logger.debug("No function definition pattern found in response")
            return None

        # Step 1: Extract all code blocks
        all_code_blocks = self._extract_all_code_blocks(response)

        # Step 2: Find the first valid function definition
        for code_block in all_code_blocks:
            if self._is_valid_function_code(code_block):
                return self._clean_code_block(code_block)

        # Step 3: Fallback to line-by-line extraction if no code blocks found
        if not all_code_blocks:
            extracted_code = self._extract_code_from_lines(response)
            if extracted_code:
                return self._clean_code_block(extracted_code)

        return None

    def _extract_all_code_blocks(self, response: str) -> List[str]:
        """Extract all code blocks from response with enhanced patterns."""
        code_blocks = []

        # Pattern 1: ```python ... ``` (flexible whitespace)
        python_blocks = re.findall(r'```python\s+(.*?)```', response, re.DOTALL)
        code_blocks.extend([block.strip() for block in python_blocks])

        # Pattern 2: ``` ... ``` (without language specification, flexible whitespace)
        generic_blocks = re.findall(r'```\s+(.*?)```', response, re.DOTALL)
        code_blocks.extend([block.strip() for block in generic_blocks])

        # Remove duplicates while preserving order
        seen = set()
        unique_blocks = []
        for block in code_blocks:
            if block and block not in seen:
                seen.add(block)
                unique_blocks.append(block)

        # Pattern 3: Code blocks without proper markdown formatting
        # Look for lines starting with 'def ' that might not be in code blocks
        if not unique_blocks:
            lines = response.split('\n')
            current_block = []
            in_function = False

            for line in lines:
                if line.strip().startswith('def ') and 'function_name' not in line:
                    in_function = True
                    current_block = [line]
                elif in_function:
                    if line.strip() == '' or line.startswith('    ') or line.startswith('\t'):
                        current_block.append(line)
                    else:
                        if current_block:
                            unique_blocks.append('\n'.join(current_block).strip())
                        in_function = False
                        current_block = []

            # Add the last block if we were still in a function
            if in_function and current_block:
                unique_blocks.append('\n'.join(current_block).strip())

        return unique_blocks

    def _clean_code_block(self, code: str) -> str:
        """Clean and normalize code block using textwrap.dedent."""
        # Remove common leading whitespace
        cleaned_code = textwrap.dedent(code).strip()
        return cleaned_code

    def _is_valid_function_code(self, code: str) -> bool:
        """Check if code contains a valid function definition with enhanced security."""
        if not code.strip() or 'function_name' in code or 'param: type' in code:
            return False

        # Performance optimization: Quick regex check before AST parsing
        if not re.search(r'\bdef\s+\w+\s*\(', code):
            return False

        # Use AST to validate the code structure and security
        try:
            tree = ast.parse(code)

            # Check if there's at least one function definition
            func_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            if len(func_defs) == 0:
                return False

            # Enhanced security validation using function_tools
            is_valid, status_msg, _ = self.function_tools.validate_function_code(code)
            if not is_valid:
                logger.warning(f"Security validation failed: {status_msg}")
                return False

            return True
        except (SyntaxError, ValueError) as e:
            logger.debug(f"Code validation failed: {e}")
            return False

    def _extract_code_from_lines(self, response: str) -> Optional[str]:
        """Fallback: extract function from response lines."""
        lines = response.split('\n')
        code_lines = []
        in_function = False
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def ') and 'function_name' not in stripped:
                in_function = True
                code_lines.append(line)
                indent_level = len(line) - len(line.lstrip())
            elif in_function:
                current_indent = len(line) - len(line.lstrip())
                if (line.strip() == '' or
                    current_indent > indent_level or
                    (current_indent == indent_level and line.strip().startswith(('"""', "'''", '#')))):
                    code_lines.append(line)
                else:
                    break

        return '\n'.join(code_lines) if code_lines else None







    def get_creation_prompt(self, specification: Dict[str, Any]) -> str:
        """Generate a prompt for function creation using PromptBuilder."""
        func_name = specification.get('name', '')
        description = specification.get('description', '')
        parameters = specification.get('parameters', [])
        return_type = specification.get('return_type', 'Any')
        examples = specification.get('examples', [])

        # Create structured prompt directly
        param_descriptions = []
        for param in parameters:
            name = param.get('name', '')
            param_type = param.get('type', 'Any')
            desc = param.get('description', 'No description')
            param_descriptions.append(f"- {name} ({param_type}): {desc}")

        param_str = "\n".join(param_descriptions) if param_descriptions else "No parameters specified."

        example_str = ""
        if examples:
            example_lines = []
            for i, example in enumerate(examples, 1):
                input_val = example.get('input', '')
                output_val = example.get('output', '')
                example_lines.append(f"{i}. Input: {input_val} → Output: {output_val}")
            example_str = "\n".join(example_lines)
        else:
            example_str = "No examples provided."

        prompt = f"""Please create a Python function with the following specifications:

Function Name: {func_name}
Description: {description}
Return Type: {return_type}

Parameters:
{param_str}

Examples:
{example_str}

Requirements:
1. Write a complete, working Python function
2. Include proper type hints for parameters and return value
3. Include a comprehensive docstring with description, parameters, and return value
4. Handle edge cases and potential errors appropriately
5. Use only safe, standard Python libraries (no file I/O, no system calls)
6. Make the function robust and well-tested
7. Avoid any dangerous operations or imports
8. Include comprehensive error handling

Please provide ONLY the Python function code, wrapped in ```python``` code blocks."""

        return prompt


