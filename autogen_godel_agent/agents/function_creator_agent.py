"""
Function Creator Agent for Self-Expanding Agent System.

This agent generates new Python functions based on specifications,
tests them, and registers them if they pass validation.
提供自动生成、验证、测试、注册 Python 函数的能力。

TODO:考虑以下几个方向继续增强系统：

1. ✅ 支持函数重试机制 / 自我修复能力
当前失败会直接返回，可新增 retry loop 或 error修复尝试：

# 在 create_function 中加入：
for attempt in range(self.max_attempts):
    ...
    if success:
        return ...
    else:
        logger.warning(f"Attempt {attempt+1} failed. Retrying with modified prompt...")
2. ✅ 保存中间产物用于审计和调试
比如保存生成的函数代码、测试用例、测试日志等，以供开发者或 LLM 后续学习改进。

可加入持久化机制（写入 JSON/DB），比如：

self._save_to_audit_log(func_name, code, test_cases, test_result)
3. ✅ 函数调用链构建支持（FunctionComposer 对接）
为自我扩展系统的下一步（函数组合）做准备，可以支持：

get_signature(func_code) 提取参数签名

get_return_type(func_code) 推断返回值类型

未来支持函数链组装的元信息抽取

4. ✅ 加强安全策略
当前禁止了 os, sys, subprocess, eval, exec 等，但建议引入更强的代码沙箱，例如：

限制运行时间（timeout）

限制最大内存

使用 subprocess + seccomp / docker sandbox 隔离执行（可选）

5. ✅ 支持注册后自动生成 API/文档
每当注册一个函数，可以自动生成：

RESTful API 或 LangChain tool wrapper

Markdown 文档或 OpenAPI schema 片段

6. ✅ 加入评估和反馈机制
为每个生成的函数打分，或者使用 LLM 再次评估其质量，如：

evaluate_code_quality(code: str) -> {"score": 4.3, "issues": [...]}
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

try:
    from ..tools.function_tools import get_function_tools
    from ..config import Config
except ImportError:
    from tools.function_tools import get_function_tools
    from config import Config




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

        # Performance optimization: Cache validation results to avoid duplicate work
        self._validation_cache = {}

        # Validate function_tools interface
        self._validate_function_tools_interface()

        # Define the system message for the function creator agent
        system_message = """You are a Function Creator Agent in a self-expanding AI system.

Your responsibilities:
1. Generate Python functions based on detailed specifications
2. Use LLM dialogue to improve and evolve functions
3. Ensure functions are safe, efficient, and well-documented
4. Handle function validation and registration

When asked to create a function:
1. Write clean, well-documented Python code
2. Include proper type hints where possible
3. Add comprehensive docstrings
4. Avoid dangerous operations (file I/O, system calls, imports of unsafe modules)
5. Focus on pure functions when possible
6. Use LLM dialogue for improvement instead of traditional testing

Code generation guidelines:
- Use only safe built-in functions and standard library modules (re, math, datetime, etc.)
- Avoid: os, sys, subprocess, eval, exec, __import__, file operations
- Include error handling where appropriate
- Write clear, readable code with good variable names
- Add docstrings explaining the function's purpose, parameters, and return value

Available tools:
- validate_function_code(code): Validate function syntax and safety
- improve_through_dialogue(func_name, code, description): Improve function through LLM dialogue
- register_function(func_name, func_code, description, task_origin):
  Register the function

Always validate functions and consider dialogue-based improvement before registration."""

        self.agent = autogen.AssistantAgent(
            name="FunctionCreator",
            system_message=system_message,
            llm_config=self.llm_config,
            function_map={
                "validate_function_code": self._validate_function_code,
                "improve_through_dialogue": self._improve_through_dialogue,
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
    
    def _improve_through_dialogue(self, func_name: str, code: str, description: str) -> str:
        """Improve function through LLM dialogue."""
        try:
            # Import dialogue evolution system
            from ..tools.llm_dialogue_evolution import evolve_code_through_dialogue

            func_spec = {
                "name": func_name,
                "description": description,
                "signature": f"def {func_name}(...)"
            }

            logger.info(f"🗣️ Starting dialogue improvement for {func_name}")

            # Run dialogue evolution in sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                evolve_code_through_dialogue(
                    func_code=code,
                    func_spec=func_spec,
                    llm_config=self.llm_config,
                    max_rounds=10
                )
            )

            loop.close()

            if result.get('success'):
                insights = result.get('insights', {})
                response = f"✅ Dialogue improvement completed for {func_name}:\n\n"

                if insights.get('issues_raised'):
                    response += f"🔍 Issues found: {len(insights['issues_raised'])}\n"
                    for issue in insights['issues_raised'][:2]:
                        response += f"  • {issue['content'][:100]}...\n"

                if insights.get('improvements'):
                    response += f"\n🛠️ Improvements suggested: {len(insights['improvements'])}\n"
                    for improvement in insights['improvements'][:2]:
                        response += f"  • {improvement['content'][:100]}...\n"

                response += f"\n📝 Total dialogue messages: {len(result.get('dialogue_history', []))}"
                return response
            else:
                return f"❌ Dialogue improvement failed: {result.get('error', 'Unknown error')}"

        except ImportError:
            return "⚠️ LLM dialogue system not available, skipping improvement"
        except Exception as e:
            logger.error(f"Error in dialogue improvement: {e}\n{traceback.format_exc()}")
            return f"❌ Error in dialogue improvement: {e}"
    


    def _register_function(self, name: str, code: str, description: str,
                          origin: str = "") -> str:
        """Register a function to the registry after dialogue improvement."""
        try:
            success = self.function_tools.register_function(
                func_name=name,
                func_code=code,
                description=description,
                task_origin=origin
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

            # Extract the generated code from the conversation with error handling
            messages = user_proxy.chat_messages.get(self.agent, [])
            if messages:
                # Look for the first message with function code (not the last one)
                code = None
                for message in messages:
                    content = message['content']
                    extracted_code = self._extract_code_from_response(content, func_name)
                    if extracted_code and 'def ' in extracted_code:
                        code = extracted_code
                        break

                if not code:
                    # Fallback to last message
                    last_message = messages[-1]['content']
                    code = self._extract_code_from_response(last_message, func_name)

                if code:
                    logger.debug(f"Code extracted successfully for {func_name}")

                    # Validate and test the generated code
                    is_valid, error_msg, extracted_name = self.function_tools.validate_function_code(code)

                    if is_valid:
                        logger.info(f"Code validation passed for {extracted_name}")

                        # Optional: Improve through dialogue
                        if hasattr(self.function_tools, 'improve_through_dialogue'):
                            try:
                                func_spec = {
                                    'name': extracted_name,
                                    'description': description,
                                    'signature': f"def {extracted_name}(...)"
                                }

                                dialogue_result = self.function_tools.improve_through_dialogue(
                                    code, func_spec, self.llm_config
                                )

                                if dialogue_result.get('success'):
                                    logger.info(f"Dialogue improvement completed for {extracted_name}")
                                else:
                                    logger.warning(f"Dialogue improvement failed: {dialogue_result.get('error', 'Unknown error')}")
                            except Exception as e:
                                logger.warning(f"Dialogue improvement error: {e}")

                        # Register the function
                        register_success = self.function_tools.register_function(
                            func_name=extracted_name,
                            func_code=code,
                            description=description,
                            task_origin=f"Auto-generated for: {description}"
                        )

                        if register_success:
                            logger.info(f"Function {extracted_name} successfully registered")
                            return True, f"Successfully created and registered function '{extracted_name}'", code
                        else:
                            logger.error(f"Registration failed for {extracted_name}")
                            return False, f"Function created but failed to register: {extracted_name}", code
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
            'improve_through_dialogue': ['func_code', 'func_spec'],
            'register_function': ['func_name', 'func_code', 'description'],
            'has_function': ['func_name']
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

    def _extract_code_from_response(self, response: str, expected_func_name: str = None) -> Optional[str]:
        """Extract Python code from LLM response with performance optimization and function name validation."""
        # Performance optimization: Pre-filter with regex before AST parsing
        if not re.search(r'\bdef\s+\w+\s*\(', response):
            logger.debug("No function definition pattern found in response")
            return None

        # Step 1: Extract all code blocks
        all_code_blocks = self._extract_all_code_blocks(response)

        # Step 2: Find the first valid function definition, preferring expected function name
        best_match = None
        for code_block in all_code_blocks:
            if self._is_valid_function_code(code_block):
                # If we have an expected function name, prioritize blocks containing it
                if expected_func_name and f"def {expected_func_name}" in code_block:
                    return self._clean_code_block(code_block)
                # Keep the first valid block as fallback
                if best_match is None:
                    best_match = code_block

        if best_match:
            return self._clean_code_block(best_match)

        # Step 3: Fallback to line-by-line extraction if no code blocks found
        if not all_code_blocks:
            extracted_code = self._extract_code_from_lines(response)
            if extracted_code:
                return self._clean_code_block(extracted_code)

        return None

    def _extract_all_code_blocks(self, response: str) -> List[str]:
        """Extract all code blocks from response with enhanced patterns."""
        code_blocks = []

        # Pattern 1: ```python ... ``` (improved whitespace handling)
        python_blocks = re.findall(r'```python\s*\n(.*?)```', response, re.DOTALL)
        code_blocks.extend([block.strip() for block in python_blocks])

        # Pattern 2: ``` ... ``` (without language specification, improved whitespace handling)
        generic_blocks = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
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
        """Check if code contains a valid function definition with enhanced security and caching."""
        # Performance optimization: Check cache first
        code_hash = hash(code)
        if code_hash in self._validation_cache:
            return self._validation_cache[code_hash]

        if not code.strip() or 'function_name' in code or 'param: type' in code:
            self._validation_cache[code_hash] = False
            return False

        # Performance optimization: Quick regex check before AST parsing
        if not re.search(r'\bdef\s+\w+\s*\(', code):
            self._validation_cache[code_hash] = False
            return False

        # Use AST to validate the code structure and security
        try:
            tree = ast.parse(code)

            # Check if there's at least one function definition
            func_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            if len(func_defs) == 0:
                self._validation_cache[code_hash] = False
                return False

            # Enhanced security validation using function_tools
            is_valid, status_msg, _ = self.function_tools.validate_function_code(code)
            if not is_valid:
                logger.warning(f"Security validation failed: {status_msg}")
                self._validation_cache[code_hash] = False
                return False

            self._validation_cache[code_hash] = True
            return True
        except (SyntaxError, ValueError) as e:
            logger.debug(f"Code validation failed: {e}")
            self._validation_cache[code_hash] = False
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
3. The function MUST return a value of type {return_type}
4. Include a comprehensive docstring with description, parameters, and return value
5. Handle edge cases and potential errors appropriately
6. Use only safe, standard Python libraries (no file I/O, no system calls)
7. Make the function robust and well-tested
8. Avoid any dangerous operations or imports
9. Include comprehensive error handling

Please provide ONLY the Python function code, wrapped in ```python``` code blocks."""

        return prompt


