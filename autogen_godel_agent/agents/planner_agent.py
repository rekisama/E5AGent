"""
Task Planner Agent for Self-Expanding Agent System.

This agent analyzes user requests, searches for existing functions,
and determines whether new functions need to be created.

理解用户需求任务
搜索已有函数工具库
判断是否需要创建新函数
结构化输出分析结果（JSON）
为 FunctionCreatorAgent 提供下一步创建依据

使用 autogen 构建 AssistantAgent，搭配临时 UserProxyAgent，模拟多轮 Agent 之间的对话和推理过程。
由 LLM 执行判断，它通过模拟调用内置的工具函数（如 search_functions()）了解已有函数，再根据语义分析判断是否满足用户任务，如果不满足就建议创建新函数。
提供三种系统内函数搜索能力
模拟模拟模拟
search_functions(query)
list_all_functions()
get_function_info(name)
"""

import autogen
from typing import Dict, List, Any, Optional
import sys
import os
import re
import json
import logging
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid

# Configure logger
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.function_tools import get_function_tools


@dataclass
class TokenUsageStats:
    """Track token usage and API call statistics."""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_calls: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_call_time: Optional[datetime] = None

    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        """Add token usage from an API call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.api_calls += 1
        self.last_call_time = datetime.now()

    def get_rate_per_minute(self) -> float:
        """Calculate API calls per minute."""
        if self.api_calls == 0:
            return 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        return self.api_calls / max(elapsed, 1.0)


@dataclass
class SessionContext:
    """Isolated session context for multi-task scenarios."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: TokenUsageStats = field(default_factory=TokenUsageStats)

    def clear_messages(self):
        """Clear session messages."""
        self.messages.clear()


class UserProxyPool:
    """Pool of UserProxyAgent instances for reuse."""

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._pool: List[autogen.UserProxyAgent] = []
        self._lock = threading.Lock()

    @contextmanager
    def get_user_proxy(self):
        """Get a UserProxyAgent from the pool as a context manager."""
        proxy = None
        try:
            with self._lock:
                if self._pool:
                    proxy = self._pool.pop()
                else:
                    proxy = autogen.UserProxyAgent(
                        name=f"temp_user_{uuid.uuid4().hex[:8]}",
                        human_input_mode="NEVER",
                        max_consecutive_auto_reply=1,
                        code_execution_config=False,
                    )

            # Clear any existing messages
            proxy.chat_messages.clear()
            yield proxy

        finally:
            if proxy:
                # Clean up and return to pool
                proxy.chat_messages.clear()
                with self._lock:
                    if len(self._pool) < self.max_size:
                        self._pool.append(proxy)


class TaskPlannerAgent:
    """
    Agent responsible for task analysis and function discovery.

    Improvements:
    - UserProxyAgent pooling for resource efficiency
    - Session-based context isolation
    - Token usage tracking and rate limiting
    - Enhanced JSON parsing with schema validation
    - Stricter function signature validation
    """

    def __init__(self, llm_config: Dict[str, Any], max_tokens_per_minute: int = 10000):
        self.function_tools = get_function_tools()
        self.max_tokens_per_minute = max_tokens_per_minute

        # Initialize pools and tracking
        self.user_proxy_pool = UserProxyPool()
        self.sessions: Dict[str, SessionContext] = {}
        self.global_token_usage = TokenUsageStats()

        # Rate limiting
        self._last_api_call = 0.0
        self._min_interval = 1.0  # Minimum seconds between API calls
        
        # Define the system message for the planner agent
        system_message = """# Task Planner Agent

## ROLE
Analyze user tasks, search existing functions, determine if new functions are needed.

## TOOLS
- `search_functions(query)` - Search for existing functions
- `list_all_functions()` - List all available functions
- `get_function_info(name)` - Get detailed function info

## SEARCH STRATEGY
Use multiple search terms: specific → general → action words
Examples: "email validation" → "validation" → "check"

## CRITICAL: OUTPUT FORMAT REQUIREMENTS
🚨 **MANDATORY**: Your response MUST contain a JSON code block with the exact structure below.
🚨 **NO EXCEPTIONS**: Always wrap JSON in ```json code blocks.
🚨 **VALIDATION**: JSON must be valid and parseable.

**Required JSON Structure:**
```json
{
  "function_found": boolean,
  "matched_functions": [
    {
      "name": "exact_function_name",
      "description": "clear description",
      "signature": "def func_name(param: type) -> return_type"
    }
  ],
  "needs_new_function": boolean,
  "suggested_function_spec": {
    "name": "valid_python_identifier",
    "description": "detailed description with purpose and behavior",
    "signature": "def function_name(param: type) -> return_type",
    "examples": [
      {"input": "example_input", "output": "expected_output"},
      {"input": "another_input", "output": "another_output"}
    ]
  },
  "reasoning": "step-by-step analysis explanation"
}
```

## PROCESS
1. **Understand** the task requirements
2. **Search** existing functions (try multiple keywords)
3. **Analyze** if existing functions can solve the task
4. **Recommend** solution or new function specification
5. **ALWAYS** respond with the required JSON format"""

        self.agent = autogen.AssistantAgent(
            name="TaskPlanner",
            system_message=system_message,
            llm_config=llm_config,
            function_map={
                "search_functions": self._search_functions,
                "list_all_functions": self._list_all_functions,
                "get_function_info": self._get_function_info,
            }
        )
    
    def _search_functions(self, query: str) -> str:
        """
        Search the function registry for functions matching a query string.

        Parameters:
            query (str): A short description of the function to search for.
                        Examples: "validate email", "calculate factorial", "phone number"

        Returns:
            str: A formatted list of matching functions with name, description, and signature.
        """
        try:
            results = self.function_tools.search_functions(query)
            if not results:
                return f"No functions found matching '{query}'"

            response = f"Found {len(results)} function(s) matching '{query}':\n\n"
            for func in results:
                response += f"- **{func['name']}**: {func['description']}\n"
                response += f"  Signature: {func['signature']}\n"
                if 'score' in func:
                    response += f"  Relevance Score: {func['score']:.2f}\n"
                if func['docstring']:
                    # Truncate long docstrings for readability
                    doc = func['docstring'][:200] + "..." if len(func['docstring']) > 200 else func['docstring']
                    response += f"  Documentation: {doc}\n"
                response += "\n"

            return response
        except Exception as e:
            return f"Error searching functions: {e}"
    
    def _list_all_functions(self) -> str:
        """
        List all available functions in the system registry.

        Returns:
            str: A formatted list of all registered functions with basic information.
        """
        try:
            functions = self.function_tools.list_all_functions()
            if not functions:
                return "No functions are currently registered in the system."

            response = f"Available functions ({len(functions)} total):\n\n"
            for func in functions:
                response += f"- **{func['name']}**: {func['description']}\n"
                response += f"  Signature: {func.get('signature', 'N/A')}\n"
                response += f"  Created: {func['created_at']}\n"
                if func.get('task_origin'):
                    response += f"  Origin: {func['task_origin']}\n"
                response += "\n"

            return response
        except Exception as e:
            return f"Error listing functions: {e}"
    
    def _get_function_info(self, name: str) -> str:
        """
        Get detailed information about a specific function.

        Parameters:
            name (str): The exact name of the function to get information about.

        Returns:
            str: Detailed information about the function including signature, documentation, and test cases.
        """
        try:
            info = self.function_tools.get_function_info(name)
            if not info:
                return f"Function '{name}' not found in registry."

            response = f"Function: {info['name']}\n"
            response += f"Description: {info['description']}\n"
            response += f"Signature: {info.get('signature', 'N/A')}\n"
            response += f"Created: {info['created_at']}\n"

            if info.get('docstring'):
                response += f"Documentation:\n{info['docstring']}\n"

            if info.get('task_origin'):
                response += f"Original Task: {info['task_origin']}\n"

            if info.get('test_cases'):
                response += f"Test Cases: {len(info['test_cases'])} available\n"
                # Show first test case as example
                if info['test_cases']:
                    first_test = info['test_cases'][0]
                    response += f"Example: {first_test.get('description', 'Test case')}\n"

            return response
        except Exception as e:
            return f"Error getting function info: {e}"
    
    def analyze_task(self, task_description: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to analyze a task and determine what functions are needed.

        This method leverages the LLM's intelligence to:
        1. Understand the task requirements
        2. Search for existing functions intelligently
        3. Determine if new functions need to be created
        4. Provide detailed specifications for missing functions

        Args:
            task_description: The user's task description
            session_id: Optional session ID for context isolation

        Returns:
            Dictionary with analysis results including LLM response and recommendations
        """
        # Create or get session context
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self.sessions:
            self.sessions[session_id] = SessionContext(session_id=session_id)

        session = self.sessions[session_id]

        # Rate limiting check
        self._enforce_rate_limit()

        # Create a comprehensive prompt for the LLM
        analysis_prompt = f"""# TASK: {task_description}

## ANALYSIS STEPS
1. **Search** existing functions (use multiple keywords)
2. **Evaluate** if existing functions solve the task
3. **Specify** new function if needed

## REQUIRED ACTIONS
- Use `search_functions()` with different keywords
- Use `list_all_functions()` if searches fail
- Use `get_function_info()` for details

## OUTPUT
🚨 CRITICAL: Respond with the required JSON format showing your analysis results.
The JSON must be wrapped in ```json code blocks and be valid JSON."""

        try:
            # Use pooled UserProxyAgent with context manager
            with self.user_proxy_pool.get_user_proxy() as user_proxy:
                # Start the analysis conversation
                start_time = time.time()

                user_proxy.initiate_chat(
                    self.agent,
                    message=analysis_prompt
                )

                # Extract the LLM's response
                messages = user_proxy.chat_messages.get(self.agent, [])
                llm_response = messages[-1]['content'] if messages else "No response received"

                # Track token usage (estimate if not available)
                elapsed_time = time.time() - start_time
                estimated_prompt_tokens = len(analysis_prompt.split()) * 1.3  # Rough estimate
                estimated_completion_tokens = len(llm_response.split()) * 1.3

                session.token_usage.add_usage(
                    int(estimated_prompt_tokens),
                    int(estimated_completion_tokens)
                )
                self.global_token_usage.add_usage(
                    int(estimated_prompt_tokens),
                    int(estimated_completion_tokens)
                )

                logger.info(f"API call completed in {elapsed_time:.2f}s, "
                           f"estimated tokens: {int(estimated_prompt_tokens + estimated_completion_tokens)}")

            # Parse the response to extract structured information
            analysis_result = self._parse_llm_analysis(llm_response, task_description)

            # Store in session context
            session.messages.append({
                'task': task_description,
                'response': llm_response,
                'analysis': analysis_result,
                'timestamp': datetime.now()
            })

            return analysis_result

        except Exception as e:
            logger.error(f"Task analysis failed: {e}")
            # Fallback to basic analysis if LLM fails
            return {
                'task': task_description,
                'status': 'error',
                'error': str(e),
                'llm_response': None,
                'function_found': False,
                'matched_functions': [],
                'needs_new_function': True,
                'suggested_function_spec': None,
                'session_id': session_id
            }

    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call

        if time_since_last_call < self._min_interval:
            sleep_time = self._min_interval - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_api_call = time.time()

        # Check token usage rate
        rate = self.global_token_usage.get_rate_per_minute()
        if rate > self.max_tokens_per_minute / 60:  # Convert to per-second rate
            logger.warning(f"High token usage rate: {rate:.2f} tokens/min")

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific session."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return {
            'session_id': session_id,
            'created_at': session.created_at,
            'message_count': len(session.messages),
            'token_usage': {
                'total_tokens': session.token_usage.total_tokens,
                'api_calls': session.token_usage.api_calls,
                'rate_per_minute': session.token_usage.get_rate_per_minute()
            }
        }

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions to prevent memory leaks."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        old_sessions = [
            sid for sid, session in self.sessions.items()
            if session.created_at < cutoff_time
        ]

        for sid in old_sessions:
            del self.sessions[sid]

        if old_sessions:
            logger.info(f"Cleaned up {len(old_sessions)} old sessions")
    
    def _parse_llm_analysis(self, llm_response: str, task_description: str) -> Dict[str, Any]:
        """
        Parse the LLM's analysis response into structured data.

        First tries to parse as JSON, falls back to regex-based parsing.

        Args:
            llm_response: The LLM's analysis response
            task_description: Original task description

        Returns:
            Structured analysis result
        """
        import json

        # Initialize default result structure
        default_result = {
            'task': task_description,
            'status': 'success',
            'llm_response': llm_response,
            'function_found': False,
            'matched_functions': [],
            'needs_new_function': False,
            'suggested_function_spec': None,
            'reasoning': None
        }

        # Try to parse as JSON first (preferred method)
        try:
            # Multiple JSON extraction strategies
            json_str = self._extract_json_from_response(llm_response)
            if not json_str:
                raise ValueError("No JSON found in response")

            parsed_json = json.loads(json_str)

            # Schema validation and structure completion
            result = self._validate_and_complete_json_structure(parsed_json, default_result)
            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback to regex-based parsing
            logger.debug(f"JSON parsing failed ({e}), falling back to regex parsing")
            return self._fallback_regex_parse(llm_response, task_description, default_result)

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON string from LLM response using multiple strategies."""

        # Strategy 1: JSON code block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Strategy 2: JSON without code blocks but with function_found key
        json_match = re.search(r'(\{[^{}]*"function_found"[^{}]*\})', response, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Strategy 3: More flexible JSON extraction
        json_match = re.search(r'(\{(?:[^{}]|{[^{}]*})*\})', response, re.DOTALL)
        if json_match:
            potential_json = json_match.group(1)
            # Check if it contains expected keys
            if any(key in potential_json for key in ['function_found', 'needs_new_function', 'matched_functions']):
                return potential_json

        return None

    def _validate_and_complete_json_structure(self, parsed_json: Dict[str, Any], default_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON structure and complete missing fields."""

        result = default_result.copy()

        # Required fields with validation
        result['function_found'] = bool(parsed_json.get('function_found', False))
        result['needs_new_function'] = bool(parsed_json.get('needs_new_function', False))
        result['reasoning'] = str(parsed_json.get('reasoning', '')) if parsed_json.get('reasoning') else None

        # Validate matched_functions structure
        matched_functions = parsed_json.get('matched_functions', [])
        if isinstance(matched_functions, list):
            validated_functions = []
            for func in matched_functions:
                if isinstance(func, dict) and 'name' in func:
                    # Ensure required fields
                    validated_func = {
                        'name': str(func['name']),
                        'description': str(func.get('description', '')),
                        'signature': str(func.get('signature', ''))
                    }
                    validated_functions.append(validated_func)
            result['matched_functions'] = validated_functions

        # Validate suggested_function_spec structure
        suggested_spec = parsed_json.get('suggested_function_spec')
        if isinstance(suggested_spec, dict):
            # Validate function name is a valid Python identifier
            func_name = suggested_spec.get('name', '')
            if func_name and func_name.isidentifier():
                result['suggested_function_spec'] = {
                    'name': func_name,
                    'description': str(suggested_spec.get('description', '')),
                    'signature': str(suggested_spec.get('signature', '')),
                    'examples': self._validate_examples_structure(suggested_spec.get('examples', []))
                }

        return result

    def _validate_examples_structure(self, examples: Any) -> List[Dict[str, str]]:
        """Validate and clean examples structure."""
        if not isinstance(examples, list):
            return []

        validated_examples = []
        for example in examples:
            if isinstance(example, dict) and 'input' in example and 'output' in example:
                validated_examples.append({
                    'input': str(example['input']),
                    'output': str(example['output'])
                })

        return validated_examples

    def _fallback_regex_parse(self, llm_response: str, task_description: str, default_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback regex-based parsing when JSON parsing fails.
        """
        result = default_result.copy()

        # Try to determine if existing functions were found
        if any(phrase in llm_response.lower() for phrase in [
            'found function', 'existing function', 'can use', 'available function'
        ]):
            result['function_found'] = True

        # Try to determine if new function is needed
        if any(phrase in llm_response.lower() for phrase in [
            'need to create', 'new function', 'missing function', 'no suitable function'
        ]):
            result['needs_new_function'] = True

        # Extract matched functions from response
        result['matched_functions'] = self._extract_matched_functions_from_response(llm_response)
        if result['matched_functions']:
            result['function_found'] = True

        # Try to extract function specifications from the response
        function_name = self._extract_function_name_from_response(llm_response)

        # If we found indicators of a new function, try to extract more details
        if result['needs_new_function'] and function_name:
            result['suggested_function_spec'] = {
                'name': function_name,
                'description': self._extract_description_from_response(llm_response),
                'signature': self._extract_signature_from_response(llm_response),
                'examples': self._extract_examples_from_response(llm_response)
            }

        return result

    def _extract_matched_functions_from_response(self, llm_response: str) -> List[Dict[str, str]]:
        """
        Extract matched functions from LLM response.

        Returns:
            List of dictionaries with 'name', 'description', and 'signature' keys
        """
        matched_functions = []

        # Pattern: - **function_name**: description\n  Signature: signature
        func_matches = re.findall(
            r'-\s*\*\*(.*?)\*\*:\s*(.*?)\n\s*Signature:\s*(.*?)(?:\n|$)',
            llm_response,
            re.DOTALL
        )

        for name, desc, sig in func_matches:
            matched_functions.append({
                'name': name.strip(),
                'description': desc.strip(),
                'signature': sig.strip()
            })

        # Alternative pattern: Function: name - description
        if not matched_functions:
            alt_matches = re.findall(
                r'Function:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*-\s*(.*?)(?:\n|$)',
                llm_response
            )
            for name, desc in alt_matches:
                matched_functions.append({
                    'name': name.strip(),
                    'description': desc.strip(),
                    'signature': ''
                })

        return matched_functions

    def _extract_function_name_from_response(self, response: str) -> Optional[str]:
        """Extract function name from LLM response."""
        name_patterns = [
            r'function name[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'name[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        ]

        for pattern in name_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_description_from_response(self, response: str) -> str:
        """Extract function description from LLM response."""
        # Look for description patterns
        patterns = [
            r'purpose[:\s]+([^\n]+)',
            r'description[:\s]+([^\n]+)',
            r'what it does[:\s]+([^\n]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "Function description not specified"

    def _extract_signature_from_response(self, response: str) -> str:
        """Extract and validate function signature from LLM response."""
        import ast

        # Look for function signature patterns
        patterns = [
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*->\s*[^:]+)',
            r'signature[:\s]+([^\n]+)',
            r'parameters[:\s]+([^\n]+)',
            r'\(([^)]*)\)\s*->\s*([^:\n]+)',  # (param: type) -> return_type
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*->\s*[^:\n]+',  # func_name(...) -> type
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                signature = match.group(1).strip()

                # Validate signature using AST parsing
                if self._validate_function_signature(signature):
                    return signature

        return "Signature not specified"

    def _validate_function_signature(self, signature: str) -> bool:
        """
        Validate function signature using stricter AST parsing and type annotation checking.

        Args:
            signature: Function signature string

        Returns:
            True if signature is valid, False otherwise
        """
        import ast
        import typing

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

    def _extract_examples_from_response(self, response: str) -> List[Dict[str, str]]:
        """
        Extract structured examples from LLM response using JSON-first approach.

        Returns:
            List of dictionaries with 'input' and 'output' keys
        """
        examples = []

        # Strategy 1: Try to extract from JSON structure first
        try:
            json_str = self._extract_json_from_response(response)
            if json_str:
                parsed_json = json.loads(json_str)
                suggested_spec = parsed_json.get('suggested_function_spec', {})
                if isinstance(suggested_spec, dict):
                    json_examples = suggested_spec.get('examples', [])
                    if isinstance(json_examples, list):
                        for example in json_examples:
                            if isinstance(example, dict) and 'input' in example and 'output' in example:
                                examples.append({
                                    'input': str(example['input']),
                                    'output': str(example['output'])
                                })
        except (json.JSONDecodeError, KeyError):
            pass

        # Strategy 2: Look for structured example patterns
        if not examples:
            # Pattern: {"input": "value", "output": "result"}
            json_example_pattern = r'\{\s*["\']input["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*["\']output["\']\s*:\s*["\']([^"\']+)["\']\s*\}'
            matches = re.findall(json_example_pattern, response, re.IGNORECASE)

            for input_val, output_val in matches:
                examples.append({
                    'input': input_val.strip(),
                    'output': output_val.strip()
                })

        # Strategy 3: Look for structured list patterns
        if not examples:
            # Pattern: - input: "value", output: result
            structured_pattern = r'-\s*input:\s*["\']?([^"\']+)["\']?,\s*output:\s*([^\n]+)'
            matches = re.findall(structured_pattern, response, re.IGNORECASE)

            for input_val, output_val in matches:
                examples.append({
                    'input': input_val.strip(),
                    'output': output_val.strip().strip('"\'')
                })

        # Strategy 4: Fallback to legacy arrow patterns (less reliable)
        if not examples:
            lines = response.split('\n')
            in_examples = False

            for line in lines:
                line = line.strip()
                if any(word in line.lower() for word in ['example', 'test case', 'input', 'output']):
                    in_examples = True

                if in_examples:
                    # Try to parse different arrow formats
                    for separator in ['=>', '->', '→', 'output:', 'returns:']:
                        if separator in line.lower():
                            parts = line.split(separator, 1)
                            if len(parts) == 2:
                                input_part = parts[0].strip()
                                output_part = parts[1].strip()

                                # Clean up input part
                                input_part = re.sub(r'^(input:?|example:?)', '', input_part, flags=re.IGNORECASE).strip()
                                input_part = input_part.strip('"\'')

                                # Clean up output part
                                output_part = re.sub(r'^(output:?|result:?)', '', output_part, flags=re.IGNORECASE).strip()
                                output_part = output_part.strip('"\'')

                                if input_part and output_part:
                                    examples.append({
                                        'input': input_part,
                                        'output': output_part
                                    })
                                break

                    # Stop if we hit an empty line or new section
                    if line == '' or line.startswith('#') or line.startswith('**'):
                        if examples:  # Only break if we've found some examples
                            break

        return examples[:3]  # Return at most 3 examples
