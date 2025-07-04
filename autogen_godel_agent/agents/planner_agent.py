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
from typing import Dict, Any, Optional
import sys
import os
import logging
import time
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.function_tools import get_function_tools
from tools.session_manager import get_session_manager
from tools.response_parser import get_response_parser
from tools.agent_pool import get_user_proxy_pool


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

        # Initialize modular components
        self.session_manager = get_session_manager(max_tokens_per_minute)
        self.response_parser = get_response_parser()
        self.user_proxy_pool = get_user_proxy_pool()
        
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
            session_id = self.session_manager.create_session()
        elif not self.session_manager.get_session(session_id):
            session_id = self.session_manager.create_session(session_id)

        session = self.session_manager.get_session(session_id)

        # Rate limiting check
        self.session_manager.enforce_rate_limit()

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

                self.session_manager.add_token_usage(
                    session_id,
                    int(estimated_prompt_tokens),
                    int(estimated_completion_tokens)
                )

                logger.info(f"API call completed in {elapsed_time:.2f}s, "
                           f"estimated tokens: {int(estimated_prompt_tokens + estimated_completion_tokens)}")

            # Parse the response to extract structured information
            analysis_result = self.response_parser.parse_llm_analysis(llm_response, task_description)

            # Store in session context
            session.add_message({
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

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific session."""
        return self.session_manager.get_session_stats(session_id)

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions to prevent memory leaks."""
        return self.session_manager.cleanup_old_sessions(max_age_hours)
    





