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
from .function_registry import FunctionRegistry, get_registry

# Import dialogue evolution if available
try:
    from .llm_dialogue_evolution import get_llm_dialogue_evolution, evolve_code_through_dialogue
    DIALOGUE_EVOLUTION_AVAILABLE = True
except ImportError:
    DIALOGUE_EVOLUTION_AVAILABLE = False

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



    def improve_through_dialogue(self, func_code: str, func_spec: Dict[str, Any],
                               llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Improve function through LLM dialogue."""
        raise NotImplementedError

    def has_function(self, func_name: str) -> bool:
        """Check if function exists in registry."""
        raise NotImplementedError

    def register_function(self, func_name: str, func_code: str, description: str,
                         task_origin: str = "") -> bool:
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
    interface, providing security validation, dialogue improvement, and registration.
    """

    def __init__(self, llm_config: Dict[str, Any] = None,
                 registry: FunctionRegistry = None):
        """
        Initialize function tools with configuration.

        Args:
            llm_config: LLM configuration for dialogue improvement
            registry: Function registry instance
        """
        self.llm_config = llm_config
        self.registry = registry or get_registry()

        # Initialize components
        self.security_validator = SecurityValidator()

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

    def improve_through_dialogue(self, func_code: str, func_spec: Dict[str, Any],
                               llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Improve function through LLM dialogue.

        Args:
            func_code: Function source code
            func_spec: Function specification
            llm_config: LLM configuration for dialogue agents

        Returns:
            Dialogue improvement result
        """
        if not DIALOGUE_EVOLUTION_AVAILABLE:
            return {
                'success': False,
                'error': 'LLM dialogue evolution not available'
            }

        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                evolve_code_through_dialogue(
                    func_code=func_code,
                    func_spec=func_spec,
                    llm_config=llm_config or self.llm_config,
                    max_rounds=10
                )
            )

            loop.close()
            return result

        except Exception as e:
            logger.error(f"Dialogue improvement failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }









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
                         task_origin: str = "") -> bool:
        """
        Register function in registry.

        Args:
            func_name: Name of the function
            func_code: Function source code
            description: Function description
            task_origin: Origin task or context

        Returns:
            True if registration successful, False otherwise
        """
        return self.registry.register_function(func_name, func_code, description, task_origin)

    def create_function_complete(self, func_name: str, task_description: str,
                               func_code: str) -> FunctionCreationResult:
        """
        Complete function creation workflow with dialogue improvement.

        This method performs the full function creation process:
        1. Validate function code
        2. Improve through LLM dialogue (optional)
        3. Register function

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

        # Step 2: Optional dialogue improvement
        dialogue_results = []
        if DIALOGUE_EVOLUTION_AVAILABLE and self.llm_config:
            try:
                func_spec = {
                    'name': func_name,
                    'description': task_description,
                    'signature': f"def {func_name}(...)"
                }

                dialogue_result = self.improve_through_dialogue(func_code, func_spec, self.llm_config)
                if dialogue_result.get('success'):
                    dialogue_results = dialogue_result.get('insights', {})
                    logger.info(f"Dialogue improvement completed for {func_name}")
                else:
                    logger.warning(f"Dialogue improvement failed: {dialogue_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Dialogue improvement error: {e}")

        # Step 3: Register function
        registration_success = False
        try:
            registration_success = self.register_function(
                func_name, func_code, task_description,
                task_origin="FunctionTools.create_function_complete"
            )
            if registration_success:
                logger.info(f"Function '{func_name}' successfully registered")
            else:
                logger.warning(f"Function '{func_name}' registration failed")
        except Exception as e:
            logger.error(f"Function registration failed: {e}")

        # Create result
        overall_success = is_valid and registration_success
        error_message = ""
        if not overall_success:
            if not is_valid:
                error_message = f"Validation failed: {status_msg}"
            elif not registration_success:
                error_message = "Registration failed"

        return FunctionCreationResult(
            success=overall_success,
            function_name=func_name,
            function_code=func_code,
            error_message=error_message,
            test_results=dialogue_results,  # Use dialogue results instead of test results
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

    def search_functions(self, query: str) -> List[Dict[str, Any]]:
        """
        Enhanced search for functions based on query string with improved accuracy.

        Args:
            query: Search query (can match function name, description, or keywords)

        Returns:
            List of matching function information dictionaries with relevance scores
        """
        try:
            all_functions = self.list_functions()
            matching_functions = []

            query_lower = query.lower().strip()
            if not query_lower:
                return []

            # Split query into words for better matching
            query_words = [word for word in query_lower.split() if len(word) > 2]

            for func_name in all_functions:
                # Skip metadata entries
                if func_name in ['metadata', 'last_updated', 'version', 'total_functions']:
                    continue

                try:
                    func_info = self.get_function_info(func_name)
                    if not func_info or not isinstance(func_info, dict):
                        continue

                    # Calculate relevance score
                    score = 0
                    match_types = []

                    func_name_lower = func_name.lower()
                    description = func_info.get('description', '').lower()
                    code = func_info.get('code', '').lower()

                    # Exact phrase matching (highest priority)
                    if query_lower in func_name_lower:
                        score += 100
                        match_types.append('name_exact')
                    elif query_lower in description:
                        score += 80
                        match_types.append('description_exact')
                    elif query_lower in code:
                        score += 60
                        match_types.append('code_exact')

                    # Word-based matching
                    for word in query_words:
                        if word in func_name_lower:
                            score += 50
                            if 'name' not in match_types:
                                match_types.append('name')
                        if word in description:
                            score += 30
                            if 'description' not in match_types:
                                match_types.append('description')
                        if word in code:
                            score += 20
                            if 'code' not in match_types:
                                match_types.append('code')

                    # Fuzzy matching for common variations
                    fuzzy_matches = self._get_fuzzy_matches(query_lower, func_name_lower, description)
                    score += fuzzy_matches * 10

                    # Only include if there's a meaningful match
                    if score > 0:
                        match_info = {
                            'name': func_name,
                            'description': func_info.get('description', 'No description'),
                            'signature': func_info.get('signature', 'No signature available'),
                            'docstring': func_info.get('docstring', ''),
                            'match_type': match_types,
                            'score': score
                        }
                        matching_functions.append(match_info)

                except Exception as e:
                    logger.debug(f"Error processing function {func_name} during search: {e}")
                    continue

            # Sort by relevance score (highest first)
            matching_functions.sort(key=lambda x: x['score'], reverse=True)

            # Filter out very low relevance matches if we have better ones
            if matching_functions and matching_functions[0]['score'] > 50:
                matching_functions = [f for f in matching_functions if f['score'] >= 20]

            logger.info(f"Enhanced search for '{query}' found {len(matching_functions)} matches")
            return matching_functions

        except Exception as e:
            logger.error(f"Function search failed: {e}")
            return []

    def _get_fuzzy_matches(self, query: str, name: str, description: str) -> int:
        """Calculate fuzzy matching score for common variations."""
        fuzzy_score = 0

        # Common word variations
        variations = {
            'validate': ['check', 'verify', 'test'],
            'generate': ['create', 'make', 'build'],
            'calculate': ['compute', 'find', 'get'],
            'email': ['mail', 'e-mail'],
            'password': ['pass', 'pwd'],
            'string': ['str', 'text'],
            'number': ['num', 'digit'],
        }

        for base_word, variants in variations.items():
            if base_word in query:
                for variant in variants:
                    if variant in name or variant in description:
                        fuzzy_score += 1
            elif any(variant in query for variant in variants):
                if base_word in name or base_word in description:
                    fuzzy_score += 1

        return fuzzy_score


# Factory function for creating FunctionTools instances
def create_function_tools(llm_config: Dict[str, Any] = None,
                         registry: FunctionRegistry = None) -> FunctionTools:
    """
    Factory function to create FunctionTools instance.

    Args:
        llm_config: LLM configuration
        registry: Function registry instance

    Returns:
        Configured FunctionTools instance
    """
    return FunctionTools(llm_config, registry)


# Global instance for backward compatibility
_global_function_tools = None


def get_function_tools() -> FunctionTools:
    """Get global FunctionTools instance."""
    global _global_function_tools
    if _global_function_tools is None:
        _global_function_tools = FunctionTools()
    return _global_function_tools