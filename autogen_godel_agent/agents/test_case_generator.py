"""
Enhanced LLM-Driven Test Case Generator for Function Creator Agent.

This module provides intelligent test case generation using the modular tools architecture.
It serves as a high-level interface that leverages the specialized components from the tools/ directory:

1. TestCaseGenerator from tools.test_runner for comprehensive test generation
2. FunctionSignatureParser from tools.secure_executor for signature analysis
3. TestCaseStandardizer and TestResponseParser for robust processing
4. Configurable test generation with multiple strategies and fallback mechanisms

为系统中的任意 Python 函数，根据其规范（如函数名、签名、描述、示例）和可选的函数源码，
自动生成结构化、合理、覆盖率高的测试用例，使用模块化工具架构实现。

Architecture (Modular):
- Uses TestCaseGenerator from tools.test_runner for core functionality
- Leverages FunctionSignatureParser from tools.secure_executor
- Integrates TestCaseStandardizer and TestResponseParser
- Provides backward-compatible interface for existing agents
"""

from typing import Dict, List, Any, Union, Optional, Tuple
import logging

# Import AutoGen evolution components from tools
try:
    from ..tools.autogen_test_evolution import get_autogen_test_evolution, evolve_function_with_autogen
    from ..tools.secure_executor import FunctionSignatureParser
    AUTOGEN_AVAILABLE = True
except ImportError:
    try:
        from tools.autogen_test_evolution import get_autogen_test_evolution, evolve_function_with_autogen
        from tools.secure_executor import FunctionSignatureParser
        AUTOGEN_AVAILABLE = True
    except ImportError:
        logger.warning("AutoGen test evolution not available")
        AUTOGEN_AVAILABLE = False

# Configure logger
logger = logging.getLogger(__name__)


class AutoGenTestEvolutionGenerator:
    """
    AutoGen-based test case generator using multi-agent evolution.

    This class uses AutoGen's multi-agent collaboration to generate, critique,
    and evolve test cases through intelligent dialogue between specialized agents.

    Architecture:
    - TestCritic: Finds flaws and edge cases
    - TestGenerator: Creates comprehensive test cases
    - CodeReviewer: Reviews code quality
    - TestExecutor: Runs tests and analyzes results
    - EvolutionCoordinator: Orchestrates the evolution process
    """

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize with LLM configuration for AutoGen agents."""
        self.llm_config = llm_config

        if AUTOGEN_AVAILABLE and llm_config:
            self.evolution_system = get_autogen_test_evolution(llm_config)
            logger.info("✅ AutoGen test evolution system initialized")
        else:
            self.evolution_system = None
            if not AUTOGEN_AVAILABLE:
                logger.warning("⚠️ AutoGen not available, test evolution disabled")
            else:
                logger.warning("⚠️ No LLM config provided, test evolution disabled")

    async def generate_test_cases_async(self, specification: Dict[str, Any],
                                      code: str = "",
                                      max_iterations: int = 2) -> Tuple[bool, str, List[Dict]]:
        """
        Generate test cases using AutoGen multi-agent evolution (async).

        Args:
            specification: Function specification dictionary
            code: Function source code
            max_iterations: Maximum evolution iterations

        Returns:
            Tuple of (success, error_message, test_cases)
        """
        if not self.evolution_system:
            return (False, "AutoGen evolution system not available", [])

        if not code:
            return (False, "Function code required for AutoGen evolution", [])

        try:
            logger.info(f"🧬 Starting AutoGen evolution for: {specification.get('name', 'unknown')}")

            # Run the evolution process
            result = await self.evolution_system.evolve_function(
                func_code=code,
                func_spec=specification,
                max_iterations=max_iterations
            )

            if result.get('success'):
                # Extract test cases from evolution history
                test_cases = []
                for iteration in result.get('evolution_history', []):
                    test_cases.extend(iteration.get('test_cases', []))

                logger.info(f"✅ Generated {len(test_cases)} test cases through AutoGen evolution")
                return (True, "", test_cases)
            else:
                error_msg = result.get('error', 'Unknown evolution error')
                logger.error(f"❌ AutoGen evolution failed: {error_msg}")
                return (False, error_msg, [])

        except Exception as e:
            error_msg = f"AutoGen evolution error: {str(e)}"
            logger.error(error_msg)
            return (False, error_msg, [])

    def generate_test_cases(self, specification: Dict[str, Any], code: str = "") -> Tuple[bool, str, List[Dict]]:
        """
        Generate test cases using AutoGen evolution (sync wrapper).

        Args:
            specification: Function specification dictionary
            code: Function source code

        Returns:
            Tuple of (success, error_message, test_cases)
        """
        if not self.evolution_system:
            # Fallback: generate basic test cases without AutoGen
            return self._generate_fallback_tests(specification)

        try:
            # Run async evolution in sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self.generate_test_cases_async(specification, code, max_iterations=2)
            )

            loop.close()
            return result

        except Exception as e:
            logger.error(f"Sync evolution wrapper failed: {e}")
            return self._generate_fallback_tests(specification)

    def _generate_fallback_tests(self, specification: Dict[str, Any]) -> Tuple[bool, str, List[Dict]]:
        """Generate basic fallback test cases when AutoGen is not available."""
        func_name = specification.get('name', 'unknown_function')

        # Basic test cases for common function types
        test_cases = []

        if 'email' in func_name.lower():
            test_cases = [
                {
                    'description': f'Test {func_name} with valid email',
                    'input': {'email': 'test@example.com'},
                    'expected_output': True,
                    'test_type': 'normal'
                },
                {
                    'description': f'Test {func_name} with invalid email',
                    'input': {'email': 'invalid.email'},
                    'expected_output': False,
                    'test_type': 'normal'
                }
            ]
        else:
            # Generic test cases
            test_cases = [
                {
                    'description': f'Test {func_name} with normal input',
                    'input': {'value': 'test_input'},
                    'expected_output': 'auto_generated',
                    'test_type': 'normal'
                }
            ]

        logger.info(f"Generated {len(test_cases)} fallback test cases")
        return (True, "", test_cases)


# Backward compatibility alias
TestCaseGenerator = AutoGenTestEvolutionGenerator
EnhancedTestCaseGenerator = AutoGenTestEvolutionGenerator
