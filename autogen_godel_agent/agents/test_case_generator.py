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

# Import modular components from tools
from ..tools.test_runner import (
    TestCaseGenerator as ModularTestCaseGenerator,
    TestGenerationConfig,
    TestCaseComplexity,
    InputFormat,
    TestResult
)
from ..tools.secure_executor import FunctionSignatureParser

# Configure logger
logger = logging.getLogger(__name__)


class EnhancedTestCaseGenerator:
    """
    Enhanced LLM-driven test case generator with modular architecture.

    This class now serves as a high-level interface that leverages the modular
    components from the tools/ directory for comprehensive test case generation.

    Architecture (Modular):
    - Uses TestCaseGenerator from tools.test_runner for core functionality
    - Leverages FunctionSignatureParser from tools.secure_executor
    - Integrates TestCaseStandardizer and TestResponseParser
    - Provides backward-compatible interface for existing agents
    """

    def __init__(self, config: Optional[TestGenerationConfig] = None):
        """Initialize with configuration."""
        self.config = config or TestGenerationConfig()

        # Initialize the modular test case generator
        self.test_generator = ModularTestCaseGenerator(config=self.config)

        # Initialize signature parser
        self.signature_parser = FunctionSignatureParser()

        logger.info(f"EnhancedTestCaseGenerator initialized with config: {self.config}")

    def generate_test_cases(self, specification: Dict[str, Any], code: str = "") -> Tuple[bool, str, List[Dict]]:
        """
        Generate test cases using the modular architecture.

        Args:
            specification: Function specification dictionary
            code: Optional function source code

        Returns:
            Tuple of (success, error_message, test_cases)
        """
        try:
            # Use the modular test generator
            test_cases = self.test_generator.generate_enhanced_test_cases(specification, code)

            # Convert to legacy tuple format for backward compatibility
            return (True, "", test_cases)

        except Exception as e:
            error_msg = f"Test case generation failed: {str(e)}"
            logger.error(error_msg)
            return (False, error_msg, [])

    def generate_enhanced_test_cases(self, specification: Dict[str, Any], code: str = "") -> List[Dict]:
        """
        Generate enhanced test cases (direct list return).

        Args:
            specification: Function specification dictionary
            code: Optional function source code

        Returns:
            List of generated test case dictionaries
        """
        return self.test_generator.generate_enhanced_test_cases(specification, code)


# Backward compatibility alias - this maintains compatibility with existing code
TestCaseGenerator = EnhancedTestCaseGenerator
