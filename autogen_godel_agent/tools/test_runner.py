"""
Test Runner and Test Case Generation Module.

This module provides comprehensive test case generation and execution capabilities
for the Function Creator Agent system. It includes:

1. TestResult: Standardized test result structure
2. TestCaseGenerator: Enhanced LLM-driven test case generator
3. Test execution and validation logic
4. Comprehensive test case generation with multiple strategies

测试运行器和测试用例生成模块，为函数创建代理系统提供全面的测试用例生成和执行能力。
包括标准化的测试结果结构、增强的LLM驱动测试用例生成器、测试执行和验证逻辑。
"""

from typing import Dict, List, Any, Optional, Union
import logging
import ast
import time
import json
import re
from dataclasses import dataclass, field
from enum import Enum
import autogen

# Import FunctionSignatureParser from secure_executor
from .secure_executor import FunctionSignatureParser

# Configure logger
logger = logging.getLogger(__name__)


class TestCaseComplexity(Enum):
    """Test case complexity levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class TestCaseComplexity(Enum):
    """Test case complexity levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class InputFormat(Enum):
    """Standardized input formats for test cases."""
    DICT = "dict"  # {"param1": value1, "param2": value2}
    ARGS_LIST = "args_list"  # [value1, value2]
    FUNCTION_CALL = "function_call"  # "func_name(value1, value2)"


@dataclass
class TestGenerationConfig:
    """Configuration for test case generation."""
    max_test_cases: int = 8
    max_combinations: int = 5
    complexity: TestCaseComplexity = TestCaseComplexity.STANDARD
    input_format: InputFormat = InputFormat.DICT
    include_edge_cases: bool = True
    include_boundary_tests: bool = True
    use_llm_generation: bool = True

    def get_test_count_by_complexity(self) -> int:
        """Get test case count based on complexity level."""
        if self.complexity == TestCaseComplexity.BASIC:
            return min(3, self.max_test_cases)
        elif self.complexity == TestCaseComplexity.STANDARD:
            return min(5, self.max_test_cases)
        else:  # COMPREHENSIVE
            return self.max_test_cases


@dataclass
class TestResult:
    """Standardized test result structure."""
    success: bool
    error_msg: str
    test_results: List[Dict]

    @classmethod
    def from_tuple(cls, result_tuple: tuple) -> 'TestResult':
        """Create TestResult from legacy tuple format."""
        if len(result_tuple) >= 3:
            return cls(
                success=result_tuple[0],
                error_msg=result_tuple[1] or "",
                test_results=result_tuple[2] or []
            )
        else:
            return cls(success=False, error_msg="Invalid result format", test_results=[])


class TestResponseParser:
    """Robust parser for LLM test case responses with multiple fallback strategies."""

    def __init__(self, max_retries: int = 2, timeout_seconds: int = 30):
        """
        Initialize the response parser.

        Args:
            max_retries: Maximum number of retry attempts for LLM calls
            timeout_seconds: Timeout for LLM responses
        """
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def parse_test_response(self, response: str) -> List[Dict]:
        """
        Parse LLM response to extract test cases with robust fallback.

        Args:
            response: Raw LLM response string

        Returns:
            List of parsed test case dictionaries
        """
        test_cases = []

        # Strategy 1: Try to parse as direct JSON
        test_cases = self._try_direct_json_parse(response)
        if test_cases:
            logger.info(f"Parsed {len(test_cases)} test cases using direct JSON parsing")
            return test_cases

        # Strategy 2: Extract JSON from markdown code blocks
        test_cases = self._try_markdown_json_parse(response)
        if test_cases:
            logger.info(f"Parsed {len(test_cases)} test cases using markdown JSON extraction")
            return test_cases

        # Strategy 3: Extract from structured text patterns
        test_cases = self._try_text_pattern_extraction(response)
        if test_cases:
            logger.info(f"Parsed {len(test_cases)} test cases using text pattern extraction")
            return test_cases

        logger.warning("Failed to parse any test cases from LLM response")
        return []

    def _try_direct_json_parse(self, response: str) -> List[Dict]:
        """Try to parse response as direct JSON."""
        try:
            parsed = json.loads(response.strip())
            return self._extract_test_cases_from_json(parsed)
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _try_markdown_json_parse(self, response: str) -> List[Dict]:
        """Try to extract JSON from markdown code blocks."""
        try:
            # Look for JSON block in response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                parsed = json.loads(json_str)
                return self._extract_test_cases_from_json(parsed)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return []

    def _extract_test_cases_from_json(self, parsed: Dict) -> List[Dict]:
        """Extract test cases from parsed JSON structure."""
        test_cases = []

        if isinstance(parsed, dict):
            if 'test_cases' in parsed and isinstance(parsed['test_cases'], list):
                test_cases = parsed['test_cases']

                # Log coverage analysis if available
                if 'coverage_analysis' in parsed:
                    coverage = parsed['coverage_analysis']
                    logger.info(f"Test coverage analysis: {coverage}")
            elif isinstance(parsed.get('tests'), list):
                test_cases = parsed['tests']
        elif isinstance(parsed, list):
            test_cases = parsed

        return test_cases

    def _try_text_pattern_extraction(self, text: str) -> List[Dict]:
        """Extract test cases from unstructured text as fallback."""
        test_cases = []

        # Look for test case patterns in text
        lines = text.split('\n')
        current_test = {}

        for line in lines:
            line = line.strip()

            # Look for test case indicators
            if any(indicator in line.lower() for indicator in ['test case', 'test:', 'description:']):
                if current_test:
                    test_cases.append(current_test)
                current_test = {'description': line, 'input': {}, 'expected_output': 'auto_generated'}

            elif 'input:' in line.lower():
                input_part = line.split(':', 1)[1].strip()
                current_test['input'] = self._parse_input_value(input_part)

            elif 'expected:' in line.lower() or 'output:' in line.lower():
                output_part = line.split(':', 1)[1].strip()
                current_test['expected_output'] = output_part

        if current_test:
            test_cases.append(current_test)

        return test_cases

    def _parse_input_value(self, input_str: str) -> Dict[str, Any]:
        """Parse input string into dictionary format."""
        try:
            # Try to evaluate as Python literal
            if input_str.startswith('{') and input_str.endswith('}'):
                return ast.literal_eval(input_str)
            else:
                # Convert to dictionary format
                return {'value': input_str}
        except (ValueError, SyntaxError):
            return {'value': input_str}


class TestCaseStandardizer:
    """Standardizes test case formats and structures."""

    def __init__(self, input_format: InputFormat = InputFormat.DICT):
        """
        Initialize the standardizer.

        Args:
            input_format: Preferred input format for test cases
        """
        self.input_format = input_format

    def standardize_test_cases(self, test_cases: List[Dict]) -> List[Dict]:
        """
        Standardize test case format to ensure consistency.

        Args:
            test_cases: List of test case dictionaries

        Returns:
            List of standardized test case dictionaries
        """
        standardized = []

        for test_case in test_cases:
            standardized_case = self._standardize_single_test_case(test_case)
            if standardized_case:
                standardized.append(standardized_case)

        return standardized

    def _standardize_single_test_case(self, test_case: Dict) -> Dict[str, Any]:
        """Standardize a single test case."""
        # Standard format: input, expected_output, description, test_type
        standardized_case = {
            'description': test_case.get('description', 'Test case'),
            'input': {},
            'expected_output': 'auto_generated',
            'test_type': test_case.get('test_type', 'normal'),
            'reasoning': test_case.get('reasoning', '')
        }

        # Standardize input format
        input_value = test_case.get('input', {})
        standardized_case['input'] = self._standardize_input(input_value)

        # Handle different expected output formats
        if 'expected_output' in test_case:
            standardized_case['expected_output'] = test_case['expected_output']
        elif 'expected_value' in test_case:
            standardized_case['expected_output'] = test_case['expected_value']
        elif 'expected_type' in test_case:
            # For type-based tests, store the expected type
            standardized_case['expected_type'] = test_case['expected_type']
            standardized_case['expected_output'] = 'type_check'
        elif 'output' in test_case:
            standardized_case['expected_output'] = test_case['output']

        return standardized_case

    def _standardize_input(self, input_value: Any) -> Dict[str, Any]:
        """Standardize input value to dictionary format."""
        if isinstance(input_value, dict):
            return input_value
        elif isinstance(input_value, (list, tuple)):
            # Convert list/tuple to indexed dictionary
            return {f'arg_{i}': val for i, val in enumerate(input_value)}
        elif isinstance(input_value, str):
            # Try to parse as dictionary or convert to single value
            try:
                if input_value.startswith('{') and input_value.endswith('}'):
                    return ast.literal_eval(input_value)
                else:
                    return {'value': input_value}
            except (ValueError, SyntaxError):
                return {'value': input_value}
        else:
            return {'value': input_value}



class TestCaseGenerator:
    """
    Enhanced test case generator with multiple strategies.

    This class provides comprehensive test case generation using both
    LLM-based intelligent generation and rule-based fallback generation.
    """

    def __init__(self, config: TestGenerationConfig = None, llm_config: Dict[str, Any] = None):
        """
        Initialize the test case generator.

        Args:
            config: Test generation configuration
            llm_config: LLM configuration for intelligent generation
        """
        self.config = config or TestGenerationConfig()
        self.llm_config = llm_config

        # Initialize sub-modules
        self.signature_parser = FunctionSignatureParser()
        self.standardizer = TestCaseStandardizer(self.config.input_format)
        self.response_parser = TestResponseParser(max_retries=2, timeout_seconds=30)

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
        specification = {
            'name': func_name,
            'description': task_description,
            'code': func_code
        }

        return self.generate_enhanced_test_cases(specification, func_code)

    def generate_enhanced_test_cases(self, specification: Dict[str, Any], code: str = "") -> List[Dict]:
        """
        Generate enhanced test cases using multiple strategies.

        Args:
            specification: Function specification with name, description, signature, etc.
            code: Optional function code for additional context

        Returns:
            List of standardized test case dictionaries
        """
        func_name = specification.get('name', '')
        description = specification.get('description', '')

        logger.info(f"Generating test cases for function: {func_name}")

        # Try LLM-based generation first if available
        if self.config.use_llm_generation and self.llm_config:
            try:
                llm_tests = self._generate_llm_test_cases(specification, code)
                if llm_tests:
                    logger.info(f"Generated {len(llm_tests)} test cases using LLM")
                    return self.standardizer.standardize_test_cases(llm_tests)
            except Exception as e:
                logger.warning(f"LLM test generation failed: {e}, falling back to rule-based generation")

        # Fallback to rule-based generation
        rule_based_tests = self._generate_rule_based_test_cases(specification, code)
        logger.info(f"Generated {len(rule_based_tests)} test cases using rule-based approach")

        return self.standardizer.standardize_test_cases(rule_based_tests)

    def _generate_llm_test_cases(self, specification: Dict[str, Any], code: str = "") -> List[Dict]:
        """Generate test cases using LLM."""
        func_name = specification.get('name', '')
        description = specification.get('description', '')
        signature = specification.get('signature', '')
        examples = specification.get('examples', [])

        # Create comprehensive prompt for LLM
        prompt = f"""# Function Analysis for Test Generation

**Function Name:** {func_name}
**Description:** {description}
**Signature:** {signature}

**Examples:** {examples if examples else "None provided"}

**Source Code:** {code if code else "Not provided"}

Please analyze this function and generate comprehensive test cases that cover:
1. Normal operation scenarios
2. Edge cases (empty inputs, None values, etc.)
3. Boundary conditions (min/max values, limits)
4. Error conditions (invalid inputs, type mismatches)
5. Special cases based on function semantics

Generate the test cases in JSON format with this structure:
{{
    "test_cases": [
        {{
            "description": "Clear description of what this test validates",
            "input": {{"param1": "value1", "param2": "value2"}},
            "expected_output": "expected_result_or_type",
            "test_type": "normal|edge_case|boundary|error_case",
            "reasoning": "Why this test case is important"
        }}
    ]
}}"""

        try:
            # Create temporary LLM agent for test generation
            llm_agent = autogen.AssistantAgent(
                name="test_generator",
                system_message="You are an expert at generating comprehensive test cases for Python functions.",
                llm_config=self.llm_config,
            )

            # Create temporary user proxy for LLM interaction
            user_proxy = autogen.UserProxyAgent(
                name="test_user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )

            # Get LLM response
            user_proxy.initiate_chat(llm_agent, message=prompt)

            # Extract response
            messages = user_proxy.chat_messages.get(llm_agent, [])
            llm_response = messages[-1]['content'] if messages else ""

            # Parse response using robust parser
            test_cases = self.response_parser.parse_test_response(llm_response)

            # Clean up
            user_proxy.chat_messages.clear()

            return test_cases

        except Exception as e:
            logger.warning(f"LLM test generation failed: {e}")
            return []



    def _generate_rule_based_test_cases(self, specification: Dict[str, Any], code: str = "") -> List[Dict]:
        """Generate test cases using rule-based approach."""
        func_name = specification.get('name', '')
        description = specification.get('description', '')
        signature = specification.get('signature', '')
        examples = specification.get('examples', [])

        test_cases = []

        # Start with existing examples if available
        if examples:
            for i, example in enumerate(examples):
                test_cases.append({
                    'description': f'Example {i + 1} from specification',
                    'input': self._parse_input_value(str(example.get('input', ''))),
                    'expected_output': str(example.get('output', 'auto_generated')),
                    'test_type': 'normal',
                    'reasoning': 'Based on provided example'
                })

        # Generate parameter-based tests if signature is available
        if signature and len(test_cases) < self.config.max_test_cases:
            param_tests = self._generate_parameter_based_tests(signature, func_name)
            test_cases.extend(param_tests)

        # Generate semantic tests based on function name and description
        if len(test_cases) < self.config.max_test_cases:
            semantic_tests = self._generate_semantic_tests(func_name, description)
            test_cases.extend(semantic_tests)

        # Add edge cases if configured
        if self.config.include_edge_cases and len(test_cases) < self.config.max_test_cases:
            edge_tests = self._generate_edge_case_tests(func_name, description)
            test_cases.extend(edge_tests)

        # Add boundary tests if configured
        if self.config.include_boundary_tests and len(test_cases) < self.config.max_test_cases:
            boundary_tests = self._generate_boundary_tests(func_name, description)
            test_cases.extend(boundary_tests)

        # Limit to configured maximum
        return test_cases[:self.config.get_test_count_by_complexity()]

    def _generate_parameter_based_tests(self, signature: str, func_name: str) -> List[Dict]:
        """Generate tests based on function signature analysis."""
        test_cases = []

        # Parse function signature
        parameters = self.signature_parser.parse_signature(signature)

        if not parameters:
            return test_cases

        # Generate basic parameter tests
        for param in parameters:
            param_name = param['name']
            param_type = param['type']

            # Generate type-appropriate test values
            test_values = self._get_test_values_for_type(param_type)

            for value in test_values[:2]:  # Limit to 2 values per parameter
                test_cases.append({
                    'description': f'Test {func_name} with {param_name}={repr(value)}',
                    'input': {param_name: value},
                    'expected_output': 'auto_generated',
                    'test_type': 'parameter_test',
                    'reasoning': f'Testing parameter {param_name} of type {param_type}'
                })

        return test_cases

    def _get_test_values_for_type(self, param_type: str) -> List[Any]:
        """Get appropriate test values for a parameter type."""
        type_lower = param_type.lower()

        if 'str' in type_lower:
            return ['test_string', '', 'special!@#$%^&*()']
        elif 'int' in type_lower:
            return [0, 1, -1, 100]
        elif 'float' in type_lower:
            return [0.0, 1.5, -2.7, 3.14159]
        elif 'bool' in type_lower:
            return [True, False]
        elif 'list' in type_lower:
            return [[], [1, 2, 3], ['a', 'b']]
        elif 'dict' in type_lower:
            return [{}, {'key': 'value'}, {'a': 1, 'b': 2}]
        else:
            return ['test_value', 0, True, []]

    def _parse_input_value(self, input_str: str) -> Dict[str, Any]:
        """Parse input string into dictionary format."""
        try:
            # Try to evaluate as Python literal
            if input_str.startswith('{') and input_str.endswith('}'):
                return ast.literal_eval(input_str)
            else:
                # Convert to dictionary format
                return {'value': input_str}
        except (ValueError, SyntaxError):
            return {'value': input_str}

    def _generate_semantic_tests(self, func_name: str, description: str) -> List[Dict]:
        """Generate tests based on semantic analysis of function name and description."""
        test_cases = []

        # Analyze function purpose from name and description
        func_lower = func_name.lower()
        desc_lower = description.lower() if description else ""

        # Common function patterns and their test scenarios
        if any(word in func_lower for word in ['validate', 'check', 'verify']):
            test_cases.extend(self._generate_validation_tests(func_name))
        elif any(word in func_lower for word in ['calculate', 'compute', 'math']):
            test_cases.extend(self._generate_calculation_tests(func_name))
        elif any(word in func_lower for word in ['format', 'convert', 'transform']):
            test_cases.extend(self._generate_transformation_tests(func_name))
        elif any(word in func_lower for word in ['parse', 'extract', 'analyze']):
            test_cases.extend(self._generate_parsing_tests(func_name))

        return test_cases

    def _generate_validation_tests(self, func_name: str) -> List[Dict]:
        """Generate tests for validation functions."""
        return [
            {
                'description': f'Test {func_name} with valid input',
                'input': {'value': 'valid_input'},
                'expected_output': True,
                'test_type': 'validation_positive',
                'reasoning': 'Testing positive validation case'
            },
            {
                'description': f'Test {func_name} with invalid input',
                'input': {'value': 'invalid_input'},
                'expected_output': False,
                'test_type': 'validation_negative',
                'reasoning': 'Testing negative validation case'
            }
        ]

    def _generate_calculation_tests(self, func_name: str) -> List[Dict]:
        """Generate tests for calculation functions."""
        return [
            {
                'description': f'Test {func_name} with positive numbers',
                'input': {'x': 5, 'y': 3},
                'expected_output': 'auto_generated',
                'test_type': 'calculation_positive',
                'reasoning': 'Testing calculation with positive values'
            },
            {
                'description': f'Test {func_name} with zero',
                'input': {'x': 0, 'y': 0},
                'expected_output': 'auto_generated',
                'test_type': 'calculation_zero',
                'reasoning': 'Testing calculation with zero values'
            }
        ]

    def _generate_transformation_tests(self, func_name: str) -> List[Dict]:
        """Generate tests for transformation functions."""
        return [
            {
                'description': f'Test {func_name} with typical input',
                'input': {'data': 'sample_data'},
                'expected_output': 'auto_generated',
                'test_type': 'transformation_normal',
                'reasoning': 'Testing transformation with normal input'
            }
        ]

    def _generate_parsing_tests(self, func_name: str) -> List[Dict]:
        """Generate tests for parsing functions."""
        return [
            {
                'description': f'Test {func_name} with well-formed input',
                'input': {'text': 'well_formed_input'},
                'expected_output': 'auto_generated',
                'test_type': 'parsing_valid',
                'reasoning': 'Testing parsing with valid input'
            }
        ]

    def _generate_edge_case_tests(self, func_name: str, description: str) -> List[Dict]:
        """Generate edge case tests."""
        return [
            {
                'description': f'Test {func_name} with empty input',
                'input': {'value': ''},
                'expected_output': 'auto_generated',
                'test_type': 'edge_case',
                'reasoning': 'Testing edge case with empty input'
            },
            {
                'description': f'Test {func_name} with None input',
                'input': {'value': None},
                'expected_output': 'auto_generated',
                'test_type': 'edge_case',
                'reasoning': 'Testing edge case with None input'
            }
        ]

    def _generate_boundary_tests(self, func_name: str, description: str) -> List[Dict]:
        """Generate boundary value tests."""
        return [
            {
                'description': f'Test {func_name} with minimum boundary',
                'input': {'value': 0},
                'expected_output': 'auto_generated',
                'test_type': 'boundary_test',
                'reasoning': 'Testing minimum boundary value'
            },
            {
                'description': f'Test {func_name} with maximum boundary',
                'input': {'value': 999999},
                'expected_output': 'auto_generated',
                'test_type': 'boundary_test',
                'reasoning': 'Testing maximum boundary value'
            }
        ]



    def normalize_test_input(self, test_cases) -> List[Dict]:
        """
        Normalize test input to handle multiple formats.

        Args:
            test_cases: Test cases in various formats (List[Dict], str, Dict)

        Returns:
            List of normalized test case dictionaries
        """
        if not test_cases:
            return []

        # Handle different input formats
        if isinstance(test_cases, str):
            try:
                import json
                # Try to parse as JSON
                parsed = json.loads(test_cases)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                # Treat as single test case description
                return [{'description': test_cases, 'input': {}, 'expected_output': 'auto_generated'}]

        elif isinstance(test_cases, dict):
            return [test_cases]

        elif isinstance(test_cases, list):
            return test_cases

        else:
            return []


# Backward compatibility alias
EnhancedTestCaseGenerator = TestCaseGenerator