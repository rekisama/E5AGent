"""
Enhanced LLM-Driven Test Case Generator for Function Creator Agent.

This module provides intelligent test case generation using LLM analysis,
addressing the following improvements:
1. LLM-based test case generation for better coverage
2. Unified test case format (standardized input/output structure)
3. Enhanced parameter type inference with description analysis
4. Configurable test case limits and complexity
5. Robust parameter serialization using repr()
6. Boundary value testing and edge case coverage
7. Context-aware test generation based on function semantics

为系统中的任意 Python 函数，根据其规范（如函数名、签名、描述、示例）和可选的函数源码，
自动生成结构化、合理、覆盖率高的测试用例，用于 Function Creator Agent 中对自动生成函数的行为进行验证。

Architecture (SRP Refactored):
- LLMBasedTestGenerator: Handles LLM interactions and intelligent test generation
- RuleBasedTestGenerator: Provides fallback rule-based test generation
- TestCaseStandardizer: Standardizes test case formats and structures
- FunctionSignatureParser: Parses function signatures and extracts parameters
- TestResponseParser: Robust parsing of LLM responses with fallback mechanisms
"""

from typing import Dict, List, Any, Union, Optional, Tuple
import json
import yaml
import logging
import re
import ast
import time
from dataclasses import dataclass, field
from enum import Enum
import autogen

# Configure logger
logger = logging.getLogger(__name__)


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


class RuleBasedTestGenerator:
    """Generates test cases using rule-based patterns and heuristics."""

    def __init__(self, config: TestGenerationConfig):
        """
        Initialize the rule-based generator.

        Args:
            config: Test generation configuration
        """
        self.config = config
        self.signature_parser = FunctionSignatureParser()

    def generate_test_cases(self, specification: Dict[str, Any], code: str = "") -> List[Dict]:
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


class LLMBasedTestGenerator:
    """Handles LLM interactions and intelligent test generation with retry and timeout."""

    def __init__(self, llm_config: Dict[str, Any], max_retries: int = 2, timeout_seconds: int = 30):
        """
        Initialize the LLM-based generator.

        Args:
            llm_config: LLM configuration dictionary
            max_retries: Maximum retry attempts for LLM calls
            timeout_seconds: Timeout for LLM responses
        """
        self.llm_config = llm_config
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.response_parser = TestResponseParser(max_retries, timeout_seconds)

        # Initialize LLM agent
        self.llm_agent = None
        self._initialize_llm_agent()

    def _initialize_llm_agent(self):
        """Initialize the LLM agent for test case generation."""
        try:
            self.llm_agent = autogen.AssistantAgent(
                name="test_generator",
                system_message=self._get_test_generation_system_message(),
                llm_config=self.llm_config,
            )
            logger.info("LLM agent initialized for test case generation")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM agent: {e}")
            raise

    def _get_test_generation_system_message(self) -> str:
        """Get system message for LLM-based test generation."""
        return """# Test Case Generation Expert

You are an expert at generating comprehensive test cases for Python functions.

## TOOLS
- Function specification analysis
- Parameter type inference
- Edge case identification
- Boundary value analysis
- Test coverage optimization

## OUTPUT FORMAT
Always respond with a JSON object in this exact format:

```json
{
    "test_cases": [
        {
            "description": "Clear description of what this test validates",
            "input": {"param1": "value1", "param2": "value2"},
            "expected_output": "expected_result_or_type",
            "test_type": "normal|edge_case|boundary|error_case",
            "reasoning": "Why this test case is important"
        }
    ],
    "coverage_analysis": "Brief analysis of test coverage achieved"
}
```

## GUIDELINES
1. Generate diverse test cases covering normal, edge, and boundary conditions
2. Use dictionary format for inputs: {"param_name": "param_value"}
3. Include clear descriptions and reasoning for each test
4. Consider function semantics from name and description
5. Ensure test cases are realistic and meaningful
6. Limit to maximum 8 test cases for comprehensive coverage
"""

    def generate_test_cases_with_retry(self, specification: Dict[str, Any], code: str = "") -> List[Dict]:
        """
        Generate test cases using LLM with retry mechanism.

        Args:
            specification: Function specification dictionary
            code: Optional function source code

        Returns:
            List of test case dictionaries
        """
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

Generate the test cases now."""

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"LLM test generation attempt {attempt + 1}/{self.max_retries + 1}")

                # Create temporary user proxy for LLM interaction
                user_proxy = autogen.UserProxyAgent(
                    name="test_user",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=1,
                    code_execution_config=False,
                )

                # Set timeout for the conversation
                start_time = time.time()

                # Get LLM response
                user_proxy.initiate_chat(self.llm_agent, message=prompt)

                # Check timeout
                if time.time() - start_time > self.timeout_seconds:
                    raise TimeoutError(f"LLM response timeout after {self.timeout_seconds} seconds")

                # Extract response
                messages = user_proxy.chat_messages.get(self.llm_agent, [])
                llm_response = messages[-1]['content'] if messages else ""

                # Parse response using robust parser
                test_cases = self.response_parser.parse_test_response(llm_response)

                # Clean up
                user_proxy.chat_messages.clear()

                if test_cases:
                    logger.info(f"Successfully generated {len(test_cases)} test cases on attempt {attempt + 1}")
                    return test_cases
                else:
                    raise ValueError("No test cases parsed from LLM response")

            except Exception as e:
                last_exception = e
                logger.warning(f"LLM test generation attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries:
                    logger.info(f"Retrying in 1 second... ({self.max_retries - attempt} attempts remaining)")
                    time.sleep(1)
                else:
                    logger.error(f"All LLM test generation attempts failed. Last error: {e}")

        # If all attempts failed, raise the last exception
        raise last_exception or Exception("LLM test generation failed with unknown error")


class EnhancedTestCaseGenerator:
    """
    Enhanced LLM-driven test case generator with modular architecture.

    Architecture (SRP Refactored):
    - Uses LLMBasedTestGenerator for intelligent LLM-driven test generation
    - Uses RuleBasedTestGenerator for fallback rule-based generation
    - Uses TestCaseStandardizer for consistent test case formatting
    - Uses TestResponseParser for robust LLM response parsing
    - Uses FunctionSignatureParser for parameter analysis

    Improvements:
    - Modular design following Single Responsibility Principle
    - Robust LLM response parsing with multiple fallback strategies
    - Retry mechanism with timeout for LLM calls
    - Enhanced error handling and logging
    - Configurable complexity levels and test limits
    """

    def __init__(self, config: TestGenerationConfig = None, llm_config: Dict[str, Any] = None):
        """
        Initialize the enhanced test case generator.

        Args:
            config: Test generation configuration
            llm_config: LLM configuration for intelligent generation
        """
        self.config = config or TestGenerationConfig()
        self.llm_config = llm_config

        # Initialize sub-modules
        self.standardizer = TestCaseStandardizer(self.config.input_format)
        self.rule_generator = RuleBasedTestGenerator(self.config)

        # Initialize LLM generator if configuration provided
        self.llm_generator = None
        if llm_config and self.config.use_llm_generation:
            try:
                self.llm_generator = LLMBasedTestGenerator(
                    llm_config=llm_config,
                    max_retries=2,
                    timeout_seconds=30
                )
                logger.info("LLM-based test generator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM generator: {e}")
                self.config.use_llm_generation = False

    def _get_test_generation_system_message(self) -> str:
        """Get system message for LLM-based test generation."""
        return """# Test Case Generation Expert

## ROLE
Generate comprehensive, intelligent test cases for Python functions based on specifications.

## CAPABILITIES
- Semantic analysis of function purpose and behavior
- Edge case and boundary value identification
- Type-aware test data generation
- Context-sensitive test scenario creation

## OUTPUT FORMAT
Always respond with JSON containing test cases:

```json
{
  "test_cases": [
    {
      "description": "Clear description of what this test validates",
      "input": {"param1": "value1", "param2": "value2"},
      "expected_output": "expected_result_or_type",
      "test_type": "normal|edge|boundary|error",
      "reasoning": "Why this test case is important"
    }
  ],
  "coverage_analysis": {
    "normal_cases": 3,
    "edge_cases": 2,
    "boundary_cases": 2,
    "error_cases": 1
  }
}
```

## REQUIREMENTS
1. **Input Format**: Always use dictionary format for inputs
2. **Parameter Serialization**: Use proper Python repr() for complex types
3. **Edge Cases**: Include boundary values, empty inputs, null values
4. **Error Cases**: Test invalid inputs and error conditions
5. **Type Awareness**: Generate type-appropriate test data
6. **Semantic Understanding**: Consider function purpose and domain

## ANALYSIS PROCESS
1. Parse function signature and parameter types
2. Analyze function description for domain context
3. Identify normal use cases and expected behaviors
4. Determine edge cases and boundary conditions
5. Consider error scenarios and invalid inputs
6. Generate diverse, meaningful test data"""

    def generate_test_cases(self, specification: Dict[str, Any], code: str = "") -> List[Dict]:
        """
        Main entry point for test case generation.

        Args:
            specification: Function specification with name, description, signature, etc.
            code: Optional function code for additional context

        Returns:
            List of standardized test case dictionaries
        """
        func_name = specification.get('name', '')
        description = specification.get('description', '')
        signature = specification.get('signature', '')

        logger.info(f"Generating test cases for function: {func_name}")

        # Try LLM-based generation first if available
        if self.config.use_llm_generation and self.llm_generator:
            try:
                llm_tests = self.llm_generator.generate_test_cases_with_retry(specification, code)
                if llm_tests:
                    logger.info(f"Generated {len(llm_tests)} test cases using LLM")
                    return self.standardizer.standardize_test_cases(llm_tests)
            except Exception as e:
                logger.warning(f"LLM test generation failed: {e}, falling back to rule-based generation")

        # Fallback to rule-based generation
        rule_based_tests = self.rule_generator.generate_test_cases(specification, code)
        logger.info(f"Generated {len(rule_based_tests)} test cases using rule-based approach")

        return self.standardizer.standardize_test_cases(rule_based_tests)


# Backward compatibility alias
TestCaseGenerator = EnhancedTestCaseGenerator
