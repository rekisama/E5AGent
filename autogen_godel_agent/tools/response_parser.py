"""
LLM Response Parser for Task Planning Agent.

This module provides robust parsing and validation of LLM responses,
with fallback strategies for different response formats.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Robust parser for LLM responses with multiple parsing strategies.
    
    Features:
    - JSON-first parsing with schema validation
    - Regex-based fallback parsing
    - Structured data validation and completion
    - Example extraction with multiple strategies
    """

    def __init__(self):
        self.required_keys = ['function_found', 'matched_functions', 'needs_new_function']
        self.optional_keys = ['suggested_function_spec', 'reasoning']

    def parse_llm_analysis(self, llm_response: str, task_description: str) -> Dict[str, Any]:
        """
        Parse the LLM's analysis response into structured data.

        First tries to parse as JSON, falls back to regex-based parsing.

        Args:
            llm_response: The LLM's analysis response
            task_description: Original task description

        Returns:
            Structured analysis result
        """
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
            if any(key in potential_json for key in self.required_keys):
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
        """Extract function signature from LLM response."""
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
                return signature

        return "Signature not specified"

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


# Global parser instance
_global_parser: Optional[ResponseParser] = None


def get_response_parser() -> ResponseParser:
    """Get or create global response parser instance."""
    global _global_parser
    if _global_parser is None:
        _global_parser = ResponseParser()
    return _global_parser
