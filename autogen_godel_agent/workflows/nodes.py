"""
LangGraph Node Functions for Workflow Processing.

This module contains the individual node functions that make up the
LangGraph workflow. Each node represents a specific step in the
task processing pipeline.
"""

import time
import logging
import json
import re
import math
from typing import Dict, Any, List, Tuple, Optional
import uuid
from collections import Counter

from .state import (
    WorkflowState, TaskStatus, WorkflowStep,
    FunctionSearchResult, CompositionResult, CreationResult, ExecutionResult
)

logger = logging.getLogger(__name__)

def log_structured(level: str, event: str, **kwargs):
    """Log structured information in JSON format."""
    log_data = {
        "event": event,
        "timestamp": time.time(),
        **kwargs
    }
    getattr(logger, level.lower())(json.dumps(log_data))

def _normalize_word(word: str) -> str:
    """Normalize words to handle common variations."""
    # Handle common word variations and synonyms
    normalizations = {
        # Action variations
        'validation': 'validate',
        'validating': 'validate',
        'validates': 'validate',
        'calculation': 'calculate',
        'calculating': 'calculate',
        'calculates': 'calculate',
        'computation': 'compute',
        'computing': 'compute',
        'computes': 'compute',
        'formatting': 'format',
        'formats': 'format',
        'processing': 'process',
        'processes': 'process',
        'creation': 'create',
        'creating': 'create',
        'creates': 'create',
        'generation': 'generate',
        'generating': 'generate',
        'generates': 'generate',
        'verification': 'verify',
        'verifying': 'verify',
        'verifies': 'verify',
        'authentication': 'authenticate',
        'authenticating': 'authenticate',
        'authenticates': 'authenticate',

        # Synonyms
        'function': 'func',
        'method': 'func',
        'procedure': 'func',
        'routine': 'func',
        'email': 'mail',
        'e-mail': 'mail',
        'electronic-mail': 'mail',
        'message': 'msg',
        'notification': 'notify',
        'notifying': 'notify',
        'address': 'addr',
        'string': 'str',
        'text': 'str',
        'data': 'info',
        'information': 'info',
        'number': 'num',
        'numeric': 'num',
        'currency': 'money',
        'monetary': 'money',
        'dollar': 'money',
        'price': 'money',
        'cost': 'money',
        'value': 'val',
        'values': 'val',
        'input': 'in',
        'output': 'out',
        'result': 'out',
        'return': 'out'
    }
    return normalizations.get(word, word)

def _calculate_semantic_similarity(words1: set, words2: set) -> float:
    """Calculate semantic similarity considering word variations."""
    # Normalize words
    norm_words1 = {_normalize_word(w) for w in words1}
    norm_words2 = {_normalize_word(w) for w in words2}

    # Calculate overlap with normalized words
    overlap = len(norm_words1 & norm_words2)
    union = len(norm_words1 | norm_words2)

    return overlap / union if union > 0 else 0.0

def _calculate_sequence_similarity(text1: str, text2: str) -> float:
    """Calculate similarity based on word sequence and order."""
    words1 = re.findall(r'\w+', text1.lower())
    words2 = re.findall(r'\w+', text2.lower())

    if not words1 or not words2:
        return 0.0

    # Use longest common subsequence approach
    max_len = max(len(words1), len(words2))
    common_sequences = 0

    # Simple n-gram overlap
    for i in range(len(words1)):
        for j in range(len(words2)):
            if words1[i] == words2[j]:
                common_sequences += 1
                break

    return common_sequences / max_len

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using improved multiple methods.

    Returns a score between 0.0 and 1.0 where 1.0 is identical.
    """
    if not text1 or not text2:
        return 0.0

    # Handle identical texts
    if text1.lower().strip() == text2.lower().strip():
        return 1.0

    text1_lower = text1.lower()
    text2_lower = text2.lower()

    # Extract words
    words1 = set(re.findall(r'\w+', text1_lower))
    words2 = set(re.findall(r'\w+', text2_lower))

    if not words1 or not words2:
        return 0.0

    # Method 1: Basic keyword overlap (Jaccard)
    basic_overlap = len(words1 & words2) / len(words1 | words2)

    # Method 2: Semantic similarity (with word normalization)
    semantic_sim = _calculate_semantic_similarity(words1, words2)

    # Method 3: Sequence similarity (word order matters)
    sequence_sim = _calculate_sequence_similarity(text1, text2)

    # Method 4: Character-level similarity for partial matches
    char_set1 = set(text1_lower.replace(' ', ''))
    char_set2 = set(text2_lower.replace(' ', ''))
    char_similarity = len(char_set1 & char_set2) / len(char_set1 | char_set2) if char_set1 or char_set2 else 0.0

    # Method 5: Length-adjusted similarity (penalize very different lengths)
    len1, len2 = len(text1_lower), len(text2_lower)
    length_factor = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

    # Weighted combination with optimized "Semantic Heavy" configuration
    final_score = (
        0.20 * basic_overlap +      # Basic word overlap
        0.60 * semantic_sim +       # Semantic similarity (maximum weight)
        0.10 * sequence_sim +       # Word sequence
        0.08 * char_similarity +    # Character level
        0.02 * length_factor        # Length adjustment
    )

    return min(1.0, max(0.0, final_score))

def calculate_function_relevance(query: str, function_info: Dict[str, Any]) -> float:
    """
    Calculate relevance score for a function based on query.

    Args:
        query: The search query/task description
        function_info: Function information dictionary

    Returns:
        Relevance score between 0.0 and 1.0
    """
    if not query or not function_info:
        return 0.0

    # Extract function details
    func_name = function_info.get('name', '')
    func_desc = function_info.get('description', '')
    func_tags = function_info.get('tags', [])

    # Calculate similarities
    name_similarity = calculate_text_similarity(query, func_name)
    desc_similarity = calculate_text_similarity(query, func_desc)

    # Tag matching
    query_words = set(re.findall(r'\w+', query.lower()))
    tag_words = set(tag.lower() for tag in func_tags if isinstance(tag, str))
    tag_overlap = len(query_words & tag_words) / len(query_words) if query_words else 0.0

    # Weighted final score
    relevance_score = (
        0.2 * name_similarity +
        0.6 * desc_similarity +
        0.2 * tag_overlap
    )

    return min(1.0, max(0.0, relevance_score))

# Import existing agents and tools
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from agents.planner_agent import TaskPlannerAgent
    from agents.function_creator_agent import FunctionCreatorAgent
    from tools.function_tools import get_function_tools
    from tools.function_composer import get_function_composer
    from config import Config
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    # Fallback imports or error handling
    TaskPlannerAgent = None
    FunctionCreatorAgent = None
    get_function_tools = None
    get_function_composer = None
    Config = None

# Initialize shared components
_llm_config = None
_planner_agent = None
_creator_agent = None
_function_tools = None
_function_composer = None

def _get_shared_components():
    """Get or initialize shared components."""
    global _llm_config, _planner_agent, _creator_agent, _function_tools, _function_composer

    if _llm_config is None:
        try:
            if Config:
                _llm_config = Config().get_llm_config()
            else:
                # Fallback configuration
                _llm_config = {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.7,
                    'max_tokens': 1000
                }

            _planner_agent = TaskPlannerAgent(_llm_config) if TaskPlannerAgent else None
            _creator_agent = FunctionCreatorAgent(_llm_config) if FunctionCreatorAgent else None
            _function_tools = get_function_tools() if get_function_tools else None
            _function_composer = get_function_composer() if get_function_composer else None

        except Exception as e:
            logger.error(f"Failed to initialize shared components: {e}")
            # Set fallback values
            _llm_config = {}
            _planner_agent = None
            _creator_agent = None
            _function_tools = None
            _function_composer = None

    return _planner_agent, _creator_agent, _function_tools, _function_composer


def analyze_task_node(state: WorkflowState) -> WorkflowState:
    """
    Analyze the user task to understand requirements and determine next steps.

    This node uses the TaskPlannerAgent to analyze the task and determine
    what functions might be needed.
    """
    log_structured("info", "task_analysis_start",
                  task_description=state.task_context.description,
                  session_id=state.task_context.session_id)
    start_time = time.time()

    try:
        state.status = TaskStatus.ANALYZING
        state.current_step = WorkflowStep.ANALYZE_TASK

        planner_agent, _, _, _ = _get_shared_components()

        if not planner_agent:
            raise Exception("TaskPlannerAgent not available")

        # Use the existing planner agent to analyze the task
        analysis_result = planner_agent.analyze_task(
            state.task_context.description,
            state.task_context.session_id
        )

        state.analysis_result = analysis_result

        # Extract token usage if available
        tokens_used = analysis_result.get('tokens_used', 0)
        if tokens_used:
            state.total_tokens_used += tokens_used

        log_structured("info", "task_analysis_completed",
                      status=analysis_result.get('status', 'unknown'),
                      tokens_used=tokens_used,
                      duration=time.time() - start_time)

    except Exception as e:
        error_msg = f"Task analysis failed: {str(e)}"
        state.add_error("analyze_task", error_msg, e)
        state.status = TaskStatus.FAILED
        log_structured("error", "task_analysis_failed",
                      error_message=error_msg,
                      duration=time.time() - start_time)

    finally:
        duration = time.time() - start_time
        state.set_step_timing("analyze_task", duration)

    return state


def search_functions_node(state: WorkflowState) -> WorkflowState:
    """
    Search for existing functions that might solve the task.

    This node uses the existing function_tools.search_functions() method
    and enhances it with advanced similarity scoring for better relevance ranking.
    """
    log_structured("info", "function_search_start",
                  search_query=state.task_context.description)
    start_time = time.time()

    try:
        state.status = TaskStatus.SEARCHING
        state.current_step = WorkflowStep.SEARCH_FUNCTIONS

        _, _, function_tools, _ = _get_shared_components()

        if not function_tools:
            raise Exception("Function tools not available")

        search_query = state.task_context.description

        # Use the existing search_functions method (it already handles basic matching)
        basic_search_results = function_tools.search_functions(search_query)

        log_structured("info", "basic_search_completed",
                      basic_matches=len(basic_search_results))

        # If basic search found results, enhance with similarity scoring
        enhanced_results = []
        if basic_search_results:
            for func_info in basic_search_results:
                # Calculate advanced relevance score
                relevance_score = calculate_function_relevance(search_query, func_info)
                func_info['relevance_score'] = relevance_score
                enhanced_results.append(func_info)

            # Sort by enhanced relevance score
            enhanced_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        # If basic search found nothing, try broader search with all functions
        if not enhanced_results:
            log_structured("info", "performing_fallback_search")

            all_functions = function_tools.list_functions()
            valid_functions = [f for f in all_functions
                              if f not in ['metadata', 'last_updated', 'version', 'total_functions']]

            # Get detailed info and calculate similarity for all functions
            for func_name in valid_functions[:20]:  # Limit to avoid performance issues
                try:
                    func_info = function_tools.get_function_info(func_name)
                    if func_info:
                        relevance_score = calculate_function_relevance(search_query, func_info)
                        if relevance_score > 0.15:  # Higher threshold for fallback search
                            func_info['relevance_score'] = relevance_score
                            enhanced_results.append(func_info)
                except Exception as e:
                    log_structured("warning", "fallback_function_processing_failed",
                                 function_name=func_name, error=str(e))

            # Sort fallback results
            enhanced_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        # Take top matches
        final_results = enhanced_results[:10]

        # Create search result object
        search_result = FunctionSearchResult(
            found_functions=final_results,
            search_query=search_query,
            total_functions=len(function_tools.list_functions()),
            search_strategy="hybrid_basic_plus_similarity"
        )

        # Store relevance scores
        for func in final_results:
            func_name = func.get('name', '')
            relevance = func.get('relevance_score', 0.0)
            search_result.relevance_scores[func_name] = relevance

        state.search_result = search_result

        log_structured("info", "function_search_completed",
                      found_functions=len(final_results),
                      search_strategy="hybrid",
                      top_score=final_results[0].get('relevance_score', 0) if final_results else 0,
                      duration=time.time() - start_time)

    except Exception as e:
        error_msg = f"Function search failed: {str(e)}"
        state.add_error("search_functions", error_msg, e)
        state.status = TaskStatus.FAILED
        log_structured("error", "function_search_failed",
                      error_message=error_msg,
                      duration=time.time() - start_time)

    finally:
        duration = time.time() - start_time
        state.set_step_timing("search_functions", duration)

    return state


def compose_functions_node(state: WorkflowState) -> WorkflowState:
    """
    Attempt to compose existing functions to solve the task.

    This node tries to combine existing functions to create a solution
    without needing to create entirely new functions.
    """
    log_structured("info", "function_composition_start",
                  task_description=state.task_context.description)
    start_time = time.time()

    try:
        state.status = TaskStatus.COMPOSING
        state.current_step = WorkflowStep.COMPOSE_FUNCTIONS

        _, _, _, function_composer = _get_shared_components()

        if not function_composer:
            raise Exception("Function composer not available")

        # Check if composition is feasible
        can_compose, reason = function_composer.can_compose_for_task(
            state.task_context.description
        )

        composition_result = CompositionResult()

        if can_compose:
            log_structured("info", "composition_feasible", reason=reason)

            # Attempt composition
            success, message, composite_func = function_composer.compose_functions(
                state.task_context.description
            )

            composition_result.success = success
            composition_result.error_message = message if not success else ""

            if success and composite_func:
                composition_result.composite_function = composite_func
                composition_result.component_functions = composite_func.get('component_functions', [])
                composition_result.composition_strategy = "llm_guided"

                log_structured("info", "function_composition_successful",
                              composite_function=composite_func.get('name', 'unnamed'),
                              component_functions=composition_result.component_functions,
                              duration=time.time() - start_time)
            else:
                log_structured("warning", "function_composition_failed",
                              error_message=message,
                              duration=time.time() - start_time)
        else:
            composition_result.success = False
            composition_result.error_message = reason
            log_structured("info", "composition_not_feasible",
                          reason=reason,
                          duration=time.time() - start_time)

        state.composition_result = composition_result

    except Exception as e:
        error_msg = f"Function composition failed: {str(e)}"
        state.add_error("compose_functions", error_msg, e)
        state.status = TaskStatus.FAILED
        log_structured("error", "function_composition_error",
                      error_message=error_msg,
                      duration=time.time() - start_time)

    finally:
        duration = time.time() - start_time
        state.set_step_timing("compose_functions", duration)

    return state


def create_function_node(state: WorkflowState) -> WorkflowState:
    """
    Create a new function to solve the task.

    This node uses the FunctionCreatorAgent to generate, test, and register
    a new function based on the task requirements.
    """
    log_structured("info", "function_creation_start",
                  task_description=state.task_context.description,
                  retry_count=state.retry_count)
    start_time = time.time()

    try:
        state.status = TaskStatus.CREATING
        state.current_step = WorkflowStep.CREATE_FUNCTION

        _, creator_agent, _, _ = _get_shared_components()

        if not creator_agent:
            raise Exception("FunctionCreatorAgent not available")

        # Extract function specification from analysis
        analysis = state.analysis_result or {}
        suggested_spec = analysis.get('suggested_function_spec', {})

        # Create a basic specification if not provided
        if not suggested_spec:
            function_name = f"task_function_{uuid.uuid4().hex[:8]}"
            suggested_spec = {
                'name': function_name,
                'description': state.task_context.description,
                'parameters': [],
                'return_type': 'Any',
                'examples': []
            }
            log_structured("info", "using_default_function_spec",
                          function_name=function_name)

        # Use creator agent to create the function
        success, message, function_code = creator_agent.create_function(suggested_spec)

        creation_result = CreationResult(
            success=success,
            function_name=suggested_spec.get('name', ''),
            function_code=function_code or "",
            error_message=message if not success else "",
            retry_count=state.retry_count
        )

        state.creation_result = creation_result

        if success:
            log_structured("info", "function_creation_successful",
                          function_name=creation_result.function_name,
                          code_length=len(function_code) if function_code else 0,
                          duration=time.time() - start_time)
        else:
            log_structured("warning", "function_creation_failed",
                          error_message=message,
                          function_name=suggested_spec.get('name', ''),
                          duration=time.time() - start_time)

    except Exception as e:
        error_msg = f"Function creation failed: {str(e)}"
        state.add_error("create_function", error_msg, e)
        state.status = TaskStatus.FAILED
        log_structured("error", "function_creation_error",
                      error_message=error_msg,
                      duration=time.time() - start_time)

    finally:
        duration = time.time() - start_time
        state.set_step_timing("create_function", duration)

    return state


def _execute_function_safely(function_tools, function_name: str, task_description: str) -> Tuple[bool, Any, str]:
    """
    Safely execute a function with proper error handling.

    Args:
        function_tools: Function tools instance
        function_name: Name of function to execute
        task_description: Task description for context

    Returns:
        Tuple of (success, result, error_message)
    """
    try:
        # Get function information
        func_info = function_tools.registry.get_function(function_name)
        if not func_info:
            return False, None, f"Function {function_name} not found"

        # Check if function has executable code
        if 'code' not in func_info:
            return False, None, f"Function {function_name} has no executable code"

        # Try to execute the function
        # This is a simplified execution - in practice, you'd need proper parameter handling
        result = function_tools.execute_function(function_name, task_description)

        return True, result, ""

    except Exception as e:
        return False, None, f"Function execution failed: {str(e)}"

def _execute_composed_function(function_tools, composite_func: Dict[str, Any], task_description: str) -> Tuple[bool, Any, str]:
    """
    Execute a composed function by running its components in sequence.

    Args:
        function_tools: Function tools instance
        composite_func: Composite function definition
        task_description: Task description

    Returns:
        Tuple of (success, result, error_message)
    """
    try:
        component_functions = composite_func.get('component_functions', [])
        if not component_functions:
            return False, None, "No component functions found"

        results = []
        current_input = task_description

        for func_name in component_functions:
            success, result, error = _execute_function_safely(function_tools, func_name, current_input)
            if not success:
                return False, None, f"Component function {func_name} failed: {error}"

            results.append(result)
            current_input = str(result)  # Use result as input for next function

        return True, results, ""

    except Exception as e:
        return False, None, f"Composed function execution failed: {str(e)}"

def execute_task_node(state: WorkflowState) -> WorkflowState:
    """
    Execute the task using available or created functions.

    This node attempts to execute the task using the best available solution
    (existing functions, composed functions, or newly created functions).
    """
    log_structured("info", "task_execution_start",
                  task_description=state.task_context.description)
    start_time = time.time()

    try:
        state.status = TaskStatus.EXECUTING
        state.current_step = WorkflowStep.EXECUTE_TASK

        _, _, function_tools, _ = _get_shared_components()
        execution_result = ExecutionResult()
        functions_used = []

        # Determine which solution to use and execute it
        if (state.composition_result and state.composition_result.is_successful):
            # Execute composed function
            composite_func = state.composition_result.composite_function
            functions_used = state.composition_result.component_functions

            log_structured("info", "executing_composed_function",
                          composite_function=composite_func.get('name', 'unnamed'),
                          component_functions=functions_used)

            success, result, error = _execute_composed_function(
                function_tools, composite_func, state.task_context.description
            )

            execution_result.success = success
            execution_result.result_data = {
                'type': 'composition',
                'function': composite_func.get('name', 'composite_function'),
                'components': functions_used,
                'result': result
            }
            execution_result.error_message = error if not success else ""

        elif (state.creation_result and state.creation_result.is_successful):
            # Execute newly created function
            function_name = state.creation_result.function_name
            functions_used = [function_name]

            log_structured("info", "executing_new_function",
                          function_name=function_name)

            success, result, error = _execute_function_safely(
                function_tools, function_name, state.task_context.description
            )

            execution_result.success = success
            execution_result.result_data = {
                'type': 'new_function',
                'function': function_name,
                'code': state.creation_result.function_code,
                'result': result
            }
            execution_result.error_message = error if not success else ""

        elif (state.search_result and state.search_result.has_matches):
            # Execute existing function
            best_match = state.search_result.best_match
            if best_match:
                function_name = best_match.get('name', 'unknown')
                functions_used = [function_name]

                log_structured("info", "executing_existing_function",
                              function_name=function_name,
                              relevance_score=best_match.get('relevance_score', 0))

                success, result, error = _execute_function_safely(
                    function_tools, function_name, state.task_context.description
                )

                execution_result.success = success
                execution_result.result_data = {
                    'type': 'existing_function',
                    'function': function_name,
                    'description': best_match.get('description', ''),
                    'result': result
                }
                execution_result.error_message = error if not success else ""

        if not execution_result.success and not execution_result.error_message:
            execution_result.error_message = "No suitable solution found"

        execution_result.functions_used = functions_used
        state.execution_result = execution_result
        state.status = TaskStatus.COMPLETED if execution_result.success else TaskStatus.FAILED

        log_structured("info", "task_execution_completed",
                      success=execution_result.success,
                      functions_used=functions_used,
                      execution_type=execution_result.result_data.get('type', 'unknown') if execution_result.result_data else 'none',
                      duration=time.time() - start_time)

    except Exception as e:
        error_msg = f"Task execution failed: {str(e)}"
        state.add_error("execute_task", error_msg, e)
        state.status = TaskStatus.FAILED
        log_structured("error", "task_execution_failed",
                      error_message=error_msg,
                      duration=time.time() - start_time)

    finally:
        duration = time.time() - start_time
        state.set_step_timing("execute_task", duration)

    return state


def error_handler_node(state: WorkflowState) -> WorkflowState:
    """
    Handle errors and determine if retry is possible.

    This node processes errors and decides whether to retry the workflow
    or terminate with failure.
    """
    error_count = len(state.error_history)
    log_structured("info", "error_handling_start",
                  error_count=error_count,
                  retry_count=state.retry_count,
                  max_retries=getattr(state, 'max_retries', 3))
    start_time = time.time()

    try:
        state.current_step = WorkflowStep.ERROR_HANDLER

        # Analyze error patterns
        error_types = {}
        for error in state.error_history:
            error_step = error.get('step', 'unknown')
            error_types[error_step] = error_types.get(error_step, 0) + 1

        # Determine if we should retry
        if state.can_retry():
            log_structured("info", "workflow_retry_initiated",
                          retry_count=state.retry_count + 1,
                          error_patterns=error_types,
                          duration=time.time() - start_time)

            state.increment_retry()
            # Reset some state for retry
            state.current_step = WorkflowStep.START
            state.next_step = WorkflowStep.ANALYZE_TASK
        else:
            log_structured("error", "workflow_terminated_max_retries",
                          final_retry_count=state.retry_count,
                          error_patterns=error_types,
                          total_errors=error_count,
                          duration=time.time() - start_time)

            state.status = TaskStatus.FAILED
            state.should_terminate = True

    except Exception as e:
        error_msg = f"Error handler failed: {str(e)}"
        state.add_error("error_handler", error_msg, e)
        state.status = TaskStatus.FAILED
        state.should_terminate = True
        log_structured("error", "error_handler_failed",
                      error_message=error_msg,
                      duration=time.time() - start_time)

    finally:
        duration = time.time() - start_time
        state.set_step_timing("error_handler", duration)

    return state
