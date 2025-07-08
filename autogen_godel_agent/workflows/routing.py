"""
Conditional Routing Logic for LangGraph Workflow.

This module contains the routing functions that determine the next step
in the workflow based on the current state and results of previous steps.
"""

import logging
from typing import Literal

from .state import WorkflowState, TaskStatus, WorkflowStep

logger = logging.getLogger(__name__)

# Type definitions for routing
RouteAfterAnalysis = Literal["search_functions", "create_function", "error_handler"]
RouteAfterSearch = Literal["execute_task", "compose_functions", "create_function", "error_handler"]
RouteAfterComposition = Literal["execute_task", "create_function", "error_handler"]
RetryRoute = Literal["analyze_task", "error_handler", "end"]


def route_after_analysis(state: WorkflowState) -> RouteAfterAnalysis:
    """
    Route after task analysis step.
    
    Determines the next step based on the analysis results:
    - If analysis failed -> error_handler
    - If analysis suggests existing functions might work -> search_functions
    - If analysis suggests new function needed -> create_function
    
    Args:
        state: Current workflow state
        
    Returns:
        Next step to execute
    """
    logger.info("🔀 Routing after task analysis")
    
    # Check for errors first
    if state.status == TaskStatus.FAILED or not state.analysis_result:
        logger.warning("Analysis failed, routing to error handler")
        return "error_handler"
    
    analysis = state.analysis_result
    
    # Check if analysis indicates existing functions might be useful
    function_found = analysis.get('function_found', False)
    matched_functions = analysis.get('matched_functions', [])
    needs_new_function = analysis.get('needs_new_function', True)
    
    # Decision logic based on analysis results
    if function_found or matched_functions:
        logger.info("📚 Analysis suggests existing functions might work, searching functions")
        return "search_functions"
    elif not needs_new_function:
        logger.info("📚 Analysis suggests searching for functions first")
        return "search_functions"
    else:
        logger.info("🛠️ Analysis suggests creating new function directly")
        return "create_function"


def route_after_search(state: WorkflowState) -> RouteAfterSearch:
    """
    Route after function search step.
    
    Determines the next step based on search results:
    - If search failed -> error_handler
    - If good matches found -> execute_task
    - If partial matches found -> compose_functions
    - If no matches found -> create_function
    
    Args:
        state: Current workflow state
        
    Returns:
        Next step to execute
    """
    logger.info("🔀 Routing after function search")
    
    # Check for errors first
    if state.status == TaskStatus.FAILED or not state.search_result:
        logger.warning("Function search failed, routing to error handler")
        return "error_handler"
    
    search_result = state.search_result
    
    # No functions found at all
    if not search_result.has_matches:
        logger.info("🛠️ No functions found, creating new function")
        return "create_function"
    
    # Check quality of matches
    best_match = search_result.best_match
    if best_match:
        best_score = search_result.relevance_scores.get(best_match.get('name', ''), 0.0)
        
        # High confidence match - execute directly
        if best_score >= 0.8:
            logger.info("⚡ High confidence match found, executing task")
            return "execute_task"
        
        # Medium confidence - try composition first
        elif best_score >= 0.5:
            logger.info("🧩 Medium confidence matches found, trying composition")
            return "compose_functions"
    
    # Low confidence matches - try composition
    if len(search_result.found_functions) >= 2:
        logger.info("🧩 Multiple functions found, trying composition")
        return "compose_functions"
    
    # Single low-confidence match - create new function
    logger.info("🛠️ Low confidence matches, creating new function")
    return "create_function"


def route_after_composition(state: WorkflowState) -> RouteAfterComposition:
    """
    Route after function composition step.
    
    Determines the next step based on composition results:
    - If composition failed -> error_handler
    - If composition successful -> execute_task
    - If composition not possible -> create_function
    
    Args:
        state: Current workflow state
        
    Returns:
        Next step to execute
    """
    logger.info("🔀 Routing after function composition")
    
    # Check for errors first
    if state.status == TaskStatus.FAILED:
        logger.warning("Function composition failed, routing to error handler")
        return "error_handler"
    
    composition_result = state.composition_result
    
    if not composition_result:
        logger.warning("No composition result available, routing to error handler")
        return "error_handler"
    
    # Composition was successful
    if composition_result.is_successful:
        logger.info("⚡ Function composition successful, executing task")
        return "execute_task"
    
    # Composition failed or not possible - create new function
    logger.info("🛠️ Function composition not successful, creating new function")
    return "create_function"


def should_retry(state: WorkflowState) -> RetryRoute:
    """
    Determine if workflow should be retried after error.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next step for retry logic
    """
    logger.info("🔀 Determining retry strategy")
    
    # Check if we can retry
    if not state.can_retry():
        logger.warning("❌ Maximum retries exceeded, ending workflow")
        return "end"
    
    # Check if we should retry
    error_count = len(state.error_history)
    
    if error_count == 0:
        logger.info("✅ No errors, ending workflow normally")
        return "end"
    
    # Analyze the types of errors
    recent_errors = state.error_history[-3:] if len(state.error_history) >= 3 else state.error_history
    error_steps = [error.get('step', '') for error in recent_errors]
    
    # If errors are in different steps, retry from beginning
    if len(set(error_steps)) > 1:
        logger.info("🔄 Multiple error types, retrying from analysis")
        return "analyze_task"
    
    # If all errors are in the same step, might be a persistent issue
    if len(recent_errors) >= 2 and all(step == error_steps[0] for step in error_steps):
        logger.warning("⚠️ Persistent errors in same step, routing to error handler")
        return "error_handler"
    
    # Default retry from beginning
    logger.info("🔄 Retrying workflow from analysis")
    return "analyze_task"


def route_to_end(state: WorkflowState) -> Literal["end"]:
    """
    Final routing function that always routes to end.
    
    This is used for terminal states where the workflow should end.
    
    Args:
        state: Current workflow state
        
    Returns:
        Always returns "end"
    """
    logger.info("🏁 Routing to workflow end")
    
    # Set final status if not already set
    if state.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        if state.execution_result and state.execution_result.is_successful:
            state.status = TaskStatus.COMPLETED
        else:
            state.status = TaskStatus.FAILED
    
    state.should_terminate = True
    return "end"


def get_next_step_from_state(state: WorkflowState) -> str:
    """
    Get the next step based on current workflow state.
    
    This is a utility function that can be used to determine the next step
    without going through the routing functions.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next step name as string
    """
    if state.next_step:
        return state.next_step.value
    
    if state.should_terminate:
        return "end"
    
    # Default routing based on current step
    current_step = state.current_step
    
    if current_step == WorkflowStep.START:
        return "analyze_task"
    elif current_step == WorkflowStep.ANALYZE_TASK:
        return route_after_analysis(state)
    elif current_step == WorkflowStep.SEARCH_FUNCTIONS:
        return route_after_search(state)
    elif current_step == WorkflowStep.COMPOSE_FUNCTIONS:
        return route_after_composition(state)
    elif current_step == WorkflowStep.CREATE_FUNCTION:
        return "execute_task"
    elif current_step == WorkflowStep.EXECUTE_TASK:
        return "end"
    elif current_step == WorkflowStep.ERROR_HANDLER:
        return should_retry(state)
    else:
        return "end"


# Routing configuration for LangGraph
ROUTING_CONFIG = {
    "analyze_task": route_after_analysis,
    "search_functions": route_after_search,
    "compose_functions": route_after_composition,
    "error_handler": should_retry,
    "execute_task": route_to_end,
    "create_function": lambda state: "execute_task"  # Always go to execute after creation
}
