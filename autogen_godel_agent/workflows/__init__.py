"""
LangGraph Workflow System for AutoGen Self-Expanding Agent System.

This module provides a state-machine based workflow orchestration system
that replaces the simple coordination logic with a more sophisticated
LangGraph-based approach.

Key Components:
- WorkflowState: Core state management for task processing
- WorkflowOrchestrator: Main workflow execution engine
- Node functions: Individual workflow steps (analyze, search, create, etc.)
- Routing logic: Conditional workflow navigation
"""

import sys
import os

# Add parent directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .state import WorkflowState, TaskContext, FunctionSearchResult, CreationResult
    from .orchestrator import WorkflowOrchestrator, get_workflow_orchestrator
except ImportError:
    # Fallback to absolute imports
    from workflows.state import WorkflowState, TaskContext, FunctionSearchResult, CreationResult
    from workflows.orchestrator import WorkflowOrchestrator, get_workflow_orchestrator
try:
    from .nodes import (
        analyze_task_node,
        search_functions_node,
        compose_functions_node,
        create_function_node,
        execute_task_node,
        error_handler_node
    )
    from .routing import (
        route_after_analysis,
        route_after_search,
        route_after_composition,
        should_retry
    )
except ImportError:
    # Fallback to absolute imports
    from workflows.nodes import (
        analyze_task_node,
        search_functions_node,
        compose_functions_node,
        create_function_node,
        execute_task_node,
        error_handler_node
    )
    from workflows.routing import (
        route_after_analysis,
        route_after_search,
        route_after_composition,
        should_retry
    )

__all__ = [
    # Core components
    'WorkflowState',
    'TaskContext', 
    'FunctionSearchResult',
    'CreationResult',
    'WorkflowOrchestrator',
    'get_workflow_orchestrator',
    
    # Node functions
    'analyze_task_node',
    'search_functions_node', 
    'compose_functions_node',
    'create_function_node',
    'execute_task_node',
    'error_handler_node',
    
    # Routing functions
    'route_after_analysis',
    'route_after_search',
    'route_after_composition',
    'should_retry'
]
