"""
LangGraph Workflow Orchestrator.

This module provides the main workflow orchestration system that replaces
the simple coordination logic in the original SelfExpandingAgentSystem
with a sophisticated LangGraph-based state machine.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, Any, Optional, List
import time

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # Fallback for when LangGraph is not available
    logging.warning("LangGraph not available, using fallback implementation")
    StateGraph = None
    END = "end"
    MemorySaver = None

try:
    from .state import WorkflowState, TaskContext, TaskStatus, WorkflowStep
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
    # Try absolute imports
    try:
        from workflows.state import WorkflowState, TaskContext, TaskStatus, WorkflowStep
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
    except ImportError as e:
        logging.error(f"Failed to import workflow components: {e}")
        # Set fallback values - use Any for type compatibility
        from typing import Any
        WorkflowState = Any
        TaskContext = Any
        TaskStatus = Any
        WorkflowStep = Any

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Main workflow orchestrator using LangGraph.
    
    This class replaces the simple coordination logic in SelfExpandingAgentSystem
    with a sophisticated state machine that provides better error handling,
    retry logic, and conditional routing.
    """
    
    def __init__(self, enable_checkpoints: bool = True):
        """
        Initialize the workflow orchestrator.
        
        Args:
            enable_checkpoints: Whether to enable workflow checkpointing
        """
        self.enable_checkpoints = enable_checkpoints
        self.workflow_graph = None
        self.checkpointer = None
        self._build_workflow()
        
        # Statistics tracking
        self.total_workflows_executed = 0
        self.successful_workflows = 0
        self.failed_workflows = 0
        self.workflow_history: List[Dict[str, Any]] = []
    
    def _build_workflow(self):
        """Build the LangGraph workflow."""
        if StateGraph is None:
            logger.warning("LangGraph not available, workflow will use fallback mode")
            return
        
        try:
            # Create the state graph
            workflow = StateGraph(WorkflowState)
            
            # Add nodes
            workflow.add_node("analyze_task", analyze_task_node)
            workflow.add_node("search_functions", search_functions_node)
            workflow.add_node("compose_functions", compose_functions_node)
            workflow.add_node("create_function", create_function_node)
            workflow.add_node("execute_task", execute_task_node)
            workflow.add_node("error_handler", error_handler_node)
            
            # Set entry point
            workflow.set_entry_point("analyze_task")
            
            # Add conditional edges (routing logic)
            workflow.add_conditional_edges(
                "analyze_task",
                route_after_analysis,
                {
                    "search_functions": "search_functions",
                    "create_function": "create_function",
                    "error_handler": "error_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "search_functions",
                route_after_search,
                {
                    "execute_task": "execute_task",
                    "compose_functions": "compose_functions",
                    "create_function": "create_function",
                    "error_handler": "error_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "compose_functions",
                route_after_composition,
                {
                    "execute_task": "execute_task",
                    "create_function": "create_function",
                    "error_handler": "error_handler"
                }
            )
            
            # Simple edges for nodes that always go to the same next step
            workflow.add_edge("create_function", "execute_task")
            workflow.add_edge("execute_task", END)
            
            # Error handler routing
            workflow.add_conditional_edges(
                "error_handler",
                should_retry,
                {
                    "analyze_task": "analyze_task",
                    "error_handler": "error_handler",
                    "end": END
                }
            )
            
            # Set up checkpointing if enabled
            if self.enable_checkpoints and MemorySaver:
                self.checkpointer = MemorySaver()
                self.workflow_graph = workflow.compile(checkpointer=self.checkpointer)
            else:
                self.workflow_graph = workflow.compile()
            
            logger.info("✅ LangGraph workflow built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build LangGraph workflow: {e}")
            self.workflow_graph = None
    
    def process_task(self, task_description: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a task using the LangGraph workflow.
        
        Args:
            task_description: Description of the task to process
            session_id: Optional session ID for context
            
        Returns:
            Dictionary containing the workflow results
        """
        logger.info(f"🎯 Processing task with LangGraph workflow: {task_description}")
        
        # Generate unique task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Create initial workflow state
        task_context = TaskContext(
            task_id=task_id,
            description=task_description,
            user_input=task_description,
            session_id=session_id
        )
        
        initial_state = WorkflowState(task_context=task_context)
        
        start_time = time.time()
        
        try:
            if self.workflow_graph:
                # Use LangGraph workflow
                result = self._execute_langgraph_workflow(initial_state)
            else:
                # Fallback to sequential execution
                result = self._execute_fallback_workflow(initial_state)
            
            # Update statistics
            self.total_workflows_executed += 1
            if result['status'] == 'completed':
                self.successful_workflows += 1
            else:
                self.failed_workflows += 1
            
            # Record workflow history
            workflow_record = {
                'task_id': task_id,
                'task_description': task_description,
                'start_time': start_time,
                'end_time': time.time(),
                'duration': time.time() - start_time,
                'status': result['status'],
                'steps_executed': result.get('steps_executed', []),
                'functions_used': result.get('functions_used', []),
                'error_count': result.get('error_count', 0)
            }
            self.workflow_history.append(workflow_record)
            
            logger.info(f"✅ Task processing completed: {result['status']}")
            return result
            
        except Exception as e:
            self.failed_workflows += 1
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': error_msg,
                'result': None,
                'workflow_type': 'langgraph' if self.workflow_graph else 'fallback'
            }
    
    def _execute_langgraph_workflow(self, initial_state: Any) -> Dict[str, Any]:
        """Execute the workflow using LangGraph."""
        logger.info("🚀 Executing LangGraph workflow")
        
        # Configure execution
        config = {"configurable": {"thread_id": initial_state.task_context.task_id}}
        
        # Execute the workflow
        final_state = None
        steps_executed = []
        
        for step_output in self.workflow_graph.stream(initial_state, config):
            for node_name, node_state in step_output.items():
                steps_executed.append(node_name)
                final_state = node_state
                logger.debug(f"Executed step: {node_name}")
        
        if final_state is None:
            raise RuntimeError("Workflow execution produced no final state")
        
        # Convert final state to result dictionary
        return self._convert_state_to_result(final_state, steps_executed, 'langgraph')
    
    def _execute_fallback_workflow(self, initial_state: Any) -> Dict[str, Any]:
        """Execute the workflow using fallback sequential logic."""
        logger.info("🔄 Executing fallback workflow")
        
        current_state = initial_state
        steps_executed = []
        max_steps = 20  # Prevent infinite loops
        step_count = 0
        
        while not current_state.should_terminate and step_count < max_steps:
            step_count += 1
            
            # Determine next step
            if current_state.current_step == WorkflowStep.START:
                current_state = analyze_task_node(current_state)
                steps_executed.append("analyze_task")
            elif current_state.current_step == WorkflowStep.ANALYZE_TASK:
                next_step = route_after_analysis(current_state)
                if next_step == "search_functions":
                    current_state = search_functions_node(current_state)
                    steps_executed.append("search_functions")
                elif next_step == "create_function":
                    current_state = create_function_node(current_state)
                    steps_executed.append("create_function")
                else:
                    current_state = error_handler_node(current_state)
                    steps_executed.append("error_handler")
            elif current_state.current_step == WorkflowStep.SEARCH_FUNCTIONS:
                next_step = route_after_search(current_state)
                if next_step == "execute_task":
                    current_state = execute_task_node(current_state)
                    steps_executed.append("execute_task")
                    break
                elif next_step == "compose_functions":
                    current_state = compose_functions_node(current_state)
                    steps_executed.append("compose_functions")
                elif next_step == "create_function":
                    current_state = create_function_node(current_state)
                    steps_executed.append("create_function")
                else:
                    current_state = error_handler_node(current_state)
                    steps_executed.append("error_handler")
            elif current_state.current_step == WorkflowStep.COMPOSE_FUNCTIONS:
                next_step = route_after_composition(current_state)
                if next_step == "execute_task":
                    current_state = execute_task_node(current_state)
                    steps_executed.append("execute_task")
                    break
                elif next_step == "create_function":
                    current_state = create_function_node(current_state)
                    steps_executed.append("create_function")
                else:
                    current_state = error_handler_node(current_state)
                    steps_executed.append("error_handler")
            elif current_state.current_step == WorkflowStep.CREATE_FUNCTION:
                current_state = execute_task_node(current_state)
                steps_executed.append("execute_task")
                break
            elif current_state.current_step == WorkflowStep.ERROR_HANDLER:
                retry_action = should_retry(current_state)
                if retry_action == "analyze_task":
                    current_state.current_step = WorkflowStep.START
                elif retry_action == "end":
                    break
                else:
                    current_state = error_handler_node(current_state)
                    steps_executed.append("error_handler")
            else:
                break
        
        return self._convert_state_to_result(current_state, steps_executed, 'fallback')
    
    def _convert_state_to_result(self, final_state: Any, steps_executed: List[str], workflow_type: str) -> Dict[str, Any]:
        """Convert workflow state to result dictionary."""
        # Determine functions used
        functions_used = []
        if final_state.execution_result:
            functions_used = final_state.execution_result.functions_used
        
        # Determine result data
        result_data = None
        if final_state.execution_result:
            result_data = final_state.execution_result.result_data
        
        return {
            'task_id': final_state.task_context.task_id,
            'status': 'completed' if final_state.status == TaskStatus.COMPLETED else 'failed',
            'result': result_data,
            'functions_used': functions_used,
            'steps_executed': steps_executed,
            'error_count': len(final_state.error_history),
            'retry_count': final_state.retry_count,
            'total_time': sum(final_state.step_timings.values()),
            'tokens_used': final_state.total_tokens_used,
            'workflow_type': workflow_type,
            'summary': final_state.get_summary()
        }
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow execution statistics."""
        success_rate = (self.successful_workflows / max(self.total_workflows_executed, 1)) * 100
        
        return {
            'total_workflows': self.total_workflows_executed,
            'successful_workflows': self.successful_workflows,
            'failed_workflows': self.failed_workflows,
            'success_rate': round(success_rate, 2),
            'average_duration': sum(w['duration'] for w in self.workflow_history) / max(len(self.workflow_history), 1),
            'workflow_type': 'langgraph' if self.workflow_graph else 'fallback'
        }


# Global orchestrator instance
_orchestrator_instance = None

def get_workflow_orchestrator(enable_checkpoints: bool = True) -> WorkflowOrchestrator:
    """Get or create the global workflow orchestrator instance."""
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        _orchestrator_instance = WorkflowOrchestrator(enable_checkpoints)
    
    return _orchestrator_instance
