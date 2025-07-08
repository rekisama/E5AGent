"""
Workflow State Management for LangGraph Integration.

This module defines the core state structures used throughout the
LangGraph workflow system, providing type-safe state management
and context preservation across workflow steps.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

# Optional: Pydantic support for enhanced validation
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    Field = None

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task processing status enumeration."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    SEARCHING = "searching"
    COMPOSING = "composing"
    CREATING = "creating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class WorkflowStep(Enum):
    """Workflow step enumeration for routing."""
    START = "start"
    ANALYZE_TASK = "analyze_task"
    SEARCH_FUNCTIONS = "search_functions"
    COMPOSE_FUNCTIONS = "compose_functions"
    CREATE_FUNCTION = "create_function"
    EXECUTE_TASK = "execute_task"
    ERROR_HANDLER = "error_handler"
    END = "end"


class IntentType(Enum):
    """Task intent classification."""
    FUNCTION_CREATION = "function_creation"
    FUNCTION_SEARCH = "function_search"
    FUNCTION_COMPOSITION = "function_composition"
    TASK_EXECUTION = "task_execution"
    UNKNOWN = "unknown"


class KeywordCategory(Enum):
    """Keyword category classification."""
    MATHEMATICAL = "mathematical"
    STRING_PROCESSING = "string_processing"
    DATA_MANIPULATION = "data_manipulation"
    FILE_OPERATIONS = "file_operations"
    NETWORK_OPERATIONS = "network_operations"
    UTILITY = "utility"
    UNKNOWN = "unknown"


class TerminateReason(Enum):
    """Workflow termination reasons."""
    SUCCESS = "success"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    USER_REQUESTED = "user_requested"
    CRITICAL_ERROR = "critical_error"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"


@dataclass
class TokenUsageDetail:
    """Detailed token usage tracking for cost optimization."""
    step_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class TaskContext:
    """Context information for the current task."""
    task_id: str
    description: str
    user_input: str
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced fields for intent and keyword analysis
    intent_type: IntentType = IntentType.UNKNOWN
    keyword_category: KeywordCategory = KeywordCategory.UNKNOWN
    confidence_score: float = 0.0
    priority: int = 1
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'description': self.description,
            'user_input': self.user_input,
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'intent_type': self.intent_type.value,
            'keyword_category': self.keyword_category.value,
            'confidence_score': self.confidence_score,
            'priority': self.priority,
            'tags': self.tags
        }


@dataclass
class FunctionSearchResult:
    """Result of function search operation."""
    found_functions: List[Dict[str, Any]] = field(default_factory=list)
    search_query: str = ""
    total_functions: int = 0
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    search_strategy: str = "semantic"  # semantic, keyword, hybrid
    
    @property
    def has_matches(self) -> bool:
        """Check if any functions were found."""
        return len(self.found_functions) > 0
    
    @property
    def best_match(self) -> Optional[Dict[str, Any]]:
        """Get the best matching function."""
        if not self.found_functions:
            return None
        return max(self.found_functions, 
                  key=lambda f: self.relevance_scores.get(f.get('name', ''), 0.0))


@dataclass
class CompositionResult:
    """Result of function composition attempt."""
    success: bool = False
    composite_function: Optional[Dict[str, Any]] = None
    component_functions: List[str] = field(default_factory=list)
    composition_strategy: str = ""
    error_message: str = ""
    
    @property
    def is_successful(self) -> bool:
        """Check if composition was successful."""
        return self.success and self.composite_function is not None


@dataclass
class CreationResult:
    """Result of function creation operation."""
    success: bool = False
    function_name: str = ""
    function_code: str = ""
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    retry_count: int = 0
    
    @property
    def is_successful(self) -> bool:
        """Check if creation was successful."""
        return self.success and bool(self.function_name) and bool(self.function_code)


@dataclass
class ExecutionResult:
    """Result of task execution."""
    success: bool = False
    result_data: Any = None
    execution_time: float = 0.0
    functions_used: List[str] = field(default_factory=list)
    error_message: str = ""
    
    @property
    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.success and self.error_message == ""


@dataclass
class WorkflowState:
    """
    Central state object for the LangGraph workflow.
    
    This class maintains all the context and results throughout
    the workflow execution, enabling stateful processing and
    conditional routing.
    """
    
    # Core task information
    task_context: TaskContext
    
    # Current workflow state
    current_step: WorkflowStep = WorkflowStep.START
    status: TaskStatus = TaskStatus.PENDING
    
    # Step results
    analysis_result: Dict[str, Any] = field(default_factory=dict)
    search_result: Optional[FunctionSearchResult] = None
    composition_result: Optional[CompositionResult] = None
    creation_result: Optional[CreationResult] = None
    execution_result: Optional[ExecutionResult] = None
    
    # Error handling
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    # Workflow control
    next_step: Optional[WorkflowStep] = None
    should_terminate: bool = False
    terminate_reason: Optional[TerminateReason] = None

    # Enhanced: Sub-task support for parallel processing
    sub_tasks: List['WorkflowState'] = field(default_factory=list)
    parent_task_id: Optional[str] = None

    # Performance tracking - Enhanced with detailed token usage
    step_timings: Dict[str, float] = field(default_factory=dict)
    total_tokens_used: int = 0
    token_usage_details: List[TokenUsageDetail] = field(default_factory=list)
    
    def add_error(self, step: str, error: str, exception: Optional[Exception] = None):
        """Add an error to the error history."""
        error_entry = {
            'step': step,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'retry_count': self.retry_count
        }
        if exception:
            error_entry['exception_type'] = type(exception).__name__
            error_entry['exception_details'] = str(exception)
        
        self.error_history.append(error_entry)
        # Enhanced logging with task_id
        logger.error(f"[{self.task_context.task_id}] Workflow error in {step}: {error}")
    
    def can_retry(self) -> bool:
        """Check if the workflow can be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment the retry counter with enhanced logging."""
        self.retry_count += 1
        self.status = TaskStatus.RETRYING
        logger.info(f"[{self.task_context.task_id}] Retrying workflow, attempt {self.retry_count}/{self.max_retries}")
    
    def set_step_timing(self, step: str, duration: float):
        """Record timing for a workflow step with enhanced logging."""
        self.step_timings[step] = duration
        # Enhanced logging with task_id and step timing
        logger.info(f"[{self.task_context.task_id}] ⏱ Step Timing: {step} = {duration:.2f}s")

    def add_token_usage(self, step_name: str, input_tokens: int = 0, output_tokens: int = 0, cost_estimate: float = 0.0):
        """Add detailed token usage for a specific step."""
        token_detail = TokenUsageDetail(
            step_name=step_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_estimate=cost_estimate
        )
        self.token_usage_details.append(token_detail)
        self.total_tokens_used += token_detail.total_tokens
        logger.info(f"[{self.task_context.task_id}] 🪙 Token Usage: {step_name} = {token_detail.total_tokens} tokens (${cost_estimate:.4f})")

    def get_token_usage_by_step(self) -> Dict[str, TokenUsageDetail]:
        """Get token usage breakdown by step."""
        return {detail.step_name: detail for detail in self.token_usage_details}

    def get_total_cost_estimate(self) -> float:
        """Get total estimated cost for all token usage."""
        return sum(detail.cost_estimate for detail in self.token_usage_details)

    def add_sub_task(self, sub_task: 'WorkflowState'):
        """Add a sub-task for parallel processing."""
        sub_task.parent_task_id = self.task_context.task_id
        self.sub_tasks.append(sub_task)
        logger.info(f"[{self.task_context.task_id}] ➕ Added sub-task: {sub_task.task_context.task_id}")

    def terminate_workflow(self, reason: TerminateReason, message: str = ""):
        """Terminate the workflow with a specific reason."""
        self.should_terminate = True
        self.terminate_reason = reason
        self.status = TaskStatus.COMPLETED if reason == TerminateReason.SUCCESS else TaskStatus.FAILED
        logger.info(f"[{self.task_context.task_id}] 🛑 Workflow terminated: {reason.value} - {message}")

    def set_intent_and_category(self, intent: IntentType, category: KeywordCategory, confidence: float = 0.0):
        """Set task intent and keyword category with confidence score."""
        self.task_context.intent_type = intent
        self.task_context.keyword_category = category
        self.task_context.confidence_score = confidence
        logger.info(f"[{self.task_context.task_id}] 🎯 Intent: {intent.value}, Category: {category.value} (confidence: {confidence:.2f})")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get an enhanced summary of the workflow state."""
        return {
            'task_id': self.task_context.task_id,
            'status': self.status.value,
            'current_step': self.current_step.value,
            'has_errors': len(self.error_history) > 0,
            'retry_count': self.retry_count,
            'functions_found': len(self.search_result.found_functions) if self.search_result else 0,
            'composition_attempted': self.composition_result is not None,
            'function_created': self.creation_result is not None and self.creation_result.is_successful,
            'execution_successful': self.execution_result is not None and self.execution_result.is_successful,
            'total_time': sum(self.step_timings.values()),
            'tokens_used': self.total_tokens_used,
            # Enhanced fields
            'intent_type': self.task_context.intent_type.value,
            'keyword_category': self.task_context.keyword_category.value,
            'confidence_score': self.task_context.confidence_score,
            'terminate_reason': self.terminate_reason.value if self.terminate_reason else None,
            'sub_tasks_count': len(self.sub_tasks),
            'total_cost_estimate': self.get_total_cost_estimate(),
            'token_usage_by_step': {detail.step_name: detail.total_tokens for detail in self.token_usage_details},
            'execution_time_by_step': self.step_timings,
            'priority': self.task_context.priority,
            'tags': self.task_context.tags
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire state to a dictionary for serialization."""
        return {
            'task_context': self.task_context.to_dict(),
            'current_step': self.current_step.value,
            'status': self.status.value,
            'analysis_result': self.analysis_result,
            'search_result': {
                'found_functions': self.search_result.found_functions,
                'total_functions': self.search_result.total_functions,
                'has_matches': self.search_result.has_matches
            } if self.search_result else None,
            'composition_result': {
                'success': self.composition_result.success,
                'component_functions': self.composition_result.component_functions,
                'error_message': self.composition_result.error_message
            } if self.composition_result else None,
            'creation_result': {
                'success': self.creation_result.success,
                'function_name': self.creation_result.function_name,
                'error_message': self.creation_result.error_message
            } if self.creation_result else None,
            'execution_result': {
                'success': self.execution_result.success,
                'functions_used': self.execution_result.functions_used,
                'error_message': self.execution_result.error_message,
                'execution_time': getattr(self.execution_result, 'execution_time', 0.0)  # Enhanced: include execution time
            } if self.execution_result else None,
            'error_count': len(self.error_history),
            'retry_count': self.retry_count,
            # Enhanced fields
            'terminate_reason': self.terminate_reason.value if self.terminate_reason else None,
            'sub_tasks': [sub_task.get_summary() for sub_task in self.sub_tasks],
            'parent_task_id': self.parent_task_id,
            'token_usage_details': [
                {
                    'step_name': detail.step_name,
                    'input_tokens': detail.input_tokens,
                    'output_tokens': detail.output_tokens,
                    'total_tokens': detail.total_tokens,
                    'cost_estimate': detail.cost_estimate,
                    'timestamp': detail.timestamp.isoformat()
                } for detail in self.token_usage_details
            ],
            'total_cost_estimate': self.get_total_cost_estimate(),
            'step_timings': self.step_timings,
            'summary': self.get_summary()
        }


# Optional: Pydantic version for enhanced validation and API support
if PYDANTIC_AVAILABLE:
    class WorkflowStatePydantic(BaseModel):
        """
        Pydantic version of WorkflowState for enhanced validation and API support.

        This provides stronger type validation, automatic serialization/deserialization,
        and better integration with FastAPI and other modern Python frameworks.
        """

        # Core task information
        task_context: Dict[str, Any]

        # Current workflow state
        current_step: str = WorkflowStep.START.value
        status: str = TaskStatus.PENDING.value

        # Step results
        analysis_result: Dict[str, Any] = Field(default_factory=dict)
        search_result: Optional[Dict[str, Any]] = None
        composition_result: Optional[Dict[str, Any]] = None
        creation_result: Optional[Dict[str, Any]] = None
        execution_result: Optional[Dict[str, Any]] = None

        # Error handling
        error_history: List[Dict[str, Any]] = Field(default_factory=list)
        retry_count: int = 0
        max_retries: int = 3

        # Workflow control
        next_step: Optional[str] = None
        should_terminate: bool = False
        terminate_reason: Optional[str] = None

        # Enhanced: Sub-task support
        sub_tasks: List[Dict[str, Any]] = Field(default_factory=list)
        parent_task_id: Optional[str] = None

        # Performance tracking
        step_timings: Dict[str, float] = Field(default_factory=dict)
        total_tokens_used: int = 0
        token_usage_details: List[Dict[str, Any]] = Field(default_factory=list)

        class Config:
            """Pydantic configuration."""
            validate_assignment = True
            extra = "allow"  # 允许额外字段以兼容 to_dict() 输出
            json_schema_extra = {
                "example": {
                    "task_context": {
                        "task_id": "task_123",
                        "description": "Create a hello world function",
                        "user_input": "Create a function that prints hello world",
                        "intent_type": "function_creation",
                        "keyword_category": "utility"
                    },
                    "current_step": "analyze_task",
                    "status": "analyzing",
                    "total_tokens_used": 150,
                    "step_timings": {"analyze_task": 2.5}
                }
            }

        @classmethod
        def from_dataclass(cls, workflow_state: WorkflowState) -> 'WorkflowStatePydantic':
            """Convert from dataclass WorkflowState to Pydantic version."""
            return cls(**workflow_state.to_dict())

        def to_dataclass(self) -> WorkflowState:
            """Convert to dataclass WorkflowState (requires proper reconstruction)."""
            # This would need more complex logic to properly reconstruct
            # the dataclass with all its nested objects
            raise NotImplementedError("Conversion to dataclass requires custom implementation")

else:
    # Fallback when Pydantic is not available
    WorkflowStatePydantic = None
