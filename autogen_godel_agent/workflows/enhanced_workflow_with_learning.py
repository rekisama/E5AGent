"""
Enhanced LangGraph Workflow with Learning Memory Integration

This module extends the existing LangGraph workflow to include learning memory
capabilities, enabling the system to learn from past experiences and make
better decisions over time.

Key Features:
- Automatic pattern recognition during workflow execution
- Real-time recommendation integration in decision nodes
- Historical success-based routing and optimization
- Continuous learning from workflow outcomes
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langgraph import StateGraph, END
from langgraph.graph import Graph

from ..tools.learning_memory_integration import LearningMemoryIntegration, LearningMemoryMiddleware
from ..tools.function_tools import FunctionTools
from ..tools.function_registry import FunctionRegistry
from ..config import Config

logger = logging.getLogger(__name__)

@dataclass
class EnhancedWorkflowState:
    """Enhanced workflow state with learning memory integration."""
    
    # Original workflow state fields
    task: str = ""
    current_step: str = "start"
    functions_used: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None
    
    # Learning memory enhancement fields
    learning_insights: Dict[str, Any] = field(default_factory=dict)
    recommendations: Dict[str, Any] = field(default_factory=dict)
    pattern_analysis: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Workflow tracking
    workflow_id: str = ""
    start_time: Optional[datetime] = None
    execution_path: List[str] = field(default_factory=list)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)
    
    # Learning feedback
    user_satisfaction: Optional[float] = None
    learning_feedback: Dict[str, Any] = field(default_factory=dict)


class EnhancedWorkflowWithLearning:
    """Enhanced workflow that integrates learning memory capabilities."""
    
    def __init__(self, function_tools: FunctionTools, function_registry: FunctionRegistry):
        self.function_tools = function_tools
        self.function_registry = function_registry
        self.learning_integration = LearningMemoryIntegration(function_tools, function_registry)
        self.learning_middleware = LearningMemoryMiddleware(self.learning_integration)
        
        # Build the enhanced workflow graph
        self.workflow = self._build_enhanced_workflow()
        
        logger.info("🚀 Enhanced Workflow with Learning Memory initialized")
    
    def _build_enhanced_workflow(self) -> StateGraph:
        """Build the enhanced workflow graph with learning integration."""
        
        # Create the state graph
        workflow = StateGraph(EnhancedWorkflowState)
        
        # Add nodes with learning enhancement
        workflow.add_node("analyze_with_learning", self._analyze_task_with_learning)
        workflow.add_node("get_recommendations", self._get_learning_recommendations)
        workflow.add_node("make_intelligent_decision", self._make_learning_informed_decision)
        workflow.add_node("execute_with_tracking", self._execute_with_learning_tracking)
        workflow.add_node("evaluate_and_learn", self._evaluate_and_learn)
        workflow.add_node("handle_failure_with_learning", self._handle_failure_with_learning)
        
        # Define the workflow edges with conditional routing
        workflow.set_entry_point("analyze_with_learning")
        
        workflow.add_edge("analyze_with_learning", "get_recommendations")
        workflow.add_edge("get_recommendations", "make_intelligent_decision")
        
        # Conditional routing based on learning insights
        workflow.add_conditional_edges(
            "make_intelligent_decision",
            self._route_based_on_learning,
            {
                "execute": "execute_with_tracking",
                "need_more_analysis": "analyze_with_learning",
                "insufficient_confidence": "handle_failure_with_learning"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_with_tracking",
            self._route_after_execution,
            {
                "success": "evaluate_and_learn",
                "failure": "handle_failure_with_learning",
                "retry": "get_recommendations"
            }
        )
        
        workflow.add_edge("evaluate_and_learn", END)
        workflow.add_edge("handle_failure_with_learning", END)
        
        return workflow.compile()
    
    async def _analyze_task_with_learning(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Analyze task with learning memory insights."""
        try:
            logger.info(f"🧠 Analyzing task with learning: {state.task}")
            
            # Initialize workflow tracking
            if not state.workflow_id:
                state.workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                state.start_time = datetime.now(timezone.utc)
            
            state.execution_path.append("analyze_with_learning")
            
            # Get learning enhancement
            enhancement = await self.learning_integration.async_enhance_task_analysis(state.task)
            state.learning_insights = enhancement
            
            # Extract pattern analysis
            if enhancement.get("enhanced"):
                state.pattern_analysis = enhancement.get("complexity_analysis", {})
                state.confidence_scores["analysis"] = enhancement.get("complexity_analysis", {}).get("confidence", 0.0)
                
                logger.info(f"📊 Task complexity: {state.pattern_analysis.get('level', 'unknown')}")
                logger.info(f"🎯 Analysis confidence: {state.confidence_scores['analysis']:.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in task analysis with learning: {e}")
            state.error_message = f"Analysis error: {str(e)}"
            return state
    
    async def _get_learning_recommendations(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Get recommendations from learning memory system."""
        try:
            logger.info("🎯 Getting learning-based recommendations")
            
            state.execution_path.append("get_recommendations")
            
            # Get function recommendations
            recommendations = self.learning_integration.get_function_recommendations(
                state.task,
                context={"workflow_id": state.workflow_id, "current_step": state.current_step}
            )
            
            state.recommendations = recommendations
            state.confidence_scores["recommendations"] = recommendations.get("confidence", 0.0)
            
            # Log recommendations
            primary_recs = recommendations.get("primary_recommendations", [])
            if primary_recs:
                rec_names = [rec["name"] for rec in primary_recs]
                logger.info(f"💡 Recommended functions: {rec_names}")
                logger.info(f"🎯 Recommendation confidence: {state.confidence_scores['recommendations']:.2f}")
            else:
                logger.info("⚠️  No specific recommendations found")
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            state.error_message = f"Recommendation error: {str(e)}"
            return state
    
    async def _make_learning_informed_decision(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Make decisions informed by learning insights."""
        try:
            logger.info("🤔 Making learning-informed decision")
            
            state.execution_path.append("make_intelligent_decision")
            
            # Analyze decision factors
            decision_factors = {
                "has_recommendations": len(state.recommendations.get("primary_recommendations", [])) > 0,
                "confidence_threshold": state.confidence_scores.get("recommendations", 0.0) >= Config.MIN_RECOMMENDATION_CONFIDENCE,
                "complexity_manageable": state.pattern_analysis.get("level") in ["simple", "moderate"],
                "historical_success": state.learning_insights.get("learning_insights", {}).get("has_historical_data", False)
            }
            
            # Calculate overall decision confidence
            decision_confidence = sum([
                0.3 if decision_factors["has_recommendations"] else 0.0,
                0.3 if decision_factors["confidence_threshold"] else 0.0,
                0.2 if decision_factors["complexity_manageable"] else 0.0,
                0.2 if decision_factors["historical_success"] else 0.0
            ])
            
            state.confidence_scores["decision"] = decision_confidence
            
            # Record decision point
            decision_point = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "factors": decision_factors,
                "confidence": decision_confidence,
                "recommendations_count": len(state.recommendations.get("primary_recommendations", [])),
                "complexity": state.pattern_analysis.get("level", "unknown")
            }
            state.decision_points.append(decision_point)
            
            logger.info(f"📊 Decision confidence: {decision_confidence:.2f}")
            logger.info(f"🎯 Decision factors: {decision_factors}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            state.error_message = f"Decision error: {str(e)}"
            return state
    
    def _route_based_on_learning(self, state: EnhancedWorkflowState) -> str:
        """Route workflow based on learning insights."""
        try:
            decision_confidence = state.confidence_scores.get("decision", 0.0)
            
            if decision_confidence >= 0.7:
                logger.info("✅ High confidence - proceeding to execution")
                return "execute"
            elif decision_confidence >= 0.4:
                logger.info("⚠️  Medium confidence - need more analysis")
                return "need_more_analysis"
            else:
                logger.info("❌ Low confidence - insufficient for execution")
                return "insufficient_confidence"
                
        except Exception as e:
            logger.error(f"Error in routing decision: {e}")
            return "insufficient_confidence"
    
    async def _execute_with_learning_tracking(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Execute functions with learning tracking."""
        try:
            logger.info("🚀 Executing with learning tracking")
            
            state.execution_path.append("execute_with_tracking")
            execution_start = datetime.now(timezone.utc)
            
            # Get recommended functions to execute
            recommended_functions = [
                rec["name"] for rec in state.recommendations.get("primary_recommendations", [])
            ]
            
            if not recommended_functions:
                # Fallback to traditional function discovery
                logger.info("🔍 No recommendations - using traditional function discovery")
                # This would integrate with existing function matching logic
                recommended_functions = []  # Placeholder
            
            # Execute functions and track results
            execution_results = []
            functions_executed = []
            
            for func_name in recommended_functions:
                try:
                    # Execute function (this would integrate with existing execution logic)
                    result = await self._execute_function_with_tracking(func_name, state.task)
                    execution_results.append(result)
                    functions_executed.append(func_name)
                    
                    if result.get("success", False):
                        logger.info(f"✅ Function {func_name} executed successfully")
                    else:
                        logger.warning(f"⚠️  Function {func_name} execution failed")
                        
                except Exception as func_error:
                    logger.error(f"❌ Error executing {func_name}: {func_error}")
                    execution_results.append({"success": False, "error": str(func_error)})
            
            # Update state with execution results
            state.functions_used.extend(functions_executed)
            state.results["execution_results"] = execution_results
            state.success = any(result.get("success", False) for result in execution_results)
            
            # Calculate execution time
            execution_time = (datetime.now(timezone.utc) - execution_start).total_seconds()
            state.results["execution_time"] = execution_time
            
            # Record execution for learning
            await self.learning_integration.async_record_task_execution(
                task_description=state.task,
                functions_used=functions_executed,
                success=state.success,
                execution_time=execution_time,
                additional_context={"workflow_id": state.workflow_id}
            )
            
            logger.info(f"📊 Execution completed: success={state.success}, time={execution_time:.2f}s")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in execution with tracking: {e}")
            state.error_message = f"Execution error: {str(e)}"
            state.success = False
            return state
    
    async def _execute_function_with_tracking(self, function_name: str, task: str) -> Dict[str, Any]:
        """Execute a single function with tracking."""
        # This is a placeholder - would integrate with actual function execution
        # For now, return a mock result
        return {
            "success": True,
            "result": f"Mock execution of {function_name} for task: {task}",
            "execution_time": 0.1
        }
    
    def _route_after_execution(self, state: EnhancedWorkflowState) -> str:
        """Route after execution based on results."""
        if state.success:
            return "success"
        elif state.error_message and "timeout" in state.error_message.lower():
            return "retry"
        else:
            return "failure"
    
    async def _evaluate_and_learn(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Evaluate results and update learning memory."""
        try:
            logger.info("📚 Evaluating results and learning")
            
            state.execution_path.append("evaluate_and_learn")
            
            # Calculate overall workflow success metrics
            total_time = (datetime.now(timezone.utc) - state.start_time).total_seconds() if state.start_time else 0
            
            # Prepare learning feedback
            learning_feedback = {
                "workflow_success": state.success,
                "total_execution_time": total_time,
                "functions_used_count": len(state.functions_used),
                "decision_points_count": len(state.decision_points),
                "recommendation_accuracy": self._calculate_recommendation_accuracy(state),
                "complexity_prediction_accuracy": self._calculate_complexity_accuracy(state)
            }
            
            state.learning_feedback = learning_feedback
            
            # Log learning insights
            logger.info(f"📊 Workflow completed successfully: {state.success}")
            logger.info(f"⏱️  Total execution time: {total_time:.2f}s")
            logger.info(f"🔧 Functions used: {len(state.functions_used)}")
            logger.info(f"🎯 Recommendation accuracy: {learning_feedback['recommendation_accuracy']:.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in evaluation and learning: {e}")
            state.error_message = f"Evaluation error: {str(e)}"
            return state
    
    async def _handle_failure_with_learning(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Handle failures with learning insights."""
        try:
            logger.info("🔧 Handling failure with learning insights")
            
            state.execution_path.append("handle_failure_with_learning")
            
            # Analyze failure patterns
            failure_analysis = {
                "confidence_too_low": state.confidence_scores.get("decision", 0.0) < 0.4,
                "no_recommendations": len(state.recommendations.get("primary_recommendations", [])) == 0,
                "execution_failed": not state.success and state.functions_used,
                "analysis_failed": state.error_message and "analysis" in state.error_message.lower()
            }
            
            # Record failure for learning
            if state.functions_used:
                await self.learning_integration.async_record_task_execution(
                    task_description=state.task,
                    functions_used=state.functions_used,
                    success=False,
                    error_message=state.error_message,
                    additional_context={
                        "workflow_id": state.workflow_id,
                        "failure_analysis": failure_analysis
                    }
                )
            
            # Provide failure insights
            state.results["failure_analysis"] = failure_analysis
            state.results["learning_suggestions"] = self._generate_learning_suggestions(failure_analysis)
            
            logger.info(f"🔍 Failure analysis: {failure_analysis}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in failure handling: {e}")
            return state
    
    def _calculate_recommendation_accuracy(self, state: EnhancedWorkflowState) -> float:
        """Calculate how accurate the recommendations were."""
        if not state.recommendations.get("primary_recommendations") or not state.functions_used:
            return 0.0
        
        recommended_names = {rec["name"] for rec in state.recommendations["primary_recommendations"]}
        used_names = set(state.functions_used)
        
        if not recommended_names:
            return 0.0
        
        # Calculate intersection over union
        intersection = len(recommended_names & used_names)
        union = len(recommended_names | used_names)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_complexity_accuracy(self, state: EnhancedWorkflowState) -> float:
        """Calculate how accurate the complexity prediction was."""
        predicted_level = state.pattern_analysis.get("level", "unknown")
        actual_success = state.success
        execution_time = state.results.get("execution_time", 0)
        
        # Simple heuristic for complexity accuracy
        if predicted_level == "simple" and actual_success and execution_time < 5:
            return 1.0
        elif predicted_level == "moderate" and execution_time < 15:
            return 0.8
        elif predicted_level == "complex" and (not actual_success or execution_time > 10):
            return 0.9
        else:
            return 0.5
    
    def _generate_learning_suggestions(self, failure_analysis: Dict[str, bool]) -> List[str]:
        """Generate suggestions based on failure analysis."""
        suggestions = []
        
        if failure_analysis.get("confidence_too_low"):
            suggestions.append("Consider gathering more historical data for similar tasks")
        
        if failure_analysis.get("no_recommendations"):
            suggestions.append("Task may be novel - consider creating new functions")
        
        if failure_analysis.get("execution_failed"):
            suggestions.append("Review function implementations and error handling")
        
        if failure_analysis.get("analysis_failed"):
            suggestions.append("Improve task analysis and pattern recognition")
        
        return suggestions
    
    async def execute_workflow(self, task: str, workflow_id: str = None) -> EnhancedWorkflowState:
        """Execute the enhanced workflow with learning."""
        try:
            # Initialize state
            initial_state = EnhancedWorkflowState(
                task=task,
                workflow_id=workflow_id or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            
            logger.info(f"🚀 Starting enhanced workflow for task: {task}")
            
            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            logger.info(f"✅ Workflow completed: {final_state.workflow_id}")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Error executing enhanced workflow: {e}")
            error_state = EnhancedWorkflowState(
                task=task,
                error_message=str(e),
                success=False
            )
            return error_state
