"""
EvoAgentX-Inspired Workflow Manager

This module provides a complete workflow management system inspired by EvoAgentX,
integrating dynamic workflow generation, execution, and evolution capabilities.

Key Features:
- Automatic workflow generation from natural language goals
- Intelligent agent creation and configuration
- Multi-pattern workflow execution (sequential, parallel, conditional, etc.)
- Performance tracking and optimization
- Self-evolving workflow capabilities
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .evo_workflow_generator import get_evo_workflow_generator
from .evo_workflow_executor import get_evo_workflow_executor, WorkflowExecutionResult, ExecutionStatus
from .evo_core_algorithms import EvoWorkflowGraph

logger = logging.getLogger(__name__)


class EvoWorkflowManager:
    """
    Complete workflow management system inspired by EvoAgentX.
    
    This class provides the main interface for:
    1. Generating workflows from natural language goals
    2. Executing workflows with different patterns
    3. Tracking performance and optimizing workflows
    4. Evolving workflows based on feedback
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.workflow_generator = get_evo_workflow_generator(llm_config)
        self.workflow_executor = get_evo_workflow_executor(llm_config)
        
        # Performance tracking
        self.workflow_history = []
        self.performance_metrics = {}
        
        logger.info("✅ EvoWorkflowManager initialized with EvoAgentX core algorithms")
    
    async def create_and_execute_workflow(self, goal: str, 
                                        context: Dict[str, Any] = None,
                                        initial_input: str = None) -> Dict[str, Any]:
        """
        Create and execute a workflow from a natural language goal.
        
        This is the main entry point that combines workflow generation and execution.
        
        Args:
            goal: Natural language description of what to achieve
            context: Additional context information
            initial_input: Initial input for workflow execution
            
        Returns:
            Dictionary containing workflow results and metadata
        """
        try:
            logger.info(f"🎯 Creating and executing workflow for: {goal}")
            
            # Step 1: Generate workflow
            workflow = await self.workflow_generator.generate_workflow(goal, context)
            
            # Step 2: Execute workflow
            execution_result = await self.workflow_executor.execute_workflow(
                workflow, initial_input
            )
            
            # Step 3: Record performance
            await self._record_workflow_performance(workflow, execution_result)
            
            # Step 4: Format response
            response = {
                "success": execution_result.status == ExecutionStatus.COMPLETED,
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "workflow_type": "evo_workflow",
                "goal": goal,
                "output": execution_result.output,
                "execution_time": execution_result.total_execution_time,
                "node_count": len(workflow.nodes),
                "errors": execution_result.errors,
                "metadata": {
                    "workflow_complexity": workflow.performance_metrics.get("complexity_score", 0),
                    "estimated_vs_actual_time": {
                        "estimated": workflow.performance_metrics.get("estimated_execution_time", 0),
                        "actual": execution_result.execution_time
                    },
                    "workflow_structure": self._get_workflow_summary(workflow)
                }
            }
            
            if response["success"]:
                logger.info(f"✅ Workflow completed successfully in {execution_result.execution_time:.2f}s")
            else:
                logger.warning(f"⚠️ Workflow completed with errors: {execution_result.errors}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Workflow creation and execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "goal": goal,
                "workflow_id": None,
                "output": None
            }
    
    async def generate_workflow_only(self, goal: str, 
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a workflow without executing it.
        
        Useful for workflow planning and visualization.
        
        Args:
            goal: Natural language description of what to achieve
            context: Additional context information
            
        Returns:
            Dictionary containing workflow structure and metadata
        """
        try:
            logger.info(f"📋 Generating workflow for: {goal}")
            
            workflow = await self.workflow_generator.generate_workflow(goal, context)
            
            return {
                "success": True,
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "workflow_type": workflow.workflow_type.value,
                "goal": goal,
                "structure": self._get_workflow_summary(workflow),
                "estimated_metrics": workflow.performance_metrics,
                "node_details": [
                    {
                        "id": node.id,
                        "name": node.name,
                        "role": node.agent_role.value,
                        "description": node.description,
                        "dependencies": node.dependencies
                    }
                    for node in workflow.nodes
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "goal": goal
            }
    
    async def execute_existing_workflow(self, workflow: EvoWorkflowGraph,
                                      initial_input: str = None) -> Dict[str, Any]:
        """
        Execute an existing workflow.
        
        Args:
            workflow: The workflow graph to execute
            initial_input: Initial input for execution
            
        Returns:
            Dictionary containing execution results
        """
        try:
            logger.info(f"▶️ Executing existing workflow: {workflow.name}")
            
            execution_result = await self.workflow_executor.execute_workflow(
                workflow, initial_input
            )
            
            await self._record_workflow_performance(workflow, execution_result)
            
            return {
                "success": execution_result.status == ExecutionStatus.COMPLETED,
                "workflow_id": workflow.id,
                "output": execution_result.final_output,
                "execution_time": execution_result.total_execution_time,
                "errors": execution_result.errors,
                "node_results": execution_result.node_results
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow.id
            }
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """
        Get performance analytics for all executed workflows.
        
        Returns:
            Dictionary containing performance metrics and insights
        """
        if not self.workflow_history:
            return {
                "total_workflows": 0,
                "success_rate": 0,
                "average_execution_time": 0,
                "insights": ["No workflows executed yet"]
            }
        
        total_workflows = len(self.workflow_history)
        successful_workflows = sum(1 for w in self.workflow_history if w["success"])
        success_rate = successful_workflows / total_workflows
        
        execution_times = [w["execution_time"] for w in self.workflow_history if w["success"]]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Workflow type distribution
        workflow_types = {}
        for w in self.workflow_history:
            wf_type = w.get("workflow_type", "unknown")
            workflow_types[wf_type] = workflow_types.get(wf_type, 0) + 1
        
        # Generate insights
        insights = []
        if success_rate >= 0.8:
            insights.append("High success rate indicates good workflow generation quality")
        elif success_rate < 0.5:
            insights.append("Low success rate suggests need for workflow optimization")
        
        if avg_execution_time > 120:  # 2 minutes
            insights.append("Long execution times may indicate overly complex workflows")
        
        most_common_type = max(workflow_types.items(), key=lambda x: x[1])[0] if workflow_types else "none"
        insights.append(f"Most common workflow type: {most_common_type}")
        
        return {
            "total_workflows": total_workflows,
            "successful_workflows": successful_workflows,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "workflow_type_distribution": workflow_types,
            "insights": insights,
            "recent_performance": self.workflow_history[-5:] if len(self.workflow_history) > 5 else self.workflow_history
        }
    
    async def optimize_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Optimize an existing workflow based on performance history.
        
        Args:
            workflow_id: ID of the workflow to optimize
            
        Returns:
            Dictionary containing optimization results
        """
        # Find workflow in history
        workflow_data = None
        for w in self.workflow_history:
            if w.get("workflow_id") == workflow_id:
                workflow_data = w
                break
        
        if not workflow_data:
            return {
                "success": False,
                "error": f"Workflow {workflow_id} not found in history"
            }
        
        # For now, provide optimization suggestions
        suggestions = []
        
        if workflow_data.get("execution_time", 0) > 60:
            suggestions.append("Consider breaking down complex tasks into smaller steps")
        
        if not workflow_data.get("success", False):
            suggestions.append("Review agent system messages for clarity and specificity")
            suggestions.append("Consider adding validation steps between major operations")
        
        node_count = workflow_data.get("node_count", 0)
        if node_count > 5:
            suggestions.append("Consider parallel execution for independent tasks")
        elif node_count < 3:
            suggestions.append("Consider adding validation or optimization steps")
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "current_performance": {
                "success": workflow_data.get("success", False),
                "execution_time": workflow_data.get("execution_time", 0),
                "node_count": node_count
            },
            "optimization_suggestions": suggestions,
            "next_steps": [
                "Implement suggested optimizations",
                "Test optimized workflow with similar goals",
                "Compare performance metrics"
            ]
        }
    
    def _get_workflow_summary(self, workflow: EvoWorkflowGraph) -> Dict[str, Any]:
        """Get a summary of the workflow structure."""
        return {
            "type": "evo_workflow",
            "node_count": len(workflow.nodes),
            "edge_count": len(workflow.edges),
            "node_names": [node.name for node in workflow.nodes],
            "complexity_score": len(workflow.nodes) * 0.1 + len(workflow.edges) * 0.05
        }
    
    async def _record_workflow_performance(self, workflow: EvoWorkflowGraph,
                                         execution_result: WorkflowExecutionResult):
        """Record workflow performance for analytics and optimization."""
        
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "workflow_type": workflow.workflow_type.value,
            "goal": workflow.goal,
            "success": execution_result.status == ExecutionStatus.COMPLETED,
            "execution_time": execution_result.total_execution_time,
            "node_count": len(workflow.nodes),
            "errors": execution_result.errors,
            "complexity_score": len(workflow.nodes) * 0.1 + len(workflow.edges) * 0.05
        }
        
        self.workflow_history.append(performance_record)
        
        # Update performance metrics
        workflow_type = "evo_workflow"
        if workflow_type not in self.performance_metrics:
            self.performance_metrics[workflow_type] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0,
                "average_time": 0
            }
        
        metrics = self.performance_metrics[workflow_type]
        metrics["total_executions"] += 1
        if execution_result.status == ExecutionStatus.COMPLETED:
            metrics["successful_executions"] += 1
        metrics["total_time"] += execution_result.total_execution_time
        metrics["average_time"] = metrics["total_time"] / metrics["total_executions"]
        
        logger.debug(f"📊 Recorded performance for workflow {workflow.id}")


# Factory function
def get_evo_workflow_manager(llm_config: Dict[str, Any]) -> EvoWorkflowManager:
    """Get an EvoWorkflowManager instance."""
    return EvoWorkflowManager(llm_config)
