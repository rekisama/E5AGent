"""
EvoAgentX-Style Workflow Generator

This module implements the core workflow generation algorithm from EvoAgentX,
adapted for our AutoGen-based system. It reuses the proven task planning
and workflow building logic while integrating with our existing components.

Key Features:
- Task planning with retry logic (from EvoAgentX)
- Automatic workflow structure generation
- Agent assignment and configuration
- Workflow validation and optimization
- Integration with AutoGen agents
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from .evo_core_algorithms import (
    EvoWorkflowGraph, EvoWorkflowNode, EvoWorkflowEdge, 
    EvoTaskPlanner, TaskPlanningOutput
)

logger = logging.getLogger(__name__)


class EvoWorkflowGenerator:
    """
    EvoAgentX-style workflow generator adapted for AutoGen.
    
    This class implements the core workflow generation algorithm from EvoAgentX
    with the following enhancements:
    - Integration with AutoGen agent system
    - Enhanced retry logic and error handling
    - Performance tracking and optimization
    - Compatibility with existing learning memory system
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.task_planner = EvoTaskPlanner(llm_config)
        self.generation_history = []
        
        logger.info("✅ EvoWorkflowGenerator initialized")
    
    def _execute_with_retry(self, operation_name: str, operation, 
                          retries_left: int = 1, **kwargs):
        """
        Helper method to execute operations with retry logic (from EvoAgentX).
        
        Args:
            operation_name: Name of the operation for logging
            operation: Callable that performs the operation
            retries_left: Number of retry attempts remaining
            **kwargs: Additional arguments to pass to the operation
            
        Returns:
            Tuple of (operation_result, number_of_retries_used)
            
        Raises:
            ValueError: If operation fails after all retries are exhausted
        """
        cur_retries = 0
        
        while cur_retries <= retries_left:
            try:
                logger.info(f"{operation_name} (attempt {cur_retries + 1}/{retries_left + 1}) ...")
                result = operation(**kwargs)
                return result, cur_retries
            except Exception as e:
                if cur_retries == retries_left:
                    raise ValueError(f"Failed to {operation_name} after {cur_retries + 1} attempts.\nError: {e}")
                
                sleep_time = 2 ** cur_retries
                logger.error(f"Failed to {operation_name} in {cur_retries + 1} attempts. Retry after {sleep_time} seconds.\nError: {e}")
                time.sleep(sleep_time)
                cur_retries += 1
    
    async def generate_workflow(self, goal: str, existing_agents: Optional[List[Dict]] = None, 
                              retry: int = 2, **kwargs) -> EvoWorkflowGraph:
        """
        Generate a workflow from a natural language goal (adapted from EvoAgentX).
        
        Args:
            goal: Natural language description of the goal
            existing_agents: Optional list of existing agents to reuse
            retry: Number of retry attempts for each operation
            **kwargs: Additional arguments
            
        Returns:
            EvoWorkflowGraph with generated workflow
        """
        # Validate input
        if not goal or len(goal.strip()) < 10:
            raise ValueError("Goal must be at least 10 characters and descriptive")
        
        logger.info(f"🎯 Generating EvoAgentX-style workflow for: {goal}")
        
        plan_history, plan_suggestion = "", ""
        
        try:
            # Step 1: Generate the initial workflow plan
            cur_retries = 0
            plan, added_retries = await self._execute_with_retry_async(
                operation_name="Generating a workflow plan",
                operation=self.generate_plan,
                retries_left=retry,
                goal=goal,
                history=plan_history,
                suggestion=plan_suggestion
            )
            cur_retries += added_retries
            
            # Step 2: Build workflow from plan
            workflow, added_retries = self._execute_with_retry(
                operation_name="Building workflow from plan",
                operation=self.build_workflow_from_plan,
                retries_left=retry - cur_retries,
                goal=goal,
                plan=plan
            )
            cur_retries += added_retries
            
            # Step 3: Validate initial workflow structure
            logger.info("Validating initial workflow structure...")
            workflow._validate_workflow_structure()
            logger.info(f"Successfully generated workflow:\n{workflow.get_workflow_description()}")
            
            # Step 4: Generate/assign agents for the workflow
            logger.info("Generating agents for the workflow...")
            workflow, added_retries = await self._execute_with_retry_async(
                operation_name="Generating agents for the workflow",
                operation=self.generate_agents,
                retries_left=retry - cur_retries,
                goal=goal,
                workflow=workflow,
                existing_agents=existing_agents
            )
            
            # Step 5: Final validation
            logger.info("Validating workflow after agent generation...")
            workflow._validate_workflow_structure()
            
            # Validate that all nodes have agents
            for node in workflow.nodes:
                if not node.agents:
                    raise ValueError(f"Node {node.name} has no agents assigned after agent generation")
            
            # Record generation history
            self._record_generation(goal, workflow, True)
            
            logger.info(f"✅ EvoAgentX workflow generation completed successfully")
            return workflow
            
        except Exception as e:
            logger.error(f"❌ EvoAgentX workflow generation failed: {e}")
            self._record_generation(goal, None, False, str(e))
            raise e
    
    async def _execute_with_retry_async(self, operation_name: str, operation, 
                                      retries_left: int = 1, **kwargs):
        """Async version of execute_with_retry."""
        cur_retries = 0
        
        while cur_retries <= retries_left:
            try:
                logger.info(f"{operation_name} (attempt {cur_retries + 1}/{retries_left + 1}) ...")
                result = await operation(**kwargs)
                return result, cur_retries
            except Exception as e:
                if cur_retries == retries_left:
                    raise ValueError(f"Failed to {operation_name} after {cur_retries + 1} attempts.\nError: {e}")
                
                sleep_time = 2 ** cur_retries
                logger.error(f"Failed to {operation_name} in {cur_retries + 1} attempts. Retry after {sleep_time} seconds.\nError: {e}")
                time.sleep(sleep_time)
                cur_retries += 1
    
    async def generate_plan(self, goal: str, history: Optional[str] = None, 
                          suggestion: Optional[str] = None) -> TaskPlanningOutput:
        """
        Generate a task plan for the goal (adapted from EvoAgentX).
        
        Args:
            goal: The goal to plan for
            history: Previous planning history
            suggestion: Suggestions for improvement
            
        Returns:
            TaskPlanningOutput with planned tasks
        """
        history = "" if history is None else history
        suggestion = "" if suggestion is None else suggestion
        
        logger.info(f"📋 Planning tasks for goal: {goal}")
        
        plan = await self.task_planner.plan_tasks(goal, history, suggestion)
        
        logger.info(f"✅ Generated plan with {len(plan.sub_tasks)} sub-tasks")
        return plan
    
    def build_workflow_from_plan(self, goal: str, plan: TaskPlanningOutput) -> EvoWorkflowGraph:
        """
        Build workflow graph from task plan (adapted from EvoAgentX).
        
        Args:
            goal: The original goal
            plan: The task planning output
            
        Returns:
            EvoWorkflowGraph with nodes and edges
        """
        logger.info(f"🏗️ Building workflow from plan with {len(plan.sub_tasks)} tasks")
        
        nodes: List[EvoWorkflowNode] = plan.sub_tasks
        
        # Infer edges from sub-tasks' inputs and outputs (EvoAgentX logic)
        edges: List[EvoWorkflowEdge] = []
        
        for node in nodes:
            for another_node in nodes:
                if node.name == another_node.name:
                    continue
                
                # Get parameter names
                node_output_params = [param.name for param in node.outputs]
                another_node_input_params = [param.name for param in another_node.inputs]
                
                # Check if any output of current node matches input of another node
                if any(param in another_node_input_params for param in node_output_params):
                    edges.append(EvoWorkflowEdge(edge_tuple=(node.name, another_node.name)))
        
        workflow = EvoWorkflowGraph(goal=goal, nodes=nodes, edges=edges)
        
        logger.info(f"✅ Built workflow with {len(nodes)} nodes and {len(edges)} edges")
        return workflow
    
    async def generate_agents(self, goal: str, workflow: EvoWorkflowGraph, 
                            existing_agents: Optional[List[Dict]] = None) -> EvoWorkflowGraph:
        """
        Generate or assign agents for workflow nodes (adapted from EvoAgentX).
        
        Args:
            goal: The original goal
            workflow: The workflow graph
            existing_agents: Optional existing agents to reuse
            
        Returns:
            Updated workflow with agents assigned
        """
        logger.info(f"🤖 Generating agents for {len(workflow.nodes)} workflow nodes")
        
        workflow_desc = workflow.get_workflow_description()
        
        for subtask in workflow.nodes:
            try:
                logger.info(f"Creating agent for subtask: {subtask.name}")
                
                # Prepare subtask data
                subtask_fields = ["name", "description", "reason", "inputs", "outputs"]
                subtask_data = {
                    key: value for key, value in subtask.to_dict(ignore=["class_name"]).items() 
                    if key in subtask_fields
                }
                subtask_desc = json.dumps(subtask_data, indent=4)
                
                # Generate agent configuration
                agent_config = await self._generate_agent_config(
                    goal, workflow_desc, subtask_desc
                )
                
                # Set agents for the subtask
                subtask.set_agents(agents=[agent_config])
                
                logger.debug(f"✅ Generated agent for {subtask.name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate agent for {subtask.name}: {e}")
                # Create a fallback agent
                fallback_agent = self._create_fallback_agent(subtask)
                subtask.set_agents(agents=[fallback_agent])
        
        logger.info(f"✅ Agent generation completed for all nodes")
        return workflow
    
    async def _generate_agent_config(self, goal: str, workflow_desc: str, 
                                   subtask_desc: str) -> Dict[str, Any]:
        """Generate agent configuration for a subtask."""
        
        system_message = """You are an expert agent designer. Create agent configurations for specific workflow tasks.

Your task is to design an agent that can effectively handle the given subtask within the workflow context."""
        
        agent_prompt = f"""
        Design an agent configuration for this workflow subtask:
        
        Overall Goal: {goal}
        Workflow Context: {workflow_desc}
        Subtask Details: {subtask_desc}
        
        Create an agent configuration that includes:
        1. A clear and specific system message for the agent
        2. Appropriate capabilities and tools
        3. Success criteria and validation methods
        
        The agent should be:
        - Specialized for the specific subtask
        - Aware of the overall workflow context
        - Capable of producing the required outputs
        - Able to work with inputs from previous tasks
        
        Respond with JSON:
        {{
            "name": "agent_name",
            "system_message": "detailed system message for the agent",
            "capabilities": ["capability1", "capability2"],
            "tools": ["tool1", "tool2"],
            "success_criteria": "how to measure success"
        }}
        """
        
        try:
            # Use task planner's LLM call method
            response = await self.task_planner._call_llm(agent_prompt, system_message)
            agent_data = json.loads(response)
            
            # Add LLM config
            agent_data["llm_config"] = self.llm_config
            
            return agent_data
            
        except Exception as e:
            logger.warning(f"Agent config generation failed: {e}")
            return self._create_fallback_agent_config()
    
    def _create_fallback_agent(self, subtask: EvoWorkflowNode) -> Dict[str, Any]:
        """Create a fallback agent configuration."""
        return {
            "name": f"agent_{subtask.name}",
            "system_message": f"You are an agent responsible for: {subtask.description}. {subtask.reason}",
            "capabilities": ["general_processing"],
            "tools": ["basic_tools"],
            "llm_config": self.llm_config,
            "success_criteria": "Complete the assigned task effectively"
        }
    
    def _create_fallback_agent_config(self) -> Dict[str, Any]:
        """Create a basic fallback agent configuration."""
        return {
            "name": "fallback_agent",
            "system_message": "You are a helpful assistant that can handle various tasks.",
            "capabilities": ["general_assistance"],
            "tools": ["basic_tools"],
            "llm_config": self.llm_config,
            "success_criteria": "Complete assigned tasks to the best of your ability"
        }
    
    def _record_generation(self, goal: str, workflow: Optional[EvoWorkflowGraph], 
                          success: bool, error: str = None):
        """Record workflow generation for analytics."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "goal": goal,
            "success": success,
            "workflow_id": workflow.id if workflow else None,
            "node_count": len(workflow.nodes) if workflow else 0,
            "edge_count": len(workflow.edges) if workflow else 0,
            "error": error
        }
        
        self.generation_history.append(record)
        logger.debug(f"📊 Recorded generation: {success}")
    
    def get_generation_analytics(self) -> Dict[str, Any]:
        """Get analytics about workflow generation performance."""
        if not self.generation_history:
            return {"total_generations": 0, "success_rate": 0}
        
        total = len(self.generation_history)
        successful = sum(1 for r in self.generation_history if r["success"])
        
        return {
            "total_generations": total,
            "successful_generations": successful,
            "success_rate": successful / total,
            "average_node_count": sum(r["node_count"] for r in self.generation_history if r["success"]) / max(successful, 1),
            "recent_generations": self.generation_history[-5:]
        }


# Factory function
def get_evo_workflow_generator(llm_config: Dict[str, Any]) -> EvoWorkflowGenerator:
    """Get an EvoWorkflowGenerator instance."""
    return EvoWorkflowGenerator(llm_config)
