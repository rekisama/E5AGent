"""
Workflow Executor

This module provides execution capabilities for dynamically generated workflows.
It can execute workflows with different patterns (sequential, parallel, conditional, etc.)
and manage agent interactions.

Inspired by EvoAgentX's execution model.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

import autogen
from .dynamic_workflow_generator import WorkflowGraph, WorkflowNode, WorkflowType, AgentRole

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of workflow execution."""
    workflow_id: str
    status: ExecutionStatus
    output: Any = None
    execution_time: float = 0.0
    node_results: Dict[str, Any] = None
    errors: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.node_results is None:
            self.node_results = {}
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}


class WorkflowExecutor:
    """
    Executes dynamically generated workflows using AutoGen agents.
    
    Supports different execution patterns:
    - Sequential: Execute nodes one by one
    - Parallel: Execute multiple nodes simultaneously
    - Conditional: Execute based on intermediate results
    - Iterative: Execute with feedback loops
    - Hierarchical: Execute with manager-worker pattern
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.active_agents = {}
        self.execution_history = []
        
        logger.info("✅ WorkflowExecutor initialized")
    
    async def execute_workflow(self, workflow: WorkflowGraph, 
                             initial_input: str = None) -> ExecutionResult:
        """
        Execute a workflow graph.
        
        Args:
            workflow: The workflow graph to execute
            initial_input: Initial input for the workflow
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Executing workflow: {workflow.name}")
            
            # Initialize execution result
            result = ExecutionResult(
                workflow_id=workflow.id,
                status=ExecutionStatus.RUNNING
            )
            
            # Create agents for the workflow
            await self._create_workflow_agents(workflow)
            
            # Execute based on workflow type
            if workflow.workflow_type == WorkflowType.SEQUENTIAL:
                output = await self._execute_sequential(workflow, initial_input or workflow.goal)
            elif workflow.workflow_type == WorkflowType.PARALLEL:
                output = await self._execute_parallel(workflow, initial_input or workflow.goal)
            elif workflow.workflow_type == WorkflowType.CONDITIONAL:
                output = await self._execute_conditional(workflow, initial_input or workflow.goal)
            elif workflow.workflow_type == WorkflowType.ITERATIVE:
                output = await self._execute_iterative(workflow, initial_input or workflow.goal)
            elif workflow.workflow_type == WorkflowType.HIERARCHICAL:
                output = await self._execute_hierarchical(workflow, initial_input or workflow.goal)
            else:
                # Default to sequential
                output = await self._execute_sequential(workflow, initial_input or workflow.goal)
            
            # Update result
            result.status = ExecutionStatus.COMPLETED
            result.output = output
            result.execution_time = time.time() - start_time
            
            logger.info(f"✅ Workflow completed in {result.execution_time:.2f}s")
            
            # Record execution history
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            
            result = ExecutionResult(
                workflow_id=workflow.id,
                status=ExecutionStatus.FAILED,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )
            
            return result
    
    async def _create_workflow_agents(self, workflow: WorkflowGraph):
        """Create AutoGen agents for each node in the workflow."""
        
        for node in workflow.nodes:
            try:
                # Create agent based on node configuration
                agent = autogen.AssistantAgent(
                    name=f"{node.name}_{node.id}",
                    system_message=node.agent_config.get("system_message", "You are a helpful assistant."),
                    llm_config=self.llm_config
                )
                
                self.active_agents[node.id] = agent
                logger.debug(f"Created agent for node: {node.id}")
                
            except Exception as e:
                logger.error(f"Failed to create agent for node {node.id}: {e}")
                raise e
    
    async def _execute_sequential(self, workflow: WorkflowGraph, initial_input: str) -> str:
        """Execute workflow sequentially."""
        
        current_input = initial_input
        
        for node in workflow.nodes:
            try:
                logger.info(f"📝 Executing node: {node.name}")
                
                # Get the agent for this node
                agent = self.active_agents[node.id]
                
                # Create user proxy for interaction
                user_proxy = autogen.UserProxyAgent(
                    name="user_proxy",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=1,
                    code_execution_config=False
                )
                
                # Execute the node
                user_proxy.initiate_chat(
                    agent,
                    message=f"Task: {current_input}\n\nPlease process this task according to your role as {node.agent_role.value}."
                )
                
                # Get the response (last message from agent)
                chat_history = user_proxy.chat_messages[agent]
                if chat_history:
                    current_input = chat_history[-1]["content"]
                
                logger.debug(f"✅ Node {node.name} completed")
                
            except Exception as e:
                logger.error(f"❌ Node {node.name} failed: {e}")
                raise e
        
        return current_input
    
    async def _execute_parallel(self, workflow: WorkflowGraph, initial_input: str) -> str:
        """Execute workflow with parallel processing."""
        
        # Find nodes that can run in parallel (no dependencies)
        parallel_nodes = [node for node in workflow.nodes if not node.dependencies]
        sequential_nodes = [node for node in workflow.nodes if node.dependencies]
        
        # Execute parallel nodes
        parallel_results = []
        
        for node in parallel_nodes:
            try:
                logger.info(f"📝 Executing parallel node: {node.name}")
                
                agent = self.active_agents[node.id]
                user_proxy = autogen.UserProxyAgent(
                    name=f"user_proxy_{node.id}",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=1,
                    code_execution_config=False
                )
                
                user_proxy.initiate_chat(
                    agent,
                    message=f"Task: {initial_input}\n\nProcess this from your perspective as {node.agent_role.value}."
                )
                
                chat_history = user_proxy.chat_messages[agent]
                if chat_history:
                    parallel_results.append(chat_history[-1]["content"])
                
            except Exception as e:
                logger.error(f"❌ Parallel node {node.name} failed: {e}")
                parallel_results.append(f"Error in {node.name}: {str(e)}")
        
        # Combine parallel results
        combined_input = f"Combined results from parallel processing:\n" + "\n".join(parallel_results)
        
        # Execute remaining sequential nodes
        current_input = combined_input
        for node in sequential_nodes:
            try:
                agent = self.active_agents[node.id]
                user_proxy = autogen.UserProxyAgent(
                    name=f"user_proxy_{node.id}",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=1,
                    code_execution_config=False
                )
                
                user_proxy.initiate_chat(
                    agent,
                    message=f"Task: {current_input}\n\nProcess this as {node.agent_role.value}."
                )
                
                chat_history = user_proxy.chat_messages[agent]
                if chat_history:
                    current_input = chat_history[-1]["content"]
                    
            except Exception as e:
                logger.error(f"❌ Sequential node {node.name} failed: {e}")
                raise e
        
        return current_input
    
    async def _execute_conditional(self, workflow: WorkflowGraph, initial_input: str) -> str:
        """Execute workflow with conditional branching."""
        # For now, implement as sequential with decision points
        return await self._execute_sequential(workflow, initial_input)
    
    async def _execute_iterative(self, workflow: WorkflowGraph, initial_input: str) -> str:
        """Execute workflow with iterative refinement."""
        
        current_input = initial_input
        max_iterations = 3  # Default max iterations
        
        for iteration in range(max_iterations):
            logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Execute all nodes in sequence
            iteration_result = await self._execute_sequential(workflow, current_input)
            
            # Check if we should continue iterating
            # For now, just do fixed iterations
            current_input = f"Iteration {iteration + 1} result: {iteration_result}\n\nPlease refine this further."
        
        return current_input
    
    async def _execute_hierarchical(self, workflow: WorkflowGraph, initial_input: str) -> str:
        """Execute workflow with hierarchical structure."""
        
        # Find manager and worker nodes
        manager_nodes = [node for node in workflow.nodes if node.agent_role == AgentRole.COORDINATOR]
        worker_nodes = [node for node in workflow.nodes if node.agent_role != AgentRole.COORDINATOR]
        
        if not manager_nodes:
            # No manager, execute sequentially
            return await self._execute_sequential(workflow, initial_input)
        
        manager_node = manager_nodes[0]
        manager_agent = self.active_agents[manager_node.id]
        
        # Manager coordinates the work
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy_manager",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False
        )
        
        # Manager analyzes and delegates
        user_proxy.initiate_chat(
            manager_agent,
            message=f"Task: {initial_input}\n\nAs a manager, analyze this task and coordinate the work."
        )
        
        chat_history = user_proxy.chat_messages[manager_agent]
        manager_output = chat_history[-1]["content"] if chat_history else initial_input
        
        # Workers execute their parts
        worker_results = []
        for worker_node in worker_nodes:
            try:
                worker_agent = self.active_agents[worker_node.id]
                worker_proxy = autogen.UserProxyAgent(
                    name=f"user_proxy_{worker_node.id}",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=1,
                    code_execution_config=False
                )
                
                worker_proxy.initiate_chat(
                    worker_agent,
                    message=f"Manager's instructions: {manager_output}\n\nExecute your part as {worker_node.agent_role.value}."
                )
                
                worker_history = worker_proxy.chat_messages[worker_agent]
                if worker_history:
                    worker_results.append(worker_history[-1]["content"])
                    
            except Exception as e:
                logger.error(f"❌ Worker {worker_node.name} failed: {e}")
                worker_results.append(f"Error in {worker_node.name}: {str(e)}")
        
        # Combine results
        final_result = f"Manager coordination: {manager_output}\n\nWorker results:\n" + "\n".join(worker_results)
        
        return final_result


# Factory function
def get_workflow_executor(llm_config: Dict[str, Any]) -> WorkflowExecutor:
    """Get a WorkflowExecutor instance."""
    return WorkflowExecutor(llm_config)
