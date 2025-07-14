"""
EvoAgentX Core Algorithms Integration

This module integrates the core algorithms from EvoAgentX project while adapting them
for our AutoGen-based system. We reuse the proven workflow generation logic and
optimization strategies from EvoAgentX.

Original EvoAgentX License: MIT License
Copyright (c) 2025 EvoAgentX Team
https://github.com/EvoAgentX/EvoAgentX

Adapted for AutoGen Self-Expanding Agent System
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TaskParameter:
    """Parameter definition for tasks (adapted from EvoAgentX)."""
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class EvoWorkflowNode:
    """Enhanced workflow node based on EvoAgentX design."""
    name: str
    description: str
    reason: str
    inputs: List[TaskParameter] = field(default_factory=list)
    outputs: List[TaskParameter] = field(default_factory=list)
    agents: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self, ignore: List[str] = None) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        ignore = ignore or []
        result = {}
        
        for key, value in self.__dict__.items():
            if key not in ignore:
                if isinstance(value, list) and value and hasattr(value[0], 'to_dict'):
                    result[key] = [item.to_dict() for item in value]
                else:
                    result[key] = value
        
        return result
    
    def set_agents(self, agents: List[Dict[str, Any]]):
        """Set agents for this node."""
        self.agents = agents


@dataclass
class EvoWorkflowEdge:
    """Workflow edge definition (adapted from EvoAgentX)."""
    edge_tuple: Tuple[str, str]
    
    @property
    def source(self) -> str:
        return self.edge_tuple[0]
    
    @property
    def target(self) -> str:
        return self.edge_tuple[1]


@dataclass
class TaskPlanningOutput:
    """Output from task planning (adapted from EvoAgentX)."""
    sub_tasks: List[EvoWorkflowNode]
    reasoning: str = ""
    confidence: float = 0.8


class EvoWorkflowGraph:
    """
    Enhanced workflow graph based on EvoAgentX design.
    
    This class maintains the core structure and validation logic from EvoAgentX
    while adapting it for our AutoGen integration.
    """
    
    def __init__(self, goal: str, nodes: List[EvoWorkflowNode] = None, 
                 edges: List[EvoWorkflowEdge] = None):
        self.goal = goal
        self.nodes = nodes or []
        self.edges = edges or []
        self.id = f"evo_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.created_at = datetime.now()
        
    def add_node(self, node: EvoWorkflowNode):
        """Add a node to the workflow."""
        self.nodes.append(node)
    
    def add_edge(self, edge: EvoWorkflowEdge):
        """Add an edge to the workflow."""
        self.edges.append(edge)
    
    def get_workflow_description(self) -> str:
        """Get a textual description of the workflow."""
        description = f"Workflow Goal: {self.goal}\n\n"
        description += f"Nodes ({len(self.nodes)}):\n"
        
        for i, node in enumerate(self.nodes, 1):
            description += f"{i}. {node.name}: {node.description}\n"
            if node.inputs:
                description += f"   Inputs: {[p.name for p in node.inputs]}\n"
            if node.outputs:
                description += f"   Outputs: {[p.name for p in node.outputs]}\n"
        
        if self.edges:
            description += f"\nEdges ({len(self.edges)}):\n"
            for edge in self.edges:
                description += f"  {edge.source} → {edge.target}\n"
        
        return description
    
    def _validate_workflow_structure(self):
        """Validate the workflow structure (adapted from EvoAgentX)."""
        if not self.nodes:
            raise ValueError("Workflow must have at least one node")
        
        # Check for duplicate node names
        node_names = [node.name for node in self.nodes]
        if len(node_names) != len(set(node_names)):
            raise ValueError("Workflow nodes must have unique names")
        
        # Validate edges reference existing nodes
        for edge in self.edges:
            source_exists = any(node.name == edge.source for node in self.nodes)
            target_exists = any(node.name == edge.target for node in self.nodes)
            
            if not source_exists:
                raise ValueError(f"Edge source '{edge.source}' does not exist in nodes")
            if not target_exists:
                raise ValueError(f"Edge target '{edge.target}' does not exist in nodes")
        
        # Check for cycles (basic check)
        self._check_for_cycles()
    
    def _check_for_cycles(self):
        """Basic cycle detection."""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_name: str) -> bool:
            visited.add(node_name)
            rec_stack.add(node_name)
            
            # Get all nodes that this node points to
            targets = [edge.target for edge in self.edges if edge.source == node_name]
            
            for target in targets:
                if target not in visited:
                    if has_cycle(target):
                        return True
                elif target in rec_stack:
                    return True
            
            rec_stack.remove(node_name)
            return False
        
        for node in self.nodes:
            if node.name not in visited:
                if has_cycle(node.name):
                    raise ValueError("Workflow contains cycles")


class EvoTaskPlanner:
    """
    Enhanced task planner based on EvoAgentX design.
    
    This class adapts EvoAgentX's task planning logic for our LLM configuration.
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        
    async def plan_tasks(self, goal: str, history: str = "", 
                        suggestion: str = "") -> TaskPlanningOutput:
        """
        Plan tasks for achieving the goal (adapted from EvoAgentX).
        
        Args:
            goal: The high-level goal to achieve
            history: Previous planning history
            suggestion: Suggestions for improvement
            
        Returns:
            TaskPlanningOutput with sub-tasks
        """
        system_message = """You are an expert task planner. Break down complex goals into manageable sub-tasks.

Your task is to analyze the given goal and create a detailed task breakdown that:
1. Identifies all necessary sub-tasks to achieve the goal
2. Defines clear inputs and outputs for each sub-task
3. Establishes proper dependencies between tasks
4. Provides reasoning for the task structure

Always respond with valid JSON in the exact format specified."""
        
        planning_prompt = f"""
        Break down this goal into manageable sub-tasks:
        
        Goal: "{goal}"
        History: {history or 'None'}
        Suggestions: {suggestion or 'None'}
        
        Create a comprehensive task breakdown with:
        1. Clear sub-task definitions
        2. Input/output specifications
        3. Logical task dependencies
        4. Reasoning for the structure
        
        Respond with JSON:
        {{
            "sub_tasks": [
                {{
                    "name": "task_name",
                    "description": "detailed description of what this task does",
                    "reason": "why this task is necessary",
                    "inputs": [
                        {{"name": "input_name", "type": "string", "description": "input description", "required": true}}
                    ],
                    "outputs": [
                        {{"name": "output_name", "type": "string", "description": "output description", "required": true}}
                    ]
                }}
            ],
            "reasoning": "overall reasoning for the task structure",
            "confidence": 0.85
        }}
        """
        
        try:
            # Use LLM for task planning
            response = await self._call_llm(planning_prompt, system_message)
            
            # Parse JSON response
            planning_data = json.loads(response)
            
            # Convert to EvoWorkflowNode objects
            sub_tasks = []
            for task_data in planning_data.get("sub_tasks", []):
                # Convert inputs and outputs
                inputs = [
                    TaskParameter(
                        name=inp["name"],
                        type=inp.get("type", "string"),
                        description=inp.get("description", ""),
                        required=inp.get("required", True)
                    )
                    for inp in task_data.get("inputs", [])
                ]
                
                outputs = [
                    TaskParameter(
                        name=out["name"],
                        type=out.get("type", "string"),
                        description=out.get("description", ""),
                        required=out.get("required", True)
                    )
                    for out in task_data.get("outputs", [])
                ]
                
                node = EvoWorkflowNode(
                    name=task_data["name"],
                    description=task_data["description"],
                    reason=task_data.get("reason", ""),
                    inputs=inputs,
                    outputs=outputs
                )
                
                sub_tasks.append(node)
            
            return TaskPlanningOutput(
                sub_tasks=sub_tasks,
                reasoning=planning_data.get("reasoning", ""),
                confidence=planning_data.get("confidence", 0.8)
            )
            
        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            # Return a simple fallback plan
            return self._create_fallback_plan(goal)
    
    async def _call_llm(self, prompt: str, system_message: str = None) -> str:
        """Call LLM for task planning."""
        try:
            import openai
            
            # Get config from llm_config
            api_key = self.llm_config.get("config_list", [{}])[0].get("api_key") if self.llm_config.get("config_list") else None
            base_url = self.llm_config.get("config_list", [{}])[0].get("base_url") if self.llm_config.get("config_list") else None
            model = "deepseek-chat"
            if self.llm_config.get("config_list"):
                model = self.llm_config["config_list"][0].get("model", "deepseek-chat")
            
            if not api_key or not base_url:
                raise Exception("LLM configuration incomplete")
            
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise e
    
    def _create_fallback_plan(self, goal: str) -> TaskPlanningOutput:
        """Create a simple fallback plan when LLM planning fails."""
        
        # Create basic analyze -> execute -> validate workflow
        analyze_task = EvoWorkflowNode(
            name="analyze_requirements",
            description="Analyze the goal and determine requirements",
            reason="Understanding requirements is essential for success",
            inputs=[TaskParameter("goal", "string", "The goal to analyze")],
            outputs=[TaskParameter("requirements", "string", "Analyzed requirements")]
        )
        
        execute_task = EvoWorkflowNode(
            name="execute_solution",
            description="Execute the solution based on requirements",
            reason="Implementation is needed to achieve the goal",
            inputs=[TaskParameter("requirements", "string", "Requirements to implement")],
            outputs=[TaskParameter("solution", "string", "Implemented solution")]
        )
        
        validate_task = EvoWorkflowNode(
            name="validate_result",
            description="Validate the solution meets the goal",
            reason="Validation ensures quality and correctness",
            inputs=[TaskParameter("solution", "string", "Solution to validate")],
            outputs=[TaskParameter("validated_result", "string", "Validated final result")]
        )
        
        return TaskPlanningOutput(
            sub_tasks=[analyze_task, execute_task, validate_task],
            reasoning="Fallback plan with basic analyze-execute-validate workflow",
            confidence=0.6
        )
