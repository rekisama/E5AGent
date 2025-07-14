"""
Dynamic Workflow Generator

Inspired by EvoAgentX, this module provides automatic workflow generation
capabilities that can create, execute, and evolve multi-agent workflows
based on natural language goals.

Key Features:
- Automatic workflow generation from natural language goals
- Dynamic agent creation and configuration
- Self-evolving workflow optimization
- Performance evaluation and improvement
- Integration with existing AutoGen agents
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import autogen
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import Config
except ImportError:
    # Fallback for when running as script
    from autogen_godel_agent.config import Config

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of workflows that can be generated."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    HIERARCHICAL = "hierarchical"


class AgentRole(Enum):
    """Roles that agents can play in workflows."""
    ANALYZER = "analyzer"
    CREATOR = "creator"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    OPTIMIZER = "optimizer"


@dataclass
class WorkflowNode:
    """A node in the workflow graph."""
    id: str
    name: str
    agent_role: AgentRole
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    agent_config: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    success_rate: float = 1.0


@dataclass
class WorkflowGraph:
    """A complete workflow graph."""
    id: str
    name: str
    description: str
    goal: str
    workflow_type: WorkflowType
    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_node(self, node: WorkflowNode):
        """Add a node to the workflow."""
        self.nodes.append(node)
    
    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between two nodes."""
        self.edges.append((from_node, to_node))
    
    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


class DynamicWorkflowGenerator:
    """
    Dynamic workflow generator inspired by EvoAgentX.
    
    This class can automatically generate multi-agent workflows
    from natural language goals and evolve them over time.
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.workflow_templates = self._load_workflow_templates()
        self.agent_templates = self._load_agent_templates()
        self.performance_history = {}
        
        logger.info("✅ DynamicWorkflowGenerator initialized")
    
    def _load_workflow_templates(self) -> Dict[WorkflowType, Dict[str, Any]]:
        """Load workflow templates for different types."""
        return {
            WorkflowType.SEQUENTIAL: {
                "description": "Sequential execution of tasks",
                "pattern": "A → B → C → Result",
                "use_cases": ["data processing", "content generation", "analysis"]
            },
            WorkflowType.PARALLEL: {
                "description": "Parallel execution with aggregation",
                "pattern": "A → [B1, B2, B3] → C → Result",
                "use_cases": ["multi-perspective analysis", "parallel processing"]
            },
            WorkflowType.CONDITIONAL: {
                "description": "Conditional branching based on results",
                "pattern": "A → Decision → [B1 | B2] → Result",
                "use_cases": ["adaptive workflows", "error handling"]
            },
            WorkflowType.ITERATIVE: {
                "description": "Iterative refinement until criteria met",
                "pattern": "A → B → Validate → [Continue | Finish]",
                "use_cases": ["optimization", "refinement", "learning"]
            },
            WorkflowType.HIERARCHICAL: {
                "description": "Hierarchical task decomposition",
                "pattern": "Manager → [Worker1, Worker2] → Aggregator",
                "use_cases": ["complex projects", "team coordination"]
            }
        }
    
    def _load_agent_templates(self) -> Dict[AgentRole, Dict[str, Any]]:
        """Load agent templates for different roles."""
        return {
            AgentRole.ANALYZER: {
                "system_message": "You are an expert analyst. Analyze the given input and provide detailed insights.",
                "capabilities": ["data analysis", "pattern recognition", "insight generation"],
                "tools": ["analysis_tools", "visualization_tools"]
            },
            AgentRole.CREATOR: {
                "system_message": "You are a creative generator. Create high-quality content based on requirements.",
                "capabilities": ["content creation", "code generation", "design"],
                "tools": ["generation_tools", "creativity_tools"]
            },
            AgentRole.VALIDATOR: {
                "system_message": "You are a quality validator. Validate and improve the given content.",
                "capabilities": ["quality assessment", "error detection", "improvement suggestions"],
                "tools": ["validation_tools", "testing_tools"]
            },
            AgentRole.COORDINATOR: {
                "system_message": "You are a workflow coordinator. Manage and coordinate multiple agents.",
                "capabilities": ["task coordination", "resource management", "decision making"],
                "tools": ["coordination_tools", "management_tools"]
            },
            AgentRole.SPECIALIST: {
                "system_message": "You are a domain specialist. Provide expert knowledge in your field.",
                "capabilities": ["domain expertise", "specialized knowledge", "technical guidance"],
                "tools": ["specialist_tools", "domain_tools"]
            },
            AgentRole.OPTIMIZER: {
                "system_message": "You are an optimization expert. Improve performance and efficiency.",
                "capabilities": ["performance optimization", "efficiency improvement", "bottleneck identification"],
                "tools": ["optimization_tools", "profiling_tools"]
            }
        }
    
    async def generate_workflow(self, goal: str, context: Dict[str, Any] = None) -> WorkflowGraph:
        """
        Generate a workflow from a natural language goal.
        
        Args:
            goal: Natural language description of the goal
            context: Additional context information
            
        Returns:
            Generated workflow graph
        """
        try:
            logger.info(f"🎯 Generating workflow for goal: {goal}")
            
            # Step 1: Analyze the goal and determine workflow type
            workflow_analysis = await self._analyze_goal(goal, context)
            
            # Step 2: Generate workflow structure
            workflow_graph = await self._generate_workflow_structure(goal, workflow_analysis)
            
            # Step 3: Create and configure agents
            await self._configure_workflow_agents(workflow_graph, workflow_analysis)
            
            # Step 4: Optimize workflow based on historical performance
            await self._optimize_workflow(workflow_graph)
            
            logger.info(f"✅ Generated workflow with {len(workflow_graph.nodes)} nodes")
            return workflow_graph
            
        except Exception as e:
            logger.error(f"❌ Workflow generation failed: {e}")
            # Return a simple fallback workflow
            return self._create_fallback_workflow(goal)
    
    async def _analyze_goal(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze the goal to determine workflow requirements using LLM."""

        system_message = """You are an expert workflow architect. Analyze goals and design optimal multi-agent workflows.

Your task is to analyze the given goal and determine:
1. Complexity level (simple, moderate, complex)
2. Best workflow type (sequential, parallel, conditional, iterative, hierarchical)
3. Required agent roles from: analyzer, creator, validator, coordinator, specialist, optimizer
4. Estimated steps and execution strategy
5. Key challenges and success requirements

Always respond with valid JSON in the exact format specified."""

        analysis_prompt = f"""
        Analyze this goal and determine the best workflow approach:

        Goal: "{goal}"
        Context: {context or 'None provided'}

        Workflow Types:
        - sequential: Linear step-by-step execution (A → B → C)
        - parallel: Multiple agents working simultaneously (A → [B1, B2, B3] → C)
        - conditional: Decision-based branching (A → Decision → [B1 | B2])
        - iterative: Refinement loops (A → B → Validate → [Continue | Finish])
        - hierarchical: Manager-worker structure (Manager → [Worker1, Worker2] → Aggregator)

        Agent Roles:
        - analyzer: Analyzes data, requirements, and provides insights
        - creator: Generates content, code, solutions, or designs
        - validator: Validates quality, correctness, and compliance
        - coordinator: Manages workflow, coordinates between agents
        - specialist: Provides domain-specific expertise
        - optimizer: Improves performance, efficiency, and quality

        Respond with JSON:
        {{
            "complexity": "simple|moderate|complex",
            "workflow_type": "sequential|parallel|conditional|iterative|hierarchical",
            "required_roles": ["role1", "role2", "role3"],
            "estimated_steps": number,
            "challenges": ["challenge1", "challenge2"],
            "requirements": ["requirement1", "requirement2"],
            "reasoning": "Brief explanation of your analysis"
        }}
        """

        try:
            # Use LLM for intelligent analysis
            response = await self._call_llm(analysis_prompt, system_message)

            # Parse JSON response
            import json
            analysis = json.loads(response)

            # Validate the response
            if self._validate_analysis(analysis):
                logger.info(f"✅ LLM analysis completed: {analysis['workflow_type']} workflow with {len(analysis['required_roles'])} roles")
                return analysis
            else:
                logger.warning("Invalid LLM analysis, using fallback")
                return self._get_default_analysis()

        except Exception as e:
            logger.warning(f"LLM goal analysis failed: {e}")
            return self._get_default_analysis()

    async def _call_llm(self, prompt: str, system_message: str = None) -> str:
        """Call LLM for analysis."""
        try:
            # Import LLM client
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
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise e

    def _validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate the LLM analysis response."""
        required_fields = ["complexity", "workflow_type", "required_roles", "estimated_steps"]

        # Check required fields
        for field in required_fields:
            if field not in analysis:
                return False

        # Validate complexity
        if analysis["complexity"] not in ["simple", "moderate", "complex"]:
            return False

        # Validate workflow type
        valid_types = ["sequential", "parallel", "conditional", "iterative", "hierarchical"]
        if analysis["workflow_type"] not in valid_types:
            return False

        # Validate roles
        valid_roles = ["analyzer", "creator", "validator", "coordinator", "specialist", "optimizer"]
        if not isinstance(analysis["required_roles"], list):
            return False

        for role in analysis["required_roles"]:
            if role not in valid_roles:
                return False

        # Validate estimated steps
        if not isinstance(analysis["estimated_steps"], int) or analysis["estimated_steps"] < 1:
            return False

        return True
    

    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis when goal analysis fails."""
        return {
            "complexity": "moderate",
            "workflow_type": "sequential",
            "required_roles": ["analyzer", "creator", "validator"],
            "estimated_steps": 3,
            "challenges": ["task completion"],
            "requirements": ["basic functionality"]
        }

    async def _generate_workflow_structure(self, goal: str, analysis: Dict[str, Any]) -> WorkflowGraph:
        """Generate the workflow structure based on analysis."""

        workflow_type = WorkflowType(analysis["workflow_type"])
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        workflow = WorkflowGraph(
            id=workflow_id,
            name=f"Workflow for: {goal[:50]}...",
            description=f"Auto-generated workflow to achieve: {goal}",
            goal=goal,
            workflow_type=workflow_type
        )

        # Generate nodes based on required roles
        required_roles = analysis["required_roles"]

        for i, role_name in enumerate(required_roles):
            role = AgentRole(role_name)
            node_id = f"node_{i+1}_{role_name}"

            node = WorkflowNode(
                id=node_id,
                name=f"{role_name.title()} Agent",
                agent_role=role,
                description=f"Performs {role_name} tasks for the workflow"
            )

            # Set dependencies (sequential by default)
            if i > 0:
                prev_node_id = f"node_{i}_{required_roles[i-1]}"
                node.dependencies.append(prev_node_id)
                workflow.add_edge(prev_node_id, node_id)

            workflow.add_node(node)

        # Adjust structure based on workflow type
        if workflow_type == WorkflowType.PARALLEL:
            self._adjust_for_parallel_workflow(workflow)
        elif workflow_type == WorkflowType.CONDITIONAL:
            self._adjust_for_conditional_workflow(workflow)
        elif workflow_type == WorkflowType.ITERATIVE:
            self._adjust_for_iterative_workflow(workflow)
        elif workflow_type == WorkflowType.HIERARCHICAL:
            self._adjust_for_hierarchical_workflow(workflow)

        return workflow

    def _adjust_for_parallel_workflow(self, workflow: WorkflowGraph):
        """Adjust workflow for parallel execution."""
        if len(workflow.nodes) >= 3:
            # Make middle nodes parallel
            middle_nodes = workflow.nodes[1:-1]
            first_node = workflow.nodes[0]
            last_node = workflow.nodes[-1]

            # Clear existing edges
            workflow.edges = []

            # Connect first node to all middle nodes
            for node in middle_nodes:
                workflow.add_edge(first_node.id, node.id)
                node.dependencies = [first_node.id]

            # Connect all middle nodes to last node
            last_node.dependencies = [node.id for node in middle_nodes]
            for node in middle_nodes:
                workflow.add_edge(node.id, last_node.id)

    def _adjust_for_conditional_workflow(self, workflow: WorkflowGraph):
        """Adjust workflow for conditional execution."""
        if len(workflow.nodes) >= 2:
            # Add a decision node
            decision_node = WorkflowNode(
                id="decision_node",
                name="Decision Agent",
                agent_role=AgentRole.COORDINATOR,
                description="Makes decisions based on intermediate results"
            )

            # Insert decision node in the middle
            middle_index = len(workflow.nodes) // 2
            workflow.nodes.insert(middle_index, decision_node)

            # Rebuild edges with conditional logic
            self._rebuild_conditional_edges(workflow)

    def _adjust_for_iterative_workflow(self, workflow: WorkflowGraph):
        """Adjust workflow for iterative execution."""
        if len(workflow.nodes) >= 2:
            # Add feedback edge from last to first node
            first_node = workflow.nodes[0]
            last_node = workflow.nodes[-1]

            # Add iteration control
            last_node.agent_config["iteration_control"] = True
            last_node.agent_config["max_iterations"] = 3

            # Add feedback edge (conceptual)
            workflow.metadata["feedback_edge"] = (last_node.id, first_node.id)

    def _adjust_for_hierarchical_workflow(self, workflow: WorkflowGraph):
        """Adjust workflow for hierarchical execution."""
        if len(workflow.nodes) >= 3:
            # Designate first node as manager
            manager_node = workflow.nodes[0]
            manager_node.agent_role = AgentRole.COORDINATOR
            manager_node.name = "Manager Agent"

            # Make other nodes workers
            for node in workflow.nodes[1:]:
                node.agent_config["managed_by"] = manager_node.id

    def _rebuild_conditional_edges(self, workflow: WorkflowGraph):
        """Rebuild edges for conditional workflow."""
        workflow.edges = []

        for i in range(len(workflow.nodes) - 1):
            current_node = workflow.nodes[i]
            next_node = workflow.nodes[i + 1]

            workflow.add_edge(current_node.id, next_node.id)
            next_node.dependencies = [current_node.id]

    async def _configure_workflow_agents(self, workflow: WorkflowGraph, analysis: Dict[str, Any]):
        """Configure agents for each node in the workflow."""

        for node in workflow.nodes:
            template = self.agent_templates[node.agent_role]

            # Customize system message based on goal and role
            customized_message = await self._customize_agent_message(
                template["system_message"],
                workflow.goal,
                node.agent_role,
                analysis
            )

            node.agent_config.update({
                "system_message": customized_message,
                "capabilities": template["capabilities"],
                "tools": template["tools"],
                "llm_config": self.llm_config
            })

    async def _customize_agent_message(self, base_message: str, goal: str,
                                     role: AgentRole, analysis: Dict[str, Any]) -> str:
        """Customize agent system message for specific goal and role using LLM."""

        system_message = """You are an expert in agent design and prompt engineering. Your task is to customize agent system messages for specific goals while maintaining their core capabilities."""

        customization_prompt = f"""
        Customize this agent system message for the specific goal:

        Base Message: {base_message}
        Goal: "{goal}"
        Agent Role: {role.value}
        Workflow Analysis: {analysis}

        Requirements:
        1. Keep the core agent capabilities and personality
        2. Add specific instructions relevant to achieving the goal
        3. Define clear success criteria for this agent's contribution
        4. Specify how this agent should interact with other agents in the workflow
        5. Include any domain-specific knowledge or constraints

        The customized message should be:
        - Specific and actionable
        - Aligned with the overall goal
        - Clear about the agent's responsibilities
        - Professional and focused

        Return only the customized system message (no explanations or formatting).
        """

        try:
            # Use LLM to customize the message
            customized = await self._call_llm(customization_prompt, system_message)

            # Validate that we got a reasonable response
            if len(customized) > 50 and "goal" in customized.lower():
                logger.debug(f"✅ Customized message for {role.value} agent")
                return customized
            else:
                logger.warning("LLM customization produced invalid result, using fallback")
                return self._create_fallback_message(base_message, goal, role)

        except Exception as e:
            logger.warning(f"LLM message customization failed: {e}")
            return self._create_fallback_message(base_message, goal, role)

    def _create_fallback_message(self, base_message: str, goal: str, role: AgentRole) -> str:
        """Create a fallback customized message when LLM fails."""
        return f"""{base_message}

SPECIFIC MISSION: {goal}

Your role as a {role.value} agent is critical to achieving this goal. Focus on:
- Applying your {role.value} expertise to advance toward the goal
- Collaborating effectively with other agents in the workflow
- Delivering high-quality results that contribute to the overall success
- Communicating clearly about your progress and any challenges

Success criteria: Your contribution should directly advance the goal while maintaining quality standards."""

    async def _optimize_workflow(self, workflow: WorkflowGraph):
        """Optimize workflow based on historical performance."""

        # Check if we have performance history for similar workflows
        similar_workflows = self._find_similar_workflows(workflow.goal)

        if similar_workflows:
            # Apply optimizations from successful similar workflows
            best_workflow = max(similar_workflows, key=lambda w: w.get("success_rate", 0))

            # Apply optimizations
            if best_workflow.get("success_rate", 0) > 0.8:
                self._apply_optimizations(workflow, best_workflow["optimizations"])

        # Set initial performance metrics
        workflow.performance_metrics = {
            "estimated_success_rate": 0.8,
            "estimated_execution_time": len(workflow.nodes) * 30,  # 30 seconds per node
            "complexity_score": self._calculate_complexity_score(workflow)
        }

    def _find_similar_workflows(self, goal: str) -> List[Dict[str, Any]]:
        """Find similar workflows from performance history."""
        # This would implement similarity search
        # For now, return empty list
        return []

    def _apply_optimizations(self, workflow: WorkflowGraph, optimizations: Dict[str, Any]):
        """Apply optimizations to the workflow."""
        # This would apply specific optimizations
        # For now, just log the intent
        logger.info(f"Applying optimizations to workflow {workflow.id}")

    def _calculate_complexity_score(self, workflow: WorkflowGraph) -> float:
        """Calculate complexity score for the workflow."""
        base_score = len(workflow.nodes) * 0.1
        edge_score = len(workflow.edges) * 0.05

        # Add complexity based on workflow type
        type_complexity = {
            WorkflowType.SEQUENTIAL: 0.1,
            WorkflowType.PARALLEL: 0.3,
            WorkflowType.CONDITIONAL: 0.4,
            WorkflowType.ITERATIVE: 0.5,
            WorkflowType.HIERARCHICAL: 0.6
        }

        return base_score + edge_score + type_complexity.get(workflow.workflow_type, 0.2)

    def _create_fallback_workflow(self, goal: str) -> WorkflowGraph:
        """Create a simple fallback workflow when generation fails."""

        workflow = WorkflowGraph(
            id="fallback_workflow",
            name="Fallback Workflow",
            description=f"Simple fallback workflow for: {goal}",
            goal=goal,
            workflow_type=WorkflowType.SEQUENTIAL
        )

        # Add basic analyzer and creator nodes
        analyzer_node = WorkflowNode(
            id="analyzer",
            name="Analyzer Agent",
            agent_role=AgentRole.ANALYZER,
            description="Analyzes the goal and requirements"
        )

        creator_node = WorkflowNode(
            id="creator",
            name="Creator Agent",
            agent_role=AgentRole.CREATOR,
            description="Creates solution based on analysis",
            dependencies=["analyzer"]
        )

        workflow.add_node(analyzer_node)
        workflow.add_node(creator_node)
        workflow.add_edge("analyzer", "creator")

        return workflow


# Factory function
def get_dynamic_workflow_generator(llm_config: Dict[str, Any]) -> DynamicWorkflowGenerator:
    """Get a DynamicWorkflowGenerator instance."""
    return DynamicWorkflowGenerator(llm_config)
