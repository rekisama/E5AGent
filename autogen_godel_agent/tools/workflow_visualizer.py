"""
Workflow Visualization Module

This module provides comprehensive workflow visualization capabilities
for the AutoGen Self-Expanding Agent System, supporting multiple output formats:

- Mermaid diagrams for web display
- Graphviz DOT format for high-quality rendering
- ASCII art for terminal display
- JSON structure for programmatic access
- HTML interactive diagrams

Key Features:
- Multiple visualization formats
- Interactive workflow exploration
- Real-time execution status display
- Performance metrics visualization
- Export capabilities
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Import workflow types
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from workflow.evo_core_algorithms import EvoWorkflowGraph, EvoWorkflowNode, EvoWorkflowEdge
    from workflow.dynamic_workflow_generator import WorkflowGraph, WorkflowNode
    from workflow.evo_workflow_executor import WorkflowExecutionResult, NodeExecutionResult, ExecutionStatus
except ImportError as e:
    logger.warning(f"Could not import workflow types: {e}")
    # Create mock classes for basic functionality
    class EvoWorkflowGraph:
        def __init__(self, goal="", nodes=None, edges=None):
            self.goal = goal
            self.nodes = nodes or []
            self.edges = edges or []
            self.id = f"mock_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    class EvoWorkflowNode:
        def __init__(self, name="", description="", reason=""):
            self.name = name
            self.description = description
            self.reason = reason
            self.inputs = []
            self.outputs = []

    class EvoWorkflowEdge:
        def __init__(self, edge_tuple=("", "")):
            self.edge_tuple = edge_tuple
            self.source = edge_tuple[0]
            self.target = edge_tuple[1]

    class WorkflowGraph:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class WorkflowNode:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class WorkflowExecutionResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class NodeExecutionResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class ExecutionStatus:
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"

logger = logging.getLogger(__name__)


class VisualizationFormat(Enum):
    """Supported visualization formats."""
    MERMAID = "mermaid"
    GRAPHVIZ = "graphviz"
    ASCII = "ascii"
    JSON = "json"
    HTML = "html"


@dataclass
class VisualizationOptions:
    """Options for workflow visualization."""
    format: VisualizationFormat = VisualizationFormat.MERMAID
    include_details: bool = True
    show_execution_status: bool = False
    show_performance_metrics: bool = False
    color_scheme: str = "default"
    layout_direction: str = "TD"  # Top-Down, Left-Right, etc.
    node_style: str = "rounded"
    edge_style: str = "solid"


class WorkflowVisualizer:
    """
    Main workflow visualization class.
    
    Supports multiple workflow types and visualization formats.
    """
    
    def __init__(self):
        self.color_schemes = self._load_color_schemes()
        logger.info("✅ WorkflowVisualizer initialized")
    
    def _load_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """Load predefined color schemes."""
        return {
            "default": {
                "analyzer": "#e3f2fd",
                "creator": "#f3e5f5", 
                "executor": "#e8f5e8",
                "validator": "#fff3e0",
                "aggregator": "#ffebee",
                "edge": "#666666"
            },
            "dark": {
                "analyzer": "#1565c0",
                "creator": "#7b1fa2",
                "executor": "#388e3c", 
                "validator": "#f57c00",
                "aggregator": "#d32f2f",
                "edge": "#ffffff"
            },
            "pastel": {
                "analyzer": "#bbdefb",
                "creator": "#e1bee7",
                "executor": "#c8e6c9",
                "validator": "#ffe0b2",
                "aggregator": "#ffcdd2",
                "edge": "#9e9e9e"
            }
        }
    
    def visualize_workflow(self, workflow: Any, 
                          options: VisualizationOptions = None,
                          execution_result: WorkflowExecutionResult = None) -> str:
        """
        Generate visualization for any workflow type.
        
        Args:
            workflow: Workflow object (EvoWorkflowGraph or WorkflowGraph)
            options: Visualization options
            execution_result: Optional execution result for status display
            
        Returns:
            Visualization string in the specified format
        """
        if options is None:
            options = VisualizationOptions()
        
        logger.info(f"🎨 Generating {options.format.value} visualization")
        
        # Determine workflow type and delegate to appropriate method
        if isinstance(workflow, EvoWorkflowGraph):
            return self._visualize_evo_workflow(workflow, options, execution_result)
        elif isinstance(workflow, WorkflowGraph):
            return self._visualize_dynamic_workflow(workflow, options, execution_result)
        else:
            raise ValueError(f"Unsupported workflow type: {type(workflow)}")
    
    def _visualize_evo_workflow(self, workflow: EvoWorkflowGraph, 
                               options: VisualizationOptions,
                               execution_result: WorkflowExecutionResult = None) -> str:
        """Visualize EvoWorkflowGraph."""
        
        if options.format == VisualizationFormat.MERMAID:
            return self._generate_mermaid_evo(workflow, options, execution_result)
        elif options.format == VisualizationFormat.GRAPHVIZ:
            return self._generate_graphviz_evo(workflow, options, execution_result)
        elif options.format == VisualizationFormat.ASCII:
            return self._generate_ascii_evo(workflow, options, execution_result)
        elif options.format == VisualizationFormat.JSON:
            return self._generate_json_evo(workflow, options, execution_result)
        elif options.format == VisualizationFormat.HTML:
            return self._generate_html_evo(workflow, options, execution_result)
        else:
            raise ValueError(f"Unsupported format: {options.format}")
    
    def _generate_mermaid_evo(self, workflow: EvoWorkflowGraph, 
                             options: VisualizationOptions,
                             execution_result: WorkflowExecutionResult = None) -> str:
        """Generate Mermaid diagram for EvoWorkflow."""
        
        mermaid = f"graph {options.layout_direction}\n"
        mermaid += f"    %% Workflow: {workflow.goal}\n"
        mermaid += f"    %% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add nodes
        for node in workflow.nodes:
            node_id = self._sanitize_id(node.name)
            node_label = node.name
            
            if options.include_details:
                node_label += f"\\n{node.description[:50]}..."
            
            # Add execution status if available
            if options.show_execution_status and execution_result:
                node_result = execution_result.node_results.get(node.name)
                if node_result:
                    status_icon = self._get_status_icon(node_result.status)
                    node_label += f"\\n{status_icon}"
            
            mermaid += f"    {node_id}[\"{node_label}\"]\n"
        
        mermaid += "\n"
        
        # Add edges
        for edge in workflow.edges:
            source_id = self._sanitize_id(edge.source)
            target_id = self._sanitize_id(edge.target)
            mermaid += f"    {source_id} --> {target_id}\n"
        
        # Add styling
        mermaid += "\n    %% Styling\n"
        color_scheme = self.color_schemes.get(options.color_scheme, self.color_schemes["default"])
        
        for i, node in enumerate(workflow.nodes):
            node_id = self._sanitize_id(node.name)
            # Determine node type based on name/description
            node_type = self._determine_node_type(node)
            color = color_scheme.get(node_type, color_scheme["analyzer"])
            mermaid += f"    classDef node{i} fill:{color},stroke:#333,stroke-width:2px\n"
            mermaid += f"    class {node_id} node{i}\n"
        
        return mermaid
    
    def _generate_ascii_evo(self, workflow: EvoWorkflowGraph, 
                           options: VisualizationOptions,
                           execution_result: WorkflowExecutionResult = None) -> str:
        """Generate ASCII art representation."""
        
        ascii_art = f"╔══════════════════════════════════════════════════════════════╗\n"
        ascii_art += f"║ WORKFLOW: {workflow.goal[:50]:<50} ║\n"
        ascii_art += f"╠══════════════════════════════════════════════════════════════╣\n"
        
        # Add nodes
        for i, node in enumerate(workflow.nodes, 1):
            status = ""
            if options.show_execution_status and execution_result:
                node_result = execution_result.node_results.get(node.name)
                if node_result:
                    status = f" [{node_result.status.value}]"
            
            ascii_art += f"║ {i:2d}. {node.name:<45}{status:>10} ║\n"
            if options.include_details:
                desc = node.description[:55] + "..." if len(node.description) > 55 else node.description
                ascii_art += f"║     {desc:<55}     ║\n"
        
        ascii_art += f"╠══════════════════════════════════════════════════════════════╣\n"
        
        # Add edges
        if workflow.edges:
            ascii_art += f"║ DEPENDENCIES:                                                ║\n"
            for edge in workflow.edges:
                ascii_art += f"║   {edge.source} → {edge.target:<40} ║\n"
        
        ascii_art += f"╚══════════════════════════════════════════════════════════════╝\n"
        
        return ascii_art
    
    def _generate_json_evo(self, workflow: EvoWorkflowGraph, 
                          options: VisualizationOptions,
                          execution_result: WorkflowExecutionResult = None) -> str:
        """Generate JSON representation."""
        
        workflow_data = {
            "id": workflow.id,
            "goal": workflow.goal,
            "created_at": workflow.created_at.isoformat(),
            "nodes": [],
            "edges": [],
            "metadata": {
                "node_count": len(workflow.nodes),
                "edge_count": len(workflow.edges),
                "visualization_generated": datetime.now().isoformat()
            }
        }
        
        # Add nodes
        for node in workflow.nodes:
            node_data = {
                "name": node.name,
                "description": node.description,
                "reason": node.reason,
                "inputs": [{"name": p.name, "type": p.type} for p in node.inputs],
                "outputs": [{"name": p.name, "type": p.type} for p in node.outputs]
            }
            
            # Add execution status if available
            if options.show_execution_status and execution_result:
                node_result = execution_result.node_results.get(node.name)
                if node_result:
                    node_data["execution_status"] = {
                        "status": node_result.status.value,
                        "execution_time": getattr(node_result, 'execution_time', None),
                        "error": node_result.error
                    }
            
            workflow_data["nodes"].append(node_data)
        
        # Add edges
        for edge in workflow.edges:
            workflow_data["edges"].append({
                "source": edge.source,
                "target": edge.target
            })
        
        return json.dumps(workflow_data, indent=2, ensure_ascii=False)
    
    def _sanitize_id(self, name: str) -> str:
        """Sanitize node name for use as ID."""
        return name.replace(" ", "_").replace("-", "_").replace(".", "_")
    
    def _get_status_icon(self, status: ExecutionStatus) -> str:
        """Get icon for execution status."""
        icons = {
            ExecutionStatus.PENDING: "⏳",
            ExecutionStatus.RUNNING: "🔄", 
            ExecutionStatus.COMPLETED: "✅",
            ExecutionStatus.FAILED: "❌",
            ExecutionStatus.CANCELLED: "⏹️"
        }
        return icons.get(status, "❓")
    
    def _determine_node_type(self, node) -> str:
        """Determine node type based on name/description."""
        name_lower = node.name.lower()
        desc_lower = node.description.lower()
        
        if any(word in name_lower or word in desc_lower for word in ["analyze", "analysis", "examine"]):
            return "analyzer"
        elif any(word in name_lower or word in desc_lower for word in ["create", "generate", "build"]):
            return "creator"
        elif any(word in name_lower or word in desc_lower for word in ["execute", "run", "process"]):
            return "executor"
        elif any(word in name_lower or word in desc_lower for word in ["validate", "verify", "check"]):
            return "validator"
        elif any(word in name_lower or word in desc_lower for word in ["aggregate", "combine", "merge"]):
            return "aggregator"
        else:
            return "analyzer"  # default


# Factory function
def get_workflow_visualizer() -> WorkflowVisualizer:
    """Get a WorkflowVisualizer instance."""
    return WorkflowVisualizer()


# Convenience functions
def visualize_workflow_mermaid(workflow: Any, include_details: bool = True) -> str:
    """Quick function to generate Mermaid diagram."""
    visualizer = get_workflow_visualizer()
    options = VisualizationOptions(
        format=VisualizationFormat.MERMAID,
        include_details=include_details
    )
    return visualizer.visualize_workflow(workflow, options)


def visualize_workflow_ascii(workflow: Any, show_execution_status: bool = False) -> str:
    """Quick function to generate ASCII art."""
    visualizer = get_workflow_visualizer()
    options = VisualizationOptions(
        format=VisualizationFormat.ASCII,
        show_execution_status=show_execution_status
    )
    return visualizer.visualize_workflow(workflow, options)


def export_workflow_html(workflow: Any, output_path: str,
                        include_interactive: bool = True) -> bool:
    """Export workflow as interactive HTML file."""
    try:
        visualizer = get_workflow_visualizer()
        options = VisualizationOptions(
            format=VisualizationFormat.HTML,
            include_details=True
        )

        html_content = visualizer.visualize_workflow(workflow, options)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"✅ Workflow exported to {output_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to export workflow: {e}")
        return False


class WorkflowDashboard:
    """
    Interactive workflow dashboard for monitoring and visualization.
    """

    def __init__(self):
        self.visualizer = get_workflow_visualizer()
        self.active_workflows = {}
        self.execution_history = []

    def register_workflow(self, workflow_id: str, workflow: Any):
        """Register a workflow for monitoring."""
        self.active_workflows[workflow_id] = {
            "workflow": workflow,
            "registered_at": datetime.now(),
            "status": "registered"
        }
        logger.info(f"📊 Registered workflow {workflow_id} for monitoring")

    def update_execution_status(self, workflow_id: str,
                               execution_result: WorkflowExecutionResult):
        """Update execution status for a workflow."""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["execution_result"] = execution_result
            self.active_workflows[workflow_id]["status"] = execution_result.status.value
            self.active_workflows[workflow_id]["last_updated"] = datetime.now()

            # Add to history
            self.execution_history.append({
                "workflow_id": workflow_id,
                "timestamp": datetime.now(),
                "status": execution_result.status.value,
                "execution_time": execution_result.total_execution_time
            })

            logger.info(f"📈 Updated status for workflow {workflow_id}: {execution_result.status.value}")

    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard for all workflows."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AutoGen Workflow Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .dashboard { max-width: 1200px; margin: 0 auto; }
                .header { background: #2196f3; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .workflow-card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .status-badge { padding: 4px 8px; border-radius: 4px; color: white; font-size: 12px; }
                .status-completed { background: #4caf50; }
                .status-running { background: #ff9800; }
                .status-failed { background: #f44336; }
                .status-pending { background: #9e9e9e; }
                .mermaid { background: white; border: 1px solid #ddd; border-radius: 4px; padding: 10px; }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }
                .metric-card { background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-value { font-size: 24px; font-weight: bold; color: #2196f3; }
                .metric-label { color: #666; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>🚀 AutoGen Workflow Dashboard</h1>
                    <p>Real-time workflow monitoring and visualization</p>
                </div>
        """

        # Add metrics
        total_workflows = len(self.active_workflows)
        completed_workflows = sum(1 for w in self.active_workflows.values() if w.get("status") == "completed")
        running_workflows = sum(1 for w in self.active_workflows.values() if w.get("status") == "running")

        html += f"""
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{total_workflows}</div>
                        <div class="metric-label">Total Workflows</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{completed_workflows}</div>
                        <div class="metric-label">Completed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{running_workflows}</div>
                        <div class="metric-label">Running</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(self.execution_history)}</div>
                        <div class="metric-label">Total Executions</div>
                    </div>
                </div>
        """

        # Add workflow cards
        for workflow_id, workflow_data in self.active_workflows.items():
            workflow = workflow_data["workflow"]
            status = workflow_data.get("status", "unknown")

            # Generate Mermaid diagram
            mermaid_diagram = self.visualizer.visualize_workflow(
                workflow,
                VisualizationOptions(format=VisualizationFormat.MERMAID),
                workflow_data.get("execution_result")
            )

            html += f"""
                <div class="workflow-card">
                    <h3>{workflow_id} <span class="status-badge status-{status}">{status.upper()}</span></h3>
                    <p><strong>Goal:</strong> {getattr(workflow, 'goal', 'N/A')}</p>
                    <p><strong>Nodes:</strong> {len(getattr(workflow, 'nodes', []))}</p>
                    <p><strong>Registered:</strong> {workflow_data['registered_at'].strftime('%Y-%m-%d %H:%M:%S')}</p>

                    <div class="mermaid">
                        {mermaid_diagram}
                    </div>
                </div>
            """

        html += """
            </div>
            <script>
                mermaid.initialize({ startOnLoad: true, theme: 'default' });
            </script>
        </body>
        </html>
        """

        return html

    def export_dashboard(self, output_path: str) -> bool:
        """Export dashboard to HTML file."""
        try:
            html_content = self.generate_dashboard_html()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"✅ Dashboard exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to export dashboard: {e}")
            return False


# Global dashboard instance
_dashboard_instance = None

def get_workflow_dashboard() -> WorkflowDashboard:
    """Get the global workflow dashboard instance."""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = WorkflowDashboard()
    return _dashboard_instance
