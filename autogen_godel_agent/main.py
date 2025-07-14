"""
Main entry point for AutoGen Self-Expanding Agent System.

This system can automatically generate, test, and register new functions
when encountering tasks it cannot complete with existing capabilities.

Enhanced with learning memory system for intelligent recommendations
and continuous improvement.
"""

import autogen
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from config import Config
from agents.planner_agent import TaskPlannerAgent
from agents.function_creator_agent import FunctionCreatorAgent
from agents.multi_file_agent import get_multi_file_agent
from workflow.evo_workflow_manager import get_evo_workflow_manager
from tools.function_tools import get_function_tools
try:
    from tools.workflow_visualizer import get_workflow_visualizer, get_workflow_dashboard, visualize_workflow_mermaid
    ADVANCED_VISUALIZER_AVAILABLE = True
except ImportError:
    from tools.simple_visualizer import (
        generate_mermaid_from_description,
        generate_ascii_from_description,
        determine_workflow_type,
        export_visualization
    )
    ADVANCED_VISUALIZER_AVAILABLE = False

# Import GUI visualizer
try:
    from tools.gui_visualizer import show_workflow_popup, get_gui_visualizer
    GUI_VISUALIZER_AVAILABLE = True
except ImportError:
    GUI_VISUALIZER_AVAILABLE = False
# Try to import learning memory integration
try:
    from tools.learning_memory_integration import LearningMemoryIntegration
    LEARNING_MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Learning memory integration not available: {e}")
    LearningMemoryIntegration = None
    LEARNING_MEMORY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SelfExpandingAgentSystem:
    """
    Main orchestrator for the self-expanding agent system.

    This system uses AutoGen agents enhanced with learning memory
    for intelligent task processing and continuous improvement.
    """

    def __init__(self):
        """Initialize the system."""
        # Validate configuration
        Config.validate_config()

        # Get LLM configuration
        self.llm_config = Config.get_llm_config()

        # Initialize core components
        self.function_tools = get_function_tools()
        self.planner_agent = TaskPlannerAgent(self.llm_config)
        self.creator_agent = FunctionCreatorAgent(self.llm_config)
        self.multi_file_agent = get_multi_file_agent(self.llm_config)
        self.evo_workflow_manager = get_evo_workflow_manager(self.llm_config)

        # Initialize visualization components
        if ADVANCED_VISUALIZER_AVAILABLE:
            self.workflow_visualizer = get_workflow_visualizer()
            self.workflow_dashboard = get_workflow_dashboard()
        else:
            self.workflow_visualizer = None
            self.workflow_dashboard = None
            logger.info("⚠️ Using simplified visualizer (advanced features unavailable)")

        # Initialize learning memory system if available
        if LEARNING_MEMORY_AVAILABLE:
            self.learning_integration = LearningMemoryIntegration(
                self.function_tools,
                self.function_tools.registry
            )
            logger.info("🧠 Learning Memory System enabled")
        else:
            self.learning_integration = None
            logger.info("⚠️ Learning Memory System disabled")

        # Create user proxy agent for complex tasks
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            system_message="You are a user proxy that facilitates communication between the user and the AI agents.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
        )

        # Create group chat for complex coordination
        self.group_chat = autogen.GroupChat(
            agents=[self.user_proxy, self.planner_agent.agent, self.creator_agent.agent, self.multi_file_agent.agent],
            messages=[],
            max_round=20,
            speaker_selection_method="round_robin",
        )

        # Create group chat manager
        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )

        # Initialize history
        self.history = self._load_history()

        logger.info("✅ Initialized Self-Expanding Agent System with Learning Memory")
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from file."""
        if os.path.exists(Config.HISTORY_FILE):
            try:
                with open(Config.HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load history: {e}")
        return []
    
    def _save_history(self):
        """Save conversation history to file."""
        try:
            os.makedirs(os.path.dirname(Config.HISTORY_FILE), exist_ok=True)
            with open(Config.HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save history: {e}")
    
    def process_task(self, task_description: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a task using the self-expanding agent system.

        Args:
            task_description: Description of the task to be completed
            session_id: Optional session ID for context preservation

        Returns:
            Dictionary containing the result and metadata
        """
        logger.info(f"🎯 Processing task: {task_description}")
        start_time = datetime.now()

        try:
            # 1. Intelligent task routing - determine the best approach
            if self._needs_evo_workflow_sync(task_description):
                logger.info("🌟 Task requires EvoAgentX-style workflow")
                result = self._process_evo_workflow_task_sync(task_description)
            elif self._needs_multi_file_generation_sync(task_description):
                logger.info("🏗️ Task requires multi-file generation")
                result = self._process_multi_file_task_sync(task_description)
            else:
                # 2. Enhanced task analysis with learning memory (if available)
                if self.learning_integration:
                    enhancement = self.learning_integration.enhance_task_analysis(task_description)

                    # 3. Intelligent decision making based on learning insights
                    if enhancement.get('enhanced', False):
                        recommendations = enhancement.get('recommendations', {})
                        confidence = recommendations.get('confidence', 0.0)

                        if confidence > getattr(Config, 'MIN_RECOMMENDATION_CONFIDENCE', 0.5):
                            logger.info(f"🧠 Using learning recommendations (confidence: {confidence:.2f})")
                            result = self._execute_with_recommendations(task_description, enhancement)
                        else:
                            logger.info("🔧 Creating new function (low confidence in recommendations)")
                            result = self._create_and_execute_function(task_description)
                    else:
                        logger.info("🔄 Using standard analysis workflow")
                        result = self._process_task_standard(task_description)

                    # 4. Record execution for learning
                    execution_time = (datetime.now() - start_time).total_seconds()
                    self.learning_integration.record_task_execution(
                        task_description=task_description,
                        functions_used=result.get('functions_used', []),
                        success=result.get('success', False),
                        execution_time=execution_time,
                        additional_context={'session_id': session_id}
                    )
                else:
                    # Fallback to standard processing without learning memory
                    logger.info("🔄 Using standard workflow (learning memory not available)")
                    result = self._process_task_standard(task_description)

            # 4. Update history
            task_record = {
                'timestamp': start_time.isoformat(),
                'task': task_description,
                'result': result,
                'execution_time': execution_time,
                'session_id': session_id
            }
            self.history.append(task_record)
            self._save_history()

            return result

        except Exception as e:
            error_msg = f'Task processing failed: {str(e)}'
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'exception': type(e).__name__,
                'system_type': 'enhanced_autogen'
            }

    def _execute_with_recommendations(self, task_description: str, enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using learning memory recommendations."""
        recommendations = enhancement.get('recommendations', {})
        recommended_functions = recommendations.get('functions', [])

        logger.info(f"📚 Executing with {len(recommended_functions)} recommended functions")

        if len(recommended_functions) == 1:
            # Single function execution
            func_name = recommended_functions[0]
            return self._execute_single_function(task_description, func_name)
        elif len(recommended_functions) > 1:
            # Function composition
            return self._compose_and_execute_functions(task_description, recommended_functions)
        else:
            # Fallback to creation
            return self._create_and_execute_function(task_description)

    def _execute_single_function(self, task_description: str, function_name: str) -> Dict[str, Any]:
        """Execute a single function."""
        try:
            # Get function info
            func_info = self.function_tools.get_function_info(function_name)
            if not func_info:
                return {'success': False, 'error': f'Function {function_name} not found'}

            # For now, return function info since direct execution needs parameter mapping
            # TODO: Implement proper parameter mapping and function execution
            result = f"Found function '{function_name}': {func_info.get('description', 'No description')}"

            return {
                'success': True,
                'result': result,
                'functions_used': [function_name],
                'execution_type': 'single_function'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _compose_and_execute_functions(self, task_description: str, function_names: List[str]) -> Dict[str, Any]:
        """Compose and execute multiple functions."""
        try:
            # Use function composer
            from tools.function_composer import get_function_composer
            composer = get_function_composer()

            success, message, composite_func = composer.compose_functions(task_description)

            if success:
                return {
                    'success': True,
                    'result': message,
                    'functions_used': function_names,
                    'execution_type': 'function_composition'
                }
            else:
                # Fallback to creation
                return self._create_and_execute_function(task_description)

        except Exception as e:
            logger.warning(f"Function composition failed: {e}")
            return self._create_and_execute_function(task_description)

    def _create_and_execute_function(self, task_description: str) -> Dict[str, Any]:
        """Create and execute a new function."""
        try:
            # Analyze task
            analysis = self.planner_agent.analyze_task(task_description)

            # Extract function specification
            spec = self._extract_function_spec(task_description, analysis)

            # Create function
            success, message, code = self.creator_agent.create_function(spec)

            if success:
                return {
                    'success': True,
                    'result': message,
                    'functions_used': [spec['name']],
                    'function_created': spec['name'],
                    'execution_type': 'new_function'
                }
            else:
                return {'success': False, 'error': message}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _process_task_standard(self, task_description: str) -> Dict[str, Any]:
        """Process task using standard AutoGen workflow."""

        logger.info("📋 Analyzing task with planner agent")

        try:
            # Step 1: Analyze task
            analysis = self.planner_agent.analyze_task(task_description)
            existing_functions = analysis.get('existing_functions', [])

            logger.info(f"Found {len(existing_functions)} existing function(s)")

            # Step 2: Determine approach
            needs_new_function = self._needs_new_function(task_description, existing_functions)

            if existing_functions and not needs_new_function:
                logger.info("✅ Using existing functions")
                result = self._execute_with_existing_functions(task_description, existing_functions)
            else:
                logger.info("🔧 Creating new function")
                result = self._create_and_execute_function(task_description)

            # Step 3: Fallback to group chat for complex coordination
            if not result.get('success', False):
                logger.info("💬 Using group chat for complex coordination")
                result = self._run_group_chat(task_description)

            return result

        except Exception as e:
            logger.error(f"Standard processing failed: {e}")
            return {'success': False, 'error': str(e)}

    def _needs_new_function(self, task_description: str, existing_functions: List[Dict]) -> bool:
        """Determine if a new function needs to be created for the task."""
        if not existing_functions:
            return True

        # Check if task is asking for a specific new function
        task_lower = task_description.lower()

        # Keywords that indicate creating something new
        creation_keywords = ['create', 'make', 'build', 'generate', 'develop', 'implement']
        if any(keyword in task_lower for keyword in creation_keywords):
            # Check if it's asking for something specific that doesn't exist
            specific_functions = {
                'phone': ['validate_phone', 'phone_validator', 'check_phone'],
                'factorial': ['factorial', 'calculate_factorial'],
                'palindrome': ['palindrome', 'is_palindrome', 'check_palindrome'],
                'temperature': ['celsius_to_fahrenheit', 'fahrenheit_to_celsius', 'convert_temperature'],
                'password': ['generate_password', 'create_password', 'password_generator']
            }

            for keyword, func_names in specific_functions.items():
                if keyword in task_lower:
                    # Check if we already have this type of function
                    existing_names = [f['name'].lower() for f in existing_functions]
                    if not any(name in existing_names for name in func_names):
                        return True

        return False

    def _execute_with_existing_functions(self, task: str, functions: List[Dict]) -> Dict[str, Any]:
        """Execute task with existing functions."""
        if not functions:
            return {'success': False, 'error': 'No functions available'}

        try:
            # Try to use the best matching function
            best_function = functions[0]  # Simplified - should use scoring
            func_name = best_function['name']

            return self._execute_single_function(task, func_name)

        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_function_spec(self, task: str, analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract function specification from task description."""

        # Generate a function name based on the task
        func_name = task.lower().replace(' ', '_').replace('-', '_')
        func_name = ''.join(c for c in func_name if c.isalnum() or c == '_')

        # Limit function name length
        if len(func_name) > 50:
            func_name = func_name[:50]

        # Add timestamp to ensure uniqueness
        import time
        func_name = f"{func_name}_{int(time.time())}"

        # Use analysis if available
        if analysis and 'suggested_function_spec' in analysis:
            spec = analysis['suggested_function_spec']
            if spec is not None:  # Check if spec is not None
                spec['name'] = func_name  # Ensure unique name
                return spec

        # Default specification
        return {
            'name': func_name,
            'description': task,
            'parameters': [],
            'return_type': 'Any',
            'examples': []
        }
    
    def _run_group_chat(self, task: str) -> Dict[str, Any]:
        """Run group chat for complex task resolution."""
        try:
            # Initialize the conversation
            self.user_proxy.initiate_chat(
                self.manager,
                message=f"Please help me complete this task: {task}\n\nAnalyze what functions are needed and create them if necessary."
            )
            
            # Get the conversation result
            messages = self.group_chat.messages
            
            return {
                'success': True,
                'message': 'Group chat completed',
                'conversation_length': len(messages)
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Group chat failed: {e}'
            }
    
    def list_available_functions(self) -> List[Dict[str, Any]]:
        """List all available functions in the system."""
        try:
            # Get function names and their info
            function_names = self.function_tools.list_functions()

            # Ensure function_names is a list
            if isinstance(function_names, str):
                logger.warning(f"list_functions returned string instead of list: {function_names}")
                return []

            if not isinstance(function_names, list):
                logger.warning(f"list_functions returned unexpected type: {type(function_names)}")
                return []

            functions_info = []

            for func_name in function_names:
                try:
                    func_info = self.function_tools.registry.get_function_info(func_name)
                    if func_info:
                        functions_info.append(func_info)
                except Exception as e:
                    logger.warning(f"Could not get info for function {func_name}: {e}")
                    continue

            return functions_info
        except Exception as e:
            logger.warning(f"Could not list functions: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        functions = self.list_available_functions()

        return {
            'total_functions': len(functions),
            'total_tasks_processed': len(self.history),
            'successful_tasks': len([h for h in self.history if h.get('status', '').startswith('completed')]),
            'failed_tasks': len([h for h in self.history if h.get('status') == 'failed']),
            'functions_by_date': self._group_functions_by_date(functions)
        }

    def visualize_workflow(self, task_description: str, format: str = "mermaid",
                          export_path: str = None, show_popup: bool = False) -> Dict[str, Any]:
        """
        Generate and optionally export workflow visualization.

        Args:
            task_description: Description of the task to visualize
            format: Visualization format (mermaid, ascii, json, html)
            export_path: Optional path to export the visualization
            show_popup: Whether to show GUI popup window

        Returns:
            Dictionary containing visualization result
        """
        try:
            logger.info(f"🎨 Generating workflow visualization for: {task_description}")

            # Determine workflow type
            if not ADVANCED_VISUALIZER_AVAILABLE:
                from tools.simple_visualizer import determine_workflow_type
                workflow_type = determine_workflow_type(task_description)
            else:
                workflow_type = "evo" if self._needs_evo_workflow_sync(task_description) else "standard"

            # Show GUI popup if requested
            if show_popup and GUI_VISUALIZER_AVAILABLE:
                success = show_workflow_popup(task_description, workflow_type)
                if success:
                    return {
                        'success': True,
                        'visualization': "GUI popup window opened",
                        'format': 'gui_popup',
                        'workflow_type': workflow_type,
                        'popup_shown': True
                    }
                else:
                    logger.warning("Failed to show GUI popup, falling back to text visualization")

            # Generate workflow without executing
            workflow_result = None

            if self._needs_evo_workflow_sync(task_description):
                # Generate EvoWorkflow
                import asyncio
                workflow_result = asyncio.run(
                    self.evo_workflow_manager.generate_workflow_only(task_description)
                )

                if workflow_result.get("success"):
                    # Create a mock workflow object for visualization
                    from workflow.evo_core_algorithms import EvoWorkflowGraph, EvoWorkflowNode

                    workflow = EvoWorkflowGraph(goal=task_description)

                    # Add nodes from the result
                    for node_detail in workflow_result.get("node_details", []):
                        node = EvoWorkflowNode(
                            name=node_detail["name"],
                            description=node_detail["description"],
                            reason=f"Required for {task_description}"
                        )
                        workflow.add_node(node)

                    # Generate visualization
                    if format.lower() == "mermaid":
                        visualization = visualize_workflow_mermaid(workflow, include_details=True)
                    elif format.lower() == "ascii":
                        from tools.workflow_visualizer import visualize_workflow_ascii
                        visualization = visualize_workflow_ascii(workflow)
                    else:
                        from tools.workflow_visualizer import VisualizationFormat, VisualizationOptions
                        viz_format = VisualizationFormat(format.lower())
                        options = VisualizationOptions(format=viz_format, include_details=True)
                        visualization = self.workflow_visualizer.visualize_workflow(workflow, options)

                    # Export if requested
                    if export_path:
                        with open(export_path, 'w', encoding='utf-8') as f:
                            f.write(visualization)
                        logger.info(f"✅ Visualization exported to: {export_path}")

                    return {
                        'success': True,
                        'visualization': visualization,
                        'format': format,
                        'workflow_type': 'evo_workflow',
                        'node_count': len(workflow.nodes),
                        'exported_to': export_path
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Failed to generate workflow: {workflow_result.get('error')}"
                    }

            else:
                # For simpler tasks, use the simple visualizer
                if format.lower() == "mermaid":
                    from tools.simple_visualizer import generate_mermaid_from_description
                    visualization = generate_mermaid_from_description(task_description, workflow_type)
                elif format.lower() == "ascii":
                    from tools.simple_visualizer import generate_ascii_from_description
                    visualization = generate_ascii_from_description(task_description, workflow_type)
                else:
                    # Fallback to basic text
                    visualization = f"""
# Task Visualization: {task_description}

## Approach: Standard Processing
1. 📋 Task Analysis (Planner Agent)
2. 🔍 Function Search (Function Registry)
3. 🔧 Function Creation (if needed)
4. ✅ Execution & Validation
5. 📝 Result Recording

## Processing Flow:
```
Task Input → Analysis → Search Functions → [Create New Function] → Execute → Result
```

Format: {format.upper()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """

                if export_path:
                    with open(export_path, 'w', encoding='utf-8') as f:
                        f.write(visualization)
                    logger.info(f"✅ Basic visualization exported to: {export_path}")

                return {
                    'success': True,
                    'visualization': visualization,
                    'format': format,
                    'workflow_type': 'standard_processing',
                    'exported_to': export_path
                }

        except Exception as e:
            error_msg = f"Workflow visualization failed: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _group_functions_by_date(self, functions: List[Dict]) -> Dict[str, int]:
        """Group functions by creation date."""
        date_counts = {}
        for func in functions:
            date = func.get('created_at', '')[:10]  # Get date part
            date_counts[date] = date_counts.get(date, 0) + 1
        return date_counts


def main():
    """Main function to run the self-expanding agent system."""
    import argparse

    parser = argparse.ArgumentParser(description="AutoGen Self-Expanding Agent System with Learning Memory")
    parser.add_argument("--task", type=str, help="Task to process")
    parser.add_argument("--session-id", type=str, help="Session ID for context preservation")
    parser.add_argument("--visualize", action="store_true", help="Generate workflow visualization")
    parser.add_argument("--viz-format", choices=["mermaid", "ascii", "json"], default="mermaid",
                       help="Visualization format")
    parser.add_argument("--export-viz", type=str, help="Export visualization to file")
    parser.add_argument("--popup", action="store_true", help="Show visualization in GUI popup window")
    parser.add_argument("--dashboard", type=str, help="Export workflow dashboard to HTML file")
    args = parser.parse_args()

    print(f"🚀 AutoGen Self-Expanding Agent System with Learning Memory")
    print("=" * 60)

    try:
        # Initialize the system
        system = SelfExpandingAgentSystem()

        # Show system stats
        stats = system.get_system_stats()
        print(f"\n📊 System Status:")
        print(f"  Total Functions: {stats['total_functions']}")
        print(f"  Tasks Processed: {stats['total_tasks_processed']}")
        print(f"  Success Rate: {stats['successful_tasks']}/{stats['total_tasks_processed']}")

        # Process task or generate visualization
        task = args.task or "Create an email validator function"

        if args.visualize:
            print(f"\n🎨 Generating Workflow Visualization: {task}")
            viz_result = system.visualize_workflow(task, args.viz_format, args.export_viz, args.popup)

            if viz_result.get('success'):
                print(f"\n✅ Visualization Generated Successfully!")
                print(f"  Format: {viz_result['format']}")
                print(f"  Workflow Type: {viz_result['workflow_type']}")
                if viz_result.get('node_count'):
                    print(f"  Nodes: {viz_result['node_count']}")
                if viz_result.get('exported_to'):
                    print(f"  Exported to: {viz_result['exported_to']}")

                # Display visualization if it's text-based
                if args.viz_format in ['ascii', 'mermaid'] and not args.export_viz:
                    print(f"\n📊 {args.viz_format.upper()} Visualization:")
                    print("-" * 60)
                    print(viz_result['visualization'])
            else:
                print(f"\n❌ Visualization Failed: {viz_result.get('error', 'Unknown error')}")

        else:
            print(f"\n🎯 Processing Task: {task}")
            result = system.process_task(task, args.session_id)

            # Display results
            if result.get('success'):
                print(f"\n✅ Task Completed Successfully!")
                if 'functions_used' in result:
                    print(f"  Functions Used: {', '.join(result['functions_used'])}")
                if 'execution_type' in result:
                    print(f"  Execution Type: {result['execution_type']}")
                if 'function_created' in result:
                    print(f"  New Function Created: {result['function_created']}")
            else:
                print(f"\n❌ Task Failed: {result.get('error', 'Unknown error')}")

        # Export dashboard if requested
        if args.dashboard:
            print(f"\n📊 Exporting Workflow Dashboard...")
            if system.workflow_dashboard.export_dashboard(args.dashboard):
                print(f"✅ Dashboard exported to: {args.dashboard}")
            else:
                print(f"❌ Failed to export dashboard to: {args.dashboard}")

        # Show updated stats
        stats = system.get_system_stats()
        print(f"\n📊 Updated System Status:")
        print(f"  Total Functions: {stats['total_functions']}")
        print(f"  Tasks Processed: {stats['total_tasks_processed']}")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")
        return 1

    return 0


# Add multi-file generation methods to SelfExpandingAgentSystem class
def _add_multi_file_methods():
    """Add multi-file generation methods to the main class."""

    def _needs_multi_file_generation_sync(self, task_description: str) -> bool:
        """
        Determine if a task requires multi-file generation.

        Args:
            task_description: Description of the task

        Returns:
            True if multi-file generation is needed
        """
        # Keywords that indicate multi-file projects
        multi_file_keywords = [
            'website', 'webpage', 'web page', 'html', 'css', 'javascript',
            'api', 'rest api', 'web service', 'backend', 'frontend',
            'project', 'application', 'app', 'dashboard', 'portfolio',
            'blog', 'documentation', 'docs', 'guide', 'manual',
            'automation script', 'workflow', 'pipeline',
            'data analysis', 'jupyter', 'notebook'
        ]

        task_lower = task_description.lower()

        # Check for explicit multi-file indicators
        for keyword in multi_file_keywords:
            if keyword in task_lower:
                return True

        # Check for phrases that suggest complexity
        complex_phrases = [
            'create a', 'build a', 'develop a', 'generate a',
            'make a', 'design a', 'set up a'
        ]

        for phrase in complex_phrases:
            if phrase in task_lower and any(kw in task_lower for kw in multi_file_keywords):
                return True

        return False

    def _process_multi_file_task_sync(self, task_description: str) -> Dict[str, Any]:
        """
        Process a task that requires multi-file generation.

        Args:
            task_description: Description of the task

        Returns:
            Dictionary containing the result
        """
        try:
            logger.info("🚀 Processing multi-file generation task")

            # Generate the multi-file project (using asyncio.run for sync context)
            import asyncio
            result = asyncio.run(self.multi_file_agent.generate_multi_file_project(task_description))

            if result['success']:
                return {
                    'success': True,
                    'result': result['instructions'],
                    'project_path': result['project_path'],
                    'generated_files': result['generated_files'],
                    'execution_type': 'multi_file_generation',
                    'metadata': result['metadata']
                }
            else:
                # Fallback to standard processing if multi-file generation fails
                logger.warning("Multi-file generation failed, falling back to standard processing")
                return self._process_task_standard(task_description)

        except Exception as e:
            logger.error(f"Multi-file task processing failed: {e}")
            # Fallback to standard processing
            return self._process_task_standard(task_description)

    def _needs_evo_workflow_sync(self, task_description: str) -> bool:
        """
        Determine if a task requires EvoAgentX-style workflow.

        Args:
            task_description: Description of the task

        Returns:
            True if EvoAgentX workflow is needed
        """
        # Keywords that indicate complex multi-agent workflows
        evo_workflow_keywords = [
            'workflow', 'multi-step', 'complex', 'comprehensive', 'system',
            'framework', 'architecture', 'multi-agent', 'coordinate', 'manage',
            'analyze and create', 'process and validate', 'research and report',
            'plan and execute', 'design and implement', 'optimize and improve'
        ]

        task_lower = task_description.lower()

        # Check for explicit workflow indicators
        for keyword in evo_workflow_keywords:
            if keyword in task_lower:
                return True

        # Check for complex task patterns
        complex_patterns = [
            'analyze', 'create', 'validate',  # Multiple verbs suggest workflow
            'step by step', 'multiple stages', 'comprehensive solution',
            'end-to-end', 'full process', 'complete system'
        ]

        pattern_count = sum(1 for pattern in complex_patterns if pattern in task_lower)

        # If multiple patterns detected, likely needs workflow
        return pattern_count >= 2

    def _process_evo_workflow_task_sync(self, task_description: str) -> Dict[str, Any]:
        """
        Process a task using EvoAgentX-style workflow.

        Args:
            task_description: Description of the task

        Returns:
            Dictionary containing the result
        """
        try:
            logger.info("🌟 Processing EvoAgentX-style workflow task")

            # Use asyncio.run for sync context
            import asyncio
            result = asyncio.run(
                self.evo_workflow_manager.create_and_execute_workflow(task_description)
            )

            if result['success']:
                return {
                    'success': True,
                    'result': result['output'],
                    'workflow_id': result['workflow_id'],
                    'workflow_type': result['workflow_type'],
                    'execution_time': result['execution_time'],
                    'execution_type': 'evo_workflow',
                    'metadata': result['metadata']
                }
            else:
                # Fallback to standard processing if workflow fails
                logger.warning("EvoAgentX workflow failed, falling back to standard processing")
                return self._process_task_standard(task_description)

        except Exception as e:
            logger.error(f"EvoAgentX workflow processing failed: {e}")
            # Fallback to standard processing
            return self._process_task_standard(task_description)

    # Add methods to the class
    SelfExpandingAgentSystem._needs_multi_file_generation_sync = _needs_multi_file_generation_sync
    SelfExpandingAgentSystem._process_multi_file_task_sync = _process_multi_file_task_sync
    SelfExpandingAgentSystem._needs_evo_workflow_sync = _needs_evo_workflow_sync
    SelfExpandingAgentSystem._process_evo_workflow_task_sync = _process_evo_workflow_task_sync

# Apply the methods
_add_multi_file_methods()


if __name__ == "__main__":
    exit(main())
