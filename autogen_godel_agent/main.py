"""
Main entry point for AutoGen Self-Expanding Agent System with LangGraph Integration.

This system can automatically generate, test, and register new functions
when encountering tasks it cannot complete with existing capabilities.

Now enhanced with LangGraph workflow orchestration for better state management,
conditional routing, and error handling.
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
from tools.function_tools import get_function_tools

logger = logging.getLogger(__name__)

# Import LangGraph workflow components
try:
    from workflows.orchestrator import get_workflow_orchestrator, WorkflowOrchestrator
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangGraph workflow components not available: {e}")
    get_workflow_orchestrator = None
    WorkflowOrchestrator = None
    LANGGRAPH_AVAILABLE = False


class SelfExpandingAgentSystem:
    """
    Main orchestrator for the self-expanding agent system with LangGraph integration.

    This class now provides both the legacy AutoGen-based workflow and the new
    LangGraph-based workflow for better state management and conditional routing.
    """

    def __init__(self, use_langgraph: bool = True):
        """
        Initialize the system.

        Args:
            use_langgraph: Whether to use LangGraph workflow (recommended) or legacy workflow
        """
        # Validate configuration
        Config.validate_config()

        # Get LLM configuration
        self.llm_config = Config.get_llm_config()
        self.use_langgraph = use_langgraph

        # Initialize function tools
        self.function_tools = get_function_tools()

        # Initialize workflow orchestrator if using LangGraph
        if self.use_langgraph and LANGGRAPH_AVAILABLE:
            self.workflow_orchestrator = get_workflow_orchestrator()
            logger.info("✅ Initialized with LangGraph workflow orchestration")
        elif self.use_langgraph and not LANGGRAPH_AVAILABLE:
            logger.warning("⚠️ LangGraph requested but not available, falling back to legacy mode")
            self.use_langgraph = False

        if not self.use_langgraph:
            # Legacy AutoGen setup
            self.planner_agent = TaskPlannerAgent(self.llm_config)
            self.creator_agent = FunctionCreatorAgent(self.llm_config)

            # Create user proxy agent
            self.user_proxy = autogen.UserProxyAgent(
                name="UserProxy",
                system_message="You are a user proxy that facilitates communication between the user and the AI agents.",
                human_input_mode="NEVER",  # Set to "ALWAYS" for interactive mode
                max_consecutive_auto_reply=10,
                is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                code_execution_config=False,
            )

            # Create group chat
            self.group_chat = autogen.GroupChat(
                agents=[self.user_proxy, self.planner_agent.agent, self.creator_agent.agent],
                messages=[],
                max_round=20,
                speaker_selection_method="round_robin",
            )

            # Create group chat manager
            self.manager = autogen.GroupChatManager(
                groupchat=self.group_chat,
                llm_config=self.llm_config,
            )

            logger.info("✅ Initialized with legacy AutoGen workflow")

        # Initialize history
        self.history = self._load_history()
    
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

        try:
            if self.use_langgraph:
                # Use LangGraph workflow orchestration
                logger.info("🚀 Using LangGraph workflow orchestration")
                result = self.workflow_orchestrator.process_task(task_description, session_id)

                # Add system metadata
                result['system_type'] = 'langgraph'
                result['workflow_stats'] = self.workflow_orchestrator.get_workflow_stats()

                return result
            else:
                # Use legacy AutoGen workflow
                logger.info("🔄 Using legacy AutoGen workflow")
                return self._process_task_legacy(task_description)

        except Exception as e:
            error_msg = f'Task processing failed: {str(e)}'
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'exception': type(e).__name__,
                'system_type': 'langgraph' if self.use_langgraph else 'legacy'
            }

    def _process_task_legacy(self, task_description: str) -> Dict[str, Any]:
        """
        Process task using the legacy AutoGen workflow.

        This method preserves the original workflow logic for backward compatibility.
        """
        print(f"\n🎯 Processing task: {task_description}")
        print("=" * 60)

        # Record task start
        task_record = {
            'timestamp': datetime.now().isoformat(),
            'task': task_description,
            'status': 'started',
            'functions_created': [],
            'result': None
        }

        try:
            # Step 1: Analyze task with planner agent
            print("\n📋 Step 1: Task Analysis")
            analysis = self.planner_agent.analyze_task(task_description)

            print(f"Found {len(analysis['existing_functions'])} existing function(s)")
            for func in analysis['existing_functions']:
                print(f"  - {func['name']}: {func['description']}")

            # Step 2: Determine if new functions are needed
            needs_new_function = self._needs_new_function(task_description, analysis['existing_functions'])

            if analysis['existing_functions'] and not needs_new_function:
                print("\n✅ Existing functions found - attempting to use them")
                # Try to execute with existing functions
                result = self._execute_with_existing_functions(task_description, analysis['existing_functions'])
                task_record['status'] = 'completed_with_existing'
                task_record['result'] = result
            else:
                print("\n🔧 No suitable functions found - creating new function")
                # Need to create new function
                result = self._create_and_execute_new_function(task_description)
                task_record['status'] = 'completed_with_new_function'
                task_record['result'] = result

            # Step 3: Start group chat for complex coordination if needed
            if not result.get('success', False):
                print("\n💬 Starting group chat for complex task resolution")
                result = self._run_group_chat(task_description)
                task_record['status'] = 'completed_with_group_chat'
                task_record['result'] = result

        except Exception as e:
            print(f"\n❌ Error processing task: {e}")
            task_record['status'] = 'failed'
            task_record['result'] = {'success': False, 'error': str(e)}

        # Save task record
        self.history.append(task_record)
        self._save_history()

        task_record['system_type'] = 'legacy'
        return task_record

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
        """Try to execute task with existing functions."""
        # This is a simplified implementation
        # In a real system, this would involve more sophisticated function composition
        
        if not functions:
            return {'success': False, 'message': 'No functions available'}
        
        # For now, just return info about available functions
        return {
            'success': True,
            'message': f'Found {len(functions)} relevant function(s)',
            'functions_used': [f['name'] for f in functions]
        }
    
    def _create_and_execute_new_function(self, task: str) -> Dict[str, Any]:
        """Create a new function for the task."""
        # Extract function specification from task
        spec = self._extract_function_spec(task)
        
        # Use creator agent to create function
        success, message, code = self.creator_agent.create_function(spec)
        
        if success:
            return {
                'success': True,
                'message': message,
                'function_created': spec['name']
            }
        else:
            return {
                'success': False,
                'message': message
            }
    
    def _extract_function_spec(self, task: str) -> Dict[str, Any]:
        """Extract function specification from task description."""
        # This is a simplified implementation
        # In a real system, this would use NLP or LLM to extract specifications
        
        # Generate a function name based on the task
        func_name = task.lower().replace(' ', '_').replace('-', '_')
        func_name = ''.join(c for c in func_name if c.isalnum() or c == '_')
        
        if 'email' in task.lower():
            func_name = 'validate_email'
        elif 'password' in task.lower():
            func_name = 'validate_password'
        elif 'url' in task.lower():
            func_name = 'validate_url'
        
        return {
            'name': func_name,
            'description': task,
            'parameters': [],
            'return_type': 'bool',
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
        return self.function_tools.list_all_functions()
    
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

    parser = argparse.ArgumentParser(description="AutoGen Self-Expanding Agent System with LangGraph")
    parser.add_argument("--legacy", action="store_true", help="Use legacy AutoGen workflow instead of LangGraph")
    parser.add_argument("--task", type=str, help="Task to process")
    parser.add_argument("--session-id", type=str, help="Session ID for context preservation")
    args = parser.parse_args()

    use_langgraph = not args.legacy
    workflow_type = "LangGraph" if use_langgraph else "Legacy AutoGen"

    print(f"🚀 AutoGen Self-Expanding Agent System ({workflow_type})")
    print("=" * 60)

    try:
        # Initialize the system
        system = SelfExpandingAgentSystem(use_langgraph=use_langgraph)

        # Show system stats
        stats = system.get_system_stats()
        print(f"\n📊 System Status:")
        print(f"  Workflow Type: {workflow_type}")
        print(f"  Total Functions: {stats['total_functions']}")
        print(f"  Tasks Processed: {stats['total_tasks_processed']}")
        print(f"  Success Rate: {stats['successful_tasks']}/{stats['total_tasks_processed']}")

        # Show workflow stats if using LangGraph
        if use_langgraph and hasattr(system, 'workflow_orchestrator'):
            workflow_stats = system.workflow_orchestrator.get_workflow_stats()
            print(f"  Workflow Success Rate: {workflow_stats['success_rate']}%")
            print(f"  Average Duration: {workflow_stats['average_duration']:.2f}s")

        # Process task
        task = args.task or "Create an email validator function"
        print(f"\n🎯 Processing Task: {task}")

        result = system.process_task(task, args.session_id)

        # Display results
        if result.get('status') == 'completed':
            print(f"\n✅ Task Completed Successfully!")
            if 'functions_used' in result:
                print(f"  Functions Used: {', '.join(result['functions_used'])}")
            if 'total_time' in result:
                print(f"  Execution Time: {result['total_time']:.2f}s")
            if 'tokens_used' in result:
                print(f"  Tokens Used: {result['tokens_used']}")
        else:
            print(f"\n❌ Task Failed: {result.get('error', 'Unknown error')}")

        # Show updated stats
        stats = system.get_system_stats()
        print(f"\n📊 Updated System Status:")
        print(f"  Total Functions: {stats['total_functions']}")
        print(f"  Tasks Processed: {stats['total_tasks_processed']}")

        # Show workflow summary if available
        if 'summary' in result:
            summary = result['summary']
            print(f"\n📋 Workflow Summary:")
            print(f"  Steps Executed: {len(result.get('steps_executed', []))}")
            print(f"  Retry Count: {summary.get('retry_count', 0)}")
            print(f"  Error Count: {summary.get('error_count', 0)}")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
