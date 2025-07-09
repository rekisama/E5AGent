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
from tools.function_tools import get_function_tools
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
            # 1. Enhanced task analysis with learning memory (if available)
            if self.learning_integration:
                enhancement = self.learning_integration.enhance_task_analysis(task_description)

                # 2. Intelligent decision making based on learning insights
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

                # 3. Record execution for learning
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

            # Execute function (simplified - in practice would need parameter mapping)
            result = self.function_tools.execute_function(function_name, task_description)

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
            functions_info = []

            for func_name in function_names:
                func_info = self.function_tools.registry.get_function_info(func_name)
                if func_info:
                    functions_info.append(func_info)

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

        # Process task
        task = args.task or "Create an email validator function"
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


if __name__ == "__main__":
    exit(main())
