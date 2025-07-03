"""
Main entry point for AutoGen Self-Expanding Agent System.

This system can automatically generate, test, and register new functions
when encountering tasks it cannot complete with existing capabilities.
"""

import autogen
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from config import Config
from agents.planner_agent import TaskPlannerAgent
from agents.function_creator_agent import FunctionCreatorAgent
from tools.function_tools import get_function_tools


class SelfExpandingAgentSystem:
    """Main orchestrator for the self-expanding agent system."""
    
    def __init__(self):
        # Validate configuration
        if not Config.validate_config():
            raise ValueError("Invalid configuration. Please check your API keys.")
        
        # Get LLM configuration
        self.llm_config = Config.get_llm_config()
        
        # Initialize function tools
        self.function_tools = get_function_tools()
        
        # Initialize agents
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
    
    def process_task(self, task_description: str) -> Dict[str, Any]:
        """
        Process a user task through the self-expanding agent system.
        
        Args:
            task_description: Description of the task to be completed
            
        Returns:
            Dictionary containing the result and any new functions created
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
    print("🚀 AutoGen Self-Expanding Agent System")
    print("=" * 50)
    
    try:
        # Initialize the system
        system = SelfExpandingAgentSystem()
        
        # Show system stats
        stats = system.get_system_stats()
        print(f"\n📊 System Status:")
        print(f"  Total Functions: {stats['total_functions']}")
        print(f"  Tasks Processed: {stats['total_tasks_processed']}")
        print(f"  Success Rate: {stats['successful_tasks']}/{stats['total_tasks_processed']}")
        
        # Example task
        example_task = "Create an email validator function"
        print(f"\n🎯 Example Task: {example_task}")
        
        result = system.process_task(example_task)
        print(f"\n✅ Task Result: {result['status']}")
        
        # Show updated stats
        stats = system.get_system_stats()
        print(f"\n📊 Updated System Status:")
        print(f"  Total Functions: {stats['total_functions']}")
        print(f"  Tasks Processed: {stats['total_tasks_processed']}")
        
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
