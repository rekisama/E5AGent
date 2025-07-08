"""
Learning Memory Integration Module

This module integrates the learning memory system with existing AutoGen agents
and LangGraph workflows to enable automatic learning and recommendation.

Key Features:
- Automatic task pattern recognition during agent conversations
- Real-time solution recommendation based on historical success
- Knowledge graph updates from function usage patterns
- Integration with existing FunctionTools and workflow systems
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from ..memory.learning_memory_system import (
    LearningMemoryManager, 
    RecommendationResult,
    TaskPattern
)
from ..config import Config
from .function_registry import FunctionRegistry
from .function_tools import FunctionTools

logger = logging.getLogger(__name__)

class LearningMemoryIntegration:
    """Integrates learning memory system with existing agent workflows."""
    
    def __init__(self, function_tools: FunctionTools, function_registry: FunctionRegistry):
        self.function_tools = function_tools
        self.function_registry = function_registry
        self.learning_manager = LearningMemoryManager(Config.MEMORY_DIR)
        self.enabled = Config.LEARNING_MEMORY_ENABLED
        
        if self.enabled:
            logger.info("🧠 Learning Memory System initialized and enabled")
        else:
            logger.info("⚠️  Learning Memory System disabled via configuration")
    
    def enhance_task_analysis(self, task_description: str) -> Dict[str, Any]:
        """Enhance task analysis with learning memory insights."""
        if not self.enabled:
            return {"enhanced": False, "reason": "Learning memory disabled"}
        
        try:
            # Get recommendations from learning memory
            recommendations = self.learning_manager.get_recommendations(
                task_description,
                available_functions=list(self.function_registry.get_all_functions().keys())
            )
            
            # Get similar patterns for context
            similar_patterns = self.learning_manager.pattern_recognizer.get_similar_patterns(
                task_description, 
                limit=Config.MAX_SIMILAR_PATTERNS
            )
            
            # Analyze task complexity based on patterns
            complexity_analysis = self._analyze_task_complexity(task_description, similar_patterns)
            
            enhancement = {
                "enhanced": True,
                "recommendations": {
                    "functions": recommendations.recommended_functions,
                    "confidence": recommendations.confidence_score,
                    "reasoning": recommendations.reasoning,
                    "estimated_success_rate": recommendations.estimated_success_rate,
                    "alternatives": recommendations.alternative_approaches
                },
                "similar_cases": recommendations.similar_cases,
                "complexity_analysis": complexity_analysis,
                "learning_insights": {
                    "pattern_count": len(similar_patterns),
                    "has_historical_data": len(similar_patterns) > 0,
                    "recommended_approach": self._get_recommended_approach(recommendations)
                }
            }
            
            logger.info(f"Enhanced task analysis for: {task_description[:50]}...")
            return enhancement
            
        except Exception as e:
            logger.error(f"Error enhancing task analysis: {e}")
            return {"enhanced": False, "error": str(e)}
    
    def _analyze_task_complexity(self, task_description: str, 
                               similar_patterns: List[Tuple[TaskPattern, float]]) -> Dict[str, Any]:
        """Analyze task complexity based on historical patterns."""
        if not similar_patterns:
            return {
                "level": "unknown",
                "confidence": 0.0,
                "reasoning": "No historical data available"
            }
        
        # Calculate average success rate of similar patterns
        avg_success_rate = sum(pattern.success_rate for pattern, _ in similar_patterns) / len(similar_patterns)
        
        # Calculate complexity based on success rate and pattern diversity
        pattern_types = set(pattern.pattern_type for pattern, _ in similar_patterns)
        type_diversity = len(pattern_types)
        
        if avg_success_rate > 0.8 and type_diversity <= 2:
            complexity = "simple"
            confidence = 0.9
            reasoning = f"High success rate ({avg_success_rate:.2f}) with consistent patterns"
        elif avg_success_rate > 0.6:
            complexity = "moderate"
            confidence = 0.7
            reasoning = f"Moderate success rate ({avg_success_rate:.2f}) with some variation"
        else:
            complexity = "complex"
            confidence = 0.8
            reasoning = f"Low success rate ({avg_success_rate:.2f}) indicating complexity"
        
        return {
            "level": complexity,
            "confidence": confidence,
            "reasoning": reasoning,
            "avg_success_rate": avg_success_rate,
            "pattern_diversity": type_diversity
        }
    
    def _get_recommended_approach(self, recommendations: RecommendationResult) -> str:
        """Get recommended approach based on recommendations."""
        if recommendations.confidence_score > Config.MIN_RECOMMENDATION_CONFIDENCE:
            if len(recommendations.recommended_functions) == 1:
                return "use_existing_function"
            elif len(recommendations.recommended_functions) > 1:
                return "compose_functions"
            else:
                return "create_new_function"
        else:
            return "explore_and_create"
    
    def record_task_execution(self, task_description: str, functions_used: List[str], 
                            success: bool, execution_time: float = 0.0,
                            error_message: str = None, additional_context: Dict[str, Any] = None):
        """Record task execution results for learning."""
        if not self.enabled:
            return
        
        try:
            # Process task completion in learning memory
            metadata = additional_context or {}
            if error_message:
                metadata['error_message'] = error_message
            
            self.learning_manager.process_task_completion(
                task_description=task_description,
                functions_used=functions_used,
                success=success,
                execution_time=execution_time,
                additional_metadata=metadata
            )
            
            # Update function usage statistics in registry
            for func_name in functions_used:
                if func_name in self.function_registry.get_all_functions():
                    # This could be enhanced to track success/failure rates per function
                    pass
            
            logger.info(f"Recorded task execution: {task_description[:50]}... -> {success}")
            
        except Exception as e:
            logger.error(f"Error recording task execution: {e}")
    
    def get_function_recommendations(self, task_description: str, 
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get function recommendations with enhanced context."""
        if not self.enabled:
            return {"recommendations": [], "reason": "Learning memory disabled"}
        
        try:
            # Get available functions
            available_functions = list(self.function_registry.get_all_functions().keys())
            
            # Get recommendations
            recommendations = self.learning_manager.get_recommendations(
                task_description, 
                available_functions
            )
            
            # Enhance with function metadata
            enhanced_recommendations = []
            for func_name in recommendations.recommended_functions:
                func_info = self.function_registry.get_function_info(func_name)
                if func_info:
                    enhanced_recommendations.append({
                        "name": func_name,
                        "description": func_info.get("description", ""),
                        "parameters": func_info.get("parameters", {}),
                        "tags": func_info.get("tags", []),
                        "success_rate": func_info.get("success_rate", 0.0)
                    })
            
            # Get related functions from knowledge graph
            related_functions = []
            for func_name in recommendations.recommended_functions:
                related = self.learning_manager.knowledge_graph.get_related_functions(
                    func_name, 
                    min_strength=Config.MIN_RELATIONSHIP_STRENGTH
                )
                for related_func, relationship in related[:3]:  # Top 3 related
                    if related_func not in recommendations.recommended_functions:
                        func_info = self.function_registry.get_function_info(related_func)
                        if func_info:
                            related_functions.append({
                                "name": related_func,
                                "relationship": relationship.relationship_type,
                                "strength": relationship.strength,
                                "description": func_info.get("description", "")
                            })
            
            return {
                "primary_recommendations": enhanced_recommendations,
                "related_functions": related_functions,
                "confidence": recommendations.confidence_score,
                "reasoning": recommendations.reasoning,
                "estimated_success_rate": recommendations.estimated_success_rate,
                "similar_cases": recommendations.similar_cases,
                "alternatives": recommendations.alternative_approaches
            }
            
        except Exception as e:
            logger.error(f"Error getting function recommendations: {e}")
            return {"error": str(e), "recommendations": []}
    
    def learn_from_user_feedback(self, task_description: str, 
                               recommended_functions: List[str],
                               user_satisfaction: float,  # 0.0 to 1.0
                               actual_functions_used: List[str] = None,
                               execution_time: float = 0.0):
        """Learn from user feedback on recommendations."""
        if not self.enabled:
            return
        
        try:
            # Convert satisfaction to success boolean
            success = user_satisfaction >= 0.7
            
            # Use actual functions if provided, otherwise use recommended
            functions_to_learn = actual_functions_used or recommended_functions
            
            # Record feedback
            self.learning_manager.recommendation_engine.learn_from_feedback(
                task_description=task_description,
                recommended_functions=functions_to_learn,
                actual_success=success,
                execution_time=execution_time
            )
            
            logger.info(f"Learned from user feedback: satisfaction={user_satisfaction}")
            
        except Exception as e:
            logger.error(f"Error learning from user feedback: {e}")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning memory system."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            stats = self.learning_manager.get_learning_statistics()
            
            # Add integration-specific insights
            insights = {
                "enabled": True,
                "statistics": stats,
                "integration_status": {
                    "functions_tracked": len(self.function_registry.get_all_functions()),
                    "learning_active": True,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                },
                "recommendations": {
                    "total_patterns": stats.get("task_patterns_count", 0),
                    "successful_solutions": len(stats.get("most_successful_solutions", [])),
                    "knowledge_connections": stats.get("knowledge_graph_edges", 0)
                }
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {"enabled": True, "error": str(e)}
    
    async def async_enhance_task_analysis(self, task_description: str) -> Dict[str, Any]:
        """Async version of enhance_task_analysis for use in async workflows."""
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.enhance_task_analysis, task_description)
    
    async def async_record_task_execution(self, task_description: str, functions_used: List[str], 
                                        success: bool, execution_time: float = 0.0,
                                        error_message: str = None, additional_context: Dict[str, Any] = None):
        """Async version of record_task_execution for use in async workflows."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.record_task_execution,
            task_description, functions_used, success, execution_time, error_message, additional_context
        )


class LearningMemoryMiddleware:
    """Middleware for automatic learning memory integration in agent conversations."""
    
    def __init__(self, learning_integration: LearningMemoryIntegration):
        self.learning_integration = learning_integration
        self.conversation_context = {}
    
    def pre_conversation_hook(self, conversation_id: str, initial_message: str) -> Dict[str, Any]:
        """Hook called before starting a conversation."""
        if not self.learning_integration.enabled:
            return {}
        
        try:
            # Analyze the initial task
            enhancement = self.learning_integration.enhance_task_analysis(initial_message)
            
            # Store context for this conversation
            self.conversation_context[conversation_id] = {
                "initial_message": initial_message,
                "enhancement": enhancement,
                "start_time": datetime.now(timezone.utc),
                "functions_used": [],
                "success": None
            }
            
            return enhancement
            
        except Exception as e:
            logger.error(f"Error in pre-conversation hook: {e}")
            return {}
    
    def post_conversation_hook(self, conversation_id: str, success: bool, 
                             functions_used: List[str] = None, error_message: str = None):
        """Hook called after completing a conversation."""
        if not self.learning_integration.enabled or conversation_id not in self.conversation_context:
            return
        
        try:
            context = self.conversation_context[conversation_id]
            execution_time = (datetime.now(timezone.utc) - context["start_time"]).total_seconds()
            
            # Record the task execution
            self.learning_integration.record_task_execution(
                task_description=context["initial_message"],
                functions_used=functions_used or context["functions_used"],
                success=success,
                execution_time=execution_time,
                error_message=error_message,
                additional_context={"conversation_id": conversation_id}
            )
            
            # Clean up context
            del self.conversation_context[conversation_id]
            
        except Exception as e:
            logger.error(f"Error in post-conversation hook: {e}")
    
    def function_usage_hook(self, conversation_id: str, function_name: str):
        """Hook called when a function is used during conversation."""
        if conversation_id in self.conversation_context:
            self.conversation_context[conversation_id]["functions_used"].append(function_name)
