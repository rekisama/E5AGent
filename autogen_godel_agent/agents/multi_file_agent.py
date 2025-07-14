"""
Multi-File Generation Agent

This agent specializes in generating complete multi-file projects
using LLM-based content generation. It integrates with the existing
AutoGen system to handle complex tasks that require multiple files.

Key Features:
- Intelligent task classification
- Dynamic project structure planning
- LLM-based content generation for all file types
- Integration with existing function registry
- Comprehensive error handling and validation
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import autogen
try:
    from autogen_godel_agent.tools.multi_file_generator import get_multi_file_generator, ProjectType, GenerationResult
    from autogen_godel_agent.config import Config
except ImportError:
    try:
        from tools.multi_file_generator import get_multi_file_generator, ProjectType, GenerationResult
        from config import Config
    except ImportError:
        # Mock imports for testing
        class ProjectType:
            WEBPAGE = "webpage"
        class GenerationResult:
            def __init__(self):
                self.success = False
        def get_multi_file_generator(config):
            return None
        class Config:
            pass

logger = logging.getLogger(__name__)


class MultiFileGeneratorAgent:
    """
    AutoGen agent specialized in multi-file project generation.
    
    This agent can handle complex tasks that require generating
    multiple files, such as websites, APIs, documentation, etc.
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.multi_file_generator = get_multi_file_generator(llm_config)
        
        # Create AutoGen agent
        self.agent = autogen.AssistantAgent(
            name="MultiFileGenerator",
            system_message=self._get_system_message(),
            llm_config=llm_config,
            function_map={
                "generate_multi_file_project": self.generate_multi_file_project,
                "classify_project_type": self.classify_project_type,
                "estimate_project_complexity": self.estimate_project_complexity
            }
        )
        
        logger.info("✅ MultiFileGeneratorAgent initialized")
    
    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        return """You are a Multi-File Project Generator Agent, specialized in creating complete projects with multiple files.

Your capabilities include:
- Generating websites (HTML, CSS, JavaScript)
- Creating API projects (Python FastAPI, Flask, etc.)
- Building data analysis projects (Python, Jupyter notebooks)
- Creating documentation projects (Markdown, guides)
- Generating automation scripts and workflows

When you receive a task that requires multiple files:
1. Analyze the task to determine if it needs multi-file generation
2. Classify the project type (webpage, API, data analysis, etc.)
3. Use the generate_multi_file_project function to create the complete project
4. Provide clear instructions on how to use the generated project

You work intelligently with LLM-based content generation to create
high-quality, functional code and documentation.

Always be helpful and provide detailed explanations of what you've generated."""
    
    async def generate_multi_file_project(self, task_description: str, 
                                        output_directory: str = None) -> Dict[str, Any]:
        """
        Generate a complete multi-file project.
        
        Args:
            task_description: Description of the project to generate
            output_directory: Optional output directory
            
        Returns:
            Dictionary with generation results
        """
        try:
            logger.info(f"🚀 MultiFileAgent generating project: {task_description}")
            
            # Generate the project
            result = await self.multi_file_generator.generate_project(
                task_description, output_directory
            )
            
            # Format response
            response = {
                "success": result.success,
                "project_path": result.project_path,
                "generated_files": result.generated_files,
                "errors": result.errors,
                "warnings": result.warnings,
                "metadata": result.metadata,
                "instructions": self._generate_usage_instructions(result)
            }
            
            if result.success:
                logger.info(f"✅ Successfully generated {len(result.generated_files)} files")
            else:
                logger.warning(f"⚠️ Generation completed with {len(result.errors)} errors")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Multi-file generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_path": "",
                "generated_files": [],
                "errors": [str(e)]
            }
    
    async def classify_project_type(self, task_description: str) -> Dict[str, Any]:
        """
        Classify the project type for a given task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Dictionary with classification results
        """
        try:
            project_type, confidence = await self.multi_file_generator.classifier.classify_with_llm(
                task_description
            )
            
            return {
                "project_type": project_type.value,
                "confidence": confidence,
                "description": self._get_project_type_description(project_type)
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "project_type": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def estimate_project_complexity(self, task_description: str) -> Dict[str, Any]:
        """
        Estimate the complexity of a project.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Dictionary with complexity estimation
        """
        try:
            # Classify project type first
            project_type, confidence = await self.multi_file_generator.classifier.classify_with_llm(
                task_description
            )
            
            # Plan structure to estimate complexity
            structure = self.multi_file_generator.planner.plan_project_structure(
                task_description, project_type
            )
            
            # Calculate complexity metrics
            complexity_score = self._calculate_complexity_score(structure, task_description)
            
            return {
                "complexity_level": self._get_complexity_level(complexity_score),
                "complexity_score": complexity_score,
                "estimated_files": len(structure.files),
                "estimated_directories": len(structure.directories),
                "project_type": project_type.value,
                "estimated_time_minutes": self._estimate_generation_time(complexity_score)
            }
            
        except Exception as e:
            logger.error(f"Complexity estimation failed: {e}")
            return {
                "complexity_level": "unknown",
                "error": str(e)
            }
    
    def _generate_usage_instructions(self, result: GenerationResult) -> str:
        """Generate usage instructions for the generated project."""
        
        if not result.success:
            return "Project generation failed. Please check the errors."
        
        project_type = result.metadata.get("project_type", "unknown")
        
        if project_type == "webpage":
            return f"""
🌐 Website Generated Successfully!

📁 Project Location: {result.project_path}

🚀 How to use:
1. Open 'index.html' in your web browser
2. Customize content in the HTML files
3. Modify styles in 'css/styles.css'
4. Add functionality in 'js/script.js'

📝 Files generated: {', '.join(result.generated_files)}

💡 Tip: You can serve this locally using:
   python -m http.server 8000
   Then visit: http://localhost:8000
"""
        
        elif project_type == "api_project":
            return f"""
🔌 API Project Generated Successfully!

📁 Project Location: {result.project_path}

🚀 How to use:
1. Install dependencies: pip install -r requirements.txt
2. Run the server: python main.py
3. Visit API docs: http://localhost:8000/docs
4. Test endpoints: http://localhost:8000

📝 Files generated: {', '.join(result.generated_files)}

💡 Tip: The API includes automatic documentation and health checks.
"""
        
        elif project_type == "data_analysis":
            return f"""
📊 Data Analysis Project Generated Successfully!

📁 Project Location: {result.project_path}

🚀 How to use:
1. Install dependencies: pip install -r requirements.txt
2. Run analysis: python analysis.py
3. Check output in 'output/' directory
4. Modify data processing in the Python files

📝 Files generated: {', '.join(result.generated_files)}

💡 Tip: Use Jupyter notebooks for interactive analysis.
"""
        
        else:
            return f"""
✅ Project Generated Successfully!

📁 Project Location: {result.project_path}
📝 Files generated: {', '.join(result.generated_files)}

Please refer to the README.md file for specific usage instructions.
"""
    
    def _get_project_type_description(self, project_type: ProjectType) -> str:
        """Get description for a project type."""
        descriptions = {
            ProjectType.WEBPAGE: "Static or interactive website with HTML, CSS, and JavaScript",
            ProjectType.API_PROJECT: "REST API backend with endpoints and documentation",
            ProjectType.DATA_ANALYSIS: "Data processing and analysis project with Python",
            ProjectType.DOCUMENTATION: "Documentation project with guides and manuals",
            ProjectType.AUTOMATION_SCRIPT: "Automation scripts and workflows",
            ProjectType.DESKTOP_APP: "Desktop application project",
            ProjectType.MOBILE_APP: "Mobile application project"
        }
        return descriptions.get(project_type, "Unknown project type")
    
    def _calculate_complexity_score(self, structure, task_description: str) -> float:
        """Calculate complexity score for a project."""
        score = 0.0
        
        # Base score from file count
        score += len(structure.files) * 0.1
        score += len(structure.directories) * 0.05
        
        # Complexity keywords in description
        complex_keywords = [
            'database', 'authentication', 'api', 'responsive', 'interactive',
            'real-time', 'dashboard', 'admin', 'user management', 'payment'
        ]
        
        for keyword in complex_keywords:
            if keyword in task_description.lower():
                score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_complexity_level(self, score: float) -> str:
        """Get complexity level from score."""
        if score < 0.3:
            return "simple"
        elif score < 0.6:
            return "moderate"
        else:
            return "complex"
    
    def _estimate_generation_time(self, complexity_score: float) -> int:
        """Estimate generation time in minutes."""
        base_time = 2  # 2 minutes base
        return int(base_time + (complexity_score * 8))  # Up to 10 minutes for complex projects


# Factory function
def get_multi_file_agent(llm_config: Dict[str, Any]) -> MultiFileGeneratorAgent:
    """Get a MultiFileGeneratorAgent instance."""
    return MultiFileGeneratorAgent(llm_config)
