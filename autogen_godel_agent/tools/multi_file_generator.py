"""
Multi-File Generation System

This module provides comprehensive multi-file generation capabilities
for complex tasks like web development, API projects, documentation, etc.

Key Features:
- Intelligent task type classification
- Dynamic file structure planning
- Template-based content generation
- Cross-file dependency management
- Validation and testing integration
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
import shutil

logger = logging.getLogger(__name__)


class ProjectType(Enum):
    """Supported project types for multi-file generation."""
    WEBPAGE = "webpage"
    API_PROJECT = "api_project"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENTATION = "documentation"
    AUTOMATION_SCRIPT = "automation_script"
    DESKTOP_APP = "desktop_app"
    MOBILE_APP = "mobile_app"


class FileType(Enum):
    """Supported file types."""
    HTML = "html"
    CSS = "css"
    JAVASCRIPT = "js"
    PYTHON = "py"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "md"
    TEXT = "txt"
    CONFIG = "config"


@dataclass
class FileSpec:
    """Specification for a single file to be generated."""
    name: str
    file_type: FileType
    content: str = ""
    dependencies: List[str] = field(default_factory=list)
    template_vars: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class ProjectStructure:
    """Complete project structure specification."""
    project_name: str
    project_type: ProjectType
    root_directory: str
    files: List[FileSpec] = field(default_factory=list)
    directories: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    build_commands: List[str] = field(default_factory=list)
    deployment_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of multi-file generation."""
    success: bool
    project_path: str
    generated_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskTypeClassifier:
    """Classifies tasks to determine appropriate project type."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        
        # Keywords for different project types
        self.type_keywords = {
            ProjectType.WEBPAGE: [
                'website', 'webpage', 'html', 'css', 'frontend', 'web page',
                'landing page', 'portfolio', 'blog', 'responsive design'
            ],
            ProjectType.API_PROJECT: [
                'api', 'rest api', 'fastapi', 'flask', 'django', 'backend',
                'web service', 'microservice', 'endpoint', 'server'
            ],
            ProjectType.DATA_ANALYSIS: [
                'data analysis', 'pandas', 'numpy', 'visualization', 'chart',
                'dashboard', 'jupyter', 'notebook', 'statistics', 'ml'
            ],
            ProjectType.DOCUMENTATION: [
                'documentation', 'readme', 'guide', 'manual', 'docs',
                'tutorial', 'help', 'wiki', 'specification'
            ],
            ProjectType.AUTOMATION_SCRIPT: [
                'automation', 'script', 'batch', 'workflow', 'task runner',
                'scheduler', 'cron', 'pipeline', 'process'
            ]
        }
    
    def classify_task(self, task_description: str) -> Tuple[ProjectType, float]:
        """
        Classify task and return project type with confidence score.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Tuple of (ProjectType, confidence_score)
        """
        task_lower = task_description.lower()
        scores = {}
        
        # Calculate keyword-based scores
        for project_type, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            if score > 0:
                scores[project_type] = score / len(keywords)
        
        if not scores:
            # Default to webpage for general requests
            return ProjectType.WEBPAGE, 0.3
        
        # Get the highest scoring type
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]
        
        return best_type, confidence
    
    async def classify_with_llm(self, task_description: str) -> Tuple[ProjectType, float]:
        """Use LLM for more sophisticated task classification."""
        
        classification_prompt = f"""
        Analyze this task and classify it into one of these project types:
        
        Task: "{task_description}"
        
        Project Types:
        1. WEBPAGE - Static or simple interactive websites
        2. API_PROJECT - REST APIs, web services, backends
        3. DATA_ANALYSIS - Data processing, visualization, analysis
        4. DOCUMENTATION - Guides, manuals, documentation
        5. AUTOMATION_SCRIPT - Scripts, workflows, automation
        6. DESKTOP_APP - Desktop applications
        7. MOBILE_APP - Mobile applications
        
        Respond with JSON:
        {{
            "project_type": "WEBPAGE",
            "confidence": 0.85,
            "reasoning": "Task involves creating HTML/CSS content"
        }}
        """
        
        try:
            # This would use the LLM to classify
            # For now, fall back to keyword-based classification
            return self.classify_task(task_description)
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return self.classify_task(task_description)


class ProjectPlanner:
    """Plans project structure based on task requirements."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.templates = self._load_project_templates()
    
    def _load_project_templates(self) -> Dict[ProjectType, Dict[str, Any]]:
        """Load project templates for different types."""
        return {
            ProjectType.WEBPAGE: {
                "directories": ["css", "js", "images", "assets"],
                "files": [
                    {"name": "index.html", "type": FileType.HTML, "required": True},
                    {"name": "css/styles.css", "type": FileType.CSS, "required": True},
                    {"name": "js/script.js", "type": FileType.JAVASCRIPT, "required": False},
                    {"name": "README.md", "type": FileType.MARKDOWN, "required": False}
                ],
                "dependencies": {}
            },
            ProjectType.API_PROJECT: {
                "directories": ["app", "tests", "docs", "config"],
                "files": [
                    {"name": "main.py", "type": FileType.PYTHON, "required": True},
                    {"name": "app/__init__.py", "type": FileType.PYTHON, "required": True},
                    {"name": "app/models.py", "type": FileType.PYTHON, "required": True},
                    {"name": "app/routes.py", "type": FileType.PYTHON, "required": True},
                    {"name": "requirements.txt", "type": FileType.TEXT, "required": True},
                    {"name": "README.md", "type": FileType.MARKDOWN, "required": True}
                ],
                "dependencies": {"fastapi": "latest", "uvicorn": "latest"}
            },
            ProjectType.DATA_ANALYSIS: {
                "directories": ["data", "notebooks", "src", "output"],
                "files": [
                    {"name": "analysis.py", "type": FileType.PYTHON, "required": True},
                    {"name": "data_processing.py", "type": FileType.PYTHON, "required": True},
                    {"name": "visualization.py", "type": FileType.PYTHON, "required": False},
                    {"name": "requirements.txt", "type": FileType.TEXT, "required": True},
                    {"name": "README.md", "type": FileType.MARKDOWN, "required": True}
                ],
                "dependencies": {"pandas": "latest", "numpy": "latest", "matplotlib": "latest"}
            }
        }
    
    def plan_project_structure(self, task_description: str, 
                             project_type: ProjectType) -> ProjectStructure:
        """
        Plan the complete project structure.
        
        Args:
            task_description: Description of the task
            project_type: Type of project to generate
            
        Returns:
            ProjectStructure with planned files and directories
        """
        template = self.templates.get(project_type, self.templates[ProjectType.WEBPAGE])
        
        # Generate project name from task
        project_name = self._generate_project_name(task_description)
        
        # Create project structure
        structure = ProjectStructure(
            project_name=project_name,
            project_type=project_type,
            root_directory=project_name,
            directories=template["directories"].copy(),
            dependencies=template["dependencies"].copy()
        )
        
        # Add files from template
        for file_template in template["files"]:
            file_spec = FileSpec(
                name=file_template["name"],
                file_type=file_template["type"]
            )
            structure.files.append(file_spec)
        
        return structure
    
    def _generate_project_name(self, task_description: str) -> str:
        """Generate a project name from task description."""
        # Simple implementation - could be enhanced with LLM
        words = task_description.lower().split()
        # Take first few meaningful words
        meaningful_words = [w for w in words[:3] if len(w) > 2 and w.isalpha()]
        if not meaningful_words:
            return "generated_project"
        
        return "_".join(meaningful_words)


class ContentGenerator:
    """Generates content for individual files using LLM."""

    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        # Import LLM client
        try:
            import openai
            # Use the config from llm_config properly
            api_key = llm_config.get("config_list", [{}])[0].get("api_key") if llm_config.get("config_list") else None
            base_url = llm_config.get("config_list", [{}])[0].get("base_url") if llm_config.get("config_list") else None

            if api_key and base_url:
                self.llm_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            else:
                logger.warning("LLM configuration incomplete, using mock generation")
                self.llm_client = None
        except ImportError:
            logger.warning("OpenAI client not available, using mock generation")
            self.llm_client = None

    async def generate_file_content(self, file_spec: FileSpec,
                                  project_context: Dict[str, Any]) -> str:
        """
        Generate content for a specific file using LLM.

        Args:
            file_spec: Specification of the file to generate
            project_context: Context information about the project

        Returns:
            Generated file content
        """
        try:
            # Generate content using LLM based on file type
            if file_spec.file_type == FileType.HTML:
                return await self._generate_html_with_llm(file_spec, project_context)
            elif file_spec.file_type == FileType.CSS:
                return await self._generate_css_with_llm(file_spec, project_context)
            elif file_spec.file_type == FileType.JAVASCRIPT:
                return await self._generate_js_with_llm(file_spec, project_context)
            elif file_spec.file_type == FileType.PYTHON:
                return await self._generate_python_with_llm(file_spec, project_context)
            elif file_spec.file_type == FileType.MARKDOWN:
                return await self._generate_markdown_with_llm(file_spec, project_context)
            elif file_spec.file_type == FileType.JSON:
                return await self._generate_json_with_llm(file_spec, project_context)
            elif file_spec.file_type == FileType.TEXT:
                return await self._generate_text_with_llm(file_spec, project_context)
            else:
                return await self._generate_generic_with_llm(file_spec, project_context)

        except Exception as e:
            logger.error(f"Failed to generate content for {file_spec.name}: {e}")
            # Fallback to basic template
            return self._generate_fallback_content(file_spec, project_context)

    async def _call_llm(self, prompt: str, system_message: str = None) -> str:
        """Call LLM with the given prompt."""
        if not self.llm_client:
            # Mock response for testing
            return f"# Generated content\n# Prompt: {prompt[:100]}..."

        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            # Get model from config_list
            model = "deepseek-chat"
            if self.llm_config.get("config_list"):
                model = self.llm_config["config_list"][0].get("model", "deepseek-chat")

            response = self.llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"# Error generating content: {str(e)}"

    async def _generate_html_with_llm(self, file_spec: FileSpec, context: Dict[str, Any]) -> str:
        """Generate HTML content using LLM."""

        system_message = """You are an expert web developer. Generate complete, modern, semantic HTML content.

        Requirements:
        - Use semantic HTML5 elements
        - Include proper meta tags and structure
        - Link to CSS and JavaScript files appropriately
        - Create responsive, accessible markup
        - Include meaningful content based on the project description
        """

        prompt = f"""
        Generate a complete HTML file for this project:

        Project Name: {context.get('project_name', 'Website')}
        Project Type: {context.get('project_type', 'webpage')}
        Description: {context.get('description', 'A website')}
        File Name: {file_spec.name}

        Requirements:
        - Complete HTML5 document structure
        - Proper DOCTYPE, head, and body sections
        - Link to 'css/styles.css' for styling
        - Link to 'js/script.js' for JavaScript
        - Create meaningful content sections based on the description
        - Use semantic HTML elements (header, nav, main, section, footer)
        - Include proper meta tags for viewport and charset
        - Make it responsive and accessible

        Generate ONLY the HTML content, no explanations.
        """

        return await self._call_llm(prompt, system_message)

    async def _generate_css_with_llm(self, file_spec: FileSpec, context: Dict[str, Any]) -> str:
        """Generate CSS content using LLM."""

        system_message = """You are an expert CSS developer. Generate modern, responsive CSS styles.

        Requirements:
        - Use modern CSS features (Grid, Flexbox, CSS Variables)
        - Create responsive design with mobile-first approach
        - Include smooth transitions and hover effects
        - Use semantic class names
        - Ensure accessibility (contrast, focus states)
        - Follow CSS best practices
        """

        prompt = f"""
        Generate CSS styles for this project:

        Project Name: {context.get('project_name', 'Website')}
        Project Type: {context.get('project_type', 'webpage')}
        Description: {context.get('description', 'A website')}
        File Name: {file_spec.name}

        Requirements:
        - Modern, responsive CSS styles
        - Mobile-first responsive design
        - Smooth animations and transitions
        - Professional color scheme
        - Typography hierarchy
        - Layout styles for header, nav, main, sections, footer
        - Form styling if applicable
        - Grid/Flexbox layouts
        - CSS variables for consistency
        - Accessibility considerations

        Generate ONLY the CSS content, no explanations.
        """

        return await self._call_llm(prompt, system_message)

    async def _generate_js_with_llm(self, file_spec: FileSpec, context: Dict[str, Any]) -> str:
        """Generate JavaScript content using LLM."""

        system_message = """You are an expert JavaScript developer. Generate modern, clean JavaScript code.

        Requirements:
        - Use modern ES6+ JavaScript features
        - Include proper error handling
        - Add interactive functionality
        - Use event delegation and best practices
        - Include accessibility considerations
        - Write clean, maintainable code
        """

        prompt = f"""
        Generate JavaScript code for this project:

        Project Name: {context.get('project_name', 'Website')}
        Project Type: {context.get('project_type', 'webpage')}
        Description: {context.get('description', 'A website')}
        File Name: {file_spec.name}

        Requirements:
        - Modern JavaScript (ES6+)
        - DOM manipulation and event handling
        - Smooth scrolling for navigation
        - Form validation and submission handling
        - Interactive animations and effects
        - Responsive behavior
        - Error handling
        - Accessibility features
        - Performance optimizations

        Generate ONLY the JavaScript content, no explanations.
        """

        return await self._call_llm(prompt, system_message)

    async def _generate_python_with_llm(self, file_spec: FileSpec, context: Dict[str, Any]) -> str:
        """Generate Python content using LLM."""

        system_message = """You are an expert Python developer. Generate clean, modern Python code.

        Requirements:
        - Follow PEP 8 style guidelines
        - Use type hints where appropriate
        - Include proper docstrings
        - Add error handling
        - Use modern Python features
        - Write maintainable, readable code
        """

        prompt = f"""
        Generate Python code for this file:

        Project Name: {context.get('project_name', 'Project')}
        Project Type: {context.get('project_type', 'python')}
        Description: {context.get('description', 'A Python project')}
        File Name: {file_spec.name}

        File-specific requirements:
        {self._get_python_file_requirements(file_spec.name, context)}

        General requirements:
        - Complete, functional Python code
        - Proper imports and dependencies
        - Type hints and docstrings
        - Error handling
        - Main execution block if appropriate
        - Follow Python best practices

        Generate ONLY the Python code, no explanations.
        """

        return await self._call_llm(prompt, system_message)

    async def _generate_markdown_with_llm(self, file_spec: FileSpec, context: Dict[str, Any]) -> str:
        """Generate Markdown content using LLM."""

        system_message = """You are an expert technical writer. Generate comprehensive, well-structured documentation.

        Requirements:
        - Use proper Markdown syntax
        - Create clear, informative content
        - Include practical examples
        - Structure with appropriate headings
        - Add relevant badges and links where appropriate
        """

        prompt = f"""
        Generate Markdown documentation for this project:

        Project Name: {context.get('project_name', 'Project')}
        Project Type: {context.get('project_type', 'project')}
        Description: {context.get('description', 'A project')}
        File Name: {file_spec.name}

        Requirements:
        - Comprehensive README.md content
        - Project description and features
        - Installation instructions
        - Usage examples
        - API documentation if applicable
        - Contributing guidelines
        - License information
        - Professional formatting

        Generate ONLY the Markdown content, no explanations.
        """

        return await self._call_llm(prompt, system_message)

    async def _generate_json_with_llm(self, file_spec: FileSpec, context: Dict[str, Any]) -> str:
        """Generate JSON content using LLM."""

        system_message = """You are an expert in configuration and data formats. Generate valid, well-structured JSON.

        Requirements:
        - Valid JSON syntax
        - Appropriate structure for the file type
        - Include relevant configuration options
        - Use meaningful keys and values
        """

        prompt = f"""
        Generate JSON content for this file:

        Project Name: {context.get('project_name', 'Project')}
        Project Type: {context.get('project_type', 'project')}
        Description: {context.get('description', 'A project')}
        File Name: {file_spec.name}

        File-specific requirements:
        {self._get_json_file_requirements(file_spec.name, context)}

        Generate ONLY valid JSON content, no explanations.
        """

        return await self._call_llm(prompt, system_message)

    async def _generate_text_with_llm(self, file_spec: FileSpec, context: Dict[str, Any]) -> str:
        """Generate text content using LLM."""

        if "requirements.txt" in file_spec.name:
            return await self._generate_requirements_txt(context)
        elif "gitignore" in file_spec.name:
            return await self._generate_gitignore(context)
        else:
            return await self._generate_generic_text(file_spec, context)

    async def _generate_generic_with_llm(self, file_spec: FileSpec, context: Dict[str, Any]) -> str:
        """Generate generic content using LLM."""

        system_message = f"""Generate appropriate content for a {file_spec.file_type.value} file."""

        prompt = f"""
        Generate content for this file:

        Project: {context.get('project_name', 'Project')}
        Type: {context.get('project_type', 'project')}
        Description: {context.get('description', 'A project')}
        File: {file_spec.name}
        File Type: {file_spec.file_type.value}

        Generate appropriate content for this file type and project context.
        """

        return await self._call_llm(prompt, system_message)

    def _get_python_file_requirements(self, filename: str, context: Dict[str, Any]) -> str:
        """Get specific requirements for Python files based on filename."""

        if "main.py" in filename:
            if context.get('project_type') == 'api_project':
                return """
                - FastAPI application setup
                - Route registration
                - CORS configuration
                - Health check endpoint
                - Main execution with uvicorn
                """
            else:
                return "- Main application entry point with proper if __name__ == '__main__' block"

        elif "models.py" in filename:
            return """
            - Pydantic models for data validation
            - Database models if applicable
            - Type definitions and schemas
            """

        elif "routes.py" in filename or "api.py" in filename:
            return """
            - FastAPI router setup
            - API endpoints with proper HTTP methods
            - Request/response models
            - Error handling
            """

        elif "test" in filename:
            return """
            - Unit tests using pytest
            - Test fixtures and mocks
            - Comprehensive test coverage
            """

        else:
            return "- Appropriate Python code for the module purpose"

    def _get_json_file_requirements(self, filename: str, context: Dict[str, Any]) -> str:
        """Get specific requirements for JSON files based on filename."""

        if "package.json" in filename:
            return """
            - NPM package configuration
            - Dependencies and devDependencies
            - Scripts for build, test, start
            - Project metadata
            """

        elif "config" in filename:
            return """
            - Application configuration
            - Environment-specific settings
            - API endpoints and keys structure
            """

        else:
            return "- Appropriate JSON structure for the file purpose"

    async def _generate_requirements_txt(self, context: Dict[str, Any]) -> str:
        """Generate requirements.txt for Python projects."""

        project_type = context.get('project_type', 'python')

        if project_type == 'api_project':
            return """fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6"""

        elif project_type == 'data_analysis':
            return """pandas>=2.1.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0"""

        else:
            return "# Add your project dependencies here"

    async def _generate_gitignore(self, context: Dict[str, Any]) -> str:
        """Generate .gitignore file."""

        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Environment variables
.env
.env.local

# Node modules
node_modules/

# Build outputs
dist/
build/"""

    async def _generate_generic_text(self, file_spec: FileSpec, context: Dict[str, Any]) -> str:
        """Generate generic text content."""

        prompt = f"""
        Generate appropriate text content for: {file_spec.name}

        Project: {context.get('project_name', 'Project')}
        Context: {context.get('description', 'A project')}

        Generate relevant content for this file.
        """

        return await self._call_llm(prompt)

    def _generate_fallback_content(self, file_spec: FileSpec, context: Dict[str, Any]) -> str:
        """Generate fallback content when LLM fails."""

        project_name = context.get('project_name', 'Generated Project')
        description = context.get('description', 'Auto-generated project')

        if file_spec.file_type == FileType.HTML:
            return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name}</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
    <header>
        <h1>{project_name}</h1>
    </header>
    <main>
        <p>{description}</p>
    </main>
    <script src="js/script.js"></script>
</body>
</html>"""

        elif file_spec.file_type == FileType.CSS:
            return f"""/* {project_name} Styles */
body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    line-height: 1.6;
}}

header {{
    background: #333;
    color: white;
    padding: 1rem;
    text-align: center;
}}"""

        elif file_spec.file_type == FileType.JAVASCRIPT:
            return f"""// {project_name} JavaScript
console.log('{project_name} loaded successfully');

document.addEventListener('DOMContentLoaded', function() {{
    // Add your JavaScript code here
}});"""

        elif file_spec.file_type == FileType.PYTHON:
            return f'''"""
{project_name}

{description}
"""

def main():
    """Main function."""
    print("Hello from {project_name}!")

if __name__ == "__main__":
    main()'''

        elif file_spec.file_type == FileType.MARKDOWN:
            return f"""# {project_name}

{description}

## Installation

Instructions for installation.

## Usage

Instructions for usage.
"""

        else:
            return f"# {project_name}\n# {description}\n# Generated content for {file_spec.name}"


class MultiFileGenerator:
    """Main class for multi-file generation orchestration."""

    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.classifier = TaskTypeClassifier(llm_config)
        self.planner = ProjectPlanner(llm_config)
        self.content_generator = ContentGenerator(llm_config)

    async def generate_project(self, task_description: str,
                             output_directory: str = None) -> GenerationResult:
        """
        Generate a complete multi-file project using LLM.

        Args:
            task_description: Description of what to generate
            output_directory: Where to save the project (optional)

        Returns:
            GenerationResult with success status and details
        """
        try:
            logger.info(f"🚀 Starting multi-file generation for: {task_description}")

            # Step 1: Classify the task
            project_type, confidence = await self.classifier.classify_with_llm(task_description)
            logger.info(f"📋 Classified as {project_type.value} (confidence: {confidence:.2f})")

            # Step 2: Plan project structure
            structure = self.planner.plan_project_structure(task_description, project_type)
            logger.info(f"🏗️ Planned structure: {len(structure.files)} files, {len(structure.directories)} directories")

            # Step 3: Prepare output directory
            if output_directory is None:
                output_directory = tempfile.mkdtemp(prefix="generated_project_")

            project_path = os.path.join(output_directory, structure.root_directory)
            os.makedirs(project_path, exist_ok=True)

            # Step 4: Create directory structure
            for directory in structure.directories:
                dir_path = os.path.join(project_path, directory)
                os.makedirs(dir_path, exist_ok=True)
                logger.debug(f"Created directory: {directory}")

            # Step 5: Generate and write files using LLM
            generated_files = []
            errors = []

            project_context = {
                "project_name": structure.project_name,
                "project_type": project_type.value,
                "description": task_description,
                "title": structure.project_name.replace("_", " ").title()
            }

            for file_spec in structure.files:
                try:
                    logger.info(f"📝 Generating {file_spec.name} with LLM")

                    # Generate file content using LLM
                    content = await self.content_generator.generate_file_content(
                        file_spec, project_context
                    )

                    # Write file
                    file_path = os.path.join(project_path, file_spec.name)

                    # Ensure directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    generated_files.append(file_spec.name)
                    logger.debug(f"✅ Generated {file_spec.name}")

                except Exception as e:
                    error_msg = f"Failed to generate {file_spec.name}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            # Step 6: Generate additional files (requirements.txt, .gitignore, etc.)
            await self._generate_additional_files(structure, project_path, project_context)

            # Step 7: Create result
            result = GenerationResult(
                success=len(errors) == 0,
                project_path=project_path,
                generated_files=generated_files,
                errors=errors,
                metadata={
                    "project_type": project_type.value,
                    "classification_confidence": confidence,
                    "total_files": len(generated_files),
                    "total_directories": len(structure.directories),
                    "generation_method": "llm_based"
                }
            )

            if result.success:
                logger.info(f"🎉 Successfully generated project at: {project_path}")
            else:
                logger.warning(f"⚠️ Project generated with {len(errors)} errors")

            return result

        except Exception as e:
            logger.error(f"❌ Multi-file generation failed: {str(e)}")
            return GenerationResult(
                success=False,
                project_path="",
                errors=[f"Generation failed: {str(e)}"]
            )

    async def _generate_additional_files(self, structure: ProjectStructure,
                                       project_path: str, context: Dict[str, Any]):
        """Generate additional project files like requirements.txt, .gitignore, etc."""

        additional_files = []

        # Add requirements.txt for Python projects
        if structure.project_type in [ProjectType.API_PROJECT, ProjectType.DATA_ANALYSIS]:
            additional_files.append(FileSpec("requirements.txt", FileType.TEXT))

        # Add .gitignore for all projects
        additional_files.append(FileSpec(".gitignore", FileType.TEXT))

        # Add package.json for web projects
        if structure.project_type == ProjectType.WEBPAGE:
            additional_files.append(FileSpec("package.json", FileType.JSON))

        # Generate additional files
        for file_spec in additional_files:
            try:
                content = await self.content_generator.generate_file_content(file_spec, context)
                file_path = os.path.join(project_path, file_spec.name)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                logger.debug(f"✅ Generated additional file: {file_spec.name}")

            except Exception as e:
                logger.warning(f"Failed to generate additional file {file_spec.name}: {e}")


# Factory function for easy access
def get_multi_file_generator(llm_config: Dict[str, Any]) -> MultiFileGenerator:
    """Get a MultiFileGenerator instance."""
    return MultiFileGenerator(llm_config)
