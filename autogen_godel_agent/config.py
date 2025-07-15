"""
Configuration settings for AutoGen Self-Expanding Agent System.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
if not os.path.exists(".env"):
    print("⚠️  Warning: .env file not found. Please create one.")

class Config:
    """Configuration class for the self-expanding agent system."""

    # === LLM Provider Configuration ===
    # DeepSeek Configuration (Preferred)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

    # Azure OpenAI Configuration (optional)
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")

    # === Agent Settings ===
    try:
        MAX_FUNCTION_GENERATION_ATTEMPTS = int(os.getenv("MAX_FUNCTION_GENERATION_ATTEMPTS", "3"))
    except ValueError:
        MAX_FUNCTION_GENERATION_ATTEMPTS = 3

    try:
        FUNCTION_TEST_TIMEOUT = int(os.getenv("FUNCTION_TEST_TIMEOUT", "120"))
    except ValueError:
        FUNCTION_TEST_TIMEOUT = 120

    # === AutoGen Conversation Settings ===
    try:
        MAX_CONSECUTIVE_AUTO_REPLY = int(os.getenv("MAX_CONSECUTIVE_AUTO_REPLY", "50"))
    except ValueError:
        MAX_CONSECUTIVE_AUTO_REPLY = 50

    try:
        MAX_GROUP_CHAT_ROUNDS = int(os.getenv("MAX_GROUP_CHAT_ROUNDS", "100"))
    except ValueError:
        MAX_GROUP_CHAT_ROUNDS = 100

    # === File Paths ===
    FUNCTION_REGISTRY_FILE = "memory/function_registry.json"
    HISTORY_FILE = "memory/history.json"

    # === Learning Memory System Settings ===
    LEARNING_MEMORY_ENABLED = os.getenv("LEARNING_MEMORY_ENABLED", "true").lower() == "true"

    # Pattern Recognition Settings
    try:
        MIN_PATTERN_SIMILARITY = float(os.getenv("MIN_PATTERN_SIMILARITY", "0.3"))
    except ValueError:
        MIN_PATTERN_SIMILARITY = 0.3

    try:
        MAX_SIMILAR_PATTERNS = int(os.getenv("MAX_SIMILAR_PATTERNS", "10"))
    except ValueError:
        MAX_SIMILAR_PATTERNS = 10

    # Knowledge Graph Settings
    try:
        MIN_RELATIONSHIP_STRENGTH = float(os.getenv("MIN_RELATIONSHIP_STRENGTH", "0.3"))
    except ValueError:
        MIN_RELATIONSHIP_STRENGTH = 0.3

    try:
        MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "3"))
    except ValueError:
        MIN_CLUSTER_SIZE = 3

    # Recommendation Engine Settings
    try:
        MIN_RECOMMENDATION_CONFIDENCE = float(os.getenv("MIN_RECOMMENDATION_CONFIDENCE", "0.5"))
    except ValueError:
        MIN_RECOMMENDATION_CONFIDENCE = 0.5

    try:
        MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "5"))
    except ValueError:
        MAX_RECOMMENDATIONS = 5

    # Learning Memory File Paths
    MEMORY_DIR = "memory"
    TASK_PATTERNS_FILE = f"{MEMORY_DIR}/task_patterns.json"
    SOLUTION_PATTERNS_FILE = f"{MEMORY_DIR}/solution_patterns.json"
    KNOWLEDGE_GRAPH_FILE = f"{MEMORY_DIR}/knowledge_graph.json"
    RECOMMENDATIONS_HISTORY_FILE = f"{MEMORY_DIR}/recommendations_history.jsonl"
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration for AutoGen agents."""
        if cls.DEEPSEEK_API_KEY:
            # Use DeepSeek API
            return {
                "config_list": [
                    {
                        "model": cls.DEEPSEEK_MODEL,
                        "api_key": cls.DEEPSEEK_API_KEY,
                        "base_url": cls.DEEPSEEK_BASE_URL,
                    }
                ],
                "temperature": 0.1,
                "timeout": 600,
            }
        elif cls.AZURE_OPENAI_API_KEY and cls.AZURE_OPENAI_ENDPOINT:
            # Use Azure OpenAI
            return {
                "config_list": [
                    {
                        "model": cls.OPENAI_MODEL,
                        "api_key": cls.AZURE_OPENAI_API_KEY,
                        "base_url": cls.AZURE_OPENAI_ENDPOINT,
                        "api_type": "azure",
                        "api_version": cls.AZURE_OPENAI_API_VERSION,
                    }
                ],
                "temperature": 0.1,
                "timeout": 600,
            }
        elif cls.OPENAI_API_KEY:
            # Use OpenAI
            return {
                "config_list": [
                    {
                        "model": cls.OPENAI_MODEL,
                        "api_key": cls.OPENAI_API_KEY,
                    }
                ],
                "temperature": 0.1,
                "timeout": 600,
            }
        else:
            raise ValueError("No valid API key found. Please set DEEPSEEK_API_KEY, OPENAI_API_KEY or Azure OpenAI credentials.")
    
    @classmethod
    def validate_config(cls) -> None:
        """
        Validate that required configuration is present.

        Raises:
            RuntimeError: If no valid LLM API key is found
        """
        if not (cls.DEEPSEEK_API_KEY or
                cls.OPENAI_API_KEY or
                (cls.AZURE_OPENAI_API_KEY and cls.AZURE_OPENAI_ENDPOINT)):

            missing_configs = []
            if not cls.DEEPSEEK_API_KEY:
                missing_configs.append("DEEPSEEK_API_KEY")
            if not cls.OPENAI_API_KEY:
                missing_configs.append("OPENAI_API_KEY")
            if not cls.AZURE_OPENAI_API_KEY or not cls.AZURE_OPENAI_ENDPOINT:
                missing_configs.append("AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT")

            raise RuntimeError(
                f"❌ No valid LLM API key found. Please set one of: {', '.join(missing_configs)}"
            )

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize and validate configuration.

        This method should be called at the start of the application
        to ensure all required configuration is present.

        Raises:
            RuntimeError: If configuration validation fails
        """
        import logging
        logger = logging.getLogger(__name__)

        # Validate configuration
        cls.validate_config()

        # Log which provider is being used
        if cls.DEEPSEEK_API_KEY:
            logger.info(f"🚀 Using DeepSeek API - Model: {cls.DEEPSEEK_MODEL}")
        elif cls.AZURE_OPENAI_API_KEY and cls.AZURE_OPENAI_ENDPOINT:
            logger.info(f"🤖 Using Azure OpenAI - Model: {cls.OPENAI_MODEL}")
        elif cls.OPENAI_API_KEY:
            logger.info(f"🤖 Using OpenAI - Model: {cls.OPENAI_MODEL}")

        logger.info("✅ Configuration validated successfully")
