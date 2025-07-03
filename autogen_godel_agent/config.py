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
        FUNCTION_TEST_TIMEOUT = int(os.getenv("FUNCTION_TEST_TIMEOUT", "30"))
    except ValueError:
        FUNCTION_TEST_TIMEOUT = 30

    # === File Paths ===
    FUNCTION_REGISTRY_FILE = "memory/function_registry.json"
    HISTORY_FILE = "memory/history.json"
    
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
                "timeout": 120,
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
                "timeout": 120,
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
                "timeout": 120,
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
