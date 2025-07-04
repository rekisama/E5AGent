"""
Logging Configuration Example for AutoGen Godel Agent

This file shows how to properly configure logging for the FunctionCreatorAgent
to ensure all log messages are properly displayed.

使用示例：在主程序入口处添加日志配置
"""

import logging
import sys
from datetime import datetime

def setup_logging(level=logging.INFO, log_to_file=False, log_file_path=None):
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to also log to file
        log_file_path: Path for log file (optional)
    """
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Optional file logging
    if log_to_file:
        if not log_file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = f"autogen_godel_agent_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        print(f"Logging to file: {log_file_path}")
    
    # Suppress overly verbose third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    print(f"Logging configured at level: {logging.getLevelName(level)}")

def setup_basic_logging():
    """Quick setup for basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )

# Example usage in main script:
if __name__ == "__main__":
    # Option 1: Basic setup
    setup_basic_logging()
    
    # Option 2: Advanced setup
    # setup_logging(level=logging.DEBUG, log_to_file=True)
    
    # Test logging
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration test successful!")
    logger.debug("Debug message test")
    logger.warning("Warning message test")
    logger.error("Error message test")
    
    # Now you can use FunctionCreatorAgent with proper logging
    from autogen_godel_agent.agents.function_creator_agent import FunctionCreatorAgent
    
    # Your agent code here...
