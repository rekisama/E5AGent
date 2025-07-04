"""
Function Registry Module.

This module provides function registration and management capabilities
for the Function Creator Agent system. It includes:

1. Function registration with metadata storage
2. Function existence checking
3. Function retrieval and management
4. Persistent storage using JSON files
5. Version control and backup functionality

函数注册模块，为函数创建代理系统提供函数注册和管理能力。
包括带元数据存储的函数注册、函数存在性检查、函数检索和管理、使用JSON文件的持久化存储。
"""

import json
import logging
import hashlib
import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


class FunctionRegistry:
    """
    Function registry for managing registered functions with persistent storage.

    This class provides comprehensive function management including registration,
    retrieval, existence checking, and persistent storage with backup functionality.
    """

    def __init__(self, registry_path: str = None):
        """
        Initialize the function registry.

        Args:
            registry_path: Path to the registry JSON file. If None, uses default path.
        """
        if registry_path is None:
            # Use default path relative to project root
            project_root = Path(__file__).parent.parent.parent
            registry_path = project_root / "memory" / "function_registry.json"

        self.registry_path = Path(registry_path)
        self.backup_path = self.registry_path.with_suffix('.backup.json')

        # Ensure directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self.functions = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load function registry from file."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} functions from registry")
                    return data
            else:
                logger.info("Registry file not found, starting with empty registry")
                return {}
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            # Try to load from backup
            try:
                if self.backup_path.exists():
                    with open(self.backup_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        logger.info(f"Loaded {len(data)} functions from backup registry")
                        return data
            except Exception as backup_e:
                logger.error(f"Failed to load backup registry: {backup_e}")

            return {}

    def _save_registry(self) -> bool:
        """Save function registry to file with backup."""
        try:
            # Create backup of existing registry
            if self.registry_path.exists():
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    backup_data = f.read()
                with open(self.backup_path, 'w', encoding='utf-8') as f:
                    f.write(backup_data)

            # Save current registry
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.functions, f, indent=2, ensure_ascii=False)

            logger.debug(f"Registry saved with {len(self.functions)} functions")
            return True

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            return False

    def has_function(self, func_name: str) -> bool:
        """
        Check if a function exists in the registry.

        Args:
            func_name: Name of the function to check

        Returns:
            True if function exists, False otherwise
        """
        return func_name in self.functions

    def register_function(self, func_name: str, func_code: str, description: str,
                         task_origin: str = "", test_cases: List[Dict] = None) -> bool:
        """
        Register a function in the registry.

        Args:
            func_name: Name of the function
            func_code: Function source code
            description: Function description
            task_origin: Origin task or context
            test_cases: List of test cases for the function

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Generate function hash for version control
            func_hash = hashlib.md5(func_code.encode('utf-8')).hexdigest()

            # Create function metadata
            function_data = {
                'name': func_name,
                'code': func_code,
                'description': description,
                'task_origin': task_origin,
                'test_cases': test_cases or [],
                'hash': func_hash,
                'created_at': datetime.datetime.now().isoformat(),
                'updated_at': datetime.datetime.now().isoformat(),
                'version': 1
            }

            # Check if function already exists
            if func_name in self.functions:
                existing_hash = self.functions[func_name].get('hash', '')
                if existing_hash == func_hash:
                    logger.info(f"Function '{func_name}' already exists with same code")
                    return True
                else:
                    # Update existing function
                    function_data['version'] = self.functions[func_name].get('version', 1) + 1
                    function_data['created_at'] = self.functions[func_name].get('created_at',
                                                                               function_data['created_at'])
                    logger.info(f"Updating function '{func_name}' to version {function_data['version']}")
            else:
                logger.info(f"Registering new function '{func_name}'")

            # Store function
            self.functions[func_name] = function_data

            # Save to file
            if self._save_registry():
                logger.info(f"Function '{func_name}' successfully registered")
                return True
            else:
                logger.error(f"Failed to save function '{func_name}' to registry")
                return False

        except Exception as e:
            logger.error(f"Failed to register function '{func_name}': {e}")
            return False

    def get_function(self, func_name: str) -> Optional[Dict[str, Any]]:
        """
        Get function data from registry.

        Args:
            func_name: Name of the function

        Returns:
            Function data dictionary or None if not found
        """
        return self.functions.get(func_name)

    def get_function_code(self, func_name: str) -> Optional[str]:
        """
        Get function source code.

        Args:
            func_name: Name of the function

        Returns:
            Function source code or None if not found
        """
        func_data = self.get_function(func_name)
        return func_data.get('code') if func_data else None

    def list_functions(self) -> List[str]:
        """
        Get list of all registered function names.

        Returns:
            List of function names
        """
        return list(self.functions.keys())

    def get_function_info(self, func_name: str) -> Optional[Dict[str, Any]]:
        """
        Get function information (metadata without code).

        Args:
            func_name: Name of the function

        Returns:
            Function info dictionary or None if not found
        """
        func_data = self.get_function(func_name)
        if func_data:
            return {
                'name': func_data.get('name'),
                'description': func_data.get('description'),
                'task_origin': func_data.get('task_origin'),
                'created_at': func_data.get('created_at'),
                'updated_at': func_data.get('updated_at'),
                'version': func_data.get('version'),
                'hash': func_data.get('hash'),
                'test_cases_count': len(func_data.get('test_cases', []))
            }
        return None

    def remove_function(self, func_name: str) -> bool:
        """
        Remove a function from the registry.

        Args:
            func_name: Name of the function to remove

        Returns:
            True if removal successful, False otherwise
        """
        try:
            if func_name in self.functions:
                del self.functions[func_name]
                if self._save_registry():
                    logger.info(f"Function '{func_name}' removed from registry")
                    return True
                else:
                    logger.error(f"Failed to save registry after removing '{func_name}'")
                    return False
            else:
                logger.warning(f"Function '{func_name}' not found in registry")
                return False
        except Exception as e:
            logger.error(f"Failed to remove function '{func_name}': {e}")
            return False

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        total_functions = len(self.functions)
        total_test_cases = sum(len(func.get('test_cases', [])) for func in self.functions.values())

        # Get creation date range
        creation_dates = [func.get('created_at') for func in self.functions.values() if func.get('created_at')]
        oldest = min(creation_dates) if creation_dates else None
        newest = max(creation_dates) if creation_dates else None

        return {
            'total_functions': total_functions,
            'total_test_cases': total_test_cases,
            'oldest_function': oldest,
            'newest_function': newest,
            'registry_path': str(self.registry_path),
            'registry_size_bytes': self.registry_path.stat().st_size if self.registry_path.exists() else 0
        }


# Global registry instance
_global_registry = None


def get_registry() -> FunctionRegistry:
    """Get the global function registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = FunctionRegistry()
    return _global_registry


# Convenience functions for backward compatibility
def has_function(func_name: str) -> bool:
    """Check if a function exists in the registry."""
    return get_registry().has_function(func_name)


def register_function(func_name: str, func_code: str, description: str,
                     task_origin: str = "", test_cases: List[Dict] = None) -> bool:
    """Register a function in the registry."""
    return get_registry().register_function(func_name, func_code, description, task_origin, test_cases)


def get_function(func_name: str) -> Optional[Dict[str, Any]]:
    """Get function data from registry."""
    return get_registry().get_function(func_name)


def get_function_code(func_name: str) -> Optional[str]:
    """Get function source code."""
    return get_registry().get_function_code(func_name)


def list_functions() -> List[str]:
    """Get list of all registered function names."""
    return get_registry().list_functions()