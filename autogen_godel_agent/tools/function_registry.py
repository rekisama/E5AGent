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
import os
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
        Initialize the function registry with hybrid storage support.

        Args:
            registry_path: Path to the registry JSON file. If None, uses default path.
        """
        if registry_path is None:
            # Use default path relative to project root
            project_root = Path(__file__).parent.parent.parent
            registry_path = project_root / "memory" / "function_registry.json"

        self.registry_path = Path(registry_path)
        self.backup_path = self.registry_path.with_suffix('.backup.json')

        # Set up function storage directories
        self.memory_dir = self.registry_path.parent
        self.functions_dir = self.memory_dir / "functions"
        self.simple_functions_dir = self.functions_dir / "simple"
        self.complex_functions_dir = self.functions_dir / "complex"
        self.composed_functions_dir = self.functions_dir / "composed"

        # Ensure directories exist
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.functions_dir.mkdir(exist_ok=True)
        self.simple_functions_dir.mkdir(exist_ok=True)
        self.complex_functions_dir.mkdir(exist_ok=True)
        self.composed_functions_dir.mkdir(exist_ok=True)

        # Load existing registry
        self.functions = self._load_registry()

    def _analyze_function_complexity(self, code: str) -> Dict[str, Any]:
        """
        Analyze function complexity to determine storage strategy.

        Args:
            code: Function source code

        Returns:
            Dictionary with complexity metrics
        """
        lines = code.split('\n')
        line_count = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        char_count = len(code)
        import_count = code.count('import ')
        class_count = code.count('class ')
        function_count = code.count('def ') - 1  # Subtract 1 for the main function
        docstring_lines = 0

        # Count docstring lines
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if '"""' in stripped or "'''" in stripped:
                docstring_lines += 1
                in_docstring = not in_docstring
            elif in_docstring:
                docstring_lines += 1

        # Calculate complexity score
        complexity_score = (
            line_count * 1.0 +
            char_count * 0.01 +
            import_count * 5.0 +
            class_count * 10.0 +
            function_count * 8.0
        )

        return {
            'line_count': line_count,
            'char_count': char_count,
            'import_count': import_count,
            'class_count': class_count,
            'function_count': function_count,
            'docstring_lines': docstring_lines,
            'complexity_score': complexity_score
        }

    def _should_store_as_file(self, code: str) -> bool:
        """
        Determine if function should be stored as separate file.

        Args:
            code: Function source code

        Returns:
            True if should store as file, False if store inline
        """
        complexity = self._analyze_function_complexity(code)

        # Thresholds for file storage
        return (
            complexity['line_count'] > 50 or
            complexity['char_count'] > 2000 or
            complexity['import_count'] > 3 or
            complexity['class_count'] > 0 or
            complexity['function_count'] > 1 or
            complexity['complexity_score'] > 100
        )

    def _get_function_file_path(self, func_name: str, storage_type: str = "complex") -> Path:
        """
        Get the file path for storing a function.

        Args:
            func_name: Function name
            storage_type: Storage type (simple, complex, composed)

        Returns:
            Path object for the function file
        """
        if storage_type == "simple":
            base_dir = self.simple_functions_dir
        elif storage_type == "composed":
            base_dir = self.composed_functions_dir
        else:
            base_dir = self.complex_functions_dir

        return base_dir / f"{func_name}.py"

    def _save_function_to_file(self, func_name: str, code: str, storage_type: str = "complex") -> bool:
        """
        Save function code to a separate file.

        Args:
            func_name: Function name
            code: Function source code
            storage_type: Storage type (simple, complex, composed)

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_function_file_path(func_name, storage_type)

            # Add header comment
            header = f'''"""
Function: {func_name}
Generated by E5Agent Function Registry
Storage Type: {storage_type}
Created: {datetime.datetime.now().isoformat()}
"""

'''

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(header + code)

            logger.info(f"Function '{func_name}' saved to file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save function '{func_name}' to file: {e}")
            return False

    def _load_function_from_file(self, func_name: str, storage_type: str = "complex") -> Optional[str]:
        """
        Load function code from file.

        Args:
            func_name: Function name
            storage_type: Storage type (simple, complex, composed)

        Returns:
            Function code or None if not found
        """
        try:
            file_path = self._get_function_file_path(func_name, storage_type)

            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Remove header comment if present
            lines = content.split('\n')
            code_start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('"""') and i > 0:
                    # Find end of docstring
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().endswith('"""'):
                            code_start = j + 1
                            break
                    break
                elif line.strip().startswith('def ') or line.strip().startswith('import ') or line.strip().startswith('from '):
                    code_start = i
                    break

            return '\n'.join(lines[code_start:]).strip()

        except Exception as e:
            logger.error(f"Failed to load function '{func_name}' from file: {e}")
            return None

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load function registry from file."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle different data structures for backward compatibility
                if isinstance(data, dict):
                    if 'functions' in data:
                        # New format with functions and metadata separated
                        functions = data['functions']
                        logger.info(f"Loaded {len(functions)} functions from registry (new format)")
                        return functions
                    elif 'metadata' in data:
                        # Old format with metadata wrapper
                        functions = data.get('metadata', {})
                        logger.info(f"Loaded {len(functions)} functions from registry (old format)")
                        return functions
                    else:
                        # Very old format or mixed format - extract only function objects
                        functions = {}
                        for k, v in data.items():
                            if isinstance(v, dict) and 'name' in v:
                                functions[k] = v
                        logger.info(f"Loaded {len(functions)} functions from registry (mixed format)")
                        return functions
                else:
                    logger.warning("Registry data is not a dictionary, starting with empty registry")
                    return {}
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
                        # Apply same logic for backup
                        if isinstance(data, dict) and 'functions' in data:
                            functions = data['functions']
                        elif isinstance(data, dict) and 'metadata' in data:
                            functions = data.get('metadata', {})
                        else:
                            functions = {k: v for k, v in data.items()
                                       if isinstance(v, dict) and 'name' in v}
                        logger.info(f"Loaded {len(functions)} functions from backup registry")
                        return functions
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

            # Save current registry in new format
            registry_data = {
                "functions": self.functions,
                "metadata": {
                    "last_updated": datetime.datetime.now().isoformat(),
                    "version": "2.0",
                    "total_functions": len(self.functions)
                }
            }

            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)

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
                         task_origin: str = "") -> bool:
        """
        Register a function in the registry with hybrid storage support.

        Args:
            func_name: Name of the function
            func_code: Function source code
            description: Function description
            task_origin: Origin task or context

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Generate function hash for version control
            func_hash = hashlib.md5(func_code.encode('utf-8')).hexdigest()

            # Analyze function complexity
            complexity = self._analyze_function_complexity(func_code)
            should_store_as_file = self._should_store_as_file(func_code)

            # Determine storage type
            storage_type = "inline"
            if should_store_as_file:
                if "compose" in func_name.lower() or "composite" in func_name.lower():
                    storage_type = "composed"
                else:
                    storage_type = "complex"
            else:
                storage_type = "simple"

            # Create function metadata
            function_data = {
                'name': func_name,
                'description': description,
                'task_origin': task_origin,
                'code_hash': func_hash,
                'created_at': datetime.datetime.now().isoformat(),
                'updated_at': datetime.datetime.now().isoformat(),
                'version': 1,
                'storage_type': storage_type,
                'complexity': complexity
            }

            # Store code based on complexity
            if storage_type == "inline" or storage_type == "simple":
                # Store simple functions inline in JSON
                function_data['code'] = func_code
                logger.info(f"Storing function '{func_name}' inline (complexity: {complexity['complexity_score']:.1f})")
            else:
                # Store complex functions as separate files
                if self._save_function_to_file(func_name, func_code, storage_type):
                    function_data['code_file'] = str(self._get_function_file_path(func_name, storage_type))
                    function_data['code'] = None  # Don't store code inline
                    logger.info(f"Storing function '{func_name}' as file (complexity: {complexity['complexity_score']:.1f})")
                else:
                    # Fallback to inline storage if file save fails
                    function_data['code'] = func_code
                    function_data['storage_type'] = "inline_fallback"
                    logger.warning(f"Failed to save '{func_name}' as file, falling back to inline storage")

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
        Get function source code with hybrid storage support.

        Args:
            func_name: Name of the function

        Returns:
            Function source code or None if not found
        """
        func_data = self.get_function(func_name)
        if not func_data:
            return None

        # Check if code is stored inline
        if func_data.get('code'):
            return func_data['code']

        # Check if code is stored in file
        storage_type = func_data.get('storage_type', 'inline')
        if storage_type in ['complex', 'composed', 'simple'] and func_data.get('code_file'):
            return self._load_function_from_file(func_name, storage_type)

        # Fallback: try to load from different storage types
        for storage_type in ['complex', 'composed', 'simple']:
            code = self._load_function_from_file(func_name, storage_type)
            if code:
                # Update metadata to reflect actual storage location
                func_data['storage_type'] = storage_type
                func_data['code_file'] = str(self._get_function_file_path(func_name, storage_type))
                self._save_registry()
                return code

        return None

    def list_functions(self) -> List[str]:
        """
        Get list of all registered function names.

        Returns:
            List of function names
        """
        return list(self.functions.keys())

    def get_all_functions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered functions with their data.

        Returns:
            Dictionary mapping function names to their data
        """
        return self.functions.copy()

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
                     task_origin: str = "") -> bool:
    """Register a function in the registry."""
    return get_registry().register_function(func_name, func_code, description, task_origin)


def get_function(func_name: str) -> Optional[Dict[str, Any]]:
    """Get function data from registry."""
    return get_registry().get_function(func_name)


def get_function_code(func_name: str) -> Optional[str]:
    """Get function source code."""
    return get_registry().get_function_code(func_name)


def list_functions() -> List[str]:
    """Get list of all registered function names."""
    return get_registry().list_functions()