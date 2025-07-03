"""
Enhanced Dynamic Function Registry for Self-Expanding Agent System

This module provides a secure, thread-safe registry for dynamically generated functions
with comprehensive metadata management, version control, and security validation.
"""

"""
TODO
1. 依赖外部模块的缺失
代码中有 from .secure_executor import get_secure_executor，这是自定义的安全执行器接口，但代码没给出实现。

需要确保 secure_executor 模块存在且稳定，且功能包括安全执行代码、代码校验、哈希计算等。

如果缺少，会导致运行报错。

2. 跨平台文件锁问题
你用了 fcntl（Unix）和 msvcrt（Windows）实现文件锁，兼容性考虑很好。

但 msvcrt.locking的用法可能需要注意锁定大小参数，当前用1字节锁定可能不够。

并且在多线程、多进程环境下，文件锁有时不够完善，建议做额外测试。

另外在某些环境（如网络挂载文件系统）文件锁可能不生效。

3. 并发性能和锁粒度
你用 threading.RLock保护整个操作，确保线程安全。

但是在高并发场景下，锁粒度较大，可能导致阻塞。

可以考虑读写锁（threading.Lock+threading.Condition 或者用第三方库readerwriterlock）提升读多写少时的性能。

4. 代码执行安全性的隐患
即使用了secure_executor，代码执行本质依然是动态的字符串执行，仍有潜在安全风险，尤其是函数代码来源不可信时。

建议在secure_executor中引入沙箱、白名单、资源限制等机制，防止恶意代码。

另外，exec执行函数代码时要注意异常捕获和报错反馈。

5. 异步保存策略的潜在风险

threading.Thread(target=self._save_registry, daemon=True).start()
这个异步保存方案虽然提高了性能，但存在多线程同时写文件冲突风险。

文件锁会起保护作用，但建议增加队列或批量写入机制避免频繁触发写操作。

6. 时间格式及比较问题
created_at, updated_at, last_used均为ISO8601字符串，虽然便于存储和展示，但在list_functions等处对时间做过滤比较时是字符串比较，容易出错。

建议统一转换成datetime对象后再比较，避免字符串比较导致逻辑错误。

7. 代码哈希计算的复用
代码中多处调用self.secure_executor.compute_code_hash(func_code)，可以封装成内部私有方法，方便维护。

8. Metadata冗余或不一致风险
version字段依赖于外部管理更新，如果更新时忘记更新版本号，可能出现不一致。

可以考虑自动版本号生成，比如基于代码哈希或时间戳。

9. 导出和备份方法健壮性
backup_registry中使用了shutil.copy2，复制前未检查目标路径是否存在同名文件导致覆盖风险。

建议加上覆盖确认或者版本号后缀。

10. 依赖管理缺少加载顺序和冲突处理
函数依赖信息被存储，但没有机制保证依赖函数已加载或版本匹配。

可设计加载依赖拓扑排序、依赖冲突检测、自动加载依赖函数等机制。

11. 缺少函数调用日志或调用者追踪
如果系统需要调试或统计函数调用上下文，可以增加调用日志和追踪支持。

12. 代码注释及文档可完善
代码整体注释还算清晰，但函数内部逻辑较复杂，建议加更多细节说明。

另外对外接口的参数和返回值说明，异常情况的描述等，可以用类型提示加文档字符串更完整。

13. 不支持网络分布式场景
目前设计是本地文件和内存管理，不支持跨机器分布式共享。

若未来要支持集群，需改用分布式存储或数据库。
"""



import inspect
import json
import os
import logging
import threading
import hashlib
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Platform-specific imports for file locking
try:
    import fcntl  # For file locking on Unix systems
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt  # For file locking on Windows
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

from .secure_executor import get_secure_executor

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class FunctionMetadata:
    """Enhanced metadata for registered functions."""
    name: str
    code: str
    code_hash: str
    description: str
    task_origin: str
    created_at: str
    updated_at: str
    version: int
    signature: str
    docstring: str
    test_cases: List[Dict[str, Any]]
    tags: List[str]
    author: str
    dependencies: List[str]
    usage_count: int
    last_used: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FunctionMetadata':
        """Create from dictionary."""
        return cls(**data)


class FileLock:
    """Cross-platform file locking utility."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.lock_file = None

    def __enter__(self):
        self.lock_file = open(self.file_path + '.lock', 'w')
        try:
            if sys.platform == 'win32' and HAS_MSVCRT:
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_LOCK, 1)
            elif HAS_FCNTL:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
            else:
                logger.warning("File locking not available on this platform")
        except Exception as e:
            logger.warning(f"Could not acquire file lock: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_file:
            try:
                if sys.platform == 'win32' and HAS_MSVCRT:
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                elif HAS_FCNTL:
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                logger.warning(f"Could not release file lock: {e}")
            finally:
                self.lock_file.close()
                # Clean up lock file
                try:
                    os.remove(self.file_path + '.lock')
                except OSError:
                    pass


class EnhancedFunctionRegistry:
    """Enhanced registry for dynamically generated and validated functions."""

    def __init__(self, registry_file: str = "memory/function_registry.json"):
        self.registry_file = Path(registry_file)
        self.functions: Dict[str, Callable] = {}
        self.metadata: Dict[str, FunctionMetadata] = {}
        self.secure_executor = get_secure_executor()
        self._registry_lock = threading.RLock()
        self._load_registry()

    def _load_registry(self):
        """Load existing functions from registry file with security validation."""
        if not self.registry_file.exists():
            logger.info(f"Registry file {self.registry_file} does not exist, starting fresh")
            return

        try:
            with FileLock(str(self.registry_file)):
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load metadata
                metadata_dict = data.get('metadata', {})
                for func_name, meta_dict in metadata_dict.items():
                    try:
                        # Convert dict to FunctionMetadata
                        metadata = FunctionMetadata.from_dict(meta_dict)

                        # Verify code integrity
                        current_hash = hashlib.sha256(metadata.code.encode('utf-8')).hexdigest()
                        if current_hash != metadata.code_hash:
                            logger.warning(f"Code integrity check failed for {func_name}, skipping")
                            continue

                        # Securely execute function code
                        success, func, error = self.secure_executor.execute_code_safely(
                            metadata.code, func_name
                        )

                        if success and func:
                            self.functions[func_name] = func
                            self.metadata[func_name] = metadata
                            logger.debug(f"Successfully loaded function: {func_name}")
                        else:
                            logger.warning(f"Failed to load function {func_name}: {error}")

                    except Exception as e:
                        logger.error(f"Failed to load function {func_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.metadata = {}

    def _save_registry(self):
        """Save current registry to file with atomic write and locking."""
        try:
            # Ensure directory exists
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for serialization
            metadata_dict = {
                name: meta.to_dict()
                for name, meta in self.metadata.items()
            }

            registry_data = {
                'metadata': metadata_dict,
                'last_updated': datetime.now().isoformat(),
                'version': '2.0',
                'total_functions': len(self.metadata)
            }

            # Atomic write with file locking
            temp_file = self.registry_file.with_suffix('.tmp')
            with FileLock(str(self.registry_file)):
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(registry_data, f, indent=2, ensure_ascii=False)

                # Atomic rename
                temp_file.replace(self.registry_file)

            logger.debug(f"Registry saved successfully with {len(self.metadata)} functions")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
    
    def has_function(self, func_name: str) -> bool:
        """Check if a function exists in the registry."""
        with self._registry_lock:
            return func_name in self.functions

    def get_function(self, func_name: str) -> Optional[Callable]:
        """Get a function from the registry and update usage statistics."""
        with self._registry_lock:
            if func_name in self.functions:
                # Update usage statistics
                if func_name in self.metadata:
                    self.metadata[func_name].usage_count += 1
                    self.metadata[func_name].last_used = datetime.now().isoformat()
                    # Save updated statistics (async to avoid blocking)
                    threading.Thread(target=self._save_registry, daemon=True).start()

                return self.functions[func_name]
            return None

    def get_function_metadata(self, func_name: str) -> Optional[FunctionMetadata]:
        """Get metadata for a specific function."""
        with self._registry_lock:
            return self.metadata.get(func_name)
    
    def register_function(self, func_name: str, func_code: str,
                         description: str, task_origin: str = "",
                         test_cases: List[Dict] = None, tags: List[str] = None,
                         author: str = "system", dependencies: List[str] = None) -> bool:
        """Register a new function to the registry with enhanced security and metadata."""
        with self._registry_lock:
            try:
                # Check if function already exists
                if func_name in self.functions:
                    logger.warning(f"Function {func_name} already exists, updating...")
                    return self.update_function(func_name, func_code, description,
                                              task_origin, test_cases, tags, author, dependencies)

                # Validate and execute code securely
                success, func, error = self.secure_executor.execute_code_safely(func_code, func_name)
                if not success:
                    logger.error(f"Failed to register function {func_name}: {error}")
                    return False

                # Compute code hash for integrity
                code_hash = self.secure_executor.compute_code_hash(func_code)

                # Create metadata
                now = datetime.now().isoformat()
                metadata = FunctionMetadata(
                    name=func_name,
                    code=func_code,
                    code_hash=code_hash,
                    description=description,
                    task_origin=task_origin,
                    created_at=now,
                    updated_at=now,
                    version=1,
                    signature=str(inspect.signature(func)),
                    docstring=inspect.getdoc(func) or "",
                    test_cases=test_cases or [],
                    tags=tags or [],
                    author=author,
                    dependencies=dependencies or [],
                    usage_count=0,
                    last_used=None
                )

                # Store function and metadata
                self.functions[func_name] = func
                self.metadata[func_name] = metadata

                # Save to file
                self._save_registry()

                logger.info(f"Successfully registered function: {func_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to register function {func_name}: {e}")
                return False
    
    def update_function(self, func_name: str, func_code: str,
                       description: str, task_origin: str = "",
                       test_cases: List[Dict] = None, tags: List[str] = None,
                       author: str = "system", dependencies: List[str] = None) -> bool:
        """Update an existing function."""
        with self._registry_lock:
            if func_name not in self.functions:
                logger.error(f"Function {func_name} does not exist, cannot update")
                return False

            try:
                # Validate and execute new code
                success, func, error = self.secure_executor.execute_code_safely(func_code, func_name)
                if not success:
                    logger.error(f"Failed to update function {func_name}: {error}")
                    return False

                # Update metadata
                old_metadata = self.metadata[func_name]
                code_hash = self.secure_executor.compute_code_hash(func_code)

                updated_metadata = FunctionMetadata(
                    name=func_name,
                    code=func_code,
                    code_hash=code_hash,
                    description=description,
                    task_origin=task_origin,
                    created_at=old_metadata.created_at,
                    updated_at=datetime.now().isoformat(),
                    version=old_metadata.version + 1,
                    signature=str(inspect.signature(func)),
                    docstring=inspect.getdoc(func) or "",
                    test_cases=test_cases or old_metadata.test_cases,
                    tags=tags or old_metadata.tags,
                    author=author,
                    dependencies=dependencies or old_metadata.dependencies,
                    usage_count=old_metadata.usage_count,
                    last_used=old_metadata.last_used
                )

                # Update function and metadata
                self.functions[func_name] = func
                self.metadata[func_name] = updated_metadata

                # Save to file
                self._save_registry()

                logger.info(f"Successfully updated function: {func_name} (v{updated_metadata.version})")
                return True

            except Exception as e:
                logger.error(f"Failed to update function {func_name}: {e}")
                return False

    def delete_function(self, func_name: str) -> bool:
        """Delete a function from the registry."""
        with self._registry_lock:
            if func_name not in self.functions:
                logger.warning(f"Function {func_name} does not exist")
                return False

            try:
                del self.functions[func_name]
                del self.metadata[func_name]
                self._save_registry()

                logger.info(f"Successfully deleted function: {func_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete function {func_name}: {e}")
                return False

    def list_functions(self, tags: List[str] = None, author: str = None,
                      created_after: str = None) -> List[Dict[str, Any]]:
        """List registered functions with optional filtering."""
        with self._registry_lock:
            results = []

            for name, meta in self.metadata.items():
                # Apply filters
                if tags and not any(tag in meta.tags for tag in tags):
                    continue
                if author and meta.author != author:
                    continue
                if created_after and meta.created_at < created_after:
                    continue

                results.append({
                    'name': name,
                    'description': meta.description,
                    'signature': meta.signature,
                    'created_at': meta.created_at,
                    'updated_at': meta.updated_at,
                    'version': meta.version,
                    'task_origin': meta.task_origin,
                    'tags': meta.tags,
                    'author': meta.author,
                    'usage_count': meta.usage_count,
                    'last_used': meta.last_used
                })

            return results

    def search_functions(self, query: str, tags: List[str] = None) -> List[str]:
        """Search functions by name, description, or tags."""
        with self._registry_lock:
            query_lower = query.lower()
            matches = []

            for func_name, meta in self.metadata.items():
                # Text search
                text_match = (
                    query_lower in func_name.lower() or
                    query_lower in meta.description.lower() or
                    query_lower in meta.docstring.lower()
                )

                # Tag filter
                tag_match = not tags or any(tag in meta.tags for tag in tags)

                if text_match and tag_match:
                    matches.append(func_name)

            return matches


    def export_function_code(self, func_name: str) -> Optional[str]:
        """Export the source code of a function."""
        with self._registry_lock:
            if func_name in self.metadata:
                return self.metadata[func_name].code
            return None

    def get_function_dependencies(self, func_name: str) -> List[str]:
        """Get the dependencies of a function."""
        with self._registry_lock:
            if func_name in self.metadata:
                return self.metadata[func_name].dependencies.copy()
            return []

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all functions."""
        with self._registry_lock:
            total_functions = len(self.metadata)
            total_usage = sum(meta.usage_count for meta in self.metadata.values())

            most_used = max(self.metadata.items(),
                          key=lambda x: x[1].usage_count,
                          default=(None, None))

            return {
                'total_functions': total_functions,
                'total_usage': total_usage,
                'average_usage': total_usage / total_functions if total_functions > 0 else 0,
                'most_used_function': most_used[0] if most_used[0] else None,
                'most_used_count': most_used[1].usage_count if most_used[1] else 0
            }

    def backup_registry(self, backup_path: str) -> bool:
        """Create a backup of the registry."""
        try:
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)

            with self._registry_lock:
                if self.registry_file.exists():
                    import shutil
                    shutil.copy2(self.registry_file, backup_file)
                    logger.info(f"Registry backed up to {backup_path}")
                    return True
                else:
                    logger.warning("Registry file does not exist, cannot backup")
                    return False

        except Exception as e:
            logger.error(f"Failed to backup registry: {e}")
            return False

    def validate_all_functions(self) -> Dict[str, bool]:
        """Validate all functions in the registry."""
        results = {}
        with self._registry_lock:
            for func_name, metadata in self.metadata.items():
                try:
                    # Validate code integrity
                    current_hash = hashlib.sha256(metadata.code.encode('utf-8')).hexdigest()
                    if current_hash != metadata.code_hash:
                        results[func_name] = False
                        continue

                    # Validate code security
                    is_valid, violations = self.secure_executor.validate_code(metadata.code)
                    results[func_name] = is_valid

                    if not is_valid:
                        logger.warning(f"Function {func_name} failed validation: {violations}")

                except Exception as e:
                    logger.error(f"Error validating function {func_name}: {e}")
                    results[func_name] = False

        return results


# Backward compatibility - keep old class name as alias
FunctionRegistry = EnhancedFunctionRegistry

# Global registry instance
_global_registry = None

def get_registry() -> EnhancedFunctionRegistry:
    """Get the global enhanced function registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = EnhancedFunctionRegistry()
    return _global_registry
