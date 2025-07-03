"""
Secure Code Executor for Function Registry.

This module provides secure execution of dynamically generated functions
with sandboxing and security validation.
"""

import ast
import hashlib
import logging
import sys
from typing import Dict, Any, Optional, Set, List, Tuple, Callable
from contextlib import contextmanager
import threading
import time

"""
1. ⚠️ threading.Timer 无法真正中断主线程执行
你使用 threading.Timer 抛 TimeoutError，但它不会影响主线程中 exec 的执行，Python 本身没有原生机制可以终止正在执行的 exec() 代码。

风险： 一旦用户提交的函数中有 while True: pass 或类似死循环，定时器虽然会触发 TimeoutError，但主线程并不会中止 exec，系统可能挂起。

✅ 建议更换为真正安全的隔离执行方式：

✅ 使用 multiprocessing.Process + Queue 实现真正的超时与进程中断。

✅ 或者使用 PyPy-Sandbox / restrictedpython 做更强隔离（更复杂）。

示例修复（简化版）：

python
复制
编辑
from multiprocessing import Process, Queue

def _exec_code(code, globals_dict, locals_dict, queue):
    try:
        exec(code, globals_dict, locals_dict)
        queue.put((True, locals_dict))
    except Exception as e:
        queue.put((False, str(e)))

def execute_code(self, code: str, timeout: int = 5) -> Tuple[bool, Any]:
    is_valid, violations = self.validate_code(code)
    if not is_valid:
        return False, {"error": "; ".join(violations)}
    
    queue = Queue()
    restricted_globals = self._create_restricted_globals()
    locals_dict = {}
    p = Process(target=_exec_code, args=(code, restricted_globals, locals_dict, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return False, {"error": "Execution timed out"}
    if not queue.empty():
        return queue.get()
    return False, {"error": "Unknown execution failure"}
⚠️ 2. __builtins__ 类型不统一兼容处理不错，但存在 eval 绕过风险
当前代码处理了 __builtins__ 既可能是 dict 也可能是模块，✅ 正确。

但用户仍可能通过例如：

python
复制
编辑
(lambda f: f.__globals__['__builtins__']['eval'])("1+1")
绕过访问 builtins。

建议增强 ast 安全扫描：

增加 visit_Lambda 检查 __globals__, __dict__, __closure__ 等。

✅ 或者在 exec 前强制：

python
复制
编辑
code = code.replace('__globals__', '').replace('__dict__', '')
或在 _create_restricted_globals 里移除更多潜在逃逸对象。

⚠️ 3. 模块白名单缺少注释或配置文件支持
allowed_modules 是硬编码的，未来难以维护。

✅ 建议：

将其配置写入 secure_executor_config.json，或者放到 settings.py 中作为白名单可修改。

或者暴露一个 set_allowed_modules(modules: Set[str]) 接口用于注册。

⚠️ 4. execute_code_safely 与 execute_code 功能有重复
二者都做了 validate → exec → return locals，差异仅在是否指定函数名。

✅ 建议：复用 execute_code 作为基础实现：

python
复制
编辑
def execute_code_safely(self, code, func_name):
    success, result = self.execute_code(code)
    if not success:
        return False, None, result.get("error", "Unknown error")
    if func_name not in result:
        return False, None, f"Function {func_name} not found"
    return True, result[func_name], ""
⚠️ 5. _execution_lock 并没有必要
当前 SecureExecutor 是单例，在多线程场景下添加锁是可以的，但：

exec 是局部执行，不需要强制串行。

你如果改为 multiprocessing，该锁完全无用。

✅ 建议移除 _execution_lock，或注明其并不影响主线程执行安全。

⚠️ 6. SecurityValidator.visit_Call 中未处理 node.func.value 是复杂对象
存在语法结构如：

python
复制
编辑
myobj.method()
此时 node.func.value 并不总是 ast.Name，直接取 .id 会报错。

✅ 建议加判断：

python
复制
编辑
if isinstance(node.func.value, ast.Name):
    if node.func.value.id == '__builtins__':
        ...
✅ 建议补充功能
✅ log_code_hash(code)：记录每段代码的 SHA256 哈希日志，追溯责任。

✅ def extract_function_names(code: str) -> List[str]：提取所有函数名，可供上游工具（如代码注册器）使用。

✅ 增加对 ast.Lambda、ast.With 等 node 的处理，避免上下文逃逸。

✅ 增加 --dry-run 模式供 debug 模拟执行环境。


"""


# Configure logger
logger = logging.getLogger(__name__)


class SecurityValidator(ast.NodeVisitor):
    """Enhanced security validator for function code."""
    
    def __init__(self):
        self.violations = []
        self.allowed_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'enumerate',
            'filter', 'float', 'format', 'frozenset', 'hex', 'int', 'len',
            'list', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'range',
            'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum',
            'tuple', 'type', 'zip', 'iter', 'next', 'divmod', 'hash',
            'isinstance', 'issubclass', 'callable', 'repr', 'ascii'
        }
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__', 'getattr', 'setattr',
            'delattr', 'hasattr', 'globals', 'locals', 'vars', 'dir',
            'open', 'file', 'input', 'raw_input', 'exit', 'quit'
        }
        self.dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'tempfile', 'pickle',
            'marshal', 'shelve', 'dbm', 'sqlite3', 'socket', 'urllib',
            'http', 'ftplib', 'smtplib', 'poplib', 'imaplib', 'threading',
            'multiprocessing', 'ctypes', 'importlib'
        }
        self.allowed_modules = {
            're', 'math', 'datetime', 'json', 'base64', 'hashlib',
            'random', 'string', 'collections', 'itertools', 'functools'
        }

    def visit_Call(self, node):
        """Check function calls for security violations."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.dangerous_functions:
                self.violations.append(f"Dangerous function call: {func_name}")
            elif func_name not in self.allowed_builtins and not func_name.startswith('_'):
                # Allow user-defined functions but flag suspicious ones
                if func_name.startswith('__'):
                    self.violations.append(f"Suspicious dunder method call: {func_name}")
        elif isinstance(node.func, ast.Attribute):
            if hasattr(node.func.value, 'id'):
                if node.func.value.id == '__builtins__':
                    self.violations.append(f"Direct __builtins__ access: {node.func.attr}")
        self.generic_visit(node)

    def visit_Import(self, node):
        """Check imports for dangerous modules."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]  # Get root module
            if module_name in self.dangerous_modules:
                self.violations.append(f"Dangerous module import: {module_name}")
            elif module_name not in self.allowed_modules:
                self.violations.append(f"Unauthorized module import: {module_name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check from imports for dangerous modules."""
        if node.module:
            module_name = node.module.split('.')[0]  # Get root module
            if module_name in self.dangerous_modules:
                self.violations.append(f"Dangerous module import: {module_name}")
            elif module_name not in self.allowed_modules:
                self.violations.append(f"Unauthorized module import: {module_name}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Check attribute access for dangerous patterns."""
        if isinstance(node.value, ast.Name):
            if node.value.id in ['__builtins__', '__globals__', '__locals__']:
                self.violations.append(f"Dangerous attribute access: {node.value.id}.{node.attr}")
        self.generic_visit(node)

    def get_violations(self) -> List[str]:
        """Return list of security violations found."""
        return self.violations


class SecureExecutor:
    """Secure executor for dynamically generated functions."""
    
    def __init__(self):
        self.validator = SecurityValidator()
        self._execution_lock = threading.Lock()
        self._execution_timeout = 30  # seconds
        
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """Validate code for security issues."""
        try:
            # Parse the code into AST
            tree = ast.parse(code)
            
            # Reset validator for new code
            self.validator = SecurityValidator()
            self.validator.visit(tree)
            
            violations = self.validator.get_violations()
            return len(violations) == 0, violations
            
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]
        except Exception as e:
            return False, [f"Validation error: {e}"]
    
    def compute_code_hash(self, code: str) -> str:
        """Compute SHA-256 hash of code for integrity checking."""
        return hashlib.sha256(code.encode('utf-8')).hexdigest()
    
    def execute_code_safely(self, code: str, expected_function_name: str) -> Tuple[bool, Optional[Callable], str]:
        """
        Execute code safely in a restricted environment.

        Returns:
            (success, function, error_message)
        """
        with self._execution_lock:
            try:
                # Validate code first
                is_valid, violations = self.validate_code(code)
                if not is_valid:
                    return False, None, f"Security violations: {'; '.join(violations)}"
                
                # Create restricted execution environment
                restricted_globals = self._create_restricted_globals()
                local_scope = {}
                
                # Execute with timeout protection
                with self._timeout_context(self._execution_timeout):
                    exec(code, restricted_globals, local_scope)
                
                # Check if expected function was created
                if expected_function_name not in local_scope:
                    return False, None, f"Function '{expected_function_name}' not found in executed code"
                
                func = local_scope[expected_function_name]
                if not callable(func):
                    return False, None, f"'{expected_function_name}' is not callable"
                
                return True, func, ""
                
            except TimeoutError:
                return False, None, "Code execution timed out"
            except Exception as e:
                logger.error(f"Code execution failed: {e}")
                return False, None, f"Execution error: {e}"
    
    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create a restricted global environment for code execution."""
        # Start with minimal builtins
        restricted_builtins = {}

        # Get builtins from the correct source
        if isinstance(__builtins__, dict):
            builtins_dict = __builtins__
        else:
            builtins_dict = __builtins__.__dict__

        for name in self.validator.allowed_builtins:
            if name in builtins_dict:
                restricted_builtins[name] = builtins_dict[name]

        # Add basic types for type annotations
        restricted_builtins.update({
            'int': int,
            'str': str,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'range': range,  # Explicitly add range
        })

        # Add safe modules
        safe_modules = {}
        for module_name in self.validator.allowed_modules:
            try:
                safe_modules[module_name] = __import__(module_name)
            except ImportError:
                pass  # Module not available, skip

        return {
            '__builtins__': restricted_builtins,
            **safe_modules
        }

    def execute_code(self, code: str, timeout: int = 10) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute code and return the local scope with all defined functions/variables.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Tuple[bool, Dict]: (success, local_scope or error_message)
        """
        try:
            # Validate code first
            is_valid, violations = self.validate_code(code)
            if not is_valid:
                return False, {"error": f"Security violations: {'; '.join(violations)}"}

            # Create restricted environment
            restricted_globals = self._create_restricted_globals()
            local_scope = {}

            # Execute with timeout protection
            with self._timeout_context(timeout):
                exec(code, restricted_globals, local_scope)

            return True, local_scope

        except TimeoutError:
            return False, {"error": "Code execution timed out"}
        except Exception as e:
            return False, {"error": str(e)}

    @contextmanager
    def _timeout_context(self, timeout_seconds: int):
        """Context manager for execution timeout."""
        def timeout_handler():
            raise TimeoutError("Code execution timed out")

        timer = threading.Timer(timeout_seconds, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()


# Global secure executor instance
_global_executor = None

def get_secure_executor() -> SecureExecutor:
    """Get the global secure executor instance."""
    global _global_executor
    if _global_executor is None:
        _global_executor = SecureExecutor()
    return _global_executor
