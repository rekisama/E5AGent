"""
AutoGen Self-Expanding Agent System - Tools Module

这个模块包含了系统的核心工具组件，采用模块化设计，支持：
- 函数注册和管理
- 安全代码验证和执行
- 测试用例生成和运行
- 函数组合和链式调用 (新增)
- 会话管理和代理池
- 响应解析

所有工具都通过统一的接口提供服务，支持依赖注入和工厂模式。
"""

# 核心工具模块
from .function_tools import FunctionToolsInterface, get_function_tools
from .function_registry import get_registry, register_function, has_function, get_function, list_functions
from .secure_executor import SecurityValidator, validate_function_code, execute_code_safely
from .autogen_test_evolution import get_autogen_test_evolution, evolve_function_with_autogen

# 函数组合模块 (新增)
from .function_composer import FunctionComposer, get_function_composer, CompositeFunction, SubTask, FunctionMatch

# 专用工具模块 (主要用于 planner_agent)
from .session_manager import SessionManager, get_session_manager
from .response_parser import ResponseParser, get_response_parser
from .agent_pool import UserProxyPool, get_user_proxy_pool, get_assistant_agent_pool

__all__ = [
    # 核心工具
    'FunctionToolsInterface',
    'get_function_tools',
    'get_registry',
    'register_function',
    'has_function',
    'get_function',
    'list_functions',
    'SecurityValidator',
    'validate_function_code',
    'execute_code_safely',
    'TestCaseGenerator',
    'TestResult',

    # 函数组合工具 (新增)
    'FunctionComposer',
    'get_function_composer',
    'CompositeFunction',
    'SubTask',
    'FunctionMatch',

    # 专用工具
    'SessionManager',
    'get_session_manager',
    'ResponseParser',
    'get_response_parser',
    'UserProxyPool',
    'get_user_proxy_pool',
    'get_assistant_agent_pool'
]
