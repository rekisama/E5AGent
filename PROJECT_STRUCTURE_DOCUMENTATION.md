# AutoGen Self-Expanding Agent System - 项目结构说明

## 📁 项目总览

```
E5Agent/
├── autogen_godel_agent/           # 主项目目录
│   ├── agents/                    # 智能代理模块
│   ├── tools/                     # 核心工具模块
│   ├── memory/                    # 持久化存储
│   ├── config.py                  # 配置管理
│   ├── main.py                    # 主入口点
│   ├── demo.py                    # 演示脚本
│   ├── interactive.py             # 交互式界面
│   └── requirements.txt           # 依赖管理
├── memory/                        # 全局存储目录
├── test_modular_refactoring.py    # 模块化测试
├── README.md                      # 项目说明
└── *.md                          # 文档文件
```

## 🤖 核心代理模块 (agents/)

### 1. `planner_agent.py` (335行)
**功能**: 任务分析和函数发现代理
**主要类**: `TaskPlannerAgent`
**核心方法**:
- `__init__(llm_config, max_tokens_per_minute=10000)` - 初始化代理
- `analyze_task(task_description, session_id=None) -> Dict[str, Any]` - 分析任务需求
- `get_session_stats(session_id) -> Dict[str, Any]` - 获取会话统计
- `cleanup_old_sessions(max_age_hours=24)` - 清理过期会话

**依赖工具**:
- `session_manager` - 会话管理和令牌跟踪
- `response_parser` - LLM响应解析
- `agent_pool` - UserProxy代理池管理
- `function_tools` - 函数工具接口

### 2. `function_creator_agent.py` (400+行)
**功能**: 函数生成、验证和注册代理
**主要类**: `FunctionCreatorAgent`
**核心方法**:
- `__init__(llm_config)` - 初始化代理
- `create_function(specification) -> Tuple[bool, str, Optional[str]]` - 创建函数
- `get_creation_prompt(specification) -> str` - 生成创建提示
- `_validate_function_code(code) -> str` - 验证函数代码
- `_generate_test_cases(func_name, code, description) -> str` - 生成测试用例
- `_test_function(code, func_name, test_cases_json) -> str` - 测试函数
- `_register_function(name, code, description, origin, test_cases_json) -> str` - 注册函数

**依赖工具**:
- `function_tools` - 统一函数工具接口
- `test_runner.TestResult` - 测试结果类型

### 3. `test_case_generator.py` (104行) ✨ **已重构**
**功能**: 智能测试用例生成（模块化架构）
**主要类**: `EnhancedTestCaseGenerator`
**核心方法**:
- `__init__(config=None)` - 初始化生成器
- `generate_test_cases(specification, code="") -> Tuple[bool, str, List[Dict]]` - 生成测试用例（兼容格式）
- `generate_enhanced_test_cases(specification, code="") -> List[Dict]` - 生成增强测试用例

**依赖工具**:
- `test_runner.TestCaseGenerator` - 模块化测试生成器
- `secure_executor.FunctionSignatureParser` - 签名解析器

## 🛠️ 核心工具模块 (tools/)

### 1. `function_tools.py` (500+行)
**功能**: 统一函数工具接口
**主要类**: 
- `FunctionToolsInterface` - 抽象接口
- `FunctionTools` - 具体实现
- `FunctionCreationResult` - 结果数据类

**核心方法**:
- `validate_function_code(code) -> Tuple[bool, str, str]` - 验证函数代码
- `execute_code_safely(code, timeout_seconds=10) -> Tuple[bool, str, Dict]` - 安全执行代码
- `generate_test_cases(func_name, func_code, task_description) -> List[Dict]` - 生成测试用例
- `run_tests(func_code, test_cases) -> TestResult` - 运行测试
- `has_function(func_name) -> bool` - 检查函数存在性
- `register_function(func_name, func_code, description, task_origin="", test_cases=None) -> bool` - 注册函数
- `create_function_complete(func_name, task_description, func_code) -> FunctionCreationResult` - 完整函数创建流程

**工厂函数**: `get_function_tools() -> FunctionTools`

### 2. `function_registry.py` (320+行)
**功能**: 函数注册和管理
**主要类**: `FunctionRegistry`
**核心方法**:
- `__init__(registry_file="memory/function_registry.json")` - 初始化注册表
- `has_function(func_name) -> bool` - 检查函数存在性
- `register_function(func_name, func_code, description, task_origin="", test_cases=None) -> bool` - 注册函数
- `get_function(func_name) -> Optional[Dict[str, Any]]` - 获取函数信息
- `get_function_code(func_name) -> Optional[str]` - 获取函数代码
- `list_functions() -> List[str]` - 列出所有函数名
- `search_functions(query) -> List[Dict[str, Any]]` - 搜索函数
- `get_registry_stats() -> Dict[str, Any]` - 获取注册表统计

**工厂函数**: `get_registry() -> FunctionRegistry`

### 3. `secure_executor.py` (578行)
**功能**: 安全验证和代码执行
**主要类**: 
- `SecurityValidator` - AST安全分析器
- `FunctionSignatureParser` - 函数签名解析器

**核心函数**:
- `validate_function_code(code) -> Tuple[bool, str, str]` - 验证函数代码
- `execute_code_safely(code, timeout_seconds=10) -> Tuple[bool, str, Dict[str, Any]]` - 安全执行代码
- `validate_function_signature(signature) -> Tuple[bool, str]` - 验证函数签名
- `extract_function_signature_from_code(code) -> Optional[str]` - 从代码提取签名

**SecurityValidator方法**:
- `visit_Call(node)` - 检查函数调用
- `visit_Import(node)` - 检查导入语句
- `get_violations() -> List[str]` - 获取安全违规

### 4. `test_runner.py` (720行)
**功能**: 测试用例生成和执行
**主要类**:
- `TestCaseGenerator` - 测试用例生成器
- `TestResponseParser` - 测试响应解析器
- `TestCaseStandardizer` - 测试用例标准化器
- `TestResult` - 测试结果数据类
- `TestGenerationConfig` - 测试生成配置

**核心方法**:
- `TestCaseGenerator.generate_enhanced_test_cases(specification, code="") -> List[Dict]` - 生成增强测试用例
- `TestResponseParser.parse_test_response(response) -> List[Dict]` - 解析测试响应
- `TestCaseStandardizer.standardize_test_cases(test_cases, input_format) -> List[Dict]` - 标准化测试用例

### 5. `session_manager.py` (282行) ⚠️ **仅planner_agent使用**
**功能**: 会话管理和令牌跟踪
**主要类**:
- `SessionManager` - 会话管理器
- `TokenUsageStats` - 令牌使用统计
- `SessionContext` - 会话上下文

**核心方法**:
- `create_session(session_id=None, metadata=None) -> str` - 创建会话
- `get_session(session_id) -> Optional[SessionContext]` - 获取会话
- `add_token_usage(session_id, prompt_tokens, completion_tokens)` - 添加令牌使用
- `enforce_rate_limit()` - 执行速率限制
- `cleanup_old_sessions(max_age_hours=24)` - 清理过期会话

**工厂函数**: `get_session_manager(max_tokens_per_minute=10000) -> SessionManager`

### 6. `response_parser.py` (380行) ⚠️ **仅planner_agent使用**
**功能**: LLM响应解析
**主要类**: `ResponseParser`
**核心方法**:
- `parse_llm_analysis(llm_response, task_description) -> Dict[str, Any]` - 解析LLM分析响应
- `_extract_json_from_response(response) -> Optional[str]` - 从响应提取JSON
- `_validate_and_complete_json_structure(parsed_json, default_result) -> Dict` - 验证和完善JSON结构

**工厂函数**: `get_response_parser() -> ResponseParser`

### 7. `agent_pool.py` (272行) ⚠️ **仅planner_agent使用**
**功能**: AutoGen代理池管理
**主要类**:
- `UserProxyPool` - UserProxy代理池
- `AssistantAgentPool` - Assistant代理池

**核心方法**:
- `@contextmanager get_user_proxy(custom_config=None)` - 获取UserProxy代理
- `get_pool_stats() -> Dict[str, Any]` - 获取池统计
- `clear_pool()` - 清空代理池

**工厂函数**: 
- `get_user_proxy_pool(max_size=5, agent_config=None) -> UserProxyPool`
- `get_assistant_agent_pool(max_size=3) -> AssistantAgentPool`

## 📋 主要入口文件

### 1. `main.py` (320+行)
**功能**: 系统主入口点
**主要类**: `SelfExpandingAgentSystem`
**核心方法**:
- `__init__()` - 初始化系统
- `process_task(task_description) -> Dict[str, Any]` - 处理任务
- `list_available_functions() -> List[Dict]` - 列出可用函数
- `get_system_stats() -> Dict[str, Any]` - 获取系统统计

### 2. `interactive.py` (200+行)
**功能**: 交互式界面
**主要函数**:
- `print_menu()` - 打印菜单
- `list_functions()` - 列出函数
- `search_functions()` - 搜索函数
- `execute_function()` - 执行函数
- `create_function()` - 创建函数
- `show_stats()` - 显示统计
- `test_function()` - 测试函数

### 3. `demo.py` (200+行)
**功能**: 系统演示脚本
**主要函数**:
- `demo_function_creation()` - 演示函数创建
- `demo_function_testing()` - 演示函数测试
- `demo_function_execution()` - 演示函数执行
- `demo_system_stats()` - 演示系统统计

### 4. `config.py` (100+行)
**功能**: 配置管理
**主要类**: `Config`
**配置项**:
- LLM提供商配置 (DeepSeek/OpenAI/Azure)
- 代理设置
- 文件路径
- 安全设置

**核心方法**:
- `get_llm_config() -> Dict[str, Any]` - 获取LLM配置
- `validate_config() -> bool` - 验证配置

## 💾 存储和测试

### 存储目录
- `memory/function_registry.json` - 函数注册表
- `memory/history.json` - 历史记录

### 测试文件
- `test_modular_refactoring.py` - 模块化重构测试套件

## 🔄 模块依赖关系

```
main.py
├── agents/planner_agent.py
│   ├── tools/session_manager.py
│   ├── tools/response_parser.py
│   ├── tools/agent_pool.py
│   └── tools/function_tools.py
├── agents/function_creator_agent.py
│   ├── tools/function_tools.py
│   └── tools/test_runner.py
└── agents/test_case_generator.py
    ├── tools/test_runner.py
    └── tools/secure_executor.py

tools/function_tools.py (统一接口)
├── tools/secure_executor.py
├── tools/test_runner.py
└── tools/function_registry.py
```

## 📊 代码统计

| 模块 | 行数 | 状态 | 重要性 |
|------|------|------|--------|
| planner_agent.py | 335 | ✅ 已重构 | 🔴 核心 |
| function_creator_agent.py | 400+ | ✅ 模块化 | 🔴 核心 |
| test_case_generator.py | 104 | ✅ 已重构 | 🔴 核心 |
| function_tools.py | 500+ | ✅ 统一接口 | 🔴 核心 |
| function_registry.py | 320+ | ✅ 完善 | 🔴 核心 |
| secure_executor.py | 578 | ✅ 增强 | 🔴 核心 |
| test_runner.py | 720 | ✅ 增强 | 🔴 核心 |
| session_manager.py | 282 | ✅ 专用 | 🟡 特定 |
| response_parser.py | 380 | ✅ 专用 | 🟡 特定 |
| agent_pool.py | 272 | ✅ 专用 | 🟡 特定 |

**总计**: ~3,500+ 行代码，完全模块化架构 🎉

## 🚀 核心功能流程

### 1. 任务处理流程
```
用户任务 → TaskPlannerAgent.analyze_task()
├── 会话管理 (SessionManager)
├── 函数搜索 (FunctionTools)
├── LLM分析 (AutoGen + ResponseParser)
└── 结果返回 (需要新函数 or 使用现有函数)

如需新函数 → FunctionCreatorAgent.create_function()
├── 代码生成 (AutoGen LLM)
├── 安全验证 (SecurityValidator)
├── 测试生成 (TestCaseGenerator)
├── 测试执行 (TestRunner)
└── 函数注册 (FunctionRegistry)
```

### 2. 函数创建流程
```
函数规范 → 代码生成 → 安全验证 → 测试生成 → 测试执行 → 注册存储
    ↓           ↓           ↓           ↓           ↓           ↓
LLM生成    AST分析    智能生成    沙箱执行    持久化    可调用
```

### 3. 测试用例生成流程
```
函数规范 → 签名解析 → LLM生成 → 格式标准化 → 测试执行
    ↓           ↓           ↓           ↓           ↓
参数分析    类型推断    智能用例    统一格式    结果验证
```

## 🔧 关键技术特性

### 安全特性
- **AST安全分析**: 检测危险函数调用和模块导入
- **沙箱执行**: 使用multiprocessing隔离代码执行
- **受限命名空间**: 仅允许安全的内置函数和模块
- **超时保护**: 防止无限循环和长时间执行

### 性能优化
- **代理池**: 重用UserProxyAgent实例，避免重复创建开销
- **会话管理**: 上下文隔离和令牌使用跟踪
- **速率限制**: 防止API调用过于频繁
- **缓存机制**: 函数注册表和测试结果缓存

### 智能特性
- **LLM驱动**: 使用大语言模型进行任务分析和代码生成
- **多策略测试**: LLM生成 + 规则回退的测试用例生成
- **语义搜索**: 基于描述的函数搜索和匹配
- **自适应生成**: 根据函数复杂度调整测试用例数量

## 📚 使用示例

### 基本使用
```python
from autogen_godel_agent.main import SelfExpandingAgentSystem

# 初始化系统
system = SelfExpandingAgentSystem()

# 处理任务
result = system.process_task("创建一个计算圆面积的函数")
print(result['status'])  # completed_with_new_function

# 查看系统统计
stats = system.get_system_stats()
print(f"总函数数: {stats['total_functions']}")
```

### 直接使用工具
```python
from autogen_godel_agent.tools.function_tools import get_function_tools

# 获取函数工具
tools = get_function_tools()

# 注册函数
success = tools.register_function(
    func_name="calculate_area",
    func_code="def calculate_area(radius: float) -> float:\n    import math\n    return math.pi * radius * radius",
    description="计算圆面积",
    task_origin="用户请求"
)

# 生成测试用例
test_cases = tools.generate_test_cases(
    func_name="calculate_area",
    func_code="def calculate_area(radius: float) -> float:\n    import math\n    return math.pi * radius * radius",
    task_description="计算圆面积"
)
```

### 交互式使用
```bash
cd autogen_godel_agent
python interactive.py
```

## 🔄 扩展指南

### 添加新的代理
1. 在 `agents/` 目录创建新文件
2. 继承或使用 AutoGen 的 Agent 类
3. 使用 `tools/` 中的模块化组件
4. 在 `main.py` 中集成新代理

### 添加新的工具
1. 在 `tools/` 目录创建新模块
2. 实现相应的接口和工厂函数
3. 在 `function_tools.py` 中集成（如需要）
4. 更新相关代理的依赖

### 扩展安全策略
1. 修改 `SecurityValidator` 的危险函数/模块列表
2. 添加新的 AST 节点访问方法
3. 更新安全命名空间配置

## 🧪 测试和验证

### 运行测试套件
```bash
python test_modular_refactoring.py
```

### 测试覆盖
- ✅ 模块导入测试
- ✅ 会话管理测试
- ✅ 响应解析测试
- ✅ 代理池测试
- ✅ 函数签名验证测试
- ✅ 集成测试

## 📈 项目演进历史

### 重构里程碑
1. **初始版本**: 单体架构，所有功能混合
2. **第一次重构**: TaskPlannerAgent 模块化 (869行 → 335行 + 4个模块)
3. **第二次重构**: TestCaseGenerator 模块化 (946行 → 104行)
4. **当前状态**: 完全模块化架构，单一职责原则

### 代码质量提升
- **代码重复**: 大幅减少，共享组件复用
- **可维护性**: 模块化设计，易于修改和扩展
- **可测试性**: 独立模块，便于单元测试
- **性能**: 代理池和缓存机制优化

---

*本文档生成时间: 2025-07-04*
*项目状态: 生产就绪，完全模块化架构* ✨
