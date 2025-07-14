"""
Simple Workflow Visualizer

A simplified version of the workflow visualizer that avoids complex imports
and provides basic visualization functionality.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_mermaid_from_description(task_description: str, workflow_type: str = "standard") -> str:
    """
    Generate a Mermaid diagram from task description.
    
    Args:
        task_description: Description of the task
        workflow_type: Type of workflow (standard, evo, multifile)
        
    Returns:
        Mermaid diagram string
    """
    
    if workflow_type == "evo":
        return _generate_evo_mermaid(task_description)
    elif workflow_type == "multifile":
        return _generate_multifile_mermaid(task_description)
    else:
        return _generate_standard_mermaid(task_description)


def _generate_standard_mermaid(task_description: str) -> str:
    """Generate Mermaid for standard task processing."""
    
    mermaid = f"""graph TD
    %% Standard Task Processing Flow
    %% Task: {task_description}
    %% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    A[用户任务输入<br/>{task_description[:30]}...] --> B[任务规划代理]
    B --> C[搜索现有函数]
    C --> D{{找到匹配函数?}}
    D -->|是| E[执行现有函数]
    D -->|否| F[函数创建代理]
    F --> G[生成函数代码]
    G --> H[安全验证]
    H --> I[生成测试用例]
    I --> J[执行测试]
    J --> K{{测试通过?}}
    K -->|是| L[注册函数]
    K -->|否| G
    L --> M[执行函数]
    E --> N[返回结果]
    M --> N
    
    %% Styling
    classDef inputBox fill:#e3f2fd,stroke:#01579b,stroke-width:2px
    classDef agentBox fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decisionBox fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef resultBox fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class A inputBox
    class B,F,G,H,I agentBox
    class D,K decisionBox
    class N resultBox
    """
    
    return mermaid


def _generate_evo_mermaid(task_description: str) -> str:
    """Generate Mermaid for EvoWorkflow processing."""
    
    mermaid = f"""graph TD
    %% EvoWorkflow Processing Flow
    %% Task: {task_description}
    %% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    A[复杂任务输入<br/>{task_description[:30]}...] --> B[EvoWorkflow管理器]
    B --> C[任务规划器]
    C --> D[LLM分析任务]
    D --> E[生成子任务列表]
    E --> F[确定依赖关系]
    F --> G[创建工作流图]
    G --> H[代理配置器]
    H --> I[创建AutoGen代理]
    I --> J[工作流执行器]
    J --> K[拓扑排序]
    K --> L[节点逐步执行]
    L --> M[结果聚合]
    M --> N[返回最终结果]
    
    %% Node details
    L --> L1[分析节点]
    L --> L2[执行节点]
    L --> L3[验证节点]
    L --> L4[聚合节点]
    
    %% Styling
    classDef managerBox fill:#e3f2fd,stroke:#0277bd,stroke-width:3px
    classDef generatorBox fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef executorBox fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef nodeBox fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef resultBox fill:#f1f8e9,stroke:#689f38,stroke-width:3px
    
    class B managerBox
    class C,D,E,F,G,H,I generatorBox
    class J,K,L executorBox
    class L1,L2,L3,L4 nodeBox
    class N resultBox
    """
    
    return mermaid


def _generate_multifile_mermaid(task_description: str) -> str:
    """Generate Mermaid for MultiFile processing."""
    
    mermaid = f"""graph TD
    %% MultiFile Generation Flow
    %% Task: {task_description}
    %% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    A[文件生成任务<br/>{task_description[:30]}...] --> B[MultiFile代理]
    B --> C[任务类型分类器]
    C --> D[项目规划器]
    D --> E[内容生成器]
    
    C --> C1{{项目类型}}
    C1 -->|网页| W1[HTML + CSS + JS]
    C1 -->|API| W2[Python + FastAPI]
    C1 -->|数据分析| W3[Jupyter + Pandas]
    
    D --> D1[创建目录结构]
    D1 --> D2[规划文件列表]
    D2 --> E
    
    E --> E1[LLM生成HTML]
    E --> E2[LLM生成CSS]
    E --> E3[LLM生成JavaScript]
    E --> E4[LLM生成README]
    
    E1 --> F[文件系统写入]
    E2 --> F
    E3 --> F
    E4 --> F
    
    F --> G[生成配置文件]
    G --> H[返回项目路径]
    
    %% Styling
    classDef inputBox fill:#e3f2fd,stroke:#01579b,stroke-width:2px
    classDef agentBox fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef typeBox fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef genBox fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef resultBox fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    
    class A inputBox
    class B,C,D,E agentBox
    class C1,W1,W2,W3 typeBox
    class E1,E2,E3,E4 genBox
    class H resultBox
    """
    
    return mermaid


def generate_ascii_from_description(task_description: str, workflow_type: str = "standard") -> str:
    """
    Generate ASCII art from task description.
    
    Args:
        task_description: Description of the task
        workflow_type: Type of workflow
        
    Returns:
        ASCII art string
    """
    
    if workflow_type == "evo":
        return _generate_evo_ascii(task_description)
    elif workflow_type == "multifile":
        return _generate_multifile_ascii(task_description)
    else:
        return _generate_standard_ascii(task_description)


def _generate_standard_ascii(task_description: str) -> str:
    """Generate ASCII for standard processing."""
    
    ascii_art = f"""
╔══════════════════════════════════════════════════════════════╗
║ 标准任务处理流程                                              ║
╠══════════════════════════════════════════════════════════════╣
║ 任务: {task_description[:50]:<50} ║
║ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<47} ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. 📋 任务规划代理 - 分析任务需求                            ║
║      ↓                                                       ║
║  2. 🔍 搜索现有函数 - 查找匹配的已注册函数                    ║
║      ↓                                                       ║
║  3. ❓ 决策点 - 是否找到合适的函数？                          ║
║      ├─ 是 → 4a. ✅ 执行现有函数                             ║
║      └─ 否 → 4b. 🔧 创建新函数                               ║
║                   ↓                                          ║
║                5. 🧪 生成测试用例                             ║
║                   ↓                                          ║
║                6. 🔒 安全验证                                 ║
║                   ↓                                          ║
║                7. 📝 注册函数                                 ║
║                   ↓                                          ║
║                8. ▶️ 执行函数                                 ║
║                                                              ║
║  9. 📊 返回结果 - 记录执行历史                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    
    return ascii_art


def _generate_evo_ascii(task_description: str) -> str:
    """Generate ASCII for EvoWorkflow."""
    
    ascii_art = f"""
╔══════════════════════════════════════════════════════════════╗
║ EvoWorkflow 复杂任务处理流程                                  ║
╠══════════════════════════════════════════════════════════════╣
║ 任务: {task_description[:50]:<50} ║
║ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<47} ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  阶段1: 工作流生成                                            ║
║  ├─ 📋 任务规划器 - LLM分析复杂任务                          ║
║  ├─ 🧩 子任务分解 - 拆解为可执行单元                         ║
║  ├─ 🔗 依赖分析 - 确定执行顺序                               ║
║  └─ 📊 工作流图 - 创建节点和边                               ║
║                                                              ║
║  阶段2: 代理配置                                              ║
║  ├─ 🤖 创建AutoGen代理 - 为每个节点分配代理                  ║
║  ├─ ⚙️ 配置代理角色 - 分析器、执行器、验证器                 ║
║  └─ 🔧 建立通信 - 代理间消息传递                             ║
║                                                              ║
║  阶段3: 执行引擎                                              ║
║  ├─ 📈 拓扑排序 - 确定最优执行顺序                           ║
║  ├─ 🔄 节点执行 - 逐步处理每个子任务                         ║
║  ├─ 🔍 结果验证 - 质量检查和错误处理                         ║
║  └─ 📊 结果聚合 - 合并所有节点输出                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    
    return ascii_art


def _generate_multifile_ascii(task_description: str) -> str:
    """Generate ASCII for MultiFile processing."""
    
    ascii_art = f"""
╔══════════════════════════════════════════════════════════════╗
║ MultiFile 项目生成流程                                        ║
╠══════════════════════════════════════════════════════════════╣
║ 任务: {task_description[:50]:<50} ║
║ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<47} ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  步骤1: 项目分析                                              ║
║  ├─ 🔍 任务分类器 - 识别项目类型                             ║
║  │   ├─ 🌐 网页项目 (HTML/CSS/JS)                            ║
║  │   ├─ 🔧 API项目 (Python/FastAPI)                          ║
║  │   └─ 📊 数据分析 (Jupyter/Pandas)                         ║
║  └─ 📋 项目规划 - 确定文件结构                               ║
║                                                              ║
║  步骤2: 内容生成                                              ║
║  ├─ 🏗️ 创建目录结构                                          ║
║  ├─ 📝 LLM生成文件内容                                        ║
║  │   ├─ index.html                                           ║
║  │   ├─ styles.css                                           ║
║  │   ├─ script.js                                            ║
║  │   └─ README.md                                            ║
║  └─ ⚙️ 生成配置文件                                           ║
║                                                              ║
║  步骤3: 项目输出                                              ║
║  ├─ 💾 写入文件系统                                           ║
║  ├─ 🔗 建立文件关联                                           ║
║  └─ 📦 返回项目路径                                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    
    return ascii_art


def export_visualization(content: str, file_path: str) -> bool:
    """
    Export visualization content to file.
    
    Args:
        content: Visualization content
        file_path: Path to export file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"✅ Visualization exported to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to export visualization: {e}")
        return False


def determine_workflow_type(task_description: str) -> str:
    """
    Determine workflow type based on task description.
    
    Args:
        task_description: Task description
        
    Returns:
        Workflow type: 'evo', 'multifile', or 'standard'
    """
    task_lower = task_description.lower()
    
    # Check for EvoWorkflow indicators
    evo_keywords = [
        'workflow', 'multi-step', 'complex', 'comprehensive', 'system',
        'analyze and create', 'process and validate', 'research and report'
    ]
    
    # Check for MultiFile indicators  
    multifile_keywords = [
        'website', 'webpage', 'html', 'css', 'javascript', 'api', 'project',
        'application', 'dashboard', 'portfolio', 'blog'
    ]
    
    if any(keyword in task_lower for keyword in evo_keywords):
        return 'evo'
    elif any(keyword in task_lower for keyword in multifile_keywords):
        return 'multifile'
    else:
        return 'standard'
