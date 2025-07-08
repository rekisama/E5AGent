#!/usr/bin/env python3
"""
任务处理工具函数模块
提供任务分解、fallback处理等通用工具函数
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

# 设置日志
logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """子任务数据类"""
    id: str
    description: str
    input_type: str
    output_type: str
    priority: int = 1
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


def create_simple_fallback_tasks(task_description: str) -> List[SubTask]:
    """
    创建简化的fallback任务分解
    
    Args:
        task_description: 任务描述
        
    Returns:
        简化的任务列表
    """
    logger.info("使用简化的任务分解策略")
    
    # 基于关键词的简单分解
    fallback_tasks = []
    
    # 检测常见的任务模式
    if any(keyword in task_description for keyword in ['读取', '文件', 'CSV', 'Excel']):
        fallback_tasks.append(SubTask(
            id="task_1",
            description="读取数据文件",
            input_type="str",
            output_type="DataFrame",
            priority=1,
            dependencies=[]
        ))
    
    if any(keyword in task_description for keyword in ['处理', '清洗', '预处理']):
        fallback_tasks.append(SubTask(
            id="task_2", 
            description="数据处理",
            input_type="DataFrame",
            output_type="DataFrame",
            priority=2,
            dependencies=["task_1"] if fallback_tasks else []
        ))
    
    if any(keyword in task_description for keyword in ['计算', '统计', '分析']):
        fallback_tasks.append(SubTask(
            id="task_3",
            description="数据分析",
            input_type="DataFrame", 
            output_type="dict",
            priority=3,
            dependencies=[fallback_tasks[-1].id] if fallback_tasks else []
        ))
    
    if any(keyword in task_description for keyword in ['生成', '报告', '输出', '保存']):
        fallback_tasks.append(SubTask(
            id="task_4",
            description="生成输出",
            input_type="dict",
            output_type="str",
            priority=4,
            dependencies=[fallback_tasks[-1].id] if fallback_tasks else []
        ))
    
    # 如果没有匹配到任何模式，创建通用任务
    if not fallback_tasks:
        fallback_tasks.append(SubTask(
            id="task_1",
            description=task_description,
            input_type="Any",
            output_type="Any", 
            priority=1,
            dependencies=[]
        ))
    
    logger.info(f"创建了 {len(fallback_tasks)} 个fallback任务")
    return fallback_tasks


def get_fallback_configs(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    获取fallback模型配置
    
    Args:
        base_config: 基础配置
        
    Returns:
        fallback配置列表
    """
    import os
    
    fallback_configs = []
    
    # Fallback 1: 降低temperature
    config1 = base_config.copy()
    config1['temperature'] = 0.1
    fallback_configs.append(config1)
    
    # Fallback 2: 使用不同的模型（如果配置了）
    if os.getenv('FALLBACK_MODEL'):
        config2 = base_config.copy()
        config2['model'] = os.getenv('FALLBACK_MODEL', 'gpt-3.5-turbo')
        
        # 更新API配置
        if os.getenv('FALLBACK_API_KEY'):
            config2['api_key'] = os.getenv('FALLBACK_API_KEY')
        if os.getenv('FALLBACK_BASE_URL'):
            config2['base_url'] = os.getenv('FALLBACK_BASE_URL')
        
        fallback_configs.append(config2)
    
    return fallback_configs


def validate_subtask_structure(subtask_data: Dict[str, Any]) -> bool:
    """
    验证子任务数据结构
    
    Args:
        subtask_data: 子任务数据字典
        
    Returns:
        是否有效
    """
    required_fields = ['id', 'description', 'input_type', 'output_type']
    
    for field in required_fields:
        if field not in subtask_data:
            logger.warning(f"子任务缺少必需字段: {field}")
            return False
    
    return True


def normalize_subtask_data(subtask_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化子任务数据
    
    Args:
        subtask_data: 原始子任务数据
        
    Returns:
        标准化后的数据
    """
    normalized = subtask_data.copy()
    
    # 设置默认值
    if 'priority' not in normalized:
        normalized['priority'] = 1
    
    if 'dependencies' not in normalized:
        normalized['dependencies'] = []
    
    # 确保dependencies是列表
    if not isinstance(normalized['dependencies'], list):
        normalized['dependencies'] = []
    
    return normalized


def create_subtask_from_dict(data: Dict[str, Any]) -> SubTask:
    """
    从字典创建SubTask对象
    
    Args:
        data: 子任务数据字典
        
    Returns:
        SubTask对象
    """
    # 验证和标准化数据
    if not validate_subtask_structure(data):
        raise ValueError(f"无效的子任务数据结构: {data}")
    
    normalized_data = normalize_subtask_data(data)
    
    return SubTask(
        id=normalized_data['id'],
        description=normalized_data['description'],
        input_type=normalized_data['input_type'],
        output_type=normalized_data['output_type'],
        priority=normalized_data.get('priority', 1),
        dependencies=normalized_data.get('dependencies', [])
    )


def analyze_task_complexity(task_description: str) -> Dict[str, Any]:
    """
    分析任务复杂度
    
    Args:
        task_description: 任务描述
        
    Returns:
        复杂度分析结果
    """
    # 简单的复杂度分析
    word_count = len(task_description.split())
    
    # 检测复杂度指标
    complexity_keywords = ['多个', '复杂', '高级', '机器学习', '深度学习', '大数据']
    complexity_score = sum(1 for keyword in complexity_keywords if keyword in task_description)
    
    # 检测步骤数量
    step_indicators = ['首先', '然后', '接着', '最后', '步骤', '阶段']
    step_count = sum(1 for indicator in step_indicators if indicator in task_description)
    
    return {
        'word_count': word_count,
        'complexity_score': complexity_score,
        'estimated_steps': max(step_count, 1),
        'is_complex': complexity_score > 2 or word_count > 50
    }


def suggest_decomposition_strategy(task_description: str) -> str:
    """
    建议任务分解策略
    
    Args:
        task_description: 任务描述
        
    Returns:
        建议的分解策略
    """
    analysis = analyze_task_complexity(task_description)
    
    if analysis['is_complex']:
        return "complex_llm_decomposition"
    elif analysis['estimated_steps'] > 3:
        return "multi_step_decomposition"
    else:
        return "simple_decomposition"


# 导出的公共接口
__all__ = [
    'SubTask',
    'create_simple_fallback_tasks',
    'get_fallback_configs',
    'validate_subtask_structure',
    'normalize_subtask_data',
    'create_subtask_from_dict',
    'analyze_task_complexity',
    'suggest_decomposition_strategy'
]
