#!/usr/bin/env python3
"""
JSON工具函数模块
提供JSON提取和解析的通用工具函数
"""

import json
import re
import logging
from typing import Optional, Dict, Any, Union, List

# 设置日志
logger = logging.getLogger(__name__)

# 尝试导入json5
try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False
    logger.debug("json5 not available, using fallback JSON parsing")


def extract_json_block(response: str) -> Optional[str]:
    """
    提取JSON块的工具函数 - 支持多种格式
    
    Args:
        response: 原始响应文本
        
    Returns:
        提取的JSON字符串，如果未找到则返回None
    """
    if not response or not isinstance(response, str):
        logger.warning("输入响应为空或不是字符串类型")
        return None
    
    # 方法1: 尝试提取markdown格式的JSON
    json_match = re.search(r'```json\s*\n(.*?)```', response, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # 方法2: 尝试提取普通代码块
    json_match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if json_match:
        content = json_match.group(1).strip()
        # 简单验证是否像JSON
        if content.startswith('{') and content.endswith('}'):
            return content
        if content.startswith('[') and content.endswith(']'):
            return content
    
    # 方法3: 尝试提取JSON对象 - 改进版本
    # 寻找平衡的大括号对
    json_candidates = []
    
    # 对象格式 {...}
    for match in re.finditer(r'\{', response):
        start = match.start()
        brace_count = 1
        i = start + 1
        
        while i < len(response) and brace_count > 0:
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
            i += 1
        
        if brace_count == 0:
            json_candidates.append(response[start:i])
    
    # 数组格式 [...]
    for match in re.finditer(r'\[', response):
        start = match.start()
        bracket_count = 1
        i = start + 1
        
        while i < len(response) and bracket_count > 0:
            if response[i] == '[':
                bracket_count += 1
            elif response[i] == ']':
                bracket_count -= 1
            i += 1
        
        if bracket_count == 0:
            json_candidates.append(response[start:i])
    
    # 返回第一个有效的JSON候选
    for candidate in json_candidates:
        if len(candidate) > 2:  # 至少要有内容
            return candidate.strip()
    
    return None


def normalize_json_quotes(json_str: str) -> str:
    """
    智能标准化JSON字符串中的引号
    
    Args:
        json_str: 原始JSON字符串
        
    Returns:
        标准化后的JSON字符串
    """
    if not json_str:
        return json_str
    
    # 更智能的单引号处理
    # 只替换作为JSON语法的单引号，不替换字符串内容中的单引号
    result = []
    in_string = False
    i = 0
    
    while i < len(json_str):
        char = json_str[i]
        
        if char == '"' and (i == 0 or json_str[i-1] != '\\'):
            in_string = not in_string
            result.append(char)
        elif char == "'" and not in_string:
            # 检查是否是JSON语法中的单引号
            if (i == 0 or json_str[i-1] in ['{', '[', ':', ',', ' ', '\n', '\t']) and \
               (i + 1 < len(json_str) and json_str[i+1] not in [' ', '\n', '\t']):
                result.append('"')
            else:
                result.append(char)
        else:
            result.append(char)
        
        i += 1
    
    return ''.join(result)


def parse_json_with_fallback(json_str: str) -> Optional[Dict[str, Any]]:
    """
    多重fallback的JSON解析
    
    Args:
        json_str: JSON字符串
        
    Returns:
        解析后的字典，如果失败则返回None
    """
    if not json_str or not isinstance(json_str, str):
        logger.warning("JSON字符串为空或不是字符串类型")
        return None
    
    # 清理输入
    json_str = json_str.strip()
    
    # 方法1: 标准JSON解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug(f"标准JSON解析失败: {e}")
    
    # 方法2: 使用json5（如果可用）
    if HAS_JSON5:
        try:
            return json5.loads(json_str)
        except Exception as e:
            logger.debug(f"JSON5解析失败: {e}")
    
    # 方法3: 尝试修复常见的JSON格式问题
    try:
        # 修复单引号问题
        fixed_json = normalize_json_quotes(json_str)
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        logger.debug(f"修复后的JSON解析失败: {e}")
    
    # 方法4: 尝试移除常见的多余字符
    try:
        # 移除可能的前后缀
        cleaned = re.sub(r'^[^{\[]*([{\[].*[}\]])[^}\]]*$', r'\1', json_str, flags=re.DOTALL)
        if cleaned != json_str:
            return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.debug(f"清理后的JSON解析失败: {e}")
    
    logger.error(f"所有JSON解析方法均失败，原始内容: {json_str[:200]}...")
    return None


def extract_and_parse_json(response: str) -> Optional[Dict[str, Any]]:
    """
    组合函数：提取并解析JSON
    
    Args:
        response: 原始响应文本
        
    Returns:
        解析后的字典，如果失败则返回None
    """
    # 提取JSON块
    json_str = extract_json_block(response)
    if not json_str:
        logger.error("未找到有效的JSON响应")
        logger.debug(f"原始响应: {response[:500]}...")
        return None
    
    # 解析JSON
    data = parse_json_with_fallback(json_str)
    if data is None:
        logger.error(f"JSON解析失败，提取的内容: {json_str[:200]}...")
        return None
    
    return data


def safe_json_extract(response: str, default_value: Any = None) -> Any:
    """
    安全的JSON提取，失败时返回默认值
    
    Args:
        response: 原始响应文本
        default_value: 失败时返回的默认值
        
    Returns:
        解析后的数据或默认值
    """
    try:
        result = extract_and_parse_json(response)
        return result if result is not None else default_value
    except Exception as e:
        logger.error(f"JSON提取失败: {e}")
        return default_value


def validate_json_structure(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    验证JSON结构是否包含必需的键
    
    Args:
        data: 解析后的JSON数据
        required_keys: 必需的键列表
        
    Returns:
        是否包含所有必需的键
    """
    if not isinstance(data, dict):
        logger.warning("数据不是字典类型")
        return False
    
    missing_keys = []
    for key in required_keys:
        if key not in data:
            missing_keys.append(key)
    
    if missing_keys:
        logger.warning(f"缺少必需的键: {missing_keys}")
        return False
    
    return True


def pretty_print_json(data: Union[Dict[str, Any], List[Any]]) -> str:
    """
    美化打印JSON数据
    
    Args:
        data: JSON数据
        
    Returns:
        格式化的JSON字符串
    """
    try:
        return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON格式化失败: {e}")
        return str(data)


def extract_nested_value(data: Dict[str, Any], key_path: str, default_value: Any = None) -> Any:
    """
    从嵌套字典中提取值
    
    Args:
        data: 嵌套字典
        key_path: 键路径，用点分隔，如 "subtasks.0.description"
        default_value: 默认值
        
    Returns:
        提取的值或默认值
    """
    if not isinstance(data, dict) or not key_path:
        return default_value
    
    try:
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if key.isdigit():
                # 数组索引
                index = int(key)
                if isinstance(current, list) and 0 <= index < len(current):
                    current = current[index]
                else:
                    return default_value
            else:
                # 字典键
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default_value
        
        return current
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.debug(f"提取嵌套值失败: {e}")
        return default_value


def merge_json_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个JSON配置（深度合并）
    
    Args:
        *configs: 多个配置字典
        
    Returns:
        合并后的配置字典
    """
    def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    result = {}
    
    for config in configs:
        if isinstance(config, dict):
            result = deep_merge(result, config)
        else:
            logger.warning(f"跳过非字典类型的配置: {type(config)}")
    
    return result


def is_valid_json(json_str: str) -> bool:
    """
    检查字符串是否为有效的JSON
    
    Args:
        json_str: 待检查的字符串
        
    Returns:
        是否为有效JSON
    """
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def minify_json(data: Union[Dict[str, Any], List[Any]]) -> str:
    """
    压缩JSON数据（移除空格和换行）
    
    Args:
        data: JSON数据
        
    Returns:
        压缩后的JSON字符串
    """
    try:
        return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON压缩失败: {e}")
        return str(data)


# 导出的公共接口
__all__ = [
    'extract_json_block',
    'parse_json_with_fallback', 
    'extract_and_parse_json',
    'safe_json_extract',
    'validate_json_structure',
    'normalize_json_quotes',
    'pretty_print_json',
    'extract_nested_value',
    'merge_json_configs',
    'is_valid_json',
    'minify_json'
]