#!/usr/bin/env python3
"""
类型兼容性检查工具模块
提供函数输入输出类型兼容性分析功能
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import inspect

# 设置日志
logger = logging.getLogger(__name__)


class TypeCompatibilityChecker:
    """类型兼容性检查器"""
    
    def __init__(self):
        # 基础类型映射
        self.type_mappings = {
            # Python基础类型
            'str': ['string', 'text', 'char', 'varchar'],
            'int': ['integer', 'number', 'numeric', 'long'],
            'float': ['double', 'decimal', 'real', 'numeric'],
            'bool': ['boolean', 'flag', 'binary'],
            'list': ['array', 'sequence', 'collection'],
            'dict': ['object', 'map', 'hash', 'record'],
            'tuple': ['pair', 'sequence'],
            
            # 数据科学类型
            'DataFrame': ['dataframe', 'data', 'table', 'dataset'],
            'Series': ['series', 'column', 'vector'],
            'ndarray': ['array', 'matrix', 'tensor'],
            
            # 文件类型
            'file': ['path', 'filename', 'filepath'],
            'image': ['img', 'picture', 'photo'],
            'json': ['dict', 'object'],
            'csv': ['dataframe', 'table'],
            
            # 通用类型
            'Any': ['any', 'object', 'unknown'],
            'None': ['null', 'void', 'empty']
        }
        
        # 类型兼容性矩阵 (source -> target 的兼容性分数)
        self.compatibility_matrix = {
            # 数值类型兼容性
            ('int', 'float'): 0.9,
            ('float', 'int'): 0.7,  # 可能丢失精度
            ('int', 'str'): 0.8,
            ('float', 'str'): 0.8,
            ('str', 'int'): 0.6,    # 需要解析
            ('str', 'float'): 0.6,
            
            # 集合类型兼容性
            ('list', 'tuple'): 0.8,
            ('tuple', 'list'): 0.8,
            ('list', 'str'): 0.5,   # 需要序列化
            ('dict', 'str'): 0.5,   # 需要序列化
            ('str', 'dict'): 0.4,   # 需要解析
            ('str', 'list'): 0.4,
            
            # 数据科学类型兼容性
            ('DataFrame', 'dict'): 0.8,
            ('dict', 'DataFrame'): 0.7,
            ('DataFrame', 'list'): 0.7,
            ('list', 'DataFrame'): 0.6,
            ('Series', 'list'): 0.9,
            ('list', 'Series'): 0.8,
            ('Series', 'ndarray'): 0.9,
            ('ndarray', 'Series'): 0.8,
            
            # 文件类型兼容性
            ('str', 'file'): 0.9,   # 路径字符串
            ('file', 'str'): 0.9,
            ('DataFrame', 'csv'): 0.9,
            ('csv', 'DataFrame'): 0.9,
            ('dict', 'json'): 0.9,
            ('json', 'dict'): 0.9,
            
            # Any类型兼容性
            ('Any', 'Any'): 1.0,
        }
    
    def normalize_type_name(self, type_str: str) -> str:
        """标准化类型名称"""
        if not type_str:
            return 'Any'
        
        # 清理类型字符串
        type_str = type_str.strip().lower()
        
        # 移除常见的修饰符
        type_str = re.sub(r'\(.*?\)', '', type_str)  # 移除括号内容
        type_str = re.sub(r'[^\w]', '', type_str)    # 移除特殊字符
        
        # 查找映射
        for standard_type, aliases in self.type_mappings.items():
            if type_str == standard_type.lower() or type_str in aliases:
                return standard_type
        
        return type_str.capitalize() if type_str else 'Any'
    
    def calculate_type_compatibility(self, source_type: str, target_type: str) -> float:
        """计算类型兼容性分数"""
        # 标准化类型名称
        source = self.normalize_type_name(source_type)
        target = self.normalize_type_name(target_type)
        
        # 完全匹配
        if source == target:
            return 1.0
        
        # Any类型兼容所有类型
        if source == 'Any' or target == 'Any':
            return 0.9
        
        # 查找兼容性矩阵
        compatibility_key = (source, target)
        if compatibility_key in self.compatibility_matrix:
            return self.compatibility_matrix[compatibility_key]
        
        # 反向查找
        reverse_key = (target, source)
        if reverse_key in self.compatibility_matrix:
            return self.compatibility_matrix[reverse_key] * 0.8  # 反向兼容性稍低
        
        # 基于字符串相似度的fallback
        similarity = self._calculate_string_similarity(source, target)
        return max(similarity * 0.5, 0.1)  # 最低0.1分
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度"""
        import difflib
        return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def parse_function_signature(self, func_info: Dict[str, Any]) -> Tuple[List[str], str]:
        """解析函数签名，提取输入输出类型"""
        input_types = []
        output_type = 'Any'
        
        try:
            # 尝试从函数代码中解析
            if 'code' in func_info:
                input_types, output_type = self._parse_from_code(func_info['code'])
            
            # 尝试从描述中解析
            elif 'description' in func_info:
                input_types, output_type = self._parse_from_description(func_info['description'])
            
            # 尝试从参数信息中解析
            elif 'parameters' in func_info:
                input_types = self._parse_from_parameters(func_info['parameters'])
            
        except Exception as e:
            logger.debug(f"函数签名解析失败: {e}")
        
        return input_types or ['Any'], output_type
    
    def _parse_from_code(self, code: str) -> Tuple[List[str], str]:
        """从函数代码中解析类型"""
        input_types = []
        output_type = 'Any'
        
        # 查找类型注解
        type_hints_pattern = r'def\s+\w+\s*\((.*?)\)\s*->\s*([^:]+):'
        match = re.search(type_hints_pattern, code, re.DOTALL)
        
        if match:
            params_str, return_type = match.groups()
            
            # 解析参数类型
            param_pattern = r'(\w+)\s*:\s*([^,)]+)'
            for param_match in re.finditer(param_pattern, params_str):
                param_name, param_type = param_match.groups()
                if param_name != 'self':
                    input_types.append(param_type.strip())
            
            # 解析返回类型
            output_type = return_type.strip()
        
        return input_types, output_type
    
    def _parse_from_description(self, description: str) -> Tuple[List[str], str]:
        """从函数描述中解析类型"""
        input_types = []
        output_type = 'Any'
        
        # 查找输入输出模式
        input_pattern = r'输入[：:]\s*([^，,。.]+)'
        output_pattern = r'输出[：:]\s*([^，,。.]+)'
        
        input_match = re.search(input_pattern, description)
        if input_match:
            input_types = [input_match.group(1).strip()]
        
        output_match = re.search(output_pattern, description)
        if output_match:
            output_type = output_match.group(1).strip()
        
        return input_types, output_type
    
    def _parse_from_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """从参数信息中解析类型"""
        input_types = []
        
        for param_name, param_info in parameters.items():
            if isinstance(param_info, dict) and 'type' in param_info:
                input_types.append(param_info['type'])
            else:
                input_types.append('Any')
        
        return input_types
    
    def check_input_compatibility(self, subtask_input_type: str, func_input_types: List[str]) -> float:
        """检查输入类型兼容性"""
        if not func_input_types:
            return 0.5  # 无类型信息时给中等分数
        
        # 计算与所有输入类型的兼容性，取最高分
        max_compatibility = 0.0
        for func_input_type in func_input_types:
            compatibility = self.calculate_type_compatibility(subtask_input_type, func_input_type)
            max_compatibility = max(max_compatibility, compatibility)
        
        return max_compatibility
    
    def check_output_compatibility(self, subtask_output_type: str, func_output_type: str) -> float:
        """检查输出类型兼容性"""
        return self.calculate_type_compatibility(func_output_type, subtask_output_type)


# 全局实例
_type_checker = None

def get_type_checker() -> TypeCompatibilityChecker:
    """获取类型检查器实例（单例模式）"""
    global _type_checker
    if _type_checker is None:
        _type_checker = TypeCompatibilityChecker()
    return _type_checker


def check_input_compatibility(subtask_input_type: str, func_info: Dict[str, Any]) -> float:
    """检查输入类型兼容性的便捷函数"""
    checker = get_type_checker()
    func_input_types, _ = checker.parse_function_signature(func_info)
    return checker.check_input_compatibility(subtask_input_type, func_input_types)


def check_output_compatibility(subtask_output_type: str, func_info: Dict[str, Any]) -> float:
    """检查输出类型兼容性的便捷函数"""
    checker = get_type_checker()
    _, func_output_type = checker.parse_function_signature(func_info)
    return checker.check_output_compatibility(subtask_output_type, func_output_type)


# 导出的公共接口
__all__ = [
    'TypeCompatibilityChecker',
    'get_type_checker',
    'check_input_compatibility',
    'check_output_compatibility'
]
