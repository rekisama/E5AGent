#!/usr/bin/env python3
"""
关键词管理工具模块
提供配置化的关键词匹配和管理功能
"""

import json
import os
import logging
from typing import Dict, List, Set, Any, Optional
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)


class KeywordManager:
    """关键词管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化关键词管理器
        
        Args:
            config_path: 关键词配置文件路径，默认使用内置配置
        """
        self.config_path = config_path or self._get_default_config_path()
        self.keywords_config = {}
        self._load_keywords_config()
    
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        current_dir = Path(__file__).parent.parent
        return str(current_dir / "config" / "keywords.json")
    
    def _load_keywords_config(self):
        """加载关键词配置"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.keywords_config = json.load(f)
                logger.info(f"成功加载关键词配置: {self.config_path}")
            else:
                logger.warning(f"关键词配置文件不存在: {self.config_path}")
                self._create_default_config()
        except Exception as e:
            logger.error(f"加载关键词配置失败: {e}")
            self._create_fallback_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        self.keywords_config = {
            "function_matching_keywords": {
                "data_processing": {
                    "chinese": ["数据", "处理", "清洗"],
                    "english": ["data", "process", "clean"]
                },
                "file_operations": {
                    "chinese": ["文件", "读取", "写入"],
                    "english": ["file", "read", "write"]
                }
            }
        }
        logger.info("使用默认关键词配置")
    
    def _create_fallback_config(self):
        """创建fallback配置"""
        self.keywords_config = {
            "function_matching_keywords": {
                "general": {
                    "chinese": ["处理", "分析", "计算"],
                    "english": ["process", "analyze", "calculate"]
                }
            }
        }
        logger.info("使用fallback关键词配置")
    
    def get_function_matching_keywords(self) -> Dict[str, List[str]]:
        """获取函数匹配关键词"""
        keywords = {}
        
        function_keywords = self.keywords_config.get("function_matching_keywords", {})
        
        for category, lang_keywords in function_keywords.items():
            all_keywords = []
            
            # 合并中英文关键词
            if isinstance(lang_keywords, dict):
                all_keywords.extend(lang_keywords.get("chinese", []))
                all_keywords.extend(lang_keywords.get("english", []))
            elif isinstance(lang_keywords, list):
                all_keywords.extend(lang_keywords)
            
            if all_keywords:
                keywords[category] = all_keywords
        
        return keywords
    
    def get_all_keywords_flat(self) -> Set[str]:
        """获取所有关键词的扁平化集合"""
        all_keywords = set()
        
        for category_keywords in self.get_function_matching_keywords().values():
            all_keywords.update(keyword.lower() for keyword in category_keywords)
        
        return all_keywords
    
    def get_keywords_by_category(self, category: str) -> List[str]:
        """根据类别获取关键词"""
        function_keywords = self.keywords_config.get("function_matching_keywords", {})
        category_data = function_keywords.get(category, {})
        
        keywords = []
        if isinstance(category_data, dict):
            keywords.extend(category_data.get("chinese", []))
            keywords.extend(category_data.get("english", []))
        elif isinstance(category_data, list):
            keywords.extend(category_data)
        
        return keywords
    
    def calculate_keyword_similarity(self, text: str, keywords: List[str]) -> float:
        """计算文本与关键词的相似度"""
        if not keywords or not text:
            return 0.0
        
        text_lower = text.lower()
        matched_keywords = 0
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched_keywords += 1
        
        return matched_keywords / len(keywords)
    
    def find_best_matching_category(self, text: str) -> tuple[str, float]:
        """找到最佳匹配的关键词类别"""
        best_category = "general"
        best_score = 0.0
        
        for category, keywords in self.get_function_matching_keywords().items():
            score = self.calculate_keyword_similarity(text, keywords)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category, best_score
    
    def get_step_indicators(self) -> List[str]:
        """获取步骤指示词"""
        step_indicators = self.keywords_config.get("step_indicators", {})
        
        indicators = []
        if isinstance(step_indicators, dict):
            indicators.extend(step_indicators.get("chinese", []))
            indicators.extend(step_indicators.get("english", []))
        elif isinstance(step_indicators, list):
            indicators.extend(step_indicators)
        
        return indicators
    
    def get_complexity_keywords(self) -> Dict[str, List[str]]:
        """获取复杂度关键词"""
        complexity_keywords = self.keywords_config.get("task_complexity_keywords", {})
        
        result = {}
        for level, lang_keywords in complexity_keywords.items():
            all_keywords = []
            if isinstance(lang_keywords, dict):
                all_keywords.extend(lang_keywords.get("chinese", []))
                all_keywords.extend(lang_keywords.get("english", []))
            elif isinstance(lang_keywords, list):
                all_keywords.extend(lang_keywords)
            
            if all_keywords:
                result[level] = all_keywords
        
        return result
    
    def analyze_task_complexity(self, text: str) -> str:
        """分析任务复杂度"""
        complexity_keywords = self.get_complexity_keywords()
        
        best_level = "simple"
        best_score = 0.0
        
        for level, keywords in complexity_keywords.items():
            score = self.calculate_keyword_similarity(text, keywords)
            if score > best_score:
                best_score = score
                best_level = level
        
        return best_level
    
    def get_operation_type_keywords(self) -> Dict[str, List[str]]:
        """获取操作类型关键词"""
        operation_keywords = self.keywords_config.get("operation_types", {})
        
        result = {}
        for op_type, lang_keywords in operation_keywords.items():
            all_keywords = []
            if isinstance(lang_keywords, dict):
                all_keywords.extend(lang_keywords.get("chinese", []))
                all_keywords.extend(lang_keywords.get("english", []))
            elif isinstance(lang_keywords, list):
                all_keywords.extend(lang_keywords)
            
            if all_keywords:
                result[op_type] = all_keywords
        
        return result
    
    def identify_operation_type(self, text: str) -> str:
        """识别操作类型"""
        operation_keywords = self.get_operation_type_keywords()
        
        best_type = "read"  # 默认为读取操作
        best_score = 0.0
        
        for op_type, keywords in operation_keywords.items():
            score = self.calculate_keyword_similarity(text, keywords)
            if score > best_score:
                best_score = score
                best_type = op_type
        
        return best_type
    
    def reload_config(self):
        """重新加载配置"""
        self._load_keywords_config()
        logger.info("关键词配置已重新加载")
    
    def add_custom_keywords(self, category: str, keywords: List[str], language: str = "custom"):
        """添加自定义关键词"""
        if "function_matching_keywords" not in self.keywords_config:
            self.keywords_config["function_matching_keywords"] = {}
        
        if category not in self.keywords_config["function_matching_keywords"]:
            self.keywords_config["function_matching_keywords"][category] = {}
        
        if language not in self.keywords_config["function_matching_keywords"][category]:
            self.keywords_config["function_matching_keywords"][category][language] = []
        
        self.keywords_config["function_matching_keywords"][category][language].extend(keywords)
        logger.info(f"添加自定义关键词到类别 {category}: {keywords}")


# 全局实例
_keyword_manager = None

def get_keyword_manager(config_path: Optional[str] = None) -> KeywordManager:
    """获取关键词管理器实例（单例模式）"""
    global _keyword_manager
    if _keyword_manager is None:
        _keyword_manager = KeywordManager(config_path)
    return _keyword_manager


def get_function_matching_keywords() -> Dict[str, List[str]]:
    """获取函数匹配关键词的便捷函数"""
    return get_keyword_manager().get_function_matching_keywords()


def calculate_keyword_similarity(text: str, keywords: List[str]) -> float:
    """计算关键词相似度的便捷函数"""
    return get_keyword_manager().calculate_keyword_similarity(text, keywords)


# 导出的公共接口
__all__ = [
    'KeywordManager',
    'get_keyword_manager',
    'get_function_matching_keywords',
    'calculate_keyword_similarity'
]
