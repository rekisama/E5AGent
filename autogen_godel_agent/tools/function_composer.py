"""
Function Composer - 函数组合器

实现智能函数组合功能，将多个已注册函数组合成新的复合函数来解决复杂任务。

核心功能:
1. 任务分解 - 将复杂任务分解为子任务
2. 函数匹配 - 为每个子任务找到最佳匹配函数
3. 组合生成 - 生成组合函数代码
4. 验证测试 - 验证组合函数的正确性

作者: AutoGen Self-Expanding Agent System
日期: 2025-07-04
"""

import re
import json
import ast
import logging
import difflib
import os
import hashlib
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import autogen

from .function_registry import get_registry
from .secure_executor import validate_function_code, execute_code_safely
from .autogen_test_evolution import get_autogen_test_evolution, evolve_function_with_autogen
from .json_utils import extract_json_block, parse_json_with_fallback
from .task_utils import SubTask, create_simple_fallback_tasks, get_fallback_configs, create_subtask_from_dict
from .type_compatibility import check_input_compatibility, check_output_compatibility
from .keyword_manager import get_keyword_manager, get_function_matching_keywords
try:
    from ..config import Config
except ImportError:
    from config import Config

# 设置日志
logger = logging.getLogger(__name__)

# 检查AutoGen版本兼容性
AUTOGEN_VERSION = getattr(autogen, '__version__', '0.0.0')
IS_AUTOGEN_ASYNC = tuple(map(int, AUTOGEN_VERSION.split('.')[:2])) >= (0, 3)
# from ..config import Config  # 暂时注释，避免导入问题

logger = logging.getLogger(__name__)

# ================================
# 数据类定义
# ================================

@dataclass
class FunctionMatch:
    """函数匹配结果"""
    func_name: str
    confidence: float  # 0.0 - 1.0
    input_compatibility: float
    output_compatibility: float
    description_similarity: float
    function_info: Dict[str, Any] = None

@dataclass
class CompositionPlan:
    """组合计划"""
    subtasks: List[SubTask]
    function_matches: Dict[str, FunctionMatch]  # subtask_id -> function_match
    execution_order: List[str]  # subtask_ids in execution order
    data_flow: Dict[str, str]  # output_subtask -> input_subtask mappings

@dataclass
class CompositeFunction:
    """组合函数结果"""
    name: str
    code: str
    description: str
    input_params: List[Dict[str, Any]]
    output_type: str
    component_functions: List[str]
    execution_plan: CompositionPlan

# ================================
# 任务分解器
# ================================

class TaskDecomposer:
    """任务分解器 - 使用LLM将复杂任务分解为子任务"""

    def __init__(self, llm_config: Dict[str, Any], max_retries: int = 3):
        self.llm_config = llm_config
        self.max_retries = max_retries
        self.fallback_configs = get_fallback_configs(llm_config)
        logger.info("TaskDecomposer 初始化完成")


    def decompose_task(self, task_description: str) -> List[SubTask]:
        """分解任务为子任务 - 支持重试和fallback"""
        # 主要尝试
        result = self._decompose_with_config(task_description, self.llm_config)
        if result:
            return result

        # 重试逻辑
        for retry_count in range(self.max_retries):
            logger.warning(f"任务分解失败，进行第 {retry_count + 1} 次重试")
            result = self._decompose_with_config(task_description, self.llm_config)
            if result:
                return result

        # Fallback模型尝试
        for i, fallback_config in enumerate(self.fallback_configs):
            logger.warning(f"使用fallback模型 {i + 1} 进行任务分解")
            result = self._decompose_with_config(task_description, fallback_config)
            if result:
                return result

        # 最后的fallback - 返回简化的任务分解
        logger.error("所有分解尝试失败，使用简化分解")
        return create_simple_fallback_tasks(task_description)

    def _decompose_with_config(self, task_description: str, config: Dict[str, Any]) -> List[SubTask]:
        """使用指定配置进行任务分解"""
        try:
            logger.info(f"开始分解任务: {task_description}")

            prompt = self._build_decomposition_prompt(task_description)

            # 使用 AutoGen 进行任务分解
            assistant = autogen.AssistantAgent(
                name="task_decomposer",
                llm_config=config,
                system_message="你是一个任务分解专家，擅长将复杂任务分解为简单的子任务。"
            )

            user_proxy = autogen.UserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                code_execution_config=False
            )

            # 发起对话，限制轮次避免无限循环
            # 兼容不同版本的AutoGen
            try:
                if IS_AUTOGEN_ASYNC:
                    # 新版本AutoGen可能需要异步调用
                    import asyncio
                    if asyncio.iscoroutinefunction(user_proxy.initiate_chat):
                        # 如果是异步函数，需要在事件循环中运行
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                        loop.run_until_complete(
                            user_proxy.initiate_chat(assistant, message=prompt, max_turns=1)
                        )
                    else:
                        user_proxy.initiate_chat(assistant, message=prompt, max_turns=1)
                else:
                    # 旧版本AutoGen
                    user_proxy.initiate_chat(assistant, message=prompt, max_turns=1)
            except Exception as chat_error:
                logger.error(f"AutoGen对话失败: {chat_error}")
                return []

            # 获取响应
            messages = user_proxy.chat_messages.get(assistant, [])
            if not messages:
                logger.error("未获取到任务分解响应")
                return []

            # 兼容不同的消息格式
            last_message = messages[-1]
            if isinstance(last_message, dict):
                response = last_message.get('content', '')
            else:
                # 可能是ChatMessage对象
                response = getattr(last_message, 'content', str(last_message))
            subtasks = self._parse_subtasks_response(response)

            if subtasks:
                logger.info(f"成功分解为 {len(subtasks)} 个子任务")
                return subtasks
            else:
                logger.error("任务分解失败")
                return []

        except Exception as e:
            logger.error(f"任务分解失败: {e}")
            return []


    
    def _build_decomposition_prompt(self, task_description: str) -> str:
        """构建任务分解提示"""
        return f"""
请将以下复杂任务分解为简单的子任务：

任务描述: {task_description}

要求:
1. 每个子任务应该是独立的、可执行的操作
2. 子任务之间应该有清晰的输入输出关系
3. 按执行顺序排列子任务
4. 为每个子任务指定输入和输出类型

请以以下JSON格式返回:
```json
{{
    "subtasks": [
        {{
            "id": "task_1",
            "description": "子任务描述",
            "input_type": "输入类型",
            "output_type": "输出类型",
            "priority": 1,
            "dependencies": []
        }}
    ]
}}
```

示例:
任务: "读取CSV文件并生成图表"
分解为:
1. 读取CSV文件 (输入: str路径, 输出: DataFrame)
2. 数据预处理 (输入: DataFrame, 输出: DataFrame) 
3. 生成图表 (输入: DataFrame, 输出: 图表文件)
"""
    


    def _parse_subtasks_response(self, response: str) -> List[SubTask]:
        """解析子任务响应 - 增强的鲁棒性解析"""
        try:
            # 提取JSON块
            json_str = extract_json_block(response)
            if not json_str:
                logger.error("未找到有效的JSON响应")
                logger.debug(f"原始响应: {response[:500]}...")
                return []

            # 解析JSON
            data = parse_json_with_fallback(json_str)
            if data is None:
                logger.error(f"JSON解析失败，原始内容: {json_str[:200]}...")
                return []

            # 解析子任务数据
            subtasks = []
            for i, task_data in enumerate(data.get('subtasks', [])):
                # 提供默认值以增强鲁棒性
                task_data_normalized = {
                    'id': task_data.get('id', f'task_{i+1}'),
                    'description': task_data.get('description', f'子任务{i+1}'),
                    'input_type': task_data.get('input_type', 'Any'),
                    'output_type': task_data.get('output_type', 'Any'),
                    'priority': task_data.get('priority', i+1),
                    'dependencies': task_data.get('dependencies', [])
                }
                subtask = create_subtask_from_dict(task_data_normalized)
                subtasks.append(subtask)

            logger.info(f"成功解析 {len(subtasks)} 个子任务")
            return subtasks

        except Exception as e:
            logger.error(f"解析子任务响应失败: {e}")
            logger.debug(f"响应内容: {response[:500]}...")
            return []

# ================================
# 函数匹配器
# ================================

class FunctionMatcher:
    """函数匹配器 - 为子任务找到最佳匹配的已注册函数"""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None, keyword_config_path: Optional[str] = None):
        self.registry = get_registry()
        self.keyword_manager = get_keyword_manager(keyword_config_path)

        # 获取配置化的关键词
        self.keywords = self.keyword_manager.get_function_matching_keywords()

        # LLM配置用于语义匹配
        self.llm_config = llm_config
        self.use_llm_matching = llm_config is not None

        # 匹配策略配置
        self.llm_threshold = 0.7  # LLM匹配的最低分数阈值
        self.max_candidates = 3   # 每个子任务最多返回的候选函数数量
        self.fallback_to_keyword = True  # 当LLM失败时是否回退到关键词匹配

        # 性能优化配置
        self.enable_prefiltering = True  # 启用候选函数预筛选
        self.prefilter_limit = 10  # 预筛选后保留的候选函数数量
        self.prefilter_threshold = 0.3  # 预筛选的最低分数阈值
        self.enable_llm_caching = True  # 启用LLM匹配结果缓存
        self.enable_concurrent_matching = True  # 启用并发LLM匹配
        self.max_concurrent_workers = 3  # 最大并发工作线程数

        # LLM匹配缓存
        self._llm_cache = {}  # 缓存LLM匹配结果
        self._cache_lock = threading.Lock()  # 缓存访问锁

    def _llm_semantic_match(self, subtask: SubTask, func_info: Dict[str, Any]) -> Dict[str, Any]:
        """使用LLM进行语义匹配 - 支持缓存优化"""
        if not self.use_llm_matching:
            return {"match": False, "score": 0.0, "reason": "LLM匹配未启用"}

        # 生成缓存键
        cache_key = None
        if self.enable_llm_caching:
            cache_key = self._generate_cache_key(subtask, func_info)

            # 检查缓存
            with self._cache_lock:
                if cache_key in self._llm_cache:
                    logger.debug(f"使用缓存的LLM匹配结果: {func_info.get('name', 'unknown')}")
                    return self._llm_cache[cache_key]

        try:
            # 构建匹配提示
            prompt = self._build_matching_prompt(subtask, func_info)

            # 调用LLM
            response = self._call_llm_for_matching(prompt)

            # 解析LLM响应
            result = self._parse_llm_response(response)

            # 缓存结果
            if self.enable_llm_caching and cache_key:
                with self._cache_lock:
                    self._llm_cache[cache_key] = result
                    # 限制缓存大小，避免内存泄漏
                    if len(self._llm_cache) > 1000:
                        # 删除最旧的一半缓存项
                        keys_to_remove = list(self._llm_cache.keys())[:500]
                        for key in keys_to_remove:
                            del self._llm_cache[key]

            logger.debug(f"LLM匹配结果: {subtask.description[:30]}... -> {func_info.get('name', 'unknown')}: {result}")
            return result

        except Exception as e:
            logger.warning(f"LLM语义匹配失败: {e}")
            return {"match": False, "score": 0.0, "reason": f"LLM调用失败: {str(e)}"}

    def _generate_cache_key(self, subtask: SubTask, func_info: Dict[str, Any]) -> str:
        """生成LLM匹配缓存键"""
        try:
            # 组合关键信息
            key_data = {
                'subtask_desc': subtask.description,
                'subtask_inputs': subtask.inputs,
                'subtask_output': subtask.output_type,
                'func_name': func_info.get('name', ''),
                'func_desc': func_info.get('description', ''),
                'func_inputs': func_info.get('inputs', {}),
                'func_output': func_info.get('output_type', '')
            }

            # 序列化并生成哈希
            key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
            cache_key = hashlib.md5(key_str.encode('utf-8')).hexdigest()

            return cache_key

        except Exception as e:
            logger.debug(f"缓存键生成失败: {e}")
            # 回退到简单键
            return f"{subtask.description[:50]}_{func_info.get('name', 'unknown')}"

    def _build_matching_prompt(self, subtask: SubTask, func_info: Dict[str, Any]) -> str:
        """构建LLM匹配提示"""
        func_name = func_info.get('name', '未知函数')
        func_desc = func_info.get('description', '无描述')
        func_code = func_info.get('code', '')

        # 提取函数签名
        func_signature = ""
        if func_code:
            import re
            signature_match = re.search(r'def\s+\w+\s*\([^)]*\)(?:\s*->\s*[^:]+)?:', func_code)
            if signature_match:
                func_signature = signature_match.group(0)

        prompt = f"""你是一个函数语义匹配专家。请判断给定的函数是否能够完成指定的子任务。

## 子任务信息
- 描述: {subtask.description}
- 输入类型: {subtask.input_type}
- 输出类型: {subtask.output_type}

## 候选函数信息
- 函数名称: {func_name}
- 函数描述: {func_desc}
- 函数签名: {func_signature}

## 评估要求
请从以下几个维度评估匹配度：
1. 功能语义匹配：函数功能是否与子任务需求一致
2. 输入输出兼容：函数的输入输出类型是否与子任务兼容
3. 实现可行性：函数是否能够实际完成子任务

## 输出格式
请严格按照以下JSON格式返回结果：
```json
{{
    "match": true/false,
    "score": 0.85,
    "reason": "详细的匹配分析说明",
    "confidence": "high/medium/low"
}}
```

其中：
- match: 是否匹配（布尔值）
- score: 匹配分数（0.0-1.0之间的浮点数）
- reason: 匹配分析的详细说明
- confidence: 匹配置信度（high/medium/low）

请仔细分析后给出判断。"""

        return prompt

    def _call_llm_for_matching(self, prompt: str) -> str:
        """调用LLM进行匹配判断 - 优化配置提升稳定性"""
        try:
            # 使用AutoGen的ConversableAgent进行LLM调用
            from autogen import ConversableAgent

            # 创建优化的LLM配置 - 降低temperature提升稳定性
            optimized_llm_config = self.llm_config.copy() if self.llm_config else {}

            # 设置低temperature以提升响应稳定性
            if 'config_list' in optimized_llm_config:
                for config in optimized_llm_config['config_list']:
                    config['temperature'] = 0.1  # 降低随机性
                    config['max_tokens'] = 500   # 限制响应长度
            else:
                optimized_llm_config.update({
                    'temperature': 0.1,
                    'max_tokens': 500
                })

            # 创建临时agent用于LLM调用
            llm_agent = ConversableAgent(
                name="function_matcher",
                llm_config=optimized_llm_config,
                system_message="你是一个专业的函数语义匹配专家，能够准确判断函数与任务的匹配度。请严格按照JSON格式返回结果。",
                human_input_mode="NEVER"
            )

            # 发送匹配请求
            response = llm_agent.generate_reply(
                messages=[{"role": "user", "content": prompt}]
            )

            return response if isinstance(response, str) else str(response)

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应 - 增强容错机制"""
        try:
            # 策略1: 尝试提取JSON块
            json_result = extract_json_block(response)
            if json_result:
                parsed = parse_json_with_fallback(json_result)
                if parsed and isinstance(parsed, dict):
                    return self._validate_and_normalize_response(parsed)

            # 策略2: 尝试直接JSON解析整个响应
            try:
                parsed = json.loads(response.strip())
                if isinstance(parsed, dict):
                    return self._validate_and_normalize_response(parsed)
            except json.JSONDecodeError:
                pass

            # 策略3: 使用正则表达式提取结构化信息
            extracted_data = self._extract_response_with_regex(response)
            if extracted_data:
                return self._validate_and_normalize_response(extracted_data)

            # 策略4: 基于关键词的启发式解析
            heuristic_data = self._heuristic_response_parsing(response)
            return self._validate_and_normalize_response(heuristic_data)

        except Exception as e:
            logger.warning(f"LLM响应解析异常: {e}")
            return self._get_default_response(f"解析异常: {str(e)}")

    def _validate_and_normalize_response(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """验证和标准化响应数据 - 提供默认值"""
        try:
            # 提取并验证score
            score = parsed.get('score', parsed.get('confidence_score', 0.0))
            if isinstance(score, str):
                # 尝试从字符串中提取数字
                import re
                score_match = re.search(r'(\d+\.?\d*)', score)
                score = float(score_match.group(1)) if score_match else 0.0
            else:
                score = float(score) if score is not None else 0.0

            # 确保score在有效范围内
            score = max(0.0, min(1.0, score))

            # 提取并验证match
            match = parsed.get('match', parsed.get('is_match', score >= self.llm_threshold))
            if isinstance(match, str):
                match = match.lower() in ['true', 'yes', '是', '匹配', 'match']
            else:
                match = bool(match)

            # 提取reason
            reason = parsed.get('reason', parsed.get('explanation', parsed.get('description', '无说明')))
            reason = str(reason) if reason else '无说明'

            # 提取confidence
            confidence = parsed.get('confidence', parsed.get('confidence_level', 'medium'))
            if confidence not in ['low', 'medium', 'high']:
                confidence = 'medium'

            return {
                "match": match,
                "score": score,
                "reason": reason,
                "confidence": confidence
            }

        except Exception as e:
            logger.debug(f"响应验证失败: {e}")
            return self._get_default_response(f"验证失败: {str(e)}")

    def _extract_response_with_regex(self, response: str) -> Optional[Dict[str, Any]]:
        """使用正则表达式提取响应信息"""
        try:
            import re

            # 提取分数
            score_patterns = [
                r'score["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)',
                r'confidence["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)',
                r'匹配度["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)',
                r'\b([0-9]*\.?[0-9]+)\b'  # 任何数字
            ]

            score = 0.0
            for pattern in score_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        if 0.0 <= score <= 1.0:
                            break
                    except ValueError:
                        continue

            # 提取匹配状态
            match_patterns = [
                r'match["\']?\s*[:=]\s*(true|false|yes|no|是|否)',
                r'is_match["\']?\s*[:=]\s*(true|false|yes|no|是|否)',
                r'(匹配|不匹配|match|no match)'
            ]

            is_match = score >= self.llm_threshold  # 默认基于分数
            for pattern in match_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    match_text = match.group(1).lower()
                    is_match = match_text in ['true', 'yes', '是', '匹配', 'match']
                    break

            # 提取原因
            reason_patterns = [
                r'reason["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                r'explanation["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                r'原因["\']?\s*[:=]\s*["\']([^"\']+)["\']'
            ]

            reason = "正则表达式提取"
            for pattern in reason_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    reason = match.group(1)
                    break

            return {
                "score": score,
                "match": is_match,
                "reason": reason
            }

        except Exception as e:
            logger.debug(f"正则表达式提取失败: {e}")
            return None

    def _heuristic_response_parsing(self, response: str) -> Dict[str, Any]:
        """启发式响应解析 - 最后的回退策略"""
        try:
            response_lower = response.lower()

            # 基于关键词判断匹配状态
            positive_keywords = ['match', 'yes', 'suitable', 'appropriate', '匹配', '适合', '合适']
            negative_keywords = ['no match', 'no', 'unsuitable', 'inappropriate', '不匹配', '不适合']

            is_match = False
            score = 0.0

            if any(keyword in response_lower for keyword in positive_keywords):
                is_match = True
                score = 0.7  # 默认中等置信度
            elif any(keyword in response_lower for keyword in negative_keywords):
                is_match = False
                score = 0.2
            else:
                # 基于响应长度和复杂度的启发式评分
                if len(response) > 50:
                    score = 0.5  # 详细响应通常表示某种程度的匹配
                else:
                    score = 0.3  # 简短响应可能表示不确定

                is_match = score >= self.llm_threshold

            return {
                "score": score,
                "match": is_match,
                "reason": "启发式解析结果"
            }

        except Exception as e:
            logger.debug(f"启发式解析失败: {e}")
            return self._get_default_response("启发式解析失败")

    def _get_default_response(self, reason: str = "解析失败") -> Dict[str, Any]:
        """获取默认响应"""
        return {
            "match": False,
            "score": 0.0,
            "reason": reason,
            "confidence": "low"
        }

    def find_matching_functions(self, subtask: SubTask) -> List[FunctionMatch]:
        """为子任务找到匹配的函数 - 支持预筛选优化"""
        try:
            logger.info(f"为子任务查找匹配函数: {subtask.description}")

            # 获取所有函数
            all_functions = self.registry.list_functions()

            # 性能优化：预筛选候选函数
            if self.enable_prefiltering and len(all_functions) > self.prefilter_limit:
                logger.debug(f"启用预筛选：从{len(all_functions)}个函数中筛选前{self.prefilter_limit}个")
                candidate_functions = self._prefilter_candidates(subtask, all_functions)
            else:
                candidate_functions = all_functions

            matches = []

            # 性能优化：并发LLM匹配
            if self.enable_concurrent_matching and self.use_llm_matching and len(candidate_functions) > 3:
                logger.debug(f"启用并发LLM匹配，候选函数数: {len(candidate_functions)}")
                matches = self._concurrent_function_matching(subtask, candidate_functions)
            else:
                # 串行匹配
                for func_name in candidate_functions:
                    # 跳过元数据字段
                    if func_name in ['metadata', 'last_updated', 'version', 'total_functions']:
                        continue

                    func_info = self.registry.get_function_info(func_name)
                    if not func_info or not isinstance(func_info, dict):
                        continue

                    # 计算匹配度
                    confidence = self._calculate_match_confidence(subtask, func_info)

                    if confidence > 0.3:  # 只保留置信度较高的匹配
                        match = FunctionMatch(
                            func_name=func_name,
                            confidence=confidence,
                            input_compatibility=self._check_input_compatibility(subtask, func_info),
                            output_compatibility=self._check_output_compatibility(subtask, func_info),
                            description_similarity=self._calculate_description_similarity(subtask, func_info),
                            function_info=func_info
                        )
                        matches.append(match)

            # 按置信度排序
            matches.sort(key=lambda x: x.confidence, reverse=True)

            logger.info(f"找到 {len(matches)} 个匹配函数")
            return matches[:5]  # 返回前5个最佳匹配

        except Exception as e:
            logger.error(f"函数匹配失败: {e}")
            return []

    def _concurrent_function_matching(self, subtask: SubTask, candidate_functions: List[str]) -> List[FunctionMatch]:
        """并发执行函数匹配 - 提升LLM匹配性能"""
        try:
            matches = []
            valid_candidates = []

            # 预处理：过滤有效候选函数
            for func_name in candidate_functions:
                if func_name in ['metadata', 'last_updated', 'version', 'total_functions']:
                    continue

                func_info = self.registry.get_function_info(func_name)
                if func_info and isinstance(func_info, dict):
                    valid_candidates.append((func_name, func_info))

            if not valid_candidates:
                return []

            # 使用线程池并发执行匹配
            with ThreadPoolExecutor(max_workers=self.max_concurrent_workers) as executor:
                # 提交所有匹配任务
                future_to_func = {
                    executor.submit(self._single_function_match, subtask, func_name, func_info): (func_name, func_info)
                    for func_name, func_info in valid_candidates
                }

                # 收集结果
                for future in as_completed(future_to_func):
                    func_name, func_info = future_to_func[future]
                    try:
                        match_result = future.result(timeout=30)  # 30秒超时
                        if match_result:
                            matches.append(match_result)
                    except Exception as e:
                        logger.warning(f"并发匹配失败 {func_name}: {e}")
                        # 回退到串行匹配这个函数
                        try:
                            fallback_match = self._single_function_match(subtask, func_name, func_info)
                            if fallback_match:
                                matches.append(fallback_match)
                        except Exception as fallback_e:
                            logger.error(f"回退匹配也失败 {func_name}: {fallback_e}")

            logger.debug(f"并发匹配完成，匹配数: {len(matches)}")
            return matches

        except Exception as e:
            logger.error(f"并发匹配异常，回退到串行: {e}")
            # 完全回退到串行匹配
            return self._serial_function_matching(subtask, candidate_functions)

    def _single_function_match(self, subtask: SubTask, func_name: str, func_info: Dict[str, Any]) -> Optional[FunctionMatch]:
        """单个函数匹配 - 用于并发执行"""
        try:
            # 计算匹配度
            confidence = self._calculate_match_confidence(subtask, func_info)

            if confidence > 0.3:  # 只保留置信度较高的匹配
                return FunctionMatch(
                    func_name=func_name,
                    confidence=confidence,
                    input_compatibility=self._check_input_compatibility(subtask, func_info),
                    output_compatibility=self._check_output_compatibility(subtask, func_info),
                    description_similarity=self._calculate_description_similarity(subtask, func_info),
                    function_info=func_info
                )
            return None

        except Exception as e:
            logger.debug(f"单个函数匹配失败 {func_name}: {e}")
            return None

    def _serial_function_matching(self, subtask: SubTask, candidate_functions: List[str]) -> List[FunctionMatch]:
        """串行函数匹配 - 回退方法"""
        matches = []

        for func_name in candidate_functions:
            if func_name in ['metadata', 'last_updated', 'version', 'total_functions']:
                continue

            func_info = self.registry.get_function_info(func_name)
            if not func_info or not isinstance(func_info, dict):
                continue

            match_result = self._single_function_match(subtask, func_name, func_info)
            if match_result:
                matches.append(match_result)

        return matches

    def _prefilter_candidates(self, subtask: SubTask, all_functions: List[str]) -> List[str]:
        """
        预筛选候选函数 - 使用轻量级方法快速筛选
        只对通过预筛选的函数进行昂贵的LLM匹配
        """
        try:
            logger.debug(f"开始预筛选，总函数数: {len(all_functions)}")

            candidates_with_scores = []

            for func_name in all_functions:
                # 跳过元数据字段
                if func_name in ['metadata', 'last_updated', 'version', 'total_functions']:
                    continue

                func_info = self.registry.get_function_info(func_name)
                if not func_info or not isinstance(func_info, dict):
                    continue

                # 轻量级评分：关键词匹配 + 类型兼容性
                prefilter_score = self._calculate_prefilter_score(subtask, func_info)

                if prefilter_score >= self.prefilter_threshold:
                    candidates_with_scores.append((func_name, prefilter_score))

            # 按预筛选分数排序，取前N个
            candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
            selected_candidates = [name for name, score in candidates_with_scores[:self.prefilter_limit]]

            logger.debug(f"预筛选完成：从{len(all_functions)}个函数中选择{len(selected_candidates)}个候选")
            return selected_candidates

        except Exception as e:
            logger.warning(f"预筛选失败，使用全部函数: {e}")
            return all_functions

    def _calculate_prefilter_score(self, subtask: SubTask, func_info: Dict[str, Any]) -> float:
        """
        计算预筛选分数 - 轻量级快速评估
        使用关键词匹配和基础类型兼容性，避免昂贵的LLM调用
        """
        try:
            # 1. 关键词匹配 (权重: 0.6)
            keyword_score = self._calculate_keyword_similarity(subtask, func_info)

            # 2. 描述相似度 (权重: 0.3)
            desc_score = self._calculate_description_similarity(subtask, func_info)

            # 3. 基础类型兼容性 (权重: 0.1)
            type_score = (self._check_input_compatibility(subtask, func_info) +
                         self._check_output_compatibility(subtask, func_info)) / 2

            # 加权平均
            prefilter_score = (keyword_score * 0.6 +
                             desc_score * 0.3 +
                             type_score * 0.1)

            return prefilter_score

        except Exception as e:
            logger.debug(f"预筛选评分失败: {e}")
            return 0.0
    
    def _calculate_match_confidence(self, subtask: SubTask, func_info: Dict[str, Any]) -> float:
        """计算匹配置信度 - 优先使用LLM语义匹配"""

        # 优先使用LLM语义匹配
        if self.use_llm_matching:
            try:
                llm_result = self._llm_semantic_match(subtask, func_info)

                if llm_result["match"] and llm_result["score"] >= self.llm_threshold:
                    # LLM匹配成功，使用LLM分数作为主要依据
                    llm_score = llm_result["score"]

                    # 仍然检查类型兼容性作为补充
                    input_compat = self._check_input_compatibility(subtask, func_info)
                    output_compat = self._check_output_compatibility(subtask, func_info)

                    # LLM分数占主导地位（权重0.7），类型兼容性作为调整（权重0.3）
                    confidence = llm_score * 0.7 + (input_compat + output_compat) / 2 * 0.3

                    logger.debug(f"LLM匹配成功: {func_info.get('name', 'unknown')} -> {confidence:.3f} "
                               f"(LLM: {llm_score:.3f}, 类型: {(input_compat + output_compat) / 2:.3f})")
                    return confidence

                elif self.fallback_to_keyword:
                    # LLM匹配失败，回退到传统方法
                    logger.debug(f"LLM匹配失败，回退到传统方法: {llm_result['reason']}")
                else:
                    # 不使用回退，直接返回低分
                    return 0.0

            except Exception as e:
                logger.warning(f"LLM匹配异常，回退到传统方法: {e}")
                if not self.fallback_to_keyword:
                    return 0.0

        # 传统匹配方法（关键词+字符串相似度）
        desc_sim = self._calculate_description_similarity(subtask, func_info)
        input_compat = self._check_input_compatibility(subtask, func_info)
        output_compat = self._check_output_compatibility(subtask, func_info)

        confidence = desc_sim * 0.5 + input_compat * 0.25 + output_compat * 0.25

        logger.debug(f"传统匹配: {func_info.get('name', 'unknown')} -> {confidence:.3f} "
                   f"(描述: {desc_sim:.3f}, 输入: {input_compat:.3f}, 输出: {output_compat:.3f})")
        return confidence
    
    def _calculate_description_similarity(self, subtask: SubTask, func_info: Dict[str, Any]) -> float:
        """计算描述相似度 - 增强版本，支持多种相似度算法"""
        subtask_desc = subtask.description.lower()
        func_desc = func_info.get('description', '').lower()

        if not subtask_desc or not func_desc:
            return 0.0

        # 方法1: 使用difflib计算序列相似度
        sequence_similarity = difflib.SequenceMatcher(None, subtask_desc, func_desc).ratio()

        # 方法2: 关键词匹配（使用配置化关键词）
        keyword_similarity = self._calculate_keyword_similarity(subtask_desc, func_desc)

        # 方法3: 字符串包含检查
        containment_similarity = 0.0
        if subtask_desc in func_desc or func_desc in subtask_desc:
            containment_similarity = 0.8
        elif any(word in func_desc for word in subtask_desc.split() if len(word) > 2):
            containment_similarity = 0.6

        # 综合相似度计算（加权平均）
        final_similarity = (
            sequence_similarity * 0.4 +
            keyword_similarity * 0.4 +
            containment_similarity * 0.2
        )

        return min(final_similarity, 1.0)

    def _calculate_keyword_similarity(self, subtask_desc: str, func_desc: str) -> float:
        """计算基于配置化关键词的相似度"""
        try:
            # 获取所有关键词的扁平化集合
            all_keywords = self.keyword_manager.get_all_keywords_flat()

            # 提取描述中的关键词
            subtask_keywords = [kw for kw in all_keywords if kw in subtask_desc.lower()]
            func_keywords = [kw for kw in all_keywords if kw in func_desc.lower()]

            if not subtask_keywords and not func_keywords:
                return 0.0

            if not subtask_keywords or not func_keywords:
                return 0.1  # 一方没有关键词时给低分

            # 计算Jaccard相似度
            intersection = set(subtask_keywords).intersection(set(func_keywords))
            union = set(subtask_keywords).union(set(func_keywords))

            return len(intersection) / len(union) if union else 0.0

        except Exception as e:
            logger.debug(f"关键词相似度计算失败: {e}")
            return 0.0

    def _check_input_compatibility(self, subtask: SubTask, func_info: Dict[str, Any]) -> float:
        """检查输入类型兼容性"""
        try:
            return check_input_compatibility(subtask.input_type, func_info)
        except Exception as e:
            logger.debug(f"输入类型兼容性检查失败: {e}")
            return 0.6  # fallback分数

    def _check_output_compatibility(self, subtask: SubTask, func_info: Dict[str, Any]) -> float:
        """检查输出类型兼容性"""
        try:
            return check_output_compatibility(subtask.output_type, func_info)
        except Exception as e:
            logger.debug(f"输出类型兼容性检查失败: {e}")
            return 0.6  # fallback分数

# ================================
# 组合生成器
# ================================

class CompositionGenerator:
    """组合生成器 - 根据组合计划生成组合函数代码"""

    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.registry = get_registry()

        # 代码生成配置
        self.use_template_engine = True  # 是否使用模板引擎
        self.use_import_mode = False     # 是否使用import模式（避免函数体重复）
        self.support_nonlinear_inputs = True  # 是否支持非线性输入处理

        # 初始化模板引擎
        self._init_template_engine()

    def _init_template_engine(self):
        """初始化模板引擎"""
        try:
            # 尝试导入jinja2，如果不可用则回退到字符串拼接
            import jinja2
            self.template_env = jinja2.Environment(
                loader=jinja2.DictLoader(self._get_templates()),
                trim_blocks=True,
                lstrip_blocks=True
            )
            self.has_jinja2 = True
            logger.info("模板引擎初始化成功 (Jinja2)")
        except ImportError:
            self.has_jinja2 = False
            self.use_template_engine = False
            logger.warning("Jinja2未安装，回退到字符串拼接模式")

    def _get_templates(self) -> Dict[str, str]:
        """获取代码生成模板"""
        return {
            'composite_function': '''# 组合函数 - 自动生成
# 生成时间: {{ generation_time }}
# 组件函数: {{ component_names | join(', ') }}

from typing import Any, Dict, List, Optional, Union
{% if use_import_mode %}
# 导入组件函数
{% for import_stmt in import_statements %}
{{ import_stmt }}
{% endfor %}
{% else %}
# 组件函数定义
{% for func_name, func_code in component_codes.items() %}

# 组件函数: {{ func_name }}
{{ func_code }}
{% endfor %}
{% endif %}

def {{ function_name }}({{ input_params_str }}) -> {{ output_type }}:
    """
    {{ description }}

    Args:
{% for param in input_params %}
        {{ param.name }} ({{ param.type }}): {{ param.description }}
{% endfor %}

    Returns:
        {{ output_type }}: {{ return_description }}
    """
{% for line in execution_logic %}
    {{ line }}
{% endfor %}
''',

            'import_function': '''# 导入模式组合函数
from typing import Any, Dict, List, Optional, Union
{% for import_stmt in import_statements %}
{{ import_stmt }}
{% endfor %}

def {{ function_name }}({{ input_params_str }}) -> {{ output_type }}:
    """{{ description }}"""
{% for line in execution_logic %}
    {{ line }}
{% endfor %}
'''
        }

    def generate_composition(self, plan: CompositionPlan) -> CompositeFunction:
        """根据组合计划生成组合函数"""
        try:
            logger.info("开始生成组合函数")

            # 构建函数名
            func_name = self._generate_function_name(plan)

            # 生成函数代码
            func_code = self._generate_function_code(plan, func_name)

            # 生成描述
            description = self._generate_description(plan)

            # 提取输入参数
            input_params = self._extract_input_params(plan)

            # 确定输出类型
            output_type = self._determine_output_type(plan)

            # 获取组件函数列表
            component_functions = list(plan.function_matches.values())
            component_names = [match.func_name for match in component_functions]

            composite_func = CompositeFunction(
                name=func_name,
                code=func_code,
                description=description,
                input_params=input_params,
                output_type=output_type,
                component_functions=component_names,
                execution_plan=plan
            )

            logger.info(f"成功生成组合函数: {func_name}")
            return composite_func

        except Exception as e:
            logger.error(f"生成组合函数失败: {e}")
            raise

    def _generate_function_name(self, plan: CompositionPlan) -> str:
        """生成函数名"""
        # 基于子任务生成简洁的函数名
        keywords = []
        for subtask in plan.subtasks:
            # 清理描述，移除中文括号和特殊字符
            desc = subtask.description.replace('（', '').replace('）', '').replace('(', '').replace(')', '')

            # 对于中文，提取关键词
            if any('\u4e00' <= char <= '\u9fff' for char in desc):
                # 中文文本，提取关键词
                chinese_keywords = ['读取', 'csv', '文件', '计算', '统计', '信息', '生成', '报告', '保存']
                for keyword in chinese_keywords:
                    if keyword in desc.lower():
                        keywords.append(keyword)
            else:
                # 英文文本，按空格分词
                words = desc.lower().split()
                for word in words:
                    if len(word) > 3 and word not in ['the', 'and', 'for', 'with']:
                        keywords.append(word)

        # 取前3个关键词，如果没有则使用默认名称
        if keywords:
            name_parts = keywords[:3]
            # 将中文关键词转换为英文
            translation_map = {
                '读取': 'read',
                'csv': 'csv',
                '文件': 'file',
                '计算': 'calculate',
                '统计': 'statistics',
                '信息': 'info',
                '生成': 'generate',
                '报告': 'report',
                '保存': 'save'
            }
            translated_parts = [translation_map.get(part, part) for part in name_parts]
            func_name = '_'.join(translated_parts)
        else:
            func_name = 'composite_function'

        # 确保函数名是有效的Python标识符
        func_name = re.sub(r'[^a-zA-Z0-9_]', '_', func_name)
        if not func_name[0].isalpha() and func_name[0] != '_':
            func_name = 'func_' + func_name

        # 确保函数名唯一
        base_name = func_name
        counter = 1
        while self.registry.has_function(func_name):
            func_name = f"{base_name}_{counter}"
            counter += 1

        return func_name

    def _generate_function_code(self, plan: CompositionPlan, func_name: str) -> str:
        """生成函数代码 - 支持模板引擎和import模式"""
        if self.use_template_engine and self.has_jinja2:
            return self._generate_code_with_template(plan, func_name)
        else:
            return self._generate_code_with_string_concat(plan, func_name)

    def _generate_code_with_template(self, plan: CompositionPlan, func_name: str) -> str:
        """使用模板引擎生成代码"""
        import datetime

        # 准备模板数据
        template_data = {
            'function_name': func_name,
            'generation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': self._generate_description(plan),
            'input_params': self._extract_input_params_enhanced(plan),
            'input_params_str': self._format_input_params_string(plan),
            'output_type': self._determine_output_type(plan),
            'return_description': self._generate_return_description(plan),
            'execution_logic': self._generate_execution_logic_enhanced(plan),
            'use_import_mode': self.use_import_mode,
            'component_names': [match.func_name for match in plan.function_matches.values()]
        }

        if self.use_import_mode:
            # 使用import模式，避免函数体重复
            template_data['import_statements'] = self._generate_import_statements(plan)
            template = self.template_env.get_template('import_function')
        else:
            # 传统模式，包含完整函数定义
            template_data['component_codes'] = self._get_component_codes(plan)
            template = self.template_env.get_template('composite_function')

        return template.render(**template_data)

    def _generate_code_with_string_concat(self, plan: CompositionPlan, func_name: str) -> str:
        """使用字符串拼接生成代码（回退方案）"""
        # 获取所有组件函数的代码
        component_codes = self._get_component_codes(plan)

        # 构建组合函数代码
        code_lines = []

        # 添加导入语句
        code_lines.append("# 组合函数 - 自动生成")
        code_lines.append("from typing import Any, Dict, List, Optional, Union")
        code_lines.append("")

        if self.use_import_mode:
            # 使用import模式
            import_statements = self._generate_import_statements(plan)
            for stmt in import_statements:
                code_lines.append(stmt)
            code_lines.append("")
        else:
            # 添加组件函数定义
            for func_name_comp, func_code in component_codes.items():
                code_lines.append(f"# 组件函数: {func_name_comp}")
                code_lines.append(func_code)
                code_lines.append("")

        # 生成主组合函数
        input_params = self._extract_input_params_enhanced(plan)
        param_str = self._format_input_params_string(plan)
        output_type = self._determine_output_type(plan)

        code_lines.append(f"def {func_name}({param_str}) -> {output_type}:")
        code_lines.append(f'    """')
        code_lines.append(f'    {self._generate_description(plan)}')
        code_lines.append(f'    """')

        # 生成执行逻辑
        execution_logic = self._generate_execution_logic_enhanced(plan)
        for line in execution_logic:
            code_lines.append(f"    {line}")

        return "\n".join(code_lines)

    def _get_component_codes(self, plan: CompositionPlan) -> Dict[str, str]:
        """获取组件函数代码，避免重复定义"""
        component_codes = {}
        seen_functions = set()

        for subtask_id, match in plan.function_matches.items():
            if match.func_name not in seen_functions:
                func_code = self.registry.get_function_code(match.func_name)
                if func_code:
                    component_codes[match.func_name] = func_code
                    seen_functions.add(match.func_name)

        return component_codes

    def _generate_import_statements(self, plan: CompositionPlan) -> List[str]:
        """生成import语句（用于import模式）"""
        import_statements = []

        for subtask_id, match in plan.function_matches.items():
            # 假设函数来自注册表模块
            import_statements.append(f"from autogen_godel_agent.registry import {match.func_name}")

        return import_statements

    def _format_input_params_string(self, plan: CompositionPlan) -> str:
        """格式化输入参数字符串"""
        input_params = self._extract_input_params_enhanced(plan)
        return ", ".join([f"{p['name']}: {p['type']}" for p in input_params])

    def _generate_return_description(self, plan: CompositionPlan) -> str:
        """生成返回值描述"""
        if not plan.subtasks:
            return "处理结果"

        last_subtask = plan.subtasks[-1]
        return f"经过{len(plan.subtasks)}步处理后的{last_subtask.output_type}类型结果"

    def _generate_execution_logic(self, plan: CompositionPlan) -> List[str]:
        """生成执行逻辑（兼容性方法）"""
        return self._generate_execution_logic_enhanced(plan)

    def _generate_execution_logic_enhanced(self, plan: CompositionPlan) -> List[str]:
        """增强的执行逻辑生成 - 支持复杂输入输出关系"""
        logic_lines = []

        if not plan.execution_order:
            logic_lines.append("return None")
            return logic_lines

        # 获取输入参数信息
        input_params = self._extract_input_params_enhanced(plan)

        # 按执行顺序处理每个子任务
        for i, subtask_id in enumerate(plan.execution_order):
            subtask = next(st for st in plan.subtasks if st.id == subtask_id)
            match = plan.function_matches.get(subtask_id)

            if not match:
                logic_lines.append(f"# 跳过子任务 {subtask_id}: 未找到匹配函数")
                continue

            # 生成函数调用
            call_args = self._determine_function_call_args(subtask, match, plan, i, input_params)
            logic_lines.append(f"result_{i} = {match.func_name}({call_args})")

            # 添加错误处理（可选）
            if self._should_add_error_handling(subtask, match):
                logic_lines.append(f"if result_{i} is None:")
                logic_lines.append(f"    raise ValueError('函数 {match.func_name} 返回了 None')")

        # 返回最后的结果
        last_index = len(plan.execution_order) - 1
        logic_lines.append(f"return result_{last_index}")

        return logic_lines

    def _determine_function_call_args(self, subtask: 'SubTask', match: 'FunctionMatch',
                                    plan: CompositionPlan, current_index: int,
                                    input_params: List[Dict[str, Any]]) -> str:
        """确定函数调用参数"""

        if current_index == 0:
            # 第一个函数：使用外部输入参数
            return self._format_first_function_args(subtask, match, input_params)
        else:
            # 后续函数：可能使用前面的结果或额外的输入
            return self._format_subsequent_function_args(subtask, match, plan, current_index, input_params)

    def _format_first_function_args(self, subtask: 'SubTask', match: 'FunctionMatch',
                                  input_params: List[Dict[str, Any]]) -> str:
        """格式化第一个函数的参数"""

        # 获取函数的参数信息
        func_info = self.registry.get_function_info(match.func_name)
        func_params = func_info.get('parameters', {}) if func_info else {}

        if len(func_params) <= 1:
            # 单参数函数，使用主要输入
            main_input = next((p for p in input_params if p.get('source') == 'external'), input_params[0])
            return main_input['name']
        else:
            # 多参数函数，尝试匹配所有参数
            args = []
            for param_name in func_params.keys():
                # 寻找匹配的输入参数
                matching_input = self._find_matching_input_param(param_name, input_params, subtask)
                if matching_input:
                    args.append(matching_input['name'])
                else:
                    # 如果找不到匹配的输入，使用默认值或主要输入
                    main_input = input_params[0] if input_params else {'name': 'input_data'}
                    args.append(main_input['name'])

            return ', '.join(args)

    def _format_subsequent_function_args(self, subtask: 'SubTask', match: 'FunctionMatch',
                                       plan: CompositionPlan, current_index: int,
                                       input_params: List[Dict[str, Any]]) -> str:
        """格式化后续函数的参数"""

        # 获取函数的参数信息
        func_info = self.registry.get_function_info(match.func_name)
        func_params = func_info.get('parameters', {}) if func_info else {}

        if len(func_params) <= 1:
            # 单参数函数，使用前一个结果
            return f"result_{current_index - 1}"
        else:
            # 多参数函数，需要智能参数映射
            args = []
            param_names = list(func_params.keys())

            # 第一个参数通常使用前一个结果
            args.append(f"result_{current_index - 1}")

            # 其他参数尝试从输入参数或更早的结果中获取
            for i, param_name in enumerate(param_names[1:], 1):
                arg = self._resolve_additional_parameter(param_name, func_params[param_name],
                                                       plan, current_index, input_params)
                args.append(arg)

            return ', '.join(args)

    def _find_matching_input_param(self, param_name: str, input_params: List[Dict[str, Any]],
                                 subtask: 'SubTask') -> Optional[Dict[str, Any]]:
        """寻找匹配的输入参数"""

        # 首先尝试精确匹配参数名
        for input_param in input_params:
            if input_param.get('param_name') == param_name:
                return input_param

        # 然后尝试类型匹配
        for input_param in input_params:
            if self._types_compatible(input_param['type'], subtask.input_type):
                return input_param

        return None

    def _resolve_additional_parameter(self, param_name: str, param_info: Dict[str, Any],
                                    plan: CompositionPlan, current_index: int,
                                    input_params: List[Dict[str, Any]]) -> str:
        """解析额外参数的来源"""

        param_type = param_info.get('type', 'Any')

        # 1. 尝试从输入参数中找到匹配的
        for input_param in input_params:
            if (input_param.get('param_name') == param_name or
                self._types_compatible(input_param['type'], param_type)):
                return input_param['name']

        # 2. 尝试从之前的结果中找到兼容的
        for i in range(current_index):
            subtask_id = plan.execution_order[i]
            subtask = next(st for st in plan.subtasks if st.id == subtask_id)
            if self._types_compatible(subtask.output_type, param_type):
                return f"result_{i}"

        # 3. 使用默认值或主要输入
        if input_params:
            return input_params[0]['name']
        else:
            return "None  # 无法解析参数来源"

    def _should_add_error_handling(self, subtask: 'SubTask', match: 'FunctionMatch') -> bool:
        """判断是否应该添加错误处理"""
        # 可以根据函数类型、重要性等因素决定
        return False  # 暂时禁用，避免代码过于复杂

    def _generate_description(self, plan: CompositionPlan) -> str:
        """生成函数描述"""
        descriptions = []
        for subtask in plan.subtasks:
            # 清理描述中的中文括号
            desc = subtask.description.replace('（', '(').replace('）', ')')
            descriptions.append(desc)
        return f"组合函数: {' -> '.join(descriptions)}"

    def _extract_input_params(self, plan: CompositionPlan) -> List[Dict[str, Any]]:
        """提取输入参数（兼容性方法）"""
        return self._extract_input_params_enhanced(plan)

    def _extract_input_params_enhanced(self, plan: CompositionPlan) -> List[Dict[str, Any]]:
        """增强的输入参数提取 - 支持非线性输入处理"""
        if not plan.subtasks:
            return []

        if not self.support_nonlinear_inputs:
            # 简单模式：只使用第一个子任务的输入
            first_subtask = plan.subtasks[0]
            input_type = self._normalize_type_string(first_subtask.input_type)
            return [{
                'name': 'input_data',
                'type': input_type,
                'description': f'输入数据 ({input_type})'
            }]

        # 高级模式：分析所有子任务的输入需求
        return self._analyze_all_input_requirements(plan)

    def _analyze_all_input_requirements(self, plan: CompositionPlan) -> List[Dict[str, Any]]:
        """分析所有输入需求，支持多输入和非线性组合"""
        input_params = []
        input_sources = set()  # 跟踪输入来源，避免重复

        # 分析执行图，找出所有外部输入
        for i, subtask_id in enumerate(plan.execution_order):
            subtask = next(st for st in plan.subtasks if st.id == subtask_id)

            if i == 0:
                # 第一个子任务的输入肯定是外部输入
                input_type = self._normalize_type_string(subtask.input_type)
                input_key = f"input_{input_type.lower()}"

                if input_key not in input_sources:
                    input_params.append({
                        'name': input_key,
                        'type': input_type,
                        'description': f'主要输入数据 ({input_type})',
                        'source': 'external',
                        'subtask_id': subtask_id
                    })
                    input_sources.add(input_key)
            else:
                # 检查是否需要额外的外部输入
                additional_inputs = self._detect_additional_inputs(subtask, plan, i)
                for add_input in additional_inputs:
                    input_key = f"input_{add_input['type'].lower()}_{len(input_params)}"
                    if input_key not in input_sources:
                        add_input['name'] = input_key
                        input_params.append(add_input)
                        input_sources.add(input_key)

        # 如果没有找到任何输入，使用默认输入
        if not input_params:
            input_params.append({
                'name': 'input_data',
                'type': 'Any',
                'description': '输入数据',
                'source': 'default'
            })

        return input_params

    def _detect_additional_inputs(self, subtask: 'SubTask', plan: CompositionPlan, current_index: int) -> List[Dict[str, Any]]:
        """检测子任务是否需要额外的外部输入"""
        additional_inputs = []

        # 获取匹配的函数信息
        match = plan.function_matches.get(subtask.id)
        if not match:
            return additional_inputs

        func_info = self.registry.get_function_info(match.func_name)
        if not func_info:
            return additional_inputs

        # 分析函数参数，看是否需要多个输入
        func_params = func_info.get('parameters', {})

        # 如果函数有多个参数，可能需要额外输入
        if len(func_params) > 1:
            for param_name, param_info in func_params.items():
                param_type = param_info.get('type', 'Any')

                # 检查这个参数是否可能来自前面的结果
                if not self._can_be_satisfied_by_previous_results(param_type, plan, current_index):
                    additional_inputs.append({
                        'type': self._normalize_type_string(param_type),
                        'description': f'额外输入参数: {param_name} ({param_type})',
                        'source': 'external_additional',
                        'subtask_id': subtask.id,
                        'param_name': param_name
                    })

        return additional_inputs

    def _can_be_satisfied_by_previous_results(self, param_type: str, plan: CompositionPlan, current_index: int) -> bool:
        """检查参数是否可以由前面的结果满足"""
        if current_index == 0:
            return False

        # 检查前面所有子任务的输出类型
        for i in range(current_index):
            subtask_id = plan.execution_order[i]
            subtask = next(st for st in plan.subtasks if st.id == subtask_id)

            # 简单的类型兼容性检查
            if self._types_compatible(subtask.output_type, param_type):
                return True

        return False

    def _types_compatible(self, output_type: str, param_type: str) -> bool:
        """简单的类型兼容性检查"""
        output_normalized = self._normalize_type_string(output_type).lower()
        param_normalized = self._normalize_type_string(param_type).lower()

        # 基本类型匹配
        if output_normalized == param_normalized:
            return True

        # 一些常见的兼容性规则
        compatible_pairs = [
            ('str', 'string'),
            ('int', 'number'),
            ('float', 'number'),
            ('int', 'float'),  # int可以转换为float
            ('list', 'array'),
            ('dict', 'object'),
            ('dataframe', 'dict'),  # DataFrame可以转换为dict
            ('any', param_normalized),  # Any类型兼容所有类型
            (output_normalized, 'any'),  # 任何类型都兼容Any
        ]

        for out_type, in_type in compatible_pairs:
            if output_normalized == out_type and param_normalized == in_type:
                return True
            if output_normalized == in_type and param_normalized == out_type:
                return True

        return False

    def _determine_output_type(self, plan: CompositionPlan) -> str:
        """确定输出类型"""
        if not plan.subtasks:
            return 'Any'

        last_subtask = plan.subtasks[-1]
        output_type = last_subtask.output_type

        # 清理和标准化输出类型
        return self._normalize_type_string(output_type)

    def _normalize_type_string(self, type_str: str) -> str:
        """标准化类型字符串，移除中文字符和无效符号"""
        if not type_str:
            return 'Any'

        # 移除中文括号和其他中文字符
        type_str = type_str.replace('（', '(').replace('）', ')')

        # 类型映射
        type_mapping = {
            'DataFrame': 'Any',  # pandas.DataFrame 简化为 Any
            'dict统计信息': 'Dict[str, Any]',
            'dict (统计信息)': 'Dict[str, Any]',
            'dict (统计信息字典)': 'Dict[str, Any]',
            'str报告文本': 'str',
            'str (报告文本)': 'str',
            'str路径': 'str',
            'str (文件路径)': 'str',
            'txt文件': 'str',
            'file (文本文件)': 'str',
            'PDF/HTML 文件': 'str',
            'str (报告文件路径) 或 PDF/HTML 文件': 'str'
        }

        # 查找匹配的类型
        for pattern, replacement in type_mapping.items():
            if pattern in type_str:
                return replacement

        # 如果没有匹配，尝试提取基本类型
        if 'str' in type_str.lower():
            return 'str'
        elif 'dict' in type_str.lower():
            return 'Dict[str, Any]'
        elif 'list' in type_str.lower():
            return 'List[Any]'
        elif 'int' in type_str.lower():
            return 'int'
        elif 'float' in type_str.lower():
            return 'float'
        elif 'bool' in type_str.lower():
            return 'bool'

        return 'Any'

# ================================
# 安全检查AST访问器
# ================================

class SecurityASTVisitor(ast.NodeVisitor):
    """
    安全检查AST访问器 - 检查代码中的潜在安全问题
    支持复杂函数调用模式，如 func.attr 和 module.func()
    """

    def __init__(self, function_matches: Dict[str, Any], registry):
        self.function_matches = function_matches
        self.registry = registry
        self.security_issues = []
        self.missing_functions = []
        self.called_functions = set()

        # 危险函数和操作的AST节点检查
        self.dangerous_functions = {
            'exec', 'eval', '__import__', 'compile',
            'globals', 'locals', 'vars', 'dir',
            'getattr', 'setattr', 'delattr', 'hasattr'
        }

        # 危险的文件操作
        self.dangerous_file_ops = {
            'open', 'file', 'input', 'raw_input'
        }

        # 内置安全函数（允许使用）
        self.safe_builtins = {
            'print', 'len', 'str', 'int', 'float', 'bool',
            'dict', 'list', 'tuple', 'set', 'range',
            'min', 'max', 'sum', 'abs', 'round',
            'type', 'isinstance', 'issubclass'
        }

    def visit_Call(self, node):
        """检查函数调用节点"""
        func_name = self._extract_function_name(node.func)

        if func_name:
            self.called_functions.add(func_name)

            # 检查危险函数
            if func_name in self.dangerous_functions:
                self.security_issues.append(f"检测到危险函数调用: {func_name}")

            # 检查危险文件操作
            elif func_name in self.dangerous_file_ops:
                self.security_issues.append(f"检测到危险文件操作: {func_name}")

            # 检查组件函数是否存在
            elif func_name not in self.safe_builtins:
                is_component_func = any(
                    match.func_name == func_name
                    for match in self.function_matches.values()
                )

                if is_component_func and not self.registry.has_function(func_name):
                    self.missing_functions.append(func_name)

        # 继续访问子节点
        self.generic_visit(node)

    def visit_Import(self, node):
        """检查import语句"""
        for alias in node.names:
            module_name = alias.name
            if self._is_dangerous_module(module_name):
                self.security_issues.append(f"检测到危险模块导入: {module_name}")

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """检查from...import语句"""
        if node.module and self._is_dangerous_module(node.module):
            self.security_issues.append(f"检测到危险模块导入: {node.module}")

        # 检查导入的具体函数
        for alias in node.names:
            if alias.name in self.dangerous_functions:
                self.security_issues.append(f"检测到危险函数导入: {alias.name}")

        self.generic_visit(node)

    def _extract_function_name(self, func_node) -> Optional[str]:
        """
        提取函数名，支持多种调用模式:
        - func() -> 'func'
        - obj.method() -> 'method'
        - module.func() -> 'func'
        - obj.attr.method() -> 'method'
        """
        if isinstance(func_node, ast.Name):
            # 简单函数调用: func()
            return func_node.id

        elif isinstance(func_node, ast.Attribute):
            # 属性调用: obj.method() 或 module.func()
            return func_node.attr

        elif isinstance(func_node, ast.Subscript):
            # 下标调用: obj[key]() - 通常不是函数调用，但要检查
            return None

        else:
            # 其他复杂情况，返回None以保守处理
            return None

    def _is_dangerous_module(self, module_name: str) -> bool:
        """检查是否为危险模块"""
        dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'tempfile',
            'pickle', 'marshal', 'shelve', 'dbm',
            'socket', 'urllib', 'http', 'ftplib',
            'importlib', 'pkgutil', 'runpy'
        }

        # 检查模块名或其父模块
        return (module_name in dangerous_modules or
                any(module_name.startswith(f"{dm}.") for dm in dangerous_modules))


# ================================
# 主要组合器类
# ================================

class FunctionComposer:
    """函数组合器主类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化函数组合器"""
        if config is None:
            config = {}

        self.config = config
        # 支持多种配置键名格式，保持向后兼容
        self.min_confidence = config.get('min_confidence', config.get('min_confidence_threshold', 0.7))
        self.max_functions = config.get('max_functions', config.get('max_functions_per_composition', 10))

        # 获取LLM配置 - 优先使用传入配置，回退到Config类
        try:
            # 优先使用传入的配置
            if 'llm_config' in config and config['llm_config']:
                llm_config = config['llm_config'].copy()
                logger.info("使用传入的LLM配置")
            else:
                # 回退到Config类的配置
                try:
                    autogen_llm_config = Config.get_llm_config()
                    # 转换AutoGen格式到简单格式
                    if 'config_list' in autogen_llm_config and autogen_llm_config['config_list']:
                        first_config = autogen_llm_config['config_list'][0]
                        llm_config = {
                            "model": first_config.get('model', Config.DEEPSEEK_MODEL),
                            "api_key": first_config.get('api_key', Config.DEEPSEEK_API_KEY),
                            "base_url": first_config.get('base_url', Config.DEEPSEEK_BASE_URL),
                            "temperature": autogen_llm_config.get('temperature', 0.7)
                        }
                    else:
                        # 直接使用Config类的属性
                        llm_config = {
                            "model": Config.DEEPSEEK_MODEL,
                            "api_key": Config.DEEPSEEK_API_KEY,
                            "base_url": Config.DEEPSEEK_BASE_URL,
                            "temperature": 0.7
                        }
                    logger.info("使用Config类LLM配置")
                except Exception as config_error:
                    logger.warning(f"Config类配置获取失败: {config_error}，使用环境变量")
                    # 最后回退到环境变量
                    llm_config = {
                        "model": os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
                        "api_key": os.getenv('DEEPSEEK_API_KEY'),
                        "base_url": os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
                        "temperature": 0.7
                    }

            # 避免在日志中泄露API密钥
            safe_config = llm_config.copy()
            if 'api_key' in safe_config and safe_config['api_key']:
                api_key = safe_config['api_key']
                safe_config['api_key'] = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            logger.info(f"LLM配置详情: {safe_config}")

        except Exception as e:
            logger.error(f"LLM配置获取失败: {e}")
            # 最终回退配置
            llm_config = {
                "model": "deepseek-chat",
                "api_key": os.getenv('DEEPSEEK_API_KEY'),
                "base_url": "https://api.deepseek.com",
                "temperature": 0.7
            }

        # 保存LLM配置为实例属性
        self.llm_config = llm_config

        # 初始化组件
        keyword_config_path = config.get('keyword_config_path')
        enable_llm_matching = config.get('enable_llm_matching', True)  # 默认启用LLM匹配

        self.task_decomposer = TaskDecomposer(llm_config)
        self.function_matcher = FunctionMatcher(
            llm_config=llm_config if enable_llm_matching else None,
            keyword_config_path=keyword_config_path
        )
        self.composition_generator = CompositionGenerator(llm_config)
        self.registry = get_registry()

        # 将FunctionMatcher的性能优化配置暴露到FunctionComposer级别
        # 这样可以保持向后兼容性，同时允许直接访问这些配置
        self.enable_prefiltering = self.function_matcher.enable_prefiltering
        self.prefilter_limit = self.function_matcher.prefilter_limit
        self.prefilter_threshold = self.function_matcher.prefilter_threshold
        self.enable_llm_caching = self.function_matcher.enable_llm_caching
        self.enable_concurrent_matching = self.function_matcher.enable_concurrent_matching
        self.max_concurrent_workers = self.function_matcher.max_concurrent_workers

        logger.info("FunctionComposer 初始化完成")

    def compose_functions(self, task_description: str) -> Tuple[bool, str, Optional[CompositeFunction]]:
        """
        组合函数主入口 - 重构为更清晰的步骤

        Args:
            task_description: 任务描述

        Returns:
            Tuple[success, message, composite_function]
        """
        try:
            logger.info(f"开始组合函数处理任务: {task_description}")

            # 1. 预检查
            can_compose, reason = self.can_compose_for_task(task_description)
            if not can_compose:
                return False, reason, None

            # 2. 任务分解
            subtasks = self._decompose_task_step(task_description)
            if not subtasks:
                return False, "任务分解失败", None

            # 3. 函数匹配
            function_matches = self._match_functions_step(subtasks)
            if not function_matches:
                return False, "函数匹配失败", None

            # 4. 构建组合计划
            plan = self._build_composition_plan(subtasks, function_matches)

            # 5. 生成并验证组合函数
            composite_func = self._generate_and_validate_step(plan, function_matches)
            if not composite_func:
                return False, "组合函数生成或验证失败", None

            # 6. 确保函数名唯一性
            self._ensure_unique_function_name(composite_func)

            # 7. 注册组合函数
            success = self._register_composite_function(composite_func, task_description)
            if success:
                logger.info(f"成功创建并注册组合函数: {composite_func.name}")
                return True, f"成功创建组合函数: {composite_func.name}", composite_func
            else:
                return False, "组合函数注册失败", None

        except Exception as e:
            logger.error(f"函数组合失败: {e}")
            return False, f"函数组合过程中发生错误: {str(e)}", None

    def _decompose_task_step(self, task_description: str) -> Optional[List[SubTask]]:
        """步骤2: 任务分解"""
        subtasks = self.task_decomposer.decompose_task(task_description)
        if not subtasks:
            logger.error("任务分解失败")
            return None

        if len(subtasks) > self.max_functions:
            logger.error(f"子任务数量过多 ({len(subtasks)} > {self.max_functions})")
            return None

        logger.info(f"任务分解成功，共{len(subtasks)}个子任务")
        return subtasks

    def _match_functions_step(self, subtasks: List[SubTask]) -> Optional[Dict[str, FunctionMatch]]:
        """步骤3: 函数匹配"""
        function_matches = {}

        for subtask in subtasks:
            matches = self.function_matcher.find_matching_functions(subtask)
            if not matches or matches[0].confidence < self.min_confidence:
                logger.error(f"未找到子任务 '{subtask.description}' 的匹配函数")
                return None

            function_matches[subtask.id] = matches[0]  # 使用最佳匹配
            logger.debug(f"子任务 '{subtask.description}' 匹配到函数: {matches[0].func_name}")

        logger.info(f"函数匹配成功，共匹配{len(function_matches)}个函数")
        return function_matches

    def _build_composition_plan(self, subtasks: List[SubTask],
                               function_matches: Dict[str, FunctionMatch]) -> CompositionPlan:
        """步骤4: 构建组合计划"""
        execution_order = [st.id for st in subtasks]  # 简化版本，按顺序执行
        data_flow = {}  # 简化版本，链式数据流

        plan = CompositionPlan(
            subtasks=subtasks,
            function_matches=function_matches,
            execution_order=execution_order,
            data_flow=data_flow
        )

        logger.info(f"组合计划构建完成，执行顺序: {execution_order}")
        return plan

    def _generate_and_validate_step(self, plan: CompositionPlan,
                                   function_matches: Dict[str, FunctionMatch]) -> Optional[CompositeFunction]:
        """步骤5: 生成并验证组合函数"""
        try:
            # 生成组合函数
            composite_func = self.composition_generator.generate_composition(plan)
            logger.info(f"组合函数生成完成: {composite_func.name}")

            # 基础语法验证
            from .secure_executor import validate_function_code
            is_valid, validation_msg, func_name = validate_function_code(composite_func.code)
            if not is_valid:
                logger.error(f"生成的组合函数验证失败: {validation_msg}")
                return None

            # 增强安全验证
            validation_result = self._validate_composite_function_security_enhanced(composite_func, function_matches)
            if not validation_result[0]:
                logger.error(f"安全验证失败: {validation_result[1]}")
                return None

            logger.info("组合函数验证通过")
            return composite_func

        except Exception as e:
            logger.error(f"组合函数生成或验证失败: {e}")
            return None

    def _ensure_unique_function_name(self, composite_func: CompositeFunction) -> None:
        """步骤6: 确保函数名唯一性"""
        if self.registry.has_function(composite_func.name):
            counter = 1
            base_name = composite_func.name
            while self.registry.has_function(f"{base_name}_{counter}"):
                counter += 1
            composite_func.name = f"{base_name}_{counter}"
            logger.info(f"函数名冲突，使用新名称: {composite_func.name}")

    def _register_composite_function(self, composite_func: CompositeFunction,
                                   task_description: str) -> bool:
        """步骤7: 注册组合函数"""
        try:
            success = self.registry.register_function(
                func_name=composite_func.name,
                func_code=composite_func.code,
                description=composite_func.description,
                task_origin=f"组合函数: {task_description}",
                test_cases=[]  # 暂时不生成测试用例
            )

            if success:
                logger.info(f"组合函数注册成功: {composite_func.name}")
            else:
                logger.error("组合函数注册失败")

            return success

        except Exception as e:
            logger.error(f"注册组合函数时发生错误: {e}")
            return False

    def _validate_composite_function_security_enhanced(self, composite_func: CompositeFunction,
                                                     function_matches: Dict[str, FunctionMatch]) -> Tuple[bool, str]:
        """
        增强安全验证 - 使用AST节点检查，支持复杂函数调用

        Args:
            composite_func: 组合函数
            function_matches: 函数匹配结果

        Returns:
            Tuple[is_valid, error_message]
        """
        try:
            import ast

            # 解析生成的代码
            try:
                tree = ast.parse(composite_func.code)
            except SyntaxError as e:
                return False, f"代码语法错误: {e}"

            # 使用AST访问器检查安全性
            security_checker = SecurityASTVisitor(function_matches, self.registry)
            security_checker.visit(tree)

            if security_checker.security_issues:
                return False, f"安全检查失败: {'; '.join(security_checker.security_issues)}"

            if security_checker.missing_functions:
                return False, f"缺少组件函数: {', '.join(security_checker.missing_functions)}"

            return True, "增强安全验证通过"

        except Exception as e:
            return False, f"安全验证过程中发生错误: {str(e)}"

    def _validate_composite_function_security(self, composite_func: CompositeFunction,
                                            function_matches: Dict[str, FunctionMatch]) -> Tuple[bool, str]:
        """
        保留原有安全验证方法以保持兼容性
        """
        return self._validate_composite_function_security_enhanced(composite_func, function_matches)

    def can_compose_for_task(self, task_description: str) -> Tuple[bool, str]:
        """
        使用LLM智能判断是否可以为任务组合函数

        Args:
            task_description: 任务描述

        Returns:
            Tuple[can_compose, reason] - (是否可以组合, 原因说明)
        """
        try:
            # 基础检查：任务描述是否为空或过短
            if not task_description or len(task_description.strip()) < 3:
                return False, "任务描述为空或过短"

            # 检查是否有足够的组件函数
            all_functions = self.registry.list_functions()
            valid_functions = [f for f in all_functions if f not in ['metadata', 'last_updated', 'version', 'total_functions']]

            if len(valid_functions) < 2:
                return False, "系统中函数数量不足，无法进行组合"

            # 使用LLM进行智能判断
            llm_result = self._llm_composition_feasibility_check(task_description, valid_functions)

            if llm_result:
                return llm_result["can_compose"], llm_result["reason"]
            else:
                # LLM判断失败时的回退逻辑
                return self._fallback_composition_check(task_description)

        except Exception as e:
            logger.error(f"检查组合可行性失败: {e}")
            return False, f"检查失败: {str(e)}"

    def _llm_composition_feasibility_check(self, task_description: str, available_functions: List[str]) -> Optional[Dict[str, Any]]:
        """
        使用LLM判断任务是否适合组合函数

        Args:
            task_description: 任务描述
            available_functions: 可用函数列表

        Returns:
            Dict包含: can_compose, reason, composition_strategy, estimated_steps
        """
        try:
            logger.info(f"使用LLM判断组合可行性: {task_description}")

            # 构建LLM提示
            prompt = self._build_composition_feasibility_prompt(task_description, available_functions)

            # 调用LLM
            response = self._call_llm_for_composition_check(prompt)

            # 解析LLM响应
            result = self._parse_composition_feasibility_response(response)

            if result:
                logger.info(f"LLM组合判断结果: {result['can_compose']} - {result['reason']}")
                return result
            else:
                logger.warning("LLM组合判断解析失败")
                return None

        except Exception as e:
            logger.error(f"LLM组合可行性检查失败: {e}")
            return None

    def _build_composition_feasibility_prompt(self, task_description: str, available_functions: List[str]) -> str:
        """构建组合可行性判断的LLM提示"""

        # 获取函数概览信息（前10个函数的简要信息）
        function_overview = self._get_function_overview(available_functions[:10])

        prompt = f"""你是一个智能函数组合分析专家。请分析给定任务是否适合通过组合现有函数来完成。

## 任务描述
{task_description}

## 系统中可用函数概览
总函数数量: {len(available_functions)}
主要函数示例:
{function_overview}

## 分析要求
请从以下维度分析任务是否适合函数组合：

1. **任务复杂度分析**：
   - 任务是否可以分解为多个子步骤？
   - 是否需要多个不同的操作？
   - 单一函数是否难以完成整个任务？

2. **组合可行性分析**：
   - 现有函数是否能覆盖任务的主要步骤？
   - 函数之间是否可能存在合理的数据流？
   - 是否存在明显的组合模式？

3. **效益评估**：
   - 组合函数相比单一函数是否有明显优势？
   - 是否已存在完全匹配的单一函数？
   - 组合的复杂度是否合理？

## 输出格式
请以JSON格式返回分析结果：

```json
{{
    "can_compose": true/false,
    "reason": "详细的判断原因",
    "composition_strategy": "如果适合组合，建议的组合策略",
    "estimated_steps": 预估的子任务数量,
    "confidence": "high/medium/low",
    "key_operations": ["主要操作1", "主要操作2", "..."]
}}
```

## 示例分析

**适合组合的任务示例**：
- "读取CSV文件，计算统计信息，生成报告并保存" → 可分解为：文件读取 + 数据分析 + 报告生成 + 文件保存
- "处理用户输入，验证数据，更新数据库，发送通知" → 可分解为：输入处理 + 数据验证 + 数据库操作 + 通知发送

**不适合组合的任务示例**：
- "计算两个数的和" → 过于简单，单一函数即可
- "获取当前时间" → 原子操作，无需组合
- "打印Hello World" → 基础操作，不需要组合

请基于以上分析框架，对给定任务进行判断。"""

        return prompt

    def _get_function_overview(self, function_names: List[str]) -> str:
        """获取函数概览信息"""
        overview_lines = []

        for i, func_name in enumerate(function_names, 1):
            try:
                func_info = self.registry.get_function_info(func_name)
                if func_info and isinstance(func_info, dict):
                    name = func_info.get('name', func_name)
                    description = func_info.get('description', '无描述')
                    # 限制描述长度
                    if len(description) > 60:
                        description = description[:60] + "..."

                    overview_lines.append(f"{i}. {name}: {description}")
                else:
                    overview_lines.append(f"{i}. {func_name}: 函数信息不可用")

            except Exception as e:
                logger.debug(f"获取函数 {func_name} 信息失败: {e}")
                overview_lines.append(f"{i}. {func_name}: 获取信息失败")

        if len(function_names) < len(self.registry.list_functions()):
            overview_lines.append("... (还有更多函数)")

        return "\n".join(overview_lines)

    def _call_llm_for_composition_check(self, prompt: str) -> str:
        """调用LLM进行组合可行性检查"""
        try:
            # 创建AutoGen配置，优化用于结构化输出
            llm_config = {
                "config_list": [{
                    "model": self.llm_config.get("model", "deepseek-chat"),
                    "api_key": self.llm_config.get("api_key"),
                    "base_url": self.llm_config.get("base_url", "https://api.deepseek.com"),
                    "temperature": 0.1,  # 低温度确保稳定输出
                    "max_tokens": 800,   # 足够的token用于详细分析
                }],
                "timeout": 30
            }

            # 创建助手代理
            assistant = autogen.AssistantAgent(
                name="composition_analyzer",
                llm_config=llm_config,
                system_message="你是一个专业的函数组合分析专家，擅长判断任务是否适合通过组合现有函数来完成。"
            )

            # 创建用户代理
            user_proxy = autogen.UserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                code_execution_config=False
            )

            # 发起对话
            if IS_AUTOGEN_ASYNC:
                import asyncio
                response = asyncio.run(user_proxy.initiate_chat(
                    assistant,
                    message=prompt,
                    max_turns=1
                ))
            else:
                response = user_proxy.initiate_chat(
                    assistant,
                    message=prompt,
                    max_turns=1
                )

            # 提取最后的消息内容
            if hasattr(response, 'chat_history') and response.chat_history:
                last_message = response.chat_history[-1]
            else:
                # 兼容不同版本的AutoGen
                last_message = response

            if isinstance(last_message, dict):
                return last_message.get('content', '')
            else:
                return getattr(last_message, 'content', str(last_message))

        except Exception as e:
            logger.error(f"LLM组合检查调用失败: {e}")
            return ""

    def _parse_composition_feasibility_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM组合可行性检查响应"""
        try:
            # 提取JSON块
            json_str = extract_json_block(response)
            if not json_str:
                logger.warning("未找到有效的JSON响应，尝试正则表达式提取")
                return self._extract_composition_result_with_regex(response)

            # 解析JSON
            data = parse_json_with_fallback(json_str)
            if data is None:
                logger.warning("JSON解析失败，尝试正则表达式提取")
                return self._extract_composition_result_with_regex(response)

            # 验证和标准化结果
            result = {
                'can_compose': bool(data.get('can_compose', False)),
                'reason': str(data.get('reason', '无具体原因')),
                'composition_strategy': str(data.get('composition_strategy', '')),
                'estimated_steps': int(data.get('estimated_steps', 2)),
                'confidence': str(data.get('confidence', 'medium')),
                'key_operations': data.get('key_operations', [])
            }

            # 验证必需字段
            if not result['reason']:
                result['reason'] = "LLM分析结果"

            # 确保confidence值有效
            if result['confidence'] not in ['high', 'medium', 'low']:
                result['confidence'] = 'medium'

            # 确保estimated_steps在合理范围内
            if result['estimated_steps'] < 1:
                result['estimated_steps'] = 2
            elif result['estimated_steps'] > 10:
                result['estimated_steps'] = 10

            logger.debug(f"组合可行性解析结果: {result}")
            return result

        except Exception as e:
            logger.error(f"解析组合可行性响应失败: {e}")
            return self._extract_composition_result_with_regex(response)

    def _extract_composition_result_with_regex(self, response: str) -> Optional[Dict[str, Any]]:
        """使用正则表达式从响应中提取组合结果"""
        try:
            import re

            # 提取can_compose
            can_compose = False

            # 先尝试结构化提取
            structured_patterns = [
                r'"can_compose":\s*(true|false)',
                r'can_compose["\']?\s*[:=]\s*(true|false)'
            ]

            found_structured = False
            for pattern in structured_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    can_compose = match.group(1).lower() == 'true'
                    found_structured = True
                    break

            # 如果没有找到结构化信息，使用语义提取
            if not found_structured:
                # 先检查否定模式，避免"不适合组合"被"适合组合"误匹配
                negative_patterns = r'不适合组合|无需组合|不建议组合|不适合函数组合|不适合.*组合'
                positive_patterns = r'(?<!不)适合组合|可以组合|建议组合|(?<!不)适合函数组合'

                if re.search(negative_patterns, response):
                    can_compose = False
                elif re.search(positive_patterns, response):
                    can_compose = True
                # 如果都没匹配到，保持默认值False

            # 提取reason
            reason = "基于LLM分析"
            reason_patterns = [
                r'"reason":\s*"([^"]+)"',
                r'reason["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                r'原因[:：]\s*([^\n]+)',
                r'判断[:：]\s*([^\n]+)'
            ]

            for pattern in reason_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    reason = match.group(1).strip()
                    break

            # 提取estimated_steps
            estimated_steps = 2
            steps_patterns = [
                r'"estimated_steps":\s*(\d+)',
                r'estimated_steps["\']?\s*[:=]\s*(\d+)',
                r'(\d+)\s*个?步骤',
                r'分解为\s*(\d+)\s*个'
            ]

            for pattern in steps_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        estimated_steps = int(match.group(1))
                        if estimated_steps < 1:
                            estimated_steps = 2
                        elif estimated_steps > 10:
                            estimated_steps = 10
                        break
                    except ValueError:
                        continue

            # 构建结果
            result = {
                'can_compose': can_compose,
                'reason': reason,
                'composition_strategy': '基于正则表达式提取',
                'estimated_steps': estimated_steps,
                'confidence': 'low',  # 正则提取的置信度较低
                'key_operations': []
            }

            logger.debug(f"正则表达式提取结果: {result}")
            return result

        except Exception as e:
            logger.error(f"正则表达式提取失败: {e}")
            return None

    def _fallback_composition_check(self, task_description: str) -> Tuple[bool, str]:
        """LLM判断失败时的回退检查逻辑"""
        try:
            logger.warning("使用回退逻辑进行组合可行性检查")

            # 基础长度检查
            if len(task_description.strip()) < 10:
                return False, "任务描述过于简单，不需要组合函数"

            # 检查组合关键词
            composition_keywords = [
                '并且', '然后', '接着', '同时', '以及', '和', '，', '、',
                'and', 'then', 'after', 'also', 'plus', ','
            ]

            has_composition_hint = any(keyword in task_description.lower() for keyword in composition_keywords)

            # 检查任务复杂度指标
            complexity_indicators = [
                '读取', '处理', '分析', '生成', '保存', '发送', '计算', '转换',
                'read', 'process', 'analyze', 'generate', 'save', 'send', 'calculate', 'convert'
            ]

            complexity_count = sum(1 for indicator in complexity_indicators if indicator in task_description.lower())

            # 判断逻辑
            if has_composition_hint and complexity_count >= 2:
                return True, "任务包含组合提示词且具有一定复杂度，适合函数组合"
            elif len(task_description) >= 30 and complexity_count >= 2:
                return True, "任务描述较复杂，包含多个操作，适合尝试函数组合"
            elif complexity_count >= 3:
                return True, "任务包含多个复杂操作，建议使用函数组合"
            else:
                return False, "任务相对简单，可能不需要函数组合"

        except Exception as e:
            logger.error(f"回退检查失败: {e}")
            return True, "无法确定，默认尝试组合"  # 保守策略，倾向于尝试组合

# ================================
# 工厂函数
# ================================

def get_function_composer(config: Optional[Dict[str, Any]] = None) -> FunctionComposer:
    """
    工厂函数：获取函数组合器实例

    Args:
        config: 组合器配置

    Returns:
        FunctionComposer 实例
    """
    return FunctionComposer(config)
