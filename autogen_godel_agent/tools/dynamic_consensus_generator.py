"""
Dynamic Consensus Generator - 根据生成的代码动态形成代理共识
"""

import ast
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """分析代码并提取共识要素"""
    
    def analyze_function_code(self, func_code: str, func_name: str, task_description: str) -> Dict[str, Any]:
        """
        分析函数代码，提取共识要素
        
        Args:
            func_code: 函数源代码
            func_name: 函数名
            task_description: 任务描述
            
        Returns:
            包含共识要素的字典
        """
        consensus_elements = {
            'naming_analysis': self._analyze_naming(func_name, task_description),
            'type_analysis': self._analyze_types(func_code),
            'error_handling_analysis': self._analyze_error_handling(func_code),
            'design_patterns': self._analyze_design_patterns(func_code),
            'system_constraints': self._identify_system_constraints(func_code, func_name),
            'code_complexity': self._analyze_complexity(func_code),
            'documentation_style': self._analyze_documentation(func_code)
        }
        
        return consensus_elements
    
    def _analyze_naming(self, func_name: str, task_description: str) -> Dict[str, Any]:
        """分析命名模式"""
        analysis = {}
        
        # 检测时间戳/哈希后缀
        timestamp_match = re.search(r'_(\d{10,})$', func_name)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            analysis['has_timestamp_suffix'] = True
            analysis['timestamp_value'] = timestamp
            analysis['timestamp_purpose'] = "用于函数版本控制和唯一性标识"
            analysis['base_name'] = func_name.replace(f'_{timestamp}', '')
        else:
            analysis['has_timestamp_suffix'] = False
            analysis['base_name'] = func_name
        
        # 分析命名风格
        if func_name.startswith('create_a_'):
            analysis['naming_style'] = 'descriptive_creation'
            analysis['style_rationale'] = '系统自动生成的描述性创建函数名'
        elif '_' in func_name:
            analysis['naming_style'] = 'snake_case'
        else:
            analysis['naming_style'] = 'other'
        
        # 分析名称长度
        analysis['name_length'] = len(func_name)
        analysis['is_long_name'] = len(func_name) > 30
        
        return analysis
    
    def _analyze_types(self, func_code: str) -> Dict[str, Any]:
        """分析类型注解"""
        analysis = {}
        
        try:
            tree = ast.parse(func_code)
            func_def = tree.body[0] if tree.body and isinstance(tree.body[0], ast.FunctionDef) else None
            
            if func_def:
                # 分析返回类型
                if func_def.returns:
                    return_type = ast.unparse(func_def.returns) if hasattr(ast, 'unparse') else str(func_def.returns)
                    analysis['return_type'] = return_type.lower()
                    analysis['uses_any_return'] = 'any' in return_type.lower()
                    
                    if analysis['uses_any_return']:
                        analysis['any_usage_context'] = '系统要求使用Any类型以保持最大灵活性'
                
                # 分析参数类型
                analysis['parameter_types'] = []
                for arg in func_def.args.args:
                    if arg.annotation:
                        param_type = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                        analysis['parameter_types'].append({
                            'name': arg.arg,
                            'type': param_type
                        })
        except Exception as e:
            logger.warning(f"类型分析失败: {e}")
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    def _analyze_error_handling(self, func_code: str) -> Dict[str, Any]:
        """分析错误处理模式"""
        analysis = {}
        
        # 检测try-except块
        has_try_except = 'try:' in func_code and 'except' in func_code
        analysis['has_exception_handling'] = has_try_except
        
        if has_try_except:
            # 检测异常类型
            if 'except Exception' in func_code:
                analysis['exception_scope'] = 'broad'
                analysis['exception_critique'] = '捕获所有异常可能掩盖真正的问题'
            elif 'except ' in func_code:
                analysis['exception_scope'] = 'specific'
                analysis['exception_critique'] = '使用了特定异常类型，这是好的实践'
            
            # 检测错误处理策略
            if 'return ""' in func_code or 'return None' in func_code:
                analysis['error_strategy'] = 'silent_fallback'
                analysis['strategy_critique'] = '静默失败可能让用户难以发现问题'
            elif 'raise' in func_code:
                analysis['error_strategy'] = 'exception_propagation'
                analysis['strategy_critique'] = '异常传播让调用者可以自行处理错误'
        
        # 检测None处理
        if 'is None' in func_code:
            analysis['handles_none_input'] = True
            if 'return ""' in func_code:
                analysis['none_handling_strategy'] = 'convert_to_empty'
                analysis['none_critique'] = 'None转换为空值可能掩盖逻辑错误'
            elif 'raise' in func_code:
                analysis['none_handling_strategy'] = 'raise_exception'
                analysis['none_critique'] = '对None抛出异常符合Python惯例'
        
        return analysis
    
    def _analyze_design_patterns(self, func_code: str) -> Dict[str, Any]:
        """分析设计模式"""
        analysis = {}
        
        # 检测函数复杂度
        line_count = len([line for line in func_code.split('\n') if line.strip()])
        analysis['line_count'] = line_count
        analysis['complexity_level'] = 'simple' if line_count < 20 else 'complex'
        
        # 检测单一职责
        operation_count = 0
        if 'return' in func_code:
            operation_count += func_code.count('return')
        if 'if' in func_code:
            operation_count += func_code.count('if')
        
        analysis['operation_count'] = operation_count
        analysis['follows_srp'] = operation_count <= 3  # 简单启发式
        
        return analysis
    
    def _identify_system_constraints(self, func_code: str, func_name: str) -> Dict[str, Any]:
        """识别系统约束"""
        constraints = {}
        
        # 自动生成的函数特征
        if re.search(r'_\d{10,}$', func_name):
            constraints['auto_generated'] = True
            constraints['naming_constraint'] = '系统自动生成带时间戳的函数名以确保唯一性'
        
        # 返回类型约束
        if '-> any' in func_code.lower() or '-> Any' in func_code:
            constraints['return_type_constraint'] = True
            constraints['return_type_rationale'] = '系统要求返回Any类型以支持多种返回值类型'
        
        # 安全约束
        dangerous_patterns = ['import os', 'import sys', 'open(', 'exec(', 'eval(']
        has_dangerous = any(pattern in func_code for pattern in dangerous_patterns)
        constraints['security_constrained'] = not has_dangerous
        
        return constraints
    
    def _analyze_complexity(self, func_code: str) -> Dict[str, Any]:
        """分析代码复杂度"""
        analysis = {}
        
        # 简单的复杂度指标
        analysis['total_lines'] = len(func_code.split('\n'))
        analysis['code_lines'] = len([line for line in func_code.split('\n') if line.strip() and not line.strip().startswith('#')])
        analysis['comment_lines'] = len([line for line in func_code.split('\n') if line.strip().startswith('#')])
        
        # 控制流复杂度
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally']
        analysis['control_flow_count'] = sum(func_code.count(keyword) for keyword in control_keywords)
        
        return analysis
    
    def _analyze_documentation(self, func_code: str) -> Dict[str, Any]:
        """分析文档风格"""
        analysis = {}
        
        # 检测docstring
        has_docstring = '"""' in func_code or "'''" in func_code
        analysis['has_docstring'] = has_docstring
        
        if has_docstring:
            # 提取docstring内容
            docstring_match = re.search(r'"""(.*?)"""', func_code, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1).strip()
                analysis['docstring_length'] = len(docstring)
                analysis['has_parameters_section'] = 'Parameters:' in docstring or 'Args:' in docstring
                analysis['has_returns_section'] = 'Returns:' in docstring
                analysis['has_examples_section'] = 'Examples:' in docstring or 'Example:' in docstring
        
        return analysis


class DynamicConsensusGenerator:
    """动态生成代理共识"""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
    
    def generate_consensus(self, func_code: str, func_name: str, task_description: str) -> Dict[str, Any]:
        """
        根据生成的代码动态生成共识
        
        Args:
            func_code: 函数源代码
            func_name: 函数名
            task_description: 任务描述
            
        Returns:
            动态生成的共识字典
        """
        logger.info(f"为函数 {func_name} 生成动态共识")
        
        # 分析代码
        analysis = self.analyzer.analyze_function_code(func_code, func_name, task_description)
        
        # 生成共识
        consensus = {
            'context_info': {
                'function_name': func_name,
                'task_description': task_description,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'technical_consensus': self._build_technical_consensus(analysis),
            'system_constraints': self._build_system_constraints(analysis),
            'design_principles': self._build_design_principles(analysis),
            'dialogue_guidelines': self._build_dialogue_guidelines(analysis)
        }
        
        return consensus
    
    def _build_technical_consensus(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """构建技术共识"""
        consensus = {}
        
        # 命名共识
        naming = analysis.get('naming_analysis', {})
        if naming.get('has_timestamp_suffix'):
            consensus['function_naming'] = {
                'timestamp_suffix': {
                    'purpose': naming.get('timestamp_purpose', '用于版本控制和唯一性'),
                    'value': naming.get('timestamp_value'),
                    'rationale': '系统自动生成，防止命名冲突并支持函数演进跟踪'
                }
            }
        
        # 类型共识
        type_analysis = analysis.get('type_analysis', {})
        if type_analysis.get('uses_any_return'):
            consensus['type_annotations'] = {
                'any_return_type': {
                    'context': type_analysis.get('any_usage_context', '系统要求'),
                    'rationale': '保持最大灵活性，支持多种返回值类型',
                    'alternatives': '在明确返回类型时可考虑使用Union类型'
                }
            }
        
        return consensus
    
    def _build_system_constraints(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """构建系统约束"""
        constraints = analysis.get('system_constraints', {})
        
        system_constraints = {}
        
        if constraints.get('auto_generated'):
            system_constraints['auto_generation'] = {
                'naming_pattern': constraints.get('naming_constraint', '自动生成命名'),
                'uniqueness_guarantee': '时间戳后缀确保函数名唯一性',
                'evolution_support': '支持函数版本演进和追踪'
            }
        
        if constraints.get('return_type_constraint'):
            system_constraints['type_flexibility'] = {
                'return_type_policy': constraints.get('return_type_rationale', '灵活返回类型'),
                'type_safety_balance': '在类型安全和灵活性之间取得平衡'
            }
        
        return system_constraints
    
    def _build_design_principles(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """构建设计原则"""
        principles = {}
        
        # 错误处理原则
        error_analysis = analysis.get('error_handling_analysis', {})
        if error_analysis.get('has_exception_handling'):
            principles['error_handling'] = {
                'current_strategy': error_analysis.get('error_strategy', 'unknown'),
                'critique': error_analysis.get('strategy_critique', ''),
                'improvement_direction': self._suggest_error_handling_improvement(error_analysis)
            }
        
        # 复杂度原则
        complexity = analysis.get('code_complexity', {})
        principles['complexity_management'] = {
            'current_level': 'simple' if complexity.get('code_lines', 0) < 20 else 'complex',
            'line_count': complexity.get('code_lines', 0),
            'maintainability_focus': '保持代码简洁和可读性'
        }
        
        return principles
    
    def _build_dialogue_guidelines(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """构建对话指导原则"""
        guidelines = {
            'context_awareness': {
                'acknowledge_constraints': '理解并承认系统约束和设计背景',
                'focus_on_improvements': '专注于在约束条件下的改进机会',
                'avoid_invalid_criticism': '避免对系统设计决策的无效批评'
            },
            'constructive_approach': {
                'build_on_analysis': '基于代码分析结果进行讨论',
                'suggest_alternatives': '在承认约束的前提下提出替代方案',
                'consider_tradeoffs': '考虑不同方案的权衡和取舍'
            }
        }
        
        # 根据具体分析添加特定指导
        naming = analysis.get('naming_analysis', {})
        if naming.get('is_long_name'):
            guidelines['specific_guidance'] = {
                'naming_discussion': '讨论长函数名的可读性影响，但要理解其必要性',
                'user_experience_focus': '考虑如何在保持系统约束的同时改善用户体验'
            }
        
        return guidelines
    
    def _suggest_error_handling_improvement(self, error_analysis: Dict[str, Any]) -> str:
        """建议错误处理改进方向"""
        if error_analysis.get('exception_scope') == 'broad':
            return '考虑使用更具体的异常类型，或评估是否真的需要异常处理'
        elif error_analysis.get('error_strategy') == 'silent_fallback':
            return '考虑是否应该让异常传播，让调用者决定如何处理'
        else:
            return '当前错误处理策略合理，可考虑在文档中说明'
    
    def format_consensus_for_dialogue(self, consensus: Dict[str, Any]) -> str:
        """格式化共识信息用于对话"""
        formatted = "**🎯 基于当前代码的技术共识：**\n\n"
        
        # 上下文信息
        context = consensus.get('context_info', {})
        formatted += f"**📋 分析对象**: {context.get('function_name', 'unknown')}\n"
        formatted += f"**📝 任务描述**: {context.get('task_description', 'unknown')}\n\n"
        
        # 技术共识
        tech_consensus = consensus.get('technical_consensus', {})
        
        # 函数命名共识
        if 'function_naming' in tech_consensus:
            naming = tech_consensus['function_naming']
            if 'timestamp_suffix' in naming:
                suffix_info = naming['timestamp_suffix']
                formatted += f"**🏷️ 函数命名**: {suffix_info.get('rationale', '系统命名策略')}\n"
                formatted += f"   - 时间戳值: {suffix_info.get('value', 'N/A')}\n"
                formatted += f"   - 目的: {suffix_info.get('purpose', '唯一性保证')}\n\n"
        
        # 类型注解共识
        if 'type_annotations' in tech_consensus:
            types = tech_consensus['type_annotations']
            if 'any_return_type' in types:
                any_info = types['any_return_type']
                formatted += f"**🔧 返回类型Any**: {any_info.get('rationale', '系统要求')}\n"
                formatted += f"   - 背景: {any_info.get('context', '灵活性需求')}\n"
                formatted += f"   - 替代方案: {any_info.get('alternatives', '考虑具体类型')}\n\n"
        
        # 系统约束
        constraints = consensus.get('system_constraints', {})
        if constraints:
            formatted += "**⚙️ 系统约束理解**:\n"
            for key, value in constraints.items():
                if isinstance(value, dict):
                    formatted += f"   - {key}: {value.get('naming_pattern', value.get('return_type_policy', '系统策略'))}\n"
        
        formatted += "\n**💡 请基于以上共识进行建设性讨论，专注于在约束条件下的改进机会！**\n\n"
        
        return formatted
