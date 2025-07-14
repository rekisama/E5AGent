"""
真正的LLM自生成共识系统
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMConsensusGenerator:
    """使用LLM自动生成代理共识"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
    
    def generate_consensus_through_llm(self, func_code: str, func_name: str, task_description: str) -> Dict[str, Any]:
        """
        通过LLM分析代码并生成共识
        
        Args:
            func_code: 函数源代码
            func_name: 函数名
            task_description: 任务描述
            
        Returns:
            LLM生成的共识字典
        """
        logger.info(f"使用LLM为函数 {func_name} 生成共识")
        
        # 构建LLM提示
        consensus_prompt = self._build_consensus_prompt(func_code, func_name, task_description)
        
        try:
            # 调用LLM生成共识
            import autogen
            
            # 创建专门的共识分析师
            consensus_analyst = autogen.AssistantAgent(
                name="ConsensusAnalyst",
                system_message="""你是一个代码分析专家，专门负责分析代码并生成技术共识。

你的任务是：
1. 深入分析给定的代码
2. 识别设计决策和约束条件
3. 理解系统架构和技术选择
4. 生成代理对话的技术共识基础

请以JSON格式返回分析结果，包含：
- technical_insights: 技术洞察
- design_rationale: 设计理由
- system_constraints: 系统约束
- improvement_opportunities: 改进机会
- dialogue_focus: 对话重点

要求：
- 客观分析，不带偏见
- 理解设计背景和约束
- 识别真正的改进机会
- 为建设性对话奠定基础""",
                llm_config=self.llm_config
            )
            
            user_proxy = autogen.UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False
            )
            
            # 启动分析对话
            user_proxy.initiate_chat(
                consensus_analyst,
                message=consensus_prompt
            )
            
            # 提取LLM的响应
            last_message = consensus_analyst.last_message()
            if last_message and 'content' in last_message:
                consensus_text = last_message['content']
                
                # 尝试解析JSON
                consensus = self._parse_llm_consensus(consensus_text)
                
                if consensus:
                    logger.info("LLM成功生成共识")
                    return self._format_consensus(consensus, func_name, task_description)
                else:
                    logger.warning("LLM共识解析失败，使用备用方案")
                    return self._fallback_consensus(func_code, func_name, task_description)
            
        except Exception as e:
            logger.error(f"LLM共识生成失败: {e}")
            return self._fallback_consensus(func_code, func_name, task_description)
    
    def _build_consensus_prompt(self, func_code: str, func_name: str, task_description: str) -> str:
        """构建LLM共识生成提示"""
        return f"""
# 代码共识分析任务

请分析以下代码并生成技术共识，为后续的代理对话奠定基础。

## 代码信息
**函数名**: {func_name}
**任务描述**: {task_description}

## 代码内容
```python
{func_code}
```

## 分析要求

请深入分析这个函数，并以JSON格式返回以下内容：

```json
{{
  "technical_insights": {{
    "naming_strategy": "分析函数命名策略和原因",
    "type_design": "分析类型注解的设计考虑",
    "algorithm_choice": "分析算法和实现选择",
    "error_handling": "分析错误处理策略"
  }},
  "design_rationale": {{
    "architecture_decisions": "理解架构决策的背景",
    "constraint_analysis": "识别设计约束和限制",
    "tradeoff_considerations": "分析设计权衡"
  }},
  "system_constraints": {{
    "auto_generation": "是否为自动生成代码及其影响",
    "compatibility_requirements": "兼容性要求",
    "security_limitations": "安全限制"
  }},
  "improvement_opportunities": {{
    "immediate_fixes": "可以立即修复的问题",
    "architectural_improvements": "架构层面的改进机会",
    "user_experience_enhancements": "用户体验改进"
  }},
  "dialogue_focus": {{
    "critical_discussion_points": "需要重点讨论的问题",
    "avoid_invalid_criticism": "应该避免的无效批评",
    "constructive_directions": "建设性讨论方向"
  }}
}}
```

## 分析原则
1. **客观理解**: 理解代码的设计背景和约束条件
2. **识别真实问题**: 区分真正的问题和系统约束
3. **建设性导向**: 为改进对话提供有价值的基础
4. **避免偏见**: 不预设立场，基于事实分析

请开始分析并返回JSON格式的共识。
"""
    
    def _parse_llm_consensus(self, consensus_text: str) -> Optional[Dict[str, Any]]:
        """解析LLM生成的共识文本"""
        try:
            # 尝试提取JSON
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', consensus_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # 如果没有代码块，尝试直接解析
            return json.loads(consensus_text)
            
        except Exception as e:
            logger.warning(f"JSON解析失败: {e}")
            return None
    
    def _format_consensus(self, llm_consensus: Dict[str, Any], func_name: str, task_description: str) -> Dict[str, Any]:
        """格式化LLM生成的共识"""
        return {
            'generation_method': 'llm_analysis',
            'context_info': {
                'function_name': func_name,
                'task_description': task_description,
                'analysis_timestamp': datetime.now().isoformat(),
                'llm_generated': True
            },
            'llm_consensus': llm_consensus,
            'formatted_consensus': self._convert_to_dialogue_format(llm_consensus)
        }
    
    def _convert_to_dialogue_format(self, llm_consensus: Dict[str, Any]) -> str:
        """将LLM共识转换为对话格式"""
        formatted = "**🤖 LLM生成的技术共识：**\n\n"
        
        # 技术洞察
        insights = llm_consensus.get('technical_insights', {})
        if insights:
            formatted += "**🔍 技术洞察**:\n"
            for key, value in insights.items():
                formatted += f"- **{key}**: {value}\n"
            formatted += "\n"
        
        # 设计理由
        rationale = llm_consensus.get('design_rationale', {})
        if rationale:
            formatted += "**🏗️ 设计理由**:\n"
            for key, value in rationale.items():
                formatted += f"- **{key}**: {value}\n"
            formatted += "\n"
        
        # 系统约束
        constraints = llm_consensus.get('system_constraints', {})
        if constraints:
            formatted += "**⚙️ 系统约束**:\n"
            for key, value in constraints.items():
                formatted += f"- **{key}**: {value}\n"
            formatted += "\n"
        
        # 对话重点
        focus = llm_consensus.get('dialogue_focus', {})
        if focus:
            formatted += "**💡 对话指导**:\n"
            for key, value in focus.items():
                if isinstance(value, list):
                    formatted += f"- **{key}**: {', '.join(value)}\n"
                else:
                    formatted += f"- **{key}**: {value}\n"
            formatted += "\n"
        
        formatted += "**🎯 请基于以上LLM分析进行建设性讨论！**\n\n"
        
        return formatted
    
    def _fallback_consensus(self, func_code: str, func_name: str, task_description: str) -> Dict[str, Any]:
        """LLM失败时的备用共识"""
        return {
            'generation_method': 'fallback',
            'context_info': {
                'function_name': func_name,
                'task_description': task_description,
                'analysis_timestamp': datetime.now().isoformat(),
                'llm_generated': False
            },
            'formatted_consensus': f"""**⚠️ 基础技术共识（备用方案）：**

**📋 分析对象**: {func_name}
**📝 任务描述**: {task_description}

**💡 请基于代码实际情况进行建设性讨论！**

"""
        }


class MultiAgentConsensusGenerator:
    """多代理协作生成共识"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
    
    def generate_consensus_through_agents(self, func_code: str, func_name: str, task_description: str) -> Dict[str, Any]:
        """
        通过多个专门代理协作生成共识
        
        这是真正的"LLM自生成共识"：
        1. 多个专门的分析代理
        2. 通过对话协商形成共识
        3. 动态调整和完善
        """
        logger.info(f"使用多代理协作为函数 {func_name} 生成共识")
        
        try:
            import autogen
            
            # 创建专门的分析代理
            code_analyst = autogen.AssistantAgent(
                name="CodeAnalyst",
                system_message="你是代码分析专家，专注于分析代码结构、算法和实现细节。",
                llm_config=self.llm_config
            )
            
            architecture_analyst = autogen.AssistantAgent(
                name="ArchitectureAnalyst", 
                system_message="你是架构分析专家，专注于设计模式、系统约束和架构决策。",
                llm_config=self.llm_config
            )
            
            ux_analyst = autogen.AssistantAgent(
                name="UXAnalyst",
                system_message="你是用户体验专家，专注于API设计、易用性和用户需求。",
                llm_config=self.llm_config
            )
            
            consensus_coordinator = autogen.AssistantAgent(
                name="ConsensusCoordinator",
                system_message="你负责协调各方观点，形成统一的技术共识。",
                llm_config=self.llm_config
            )
            
            # 创建群聊进行共识生成
            agents_list = [code_analyst, architecture_analyst, ux_analyst, consensus_coordinator]
            
            group_chat = autogen.GroupChat(
                agents=agents_list,
                messages=[],
                max_round=8,  # 限制轮次
                speaker_selection_method="round_robin"
            )
            
            manager = autogen.GroupChatManager(
                groupchat=group_chat,
                llm_config=self.llm_config
            )
            
            # 启动共识生成对话
            consensus_prompt = f"""
# 多代理共识生成任务

请各位专家协作分析以下代码，并形成技术共识：

**函数名**: {func_name}
**任务**: {task_description}

```python
{func_code}
```

## 分析要求
1. **CodeAnalyst**: 分析代码实现、算法选择、技术细节
2. **ArchitectureAnalyst**: 分析架构设计、系统约束、设计模式
3. **UXAnalyst**: 分析用户体验、API设计、易用性
4. **ConsensusCoordinator**: 综合各方观点，形成统一共识

## 输出要求
最终由ConsensusCoordinator输出JSON格式的共识，包含：
- 技术背景理解
- 设计约束识别  
- 改进机会分析
- 对话指导原则

开始协作分析！
"""
            
            code_analyst.initiate_chat(manager, message=consensus_prompt)
            
            # 提取最终共识
            messages = group_chat.messages
            final_consensus = self._extract_consensus_from_dialogue(messages)
            
            return {
                'generation_method': 'multi_agent_collaboration',
                'context_info': {
                    'function_name': func_name,
                    'task_description': task_description,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'llm_generated': True,
                    'agent_count': len(agents_list)
                },
                'dialogue_history': [msg.get('content', '') for msg in messages],
                'consensus': final_consensus,
                'formatted_consensus': self._format_multi_agent_consensus(final_consensus)
            }
            
        except Exception as e:
            logger.error(f"多代理共识生成失败: {e}")
            return self._fallback_consensus(func_code, func_name, task_description)
    
    def _extract_consensus_from_dialogue(self, messages: List[Dict]) -> Dict[str, Any]:
        """从对话中提取共识"""
        # 查找ConsensusCoordinator的最终输出
        for msg in reversed(messages):
            if msg.get('name') == 'ConsensusCoordinator':
                content = msg.get('content', '')
                # 尝试提取JSON
                consensus = self._parse_consensus_json(content)
                if consensus:
                    return consensus
        
        # 如果没有找到，返回基础共识
        return {'status': 'partial_consensus', 'summary': 'Multi-agent analysis completed'}
    
    def _parse_consensus_json(self, content: str) -> Optional[Dict[str, Any]]:
        """解析共识JSON"""
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return None
        except:
            return None
    
    def _format_multi_agent_consensus(self, consensus: Dict[str, Any]) -> str:
        """格式化多代理共识"""
        return f"""**🤖 多代理协作生成的共识：**

{json.dumps(consensus, indent=2, ensure_ascii=False)}

**🎯 请基于多代理分析结果进行讨论！**

"""
    
    def _fallback_consensus(self, func_code: str, func_name: str, task_description: str) -> Dict[str, Any]:
        """备用共识"""
        return {
            'generation_method': 'fallback',
            'context_info': {
                'function_name': func_name,
                'task_description': task_description,
                'analysis_timestamp': datetime.now().isoformat(),
                'llm_generated': False
            },
            'formatted_consensus': f"""**⚠️ 基础共识（备用方案）：**

**函数**: {func_name}
**任务**: {task_description}

**请基于实际代码进行讨论！**

"""
        }
