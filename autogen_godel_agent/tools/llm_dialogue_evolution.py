"""
LLM Dialogue Evolution System

Pure LLM-to-LLM dialogue system for code improvement without traditional test cases.
Multiple AI agents collaborate through conversation to critique, improve, and evolve code.

Key Concept:
- No predefined test cases
- Pure dialogue-based improvement
- Agents challenge each other's assumptions
- Continuous refinement through conversation
- Natural language reasoning about code quality
"""

import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import autogen

try:
    from .dynamic_consensus_generator import DynamicConsensusGenerator
    from .llm_consensus_generator import LLMConsensusGenerator, MultiAgentConsensusGenerator
except ImportError:
    from dynamic_consensus_generator import DynamicConsensusGenerator
    from llm_consensus_generator import LLMConsensusGenerator, MultiAgentConsensusGenerator

logger = logging.getLogger(__name__)


class LLMDialogueEvolution:
    """
    Pure LLM dialogue-based code evolution system.
    
    Agents engage in natural conversation to:
    1. Critique code from different perspectives
    2. Propose improvements through dialogue
    3. Challenge each other's suggestions
    4. Reach consensus on best practices
    5. Evolve code through iterative discussion
    """
    
    def __init__(self, llm_config: Dict[str, Any], consensus_method: str = "rule_based"):
        self.llm_config = llm_config
        self.agents = {}
        self.consensus_method = consensus_method

        # 初始化不同的共识生成器
        self.rule_based_generator = DynamicConsensusGenerator()
        self.llm_generator = LLMConsensusGenerator(llm_config)
        self.multi_agent_generator = MultiAgentConsensusGenerator(llm_config)

        self._setup_dialogue_agents()

        logger.info(f"✅ LLM Dialogue Evolution System initialized (consensus: {consensus_method})")
    
    def _setup_dialogue_agents(self):
        """Setup specialized dialogue agents for code evolution."""
        
        # 1. Code Critic - 专门挑刺的代理
        self.agents['critic'] = autogen.AssistantAgent(
            name="CodeCritic",
            system_message="""你是一个严格的代码评论家，专门通过对话来挑刺和发现问题。

你的对话风格：
- 直接指出代码中的问题和潜在bug
- 质疑代码的假设和实现方式
- 提出尖锐但建设性的问题
- 挑战其他代理的建议
- 从用户角度思考边界情况

对话重点：
- "这个函数在xxx情况下会失败"
- "你考虑过xxx场景吗？"
- "这里的逻辑有漏洞"
- "性能方面有问题"
- "安全性考虑不足"

请用自然语言对话，不要生成测试用例，而是通过讨论来发现问题。""",
            llm_config=self.llm_config
        )
        
        # 2. Code Improver - 提出改进建议的代理
        self.agents['improver'] = autogen.AssistantAgent(
            name="CodeImprover",
            system_message="""你是一个代码改进专家，专门通过对话来提出改进建议。

你的对话风格：
- 针对Critic提出的问题给出解决方案
- 提出更好的实现方式
- 建议代码重构和优化
- 解释改进的理由和好处
- 与其他代理讨论最佳实践

对话重点：
- "我建议这样改进..."
- "更好的做法是..."
- "这样可以解决xxx问题"
- "从性能角度考虑..."
- "这种实现更安全"

请通过对话来建议改进，而不是直接给出代码。""",
            llm_config=self.llm_config
        )
        
        # 3. Architecture Reviewer - 架构和设计评审
        self.agents['architect'] = autogen.AssistantAgent(
            name="ArchitectReviewer", 
            system_message="""你是一个软件架构师，专门从设计角度评审代码。

你的对话风点：
- 评估代码的整体设计
- 讨论可维护性和可扩展性
- 关注代码的清晰度和可读性
- 建议更好的设计模式
- 考虑长期维护成本

对话重点：
- "从架构角度来看..."
- "这个设计的问题是..."
- "更好的设计模式是..."
- "考虑到可维护性..."
- "这样设计更清晰"

通过对话来讨论设计理念，而不是具体的测试。""",
            llm_config=self.llm_config
        )
        
        # 4. User Advocate - 从用户角度思考
        self.agents['user_advocate'] = autogen.AssistantAgent(
            name="UserAdvocate",
            system_message="""你是用户体验倡导者，从实际使用角度评估代码。

你的对话风格：
- 从用户角度思考函数的易用性
- 考虑常见的使用场景和误用
- 关注错误处理和用户友好性
- 讨论API设计的直观性
- 提出实际使用中的问题

对话重点：
- "用户可能会这样使用..."
- "这种情况下用户会困惑"
- "错误信息不够清楚"
- "这个API不够直观"
- "实际使用中会遇到..."

通过对话来代表用户利益，而不是写测试用例。""",
            llm_config=self.llm_config
        )
        
        # 5. Synthesis Coordinator - 综合协调者
        self.agents['coordinator'] = autogen.AssistantAgent(
            name="SynthesisCoordinator",
            system_message="""你是对话协调者，负责综合各方观点并推动改进。

你的职责：
- 总结各代理的观点和建议
- 识别共识和分歧点
- 推动深入讨论
- 协调改进方案
- 决定何时结束讨论

对话风格：
- "根据大家的讨论..."
- "我们达成的共识是..."
- "还需要进一步讨论..."
- "综合考虑各方观点..."
- "最终的改进方案是..."

通过对话来协调和综合，产出最终的改进建议。""",
            llm_config=self.llm_config
        )
        
        # User proxy for interaction
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )
    
    async def evolve_through_dialogue(self, func_code: str, func_spec: Dict[str, Any], 
                                    max_rounds: int = 15) -> Dict[str, Any]:
        """
        Evolve function through pure LLM dialogue.
        
        Args:
            func_code: The function code to evolve
            func_spec: Function specification
            max_rounds: Maximum dialogue rounds
            
        Returns:
            Evolution result with dialogue history and improvements
        """
        
        logger.info(f"🗣️ Starting LLM dialogue evolution: {func_spec.get('name', 'unknown')}")
        
        # Create group chat for dialogue
        agents_list = [
            self.agents['critic'],
            self.agents['improver'], 
            self.agents['architect'],
            self.agents['user_advocate'],
            self.agents['coordinator'],
            self.user_proxy
        ]
        
        group_chat = autogen.GroupChat(
            agents=agents_list,
            messages=[],
            max_round=max_rounds,
            speaker_selection_method="round_robin"
        )
        
        manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config
        )
        
        # Generate consensus based on selected method
        consensus_data = self._generate_consensus(
            func_code, func_spec.get('name', 'unknown'), func_spec.get('description', '')
        )

        # Start the dialogue
        initial_message = f"""
# 代码改进对话 - 纯LLM讨论

## 技术共识（{self.consensus_method}）：
{consensus_data.get('formatted_consensus', '基础共识')}

## 当前函数代码：
```python
{func_code}
```

## 函数规格：
- 名称: {func_spec.get('name', 'unknown')}
- 描述: {func_spec.get('description', 'No description')}
- 目标: {func_spec.get('signature', 'No signature')}

## 对话目标：
请各位代理通过**自然对话**来改进这个函数，而不是写测试用例。

**对话方式：**
1. CodeCritic: 请先挑刺，指出代码问题（考虑技术共识）
2. CodeImprover: 针对问题提出改进建议
3. ArchitectReviewer: 从设计角度评估
4. UserAdvocate: 从用户角度思考
5. SynthesisCoordinator: 综合各方观点

**重要：请用对话讨论，不要生成测试用例！基于技术共识进行讨论！**

开始对话吧！
"""
        
        try:
            # Start the group discussion
            await self.user_proxy.a_initiate_chat(
                manager,
                message=initial_message
            )
            
            # Extract insights from dialogue
            dialogue_result = self._extract_dialogue_insights(group_chat.messages)
            
            return {
                'success': True,
                'original_code': func_code,
                'dialogue_history': group_chat.messages,
                'insights': dialogue_result,
                'improvement_suggestions': dialogue_result.get('improvements', []),
                'consensus_points': dialogue_result.get('consensus', []),
                'remaining_concerns': dialogue_result.get('concerns', [])
            }
            
        except Exception as e:
            logger.error(f"Dialogue evolution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_code': func_code
            }

    def _generate_consensus(self, func_code: str, func_name: str, task_description: str) -> Dict[str, Any]:
        """根据选择的方法生成共识"""
        try:
            if self.consensus_method == "llm_analysis":
                logger.info("使用LLM分析生成共识")
                return self.llm_generator.generate_consensus_through_llm(func_code, func_name, task_description)

            elif self.consensus_method == "multi_agent":
                logger.info("使用多代理协作生成共识")
                return self.multi_agent_generator.generate_consensus_through_agents(func_code, func_name, task_description)

            else:  # rule_based (default)
                logger.info("使用规则引擎生成共识")
                consensus = self.rule_based_generator.generate_consensus(func_code, func_name, task_description)
                return {
                    'generation_method': 'rule_based',
                    'formatted_consensus': self.rule_based_generator.format_consensus_for_dialogue(consensus)
                }

        except Exception as e:
            logger.error(f"共识生成失败: {e}")
            return {
                'generation_method': 'fallback',
                'formatted_consensus': f"""**⚠️ 基础共识：**

**函数**: {func_name}
**任务**: {task_description}

**请基于实际代码进行讨论！**
"""
            }

    def _load_agent_consensus(self) -> Dict[str, Any]:
        """Load agent consensus from configuration file."""
        try:
            consensus_path = Path(__file__).parent.parent / "config" / "agent_consensus.json"
            if consensus_path.exists():
                with open(consensus_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load agent consensus: {e}")

        # Return default consensus if file not found
        return {
            "technical_consensus": {
                "function_naming": {
                    "timestamp_suffix": {
                        "purpose": "Functions include timestamp/hash suffix for uniqueness",
                        "rationale": "Prevents naming conflicts and enables function evolution tracking"
                    }
                }
            }
        }

    def _format_consensus_for_dialogue(self, consensus: Dict[str, Any]) -> str:
        """Format consensus information for dialogue context."""
        formatted = "**重要技术背景和共识：**\n"

        tech_consensus = consensus.get("technical_consensus", {})

        # Function naming consensus
        naming = tech_consensus.get("function_naming", {})
        if "timestamp_suffix" in naming:
            suffix_info = naming["timestamp_suffix"]
            formatted += f"- **函数命名**: {suffix_info.get('purpose', '函数名包含时间戳后缀用于唯一性')}\n"

        # Type annotations consensus
        types = tech_consensus.get("type_annotations", {})
        if "any_type_usage" in types:
            any_info = types["any_type_usage"]
            formatted += f"- **返回类型Any**: {any_info.get('context', '系统要求使用Any类型以保持灵活性')}\n"

        # System constraints
        constraints = consensus.get("system_constraints", {})
        if "auto_generation" in constraints:
            auto_info = constraints["auto_generation"]
            formatted += "- **系统约束**: 这是自动生成的函数，某些设计选择受系统限制\n"

        formatted += "\n**请基于这些共识进行讨论，避免对系统设计选择的无效批评。**\n\n"

        return formatted
    
    def _extract_dialogue_insights(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract insights from agent dialogue."""
        
        insights = {
            'issues_raised': [],
            'improvements': [],
            'design_suggestions': [],
            'user_concerns': [],
            'consensus': [],
            'concerns': []
        }
        
        for message in messages:
            content = message.get('content', '')
            sender = message.get('name', '')
            
            # Categorize insights by agent type
            if sender == 'CodeCritic':
                # Extract criticism and issues
                if any(word in content.lower() for word in ['问题', '错误', '漏洞', 'bug', 'issue']):
                    insights['issues_raised'].append({
                        'agent': sender,
                        'content': content[:200] + '...' if len(content) > 200 else content
                    })
            
            elif sender == 'CodeImprover':
                # Extract improvement suggestions
                if any(word in content.lower() for word in ['建议', '改进', '优化', 'improve', 'better']):
                    insights['improvements'].append({
                        'agent': sender,
                        'content': content[:200] + '...' if len(content) > 200 else content
                    })
            
            elif sender == 'ArchitectReviewer':
                # Extract design suggestions
                if any(word in content.lower() for word in ['设计', '架构', 'design', 'architecture']):
                    insights['design_suggestions'].append({
                        'agent': sender,
                        'content': content[:200] + '...' if len(content) > 200 else content
                    })
            
            elif sender == 'UserAdvocate':
                # Extract user concerns
                if any(word in content.lower() for word in ['用户', '使用', 'user', 'usage']):
                    insights['user_concerns'].append({
                        'agent': sender,
                        'content': content[:200] + '...' if len(content) > 200 else content
                    })
            
            elif sender == 'SynthesisCoordinator':
                # Extract consensus and final thoughts
                if any(word in content.lower() for word in ['共识', '总结', 'consensus', 'summary']):
                    insights['consensus'].append({
                        'agent': sender,
                        'content': content[:200] + '...' if len(content) > 200 else content
                    })
        
        return insights


# Factory function
def get_llm_dialogue_evolution(llm_config: Dict[str, Any]) -> LLMDialogueEvolution:
    """Get LLM dialogue evolution system instance."""
    return LLMDialogueEvolution(llm_config)


# Convenience function for quick evolution
async def evolve_code_through_dialogue(func_code: str, func_spec: Dict[str, Any], 
                                     llm_config: Dict[str, Any], 
                                     max_rounds: int = 15) -> Dict[str, Any]:
    """
    Quick code evolution using pure LLM dialogue.
    
    Args:
        func_code: Function code to evolve
        func_spec: Function specification
        llm_config: LLM configuration
        max_rounds: Maximum dialogue rounds
        
    Returns:
        Evolution result with dialogue insights
    """
    evolution_system = get_llm_dialogue_evolution(llm_config)
    return await evolution_system.evolve_through_dialogue(func_code, func_spec, max_rounds)
