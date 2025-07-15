"""
AutoGen Test Evolution System

This module implements a multi-agent testing system where different agents
collaborate to test, critique, and evolve code through intelligent dialogue.

Key Features:
- TestCritic Agent: Finds flaws and edge cases
- TestGenerator Agent: Creates comprehensive test cases
- CodeReviewer Agent: Reviews code quality and suggests improvements
- TestExecutor Agent: Runs tests and analyzes results
- EvolutionCoordinator: Orchestrates the evolution process
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import autogen

logger = logging.getLogger(__name__)


class AutoGenTestEvolution:
    """
    Multi-agent test evolution system using AutoGen.
    
    Agents collaborate to:
    1. Generate comprehensive test cases
    2. Find edge cases and potential bugs
    3. Critique code quality
    4. Suggest improvements
    5. Evolve code through iterations
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.agents = {}
        self.group_chat = None
        self.manager = None
        self._setup_agents()
        
        logger.info("✅ AutoGen Test Evolution System initialized")
    
    def _setup_agents(self):
        """Setup all the specialized agents."""
        
        # 1. Test Critic Agent - 专门挑刺找问题
        self.agents['test_critic'] = autogen.AssistantAgent(
            name="TestCritic",
            system_message="""你是一个严格的测试评论家，专门找代码的问题和漏洞。

你的职责：
1. 仔细分析代码，找出潜在的bug和边界情况
2. 识别代码中的逻辑错误、性能问题、安全漏洞
3. 提出尖锐但建设性的批评
4. 建议需要测试的边界情况和异常情况
5. 质疑代码的假设和实现方式

你的风格：
- 严格但公正
- 关注细节
- 提出具体的反例
- 不放过任何可疑的地方

回复格式：
```json
{
  "issues_found": [
    {
      "type": "logic_error|performance|security|edge_case",
      "severity": "high|medium|low", 
      "description": "具体问题描述",
      "example": "能触发问题的具体例子",
      "suggestion": "修复建议"
    }
  ],
  "edge_cases": ["边界情况1", "边界情况2"],
  "overall_assessment": "整体评价"
}
```""",
            llm_config=self.llm_config
        )
        
        # 2. Test Generator Agent - 生成全面的测试用例
        self.agents['test_generator'] = autogen.AssistantAgent(
            name="TestGenerator", 
            system_message="""你是一个测试用例生成专家，能够创建全面、智能的测试用例。

你的职责：
1. 根据函数功能生成正常情况测试
2. 根据TestCritic的建议生成边界测试
3. 创建异常情况和错误处理测试
4. 确保测试覆盖率和质量
5. 生成性能测试用例

测试类型：
- 正常功能测试
- 边界值测试  
- 异常输入测试
- 性能压力测试
- 安全性测试

回复格式：
```json
{
  "test_cases": [
    {
      "name": "测试名称",
      "type": "normal|boundary|exception|performance|security",
      "input": {"参数名": "参数值"},
      "expected_output": "期望结果",
      "description": "测试目的",
      "reasoning": "为什么需要这个测试"
    }
  ],
  "coverage_analysis": "测试覆盖率分析",
  "additional_suggestions": "额外建议"
}
```""",
            llm_config=self.llm_config
        )
        
        # 3. Code Reviewer Agent - 代码质量审查
        self.agents['code_reviewer'] = autogen.AssistantAgent(
            name="CodeReviewer",
            system_message="""你是一个资深的代码审查专家，专注于代码质量和最佳实践。

你的职责：
1. 评估代码的可读性、可维护性
2. 检查是否遵循编程最佳实践
3. 建议代码结构和设计改进
4. 评估错误处理和异常管理
5. 提出重构建议

评审重点：
- 代码清晰度和可读性
- 错误处理完整性
- 性能优化机会
- 安全性考虑
- 代码复用性

回复格式：
```json
{
  "code_quality_score": "1-10分",
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["缺点1", "缺点2"], 
  "improvements": [
    {
      "area": "改进领域",
      "suggestion": "具体建议",
      "priority": "high|medium|low",
      "example": "改进示例代码"
    }
  ],
  "best_practices": "最佳实践建议"
}
```""",
            llm_config=self.llm_config
        )
        
        # 4. Test Executor Agent - 执行测试并分析结果
        self.agents['test_executor'] = autogen.AssistantAgent(
            name="TestExecutor",
            system_message="""你是测试执行和结果分析专家。

你的职责：
1. 执行测试用例并收集结果
2. 分析测试失败的原因
3. 识别测试结果中的模式
4. 提供测试报告和建议
5. 建议下一步的测试策略

分析重点：
- 测试通过率
- 失败原因分析
- 性能指标
- 覆盖率统计
- 改进建议

回复格式：
```json
{
  "execution_summary": {
    "total_tests": "总测试数",
    "passed": "通过数",
    "failed": "失败数",
    "pass_rate": "通过率"
  },
  "failure_analysis": [
    {
      "test_name": "失败测试名",
      "error": "错误信息", 
      "root_cause": "根本原因",
      "fix_suggestion": "修复建议"
    }
  ],
  "performance_metrics": "性能指标",
  "next_steps": "下一步建议"
}
```""",
            llm_config=self.llm_config
        )
        
        # 5. Evolution Coordinator - 协调进化过程
        self.agents['coordinator'] = autogen.AssistantAgent(
            name="EvolutionCoordinator",
            system_message="""你是代码进化过程的协调者，负责整合所有反馈并指导改进。

你的职责：
1. 整合所有代理的反馈
2. 制定代码改进计划
3. 协调多轮进化迭代
4. 决定何时停止进化
5. 生成最终的改进报告

决策原则：
- 优先修复高严重性问题
- 平衡功能性和性能
- 确保向后兼容性
- 控制复杂度增长

回复格式：
```json
{
  "evolution_plan": [
    {
      "iteration": "迭代次数",
      "focus": "改进重点",
      "changes": ["具体改动"],
      "expected_outcome": "预期结果"
    }
  ],
  "priority_issues": ["优先问题列表"],
  "stop_criteria": "停止条件",
  "final_recommendation": "最终建议"
}
```""",
            llm_config=self.llm_config
        )
        
        # Setup user proxy for interaction
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )
    
    async def evolve_function(self, func_code: str, func_spec: Dict[str, Any], 
                            max_iterations: int = 3) -> Dict[str, Any]:
        """
        Evolve a function through multi-agent collaboration.
        
        Args:
            func_code: The function code to evolve
            func_spec: Function specification
            max_iterations: Maximum evolution iterations
            
        Returns:
            Evolution result with improved code and analysis
        """
        
        logger.info(f"🧬 Starting function evolution: {func_spec.get('name', 'unknown')}")
        
        evolution_history = []
        current_code = func_code
        
        for iteration in range(max_iterations):
            logger.info(f"🔄 Evolution iteration {iteration + 1}/{max_iterations}")
            
            # Create group chat for this iteration
            agents_list = [
                self.agents['test_critic'],
                self.agents['test_generator'], 
                self.agents['code_reviewer'],
                self.agents['test_executor'],
                self.agents['coordinator'],
                self.user_proxy
            ]
            
            group_chat = autogen.GroupChat(
                agents=agents_list,
                messages=[],
                max_round=50,  # Increased from 20 to 50
                speaker_selection_method="round_robin"
            )
            
            manager = autogen.GroupChatManager(
                groupchat=group_chat,
                llm_config=self.llm_config
            )
            
            # Start the evolution discussion
            initial_message = f"""
# 代码进化任务 - 第 {iteration + 1} 轮

## 当前函数代码：
```python
{current_code}
```

## 函数规格：
- 名称: {func_spec.get('name', 'unknown')}
- 描述: {func_spec.get('description', 'No description')}
- 签名: {func_spec.get('signature', 'No signature')}

## 任务目标：
1. TestCritic: 找出代码问题和边界情况
2. TestGenerator: 生成全面的测试用例
3. CodeReviewer: 评估代码质量并建议改进
4. TestExecutor: 分析测试结果
5. EvolutionCoordinator: 制定改进计划

请开始分析和讨论！
"""
            
            try:
                # Start the group discussion
                await self.user_proxy.a_initiate_chat(
                    manager,
                    message=initial_message
                )
                
                # Extract insights from the conversation
                iteration_result = self._extract_evolution_insights(group_chat.messages)
                evolution_history.append(iteration_result)
                
                # Check if we should continue evolving
                if iteration_result.get('should_stop', False):
                    logger.info("🏁 Evolution completed based on agent recommendation")
                    break
                    
                # Apply improvements for next iteration
                if iteration_result.get('improved_code'):
                    current_code = iteration_result['improved_code']
                
            except Exception as e:
                logger.error(f"Evolution iteration {iteration + 1} failed: {e}")
                break
        
        return {
            'success': True,
            'original_code': func_code,
            'evolved_code': current_code,
            'evolution_history': evolution_history,
            'iterations_completed': len(evolution_history),
            'final_analysis': evolution_history[-1] if evolution_history else {}
        }
    
    def _extract_evolution_insights(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract insights from agent conversation."""
        
        insights = {
            'issues_found': [],
            'test_cases': [],
            'code_improvements': [],
            'execution_results': {},
            'evolution_plan': {},
            'should_stop': False,
            'improved_code': None
        }
        
        for message in messages:
            content = message.get('content', '')
            sender = message.get('name', '')
            
            # Extract JSON responses from agents
            if '```json' in content:
                try:
                    json_start = content.find('```json') + 7
                    json_end = content.find('```', json_start)
                    json_content = content[json_start:json_end].strip()
                    data = json.loads(json_content)
                    
                    if sender == 'TestCritic':
                        insights['issues_found'].extend(data.get('issues_found', []))
                    elif sender == 'TestGenerator':
                        insights['test_cases'].extend(data.get('test_cases', []))
                    elif sender == 'CodeReviewer':
                        insights['code_improvements'].append(data)
                    elif sender == 'TestExecutor':
                        insights['execution_results'] = data
                    elif sender == 'EvolutionCoordinator':
                        insights['evolution_plan'] = data
                        insights['should_stop'] = data.get('stop_evolution', False)
                        
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse JSON from {sender}: {e}")
            
            # Extract improved code if present
            if '```python' in content and sender == 'EvolutionCoordinator':
                try:
                    code_start = content.find('```python') + 9
                    code_end = content.find('```', code_start)
                    improved_code = content[code_start:code_end].strip()
                    if improved_code:
                        insights['improved_code'] = improved_code
                except Exception as e:
                    logger.warning(f"Failed to extract improved code: {e}")
        
        return insights


# Factory function
def get_autogen_test_evolution(llm_config: Dict[str, Any]) -> AutoGenTestEvolution:
    """Get AutoGen test evolution system instance."""
    return AutoGenTestEvolution(llm_config)


# Convenience function for quick evolution
async def evolve_function_with_autogen(func_code: str, func_spec: Dict[str, Any], 
                                     llm_config: Dict[str, Any], 
                                     max_iterations: int = 3) -> Dict[str, Any]:
    """
    Quick function evolution using AutoGen agents.
    
    Args:
        func_code: Function code to evolve
        func_spec: Function specification
        llm_config: LLM configuration
        max_iterations: Maximum evolution iterations
        
    Returns:
        Evolution result
    """
    evolution_system = get_autogen_test_evolution(llm_config)
    return await evolution_system.evolve_function(func_code, func_spec, max_iterations)
