#!/usr/bin/env python3
"""
交互式轮次管理器 - 在达到上限时请求用户确认是否继续
保存当前进展，让用户决定是否增加轮次继续执行
"""

import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RoundLimitState:
    """轮次限制状态"""
    task_id: str
    current_round: int
    original_limit: int
    current_limit: int
    extensions_granted: int
    total_extensions: int
    task_description: str
    current_progress: Dict[str, Any]
    dialogue_history: List[Dict[str, Any]]
    timestamp: str
    user_decision_pending: bool = False

class InteractiveRoundManager:
    """交互式轮次管理器"""
    
    def __init__(self, save_dir: str = "memory/round_states"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.warning_threshold = 0.8  # 80%时发出警告
        self.critical_threshold = 0.95  # 95%时请求确认
        
        self.active_states: Dict[str, RoundLimitState] = {}
        self.user_confirmation_callback: Optional[Callable] = None
        
        # 默认扩展选项
        self.extension_options = {
            'small': 25,    # 小幅扩展
            'medium': 50,   # 中等扩展
            'large': 100,   # 大幅扩展
            'custom': 0     # 自定义数量
        }
    
    def set_user_confirmation_callback(self, callback: Callable):
        """设置用户确认回调函数"""
        self.user_confirmation_callback = callback
    
    def monitor_round_usage(self, task_id: str, current_round: int, 
                          max_rounds: int, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """监控轮次使用情况"""
        
        usage_ratio = current_round / max_rounds
        
        # 获取或创建状态
        if task_id not in self.active_states:
            self.active_states[task_id] = RoundLimitState(
                task_id=task_id,
                current_round=current_round,
                original_limit=max_rounds,
                current_limit=max_rounds,
                extensions_granted=0,
                total_extensions=0,
                task_description=task_context.get('task_description', ''),
                current_progress=task_context.get('progress', {}),
                dialogue_history=task_context.get('dialogue_history', []),
                timestamp=datetime.now().isoformat()
            )
        
        state = self.active_states[task_id]
        state.current_round = current_round
        state.current_progress = task_context.get('progress', {})
        
        if usage_ratio >= self.critical_threshold:
            return self._handle_critical_threshold(state)
        elif usage_ratio >= self.warning_threshold:
            return self._handle_warning_threshold(state, usage_ratio)
        else:
            return {
                'status': 'normal',
                'action': 'continue',
                'message': f'轮次使用正常 ({usage_ratio:.1%})',
                'remaining_rounds': max_rounds - current_round
            }
    
    def _handle_warning_threshold(self, state: RoundLimitState, usage_ratio: float) -> Dict[str, Any]:
        """处理警告阈值"""
        remaining = state.current_limit - state.current_round
        
        return {
            'status': 'warning',
            'action': 'continue_with_warning',
            'message': f'⚠️ 轮次使用接近上限 ({usage_ratio:.1%})，剩余 {remaining} 轮',
            'remaining_rounds': remaining,
            'suggestion': '建议准备保存当前进展或考虑任务分解'
        }
    
    def _handle_critical_threshold(self, state: RoundLimitState) -> Dict[str, Any]:
        """处理临界阈值 - 请求用户确认"""
        remaining = state.current_limit - state.current_round
        
        if remaining <= 2:
            # 即将达到上限，暂停并请求用户确认
            return self._request_user_confirmation(state)
        else:
            return {
                'status': 'critical',
                'action': 'prepare_for_limit',
                'message': f'🚨 即将达到轮次上限，剩余 {remaining} 轮',
                'remaining_rounds': remaining,
                'suggestion': '准备保存进展并请求用户确认是否继续'
            }
    
    def _request_user_confirmation(self, state: RoundLimitState) -> Dict[str, Any]:
        """请求用户确认是否继续"""
        
        # 保存当前状态
        self._save_state(state)
        
        # 标记等待用户决策
        state.user_decision_pending = True
        
        # 生成确认信息
        confirmation_info = self._generate_confirmation_info(state)
        
        return {
            'status': 'awaiting_confirmation',
            'action': 'pause_for_confirmation',
            'message': '🛑 已达到轮次上限，需要用户确认是否继续',
            'confirmation_info': confirmation_info,
            'state_saved': True,
            'task_id': state.task_id
        }
    
    def _generate_confirmation_info(self, state: RoundLimitState) -> Dict[str, Any]:
        """生成用户确认信息"""
        
        # 分析当前进展
        progress_analysis = self._analyze_progress(state)
        
        # 估算完成所需轮次
        estimated_remaining = self._estimate_remaining_rounds(state)
        
        return {
            'task_description': state.task_description,
            'current_round': state.current_round,
            'original_limit': state.original_limit,
            'extensions_used': state.extensions_granted,
            'progress_analysis': progress_analysis,
            'estimated_remaining_rounds': estimated_remaining,
            'extension_options': self.extension_options,
            'recommendation': self._get_recommendation(state, estimated_remaining)
        }
    
    def _analyze_progress(self, state: RoundLimitState) -> Dict[str, Any]:
        """分析当前进展"""
        progress = state.current_progress
        
        # 基本进展指标
        functions_created = progress.get('functions_created', 0)
        errors_encountered = progress.get('errors_count', 0)
        tasks_completed = progress.get('completed_tasks', 0)
        
        # 分析对话内容
        recent_activity = self._analyze_recent_activity(state.dialogue_history[-10:])
        
        return {
            'functions_created': functions_created,
            'errors_encountered': errors_encountered,
            'tasks_completed': tasks_completed,
            'recent_activity': recent_activity,
            'progress_percentage': min(state.current_round / state.original_limit * 100, 100),
            'efficiency_score': self._calculate_efficiency_score(state)
        }
    
    def _analyze_recent_activity(self, recent_dialogue: List[Dict[str, Any]]) -> str:
        """分析最近的活动"""
        if not recent_dialogue:
            return "无最近活动记录"
        
        # 简单的关键词分析
        recent_content = " ".join([str(msg.get('content', '')) for msg in recent_dialogue]).lower()
        
        if 'function' in recent_content and 'created' in recent_content:
            return "正在创建函数"
        elif 'error' in recent_content or 'failed' in recent_content:
            return "正在处理错误"
        elif 'test' in recent_content:
            return "正在测试功能"
        elif 'complete' in recent_content or '完成' in recent_content:
            return "接近完成"
        else:
            return "正在进行常规处理"
    
    def _calculate_efficiency_score(self, state: RoundLimitState) -> float:
        """计算效率评分"""
        if state.current_round == 0:
            return 1.0
        
        progress = state.current_progress
        functions_created = progress.get('functions_created', 0)
        tasks_completed = progress.get('completed_tasks', 0)
        
        # 简单的效率计算
        productivity = (functions_created * 10 + tasks_completed * 5) / state.current_round
        return min(productivity, 1.0)
    
    def _estimate_remaining_rounds(self, state: RoundLimitState) -> int:
        """估算完成任务还需要的轮次"""
        
        # 基于当前进展和效率估算
        efficiency = self._calculate_efficiency_score(state)
        
        if efficiency > 0.7:
            return 10  # 高效率，预计10轮完成
        elif efficiency > 0.4:
            return 20  # 中等效率，预计20轮完成
        else:
            return 30  # 低效率，预计30轮完成
    
    def _get_recommendation(self, state: RoundLimitState, estimated_remaining: int) -> str:
        """获取推荐建议"""
        
        if estimated_remaining <= 25:
            return f"建议增加 {self.extension_options['small']} 轮（小幅扩展）"
        elif estimated_remaining <= 50:
            return f"建议增加 {self.extension_options['medium']} 轮（中等扩展）"
        else:
            return f"建议增加 {self.extension_options['large']} 轮（大幅扩展）或考虑任务分解"
    
    def process_user_decision(self, task_id: str, decision: str, 
                            custom_extension: int = 0) -> Dict[str, Any]:
        """处理用户决策"""
        
        if task_id not in self.active_states:
            return {
                'success': False,
                'error': f'未找到任务 {task_id} 的状态'
            }
        
        state = self.active_states[task_id]
        
        if decision == 'continue':
            # 用户选择继续，使用推荐的扩展
            estimated = self._estimate_remaining_rounds(state)
            if estimated <= 25:
                extension = self.extension_options['small']
            elif estimated <= 50:
                extension = self.extension_options['medium']
            else:
                extension = self.extension_options['large']
        
        elif decision == 'extend_small':
            extension = self.extension_options['small']
        elif decision == 'extend_medium':
            extension = self.extension_options['medium']
        elif decision == 'extend_large':
            extension = self.extension_options['large']
        elif decision == 'extend_custom':
            extension = custom_extension
        elif decision == 'stop':
            return self._handle_user_stop(state)
        else:
            return {
                'success': False,
                'error': f'未知的决策选项: {decision}'
            }
        
        # 应用扩展
        state.current_limit += extension
        state.extensions_granted += 1
        state.total_extensions += extension
        state.user_decision_pending = False
        
        # 保存更新后的状态
        self._save_state(state)
        
        logger.info(f"用户确认继续任务 {task_id}，增加 {extension} 轮")
        
        return {
            'success': True,
            'action': 'continue_execution',
            'new_limit': state.current_limit,
            'extension_granted': extension,
            'message': f'✅ 已增加 {extension} 轮，新上限: {state.current_limit}'
        }
    
    def _handle_user_stop(self, state: RoundLimitState) -> Dict[str, Any]:
        """处理用户选择停止"""
        
        # 生成最终报告
        final_report = {
            'task_id': state.task_id,
            'task_description': state.task_description,
            'total_rounds_used': state.current_round,
            'original_limit': state.original_limit,
            'extensions_used': state.extensions_granted,
            'final_progress': state.current_progress,
            'stopped_by_user': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存最终报告
        report_path = self.save_dir / f"{state.task_id}_final_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        # 清理活动状态
        del self.active_states[state.task_id]
        
        logger.info(f"用户选择停止任务 {state.task_id}")
        
        return {
            'success': True,
            'action': 'stop_execution',
            'message': '🛑 任务已按用户要求停止',
            'final_report': final_report,
            'report_saved': str(report_path)
        }
    
    def _save_state(self, state: RoundLimitState):
        """保存状态到文件"""
        state_path = self.save_dir / f"{state.task_id}_state.json"
        
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(state), f, indent=2, ensure_ascii=False)
    
    def load_state(self, task_id: str) -> Optional[RoundLimitState]:
        """从文件加载状态"""
        state_path = self.save_dir / f"{task_id}_state.json"
        
        if not state_path.exists():
            return None
        
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = RoundLimitState(**data)
            self.active_states[task_id] = state
            return state
            
        except Exception as e:
            logger.error(f"加载状态失败 {task_id}: {e}")
            return None
    
    def get_pending_confirmations(self) -> List[Dict[str, Any]]:
        """获取等待确认的任务"""
        pending = []
        
        for task_id, state in self.active_states.items():
            if state.user_decision_pending:
                pending.append({
                    'task_id': task_id,
                    'task_description': state.task_description,
                    'current_round': state.current_round,
                    'current_limit': state.current_limit,
                    'timestamp': state.timestamp
                })
        
        return pending
    
    def cleanup_old_states(self, max_age_hours: int = 24):
        """清理旧状态"""
        current_time = datetime.now()
        to_remove = []
        
        for task_id, state in self.active_states.items():
            state_time = datetime.fromisoformat(state.timestamp)
            age_hours = (current_time - state_time).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.active_states[task_id]
            # 删除文件
            state_path = self.save_dir / f"{task_id}_state.json"
            if state_path.exists():
                state_path.unlink()
        
        if to_remove:
            logger.info(f"清理了 {len(to_remove)} 个过期状态")

# 全局实例
_global_round_manager: Optional[InteractiveRoundManager] = None

def get_round_manager() -> InteractiveRoundManager:
    """获取全局轮次管理器"""
    global _global_round_manager
    if _global_round_manager is None:
        _global_round_manager = InteractiveRoundManager()
    return _global_round_manager
