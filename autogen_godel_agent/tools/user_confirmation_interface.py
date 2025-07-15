#!/usr/bin/env python3
"""
用户确认界面 - 提供友好的用户交互界面来处理轮次上限确认
"""

import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

class UserConfirmationInterface:
    """用户确认界面"""
    
    def __init__(self):
        self.extension_options = {
            '1': ('小幅扩展', 25, '适合即将完成的任务'),
            '2': ('中等扩展', 50, '适合需要更多时间的复杂任务'),
            '3': ('大幅扩展', 100, '适合非常复杂的长期任务'),
            '4': ('自定义扩展', 0, '自己指定扩展轮次数'),
            '5': ('停止任务', 0, '保存当前进展并停止执行')
        }
    
    def show_round_limit_warning(self, confirmation_info: Dict[str, Any]) -> Dict[str, Any]:
        """显示轮次上限警告并获取用户选择"""
        
        print("\n" + "="*70)
        print("🚨 轮次上限警告")
        print("="*70)
        
        # 显示任务信息
        self._show_task_info(confirmation_info)
        
        # 显示进展分析
        self._show_progress_analysis(confirmation_info)
        
        # 显示选项
        self._show_options(confirmation_info)
        
        # 获取用户选择
        return self._get_user_choice(confirmation_info)
    
    def _show_task_info(self, info: Dict[str, Any]):
        """显示任务信息"""
        print(f"\n📋 任务信息:")
        print(f"  任务描述: {info['task_description']}")
        print(f"  当前轮次: {info['current_round']}")
        print(f"  原始上限: {info['original_limit']}")
        print(f"  已使用扩展: {info['extensions_used']} 次")
    
    def _show_progress_analysis(self, info: Dict[str, Any]):
        """显示进展分析"""
        progress = info['progress_analysis']
        
        print(f"\n📊 当前进展:")
        print(f"  已创建函数: {progress['functions_created']} 个")
        print(f"  遇到错误: {progress['errors_encountered']} 个")
        print(f"  完成任务: {progress['tasks_completed']} 个")
        print(f"  进展百分比: {progress['progress_percentage']:.1f}%")
        print(f"  效率评分: {progress['efficiency_score']:.2f}")
        print(f"  最近活动: {progress['recent_activity']}")
        
        estimated = info['estimated_remaining_rounds']
        print(f"\n🎯 预估还需要: {estimated} 轮完成")
    
    def _show_options(self, info: Dict[str, Any]):
        """显示选项"""
        print(f"\n💡 推荐方案: {info['recommendation']}")
        
        print(f"\n🔧 可选操作:")
        for key, (name, rounds, desc) in self.extension_options.items():
            if rounds > 0:
                print(f"  {key}. {name} (+{rounds} 轮) - {desc}")
            else:
                print(f"  {key}. {name} - {desc}")
    
    def _get_user_choice(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """获取用户选择"""
        
        while True:
            try:
                print(f"\n❓ 请选择操作 (1-5): ", end="")
                choice = input().strip()
                
                if choice not in self.extension_options:
                    print("❌ 无效选择，请输入 1-5")
                    continue
                
                option_name, rounds, desc = self.extension_options[choice]
                
                if choice == '4':  # 自定义扩展
                    return self._handle_custom_extension()
                elif choice == '5':  # 停止任务
                    return self._handle_stop_confirmation()
                else:
                    return self._handle_extension_confirmation(choice, option_name, rounds)
                    
            except KeyboardInterrupt:
                print("\n\n🛑 用户中断操作")
                return {'decision': 'stop', 'confirmed': True}
            except Exception as e:
                print(f"❌ 输入错误: {e}")
                continue
    
    def _handle_custom_extension(self) -> Dict[str, Any]:
        """处理自定义扩展"""
        while True:
            try:
                print("请输入要增加的轮次数 (1-200): ", end="")
                custom_rounds = int(input().strip())
                
                if custom_rounds < 1 or custom_rounds > 200:
                    print("❌ 轮次数必须在 1-200 之间")
                    continue
                
                print(f"\n确认增加 {custom_rounds} 轮？(y/n): ", end="")
                confirm = input().strip().lower()
                
                if confirm in ['y', 'yes', '是', '确认']:
                    return {
                        'decision': 'extend_custom',
                        'custom_extension': custom_rounds,
                        'confirmed': True
                    }
                elif confirm in ['n', 'no', '否', '取消']:
                    continue
                else:
                    print("❌ 请输入 y/n")
                    continue
                    
            except ValueError:
                print("❌ 请输入有效的数字")
                continue
            except KeyboardInterrupt:
                return {'decision': 'stop', 'confirmed': True}
    
    def _handle_stop_confirmation(self) -> Dict[str, Any]:
        """处理停止确认"""
        print(f"\n⚠️ 确认停止任务？当前进展将被保存，但任务不会完成。")
        print(f"停止后可以稍后恢复执行。")
        print(f"确认停止？(y/n): ", end="")
        
        try:
            confirm = input().strip().lower()
            
            if confirm in ['y', 'yes', '是', '确认']:
                return {'decision': 'stop', 'confirmed': True}
            else:
                print("❌ 已取消停止操作，请重新选择")
                return {'decision': 'cancel', 'confirmed': False}
                
        except KeyboardInterrupt:
            return {'decision': 'stop', 'confirmed': True}
    
    def _handle_extension_confirmation(self, choice: str, option_name: str, rounds: int) -> Dict[str, Any]:
        """处理扩展确认"""
        print(f"\n确认选择 '{option_name}' (+{rounds} 轮)？(y/n): ", end="")
        
        try:
            confirm = input().strip().lower()
            
            if confirm in ['y', 'yes', '是', '确认']:
                decision_map = {
                    '1': 'extend_small',
                    '2': 'extend_medium', 
                    '3': 'extend_large'
                }
                
                return {
                    'decision': decision_map[choice],
                    'confirmed': True
                }
            else:
                print("❌ 已取消选择，请重新选择")
                return {'decision': 'cancel', 'confirmed': False}
                
        except KeyboardInterrupt:
            return {'decision': 'stop', 'confirmed': True}
    
    def show_extension_result(self, result: Dict[str, Any]):
        """显示扩展结果"""
        if result['success']:
            print(f"\n✅ {result['message']}")
            if 'new_limit' in result:
                print(f"📊 新的轮次上限: {result['new_limit']}")
        else:
            print(f"\n❌ 操作失败: {result.get('error', '未知错误')}")
    
    def show_stop_result(self, result: Dict[str, Any]):
        """显示停止结果"""
        if result['success']:
            print(f"\n🛑 {result['message']}")
            if 'final_report' in result:
                report = result['final_report']
                print(f"📊 最终统计:")
                print(f"  使用轮次: {report['total_rounds_used']}")
                print(f"  扩展次数: {report['extensions_used']}")
                print(f"  报告已保存: {result.get('report_saved', 'N/A')}")
        else:
            print(f"\n❌ 停止失败: {result.get('error', '未知错误')}")
    
    def show_pending_confirmations(self, pending_list: list):
        """显示等待确认的任务"""
        if not pending_list:
            print("📋 当前没有等待确认的任务")
            return
        
        print(f"\n📋 等待确认的任务 ({len(pending_list)} 个):")
        print("-" * 60)
        
        for i, task in enumerate(pending_list, 1):
            print(f"{i}. 任务ID: {task['task_id']}")
            print(f"   描述: {task['task_description'][:50]}...")
            print(f"   轮次: {task['current_round']}/{task['current_limit']}")
            print(f"   时间: {task['timestamp']}")
            print()

def create_user_interface() -> UserConfirmationInterface:
    """创建用户界面实例"""
    return UserConfirmationInterface()

# 示例使用
if __name__ == "__main__":
    # 模拟确认信息
    mock_info = {
        'task_description': '创建一个数据分析工具',
        'current_round': 48,
        'original_limit': 50,
        'extensions_used': 0,
        'progress_analysis': {
            'functions_created': 2,
            'errors_encountered': 1,
            'tasks_completed': 1,
            'progress_percentage': 85.0,
            'efficiency_score': 0.75,
            'recent_activity': '正在测试功能'
        },
        'estimated_remaining_rounds': 15,
        'recommendation': '建议增加 25 轮（小幅扩展）'
    }
    
    interface = create_user_interface()
    result = interface.show_round_limit_warning(mock_info)
    print(f"\n用户选择结果: {result}")
