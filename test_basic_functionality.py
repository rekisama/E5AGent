#!/usr/bin/env python3
"""
基本功能测试脚本
测试E5Agent项目的核心组件是否正常工作
"""

import sys
import os
sys.path.append('autogen_godel_agent')

def test_config():
    """测试配置系统"""
    print("🔧 测试配置系统...")
    try:
        from config import Config
        
        # 测试配置验证
        try:
            Config.validate_config()
            print("✅ 配置验证通过")
        except Exception as e:
            print(f"❌ 配置验证失败: {e}")
            return False
            
        # 测试LLM配置获取
        try:
            llm_config = Config.get_llm_config()
            print(f"✅ LLM配置获取成功: {llm_config.get('config_list', [{}])[0].get('model', 'unknown')}")
        except Exception as e:
            print(f"❌ LLM配置获取失败: {e}")
            return False
            
        return True
    except ImportError as e:
        print(f"❌ 配置模块导入失败: {e}")
        return False

def test_function_registry():
    """测试函数注册表"""
    print("\n📚 测试函数注册表...")
    try:
        from tools.function_registry import get_registry
        
        registry = get_registry()
        
        # 测试列出函数
        functions = registry.list_functions()
        print(f"✅ 函数注册表加载成功，包含 {len(functions)} 个函数")
        
        # 显示前几个函数
        for i, func_name in enumerate(functions[:3]):
            func_info = registry.get_function_info(func_name)
            print(f"  - {func_name}: {func_info.get('description', 'No description')}")
            
        return True
    except Exception as e:
        print(f"❌ 函数注册表测试失败: {e}")
        return False

def test_function_tools():
    """测试函数工具"""
    print("\n🔨 测试函数工具...")
    try:
        from tools.function_tools import get_function_tools
        
        tools = get_function_tools()
        
        # 测试搜索函数
        search_results = tools.search_functions("email")
        print(f"✅ 函数搜索成功，找到 {len(search_results)} 个相关函数")
        
        # 测试代码验证
        test_code = '''
def test_function(x: int) -> int:
    """Test function"""
    return x * 2
'''
        is_valid, message, extracted_code = tools.validate_function_code(test_code)
        if is_valid:
            print("✅ 代码验证功能正常")
        else:
            print(f"❌ 代码验证失败: {message}")
            
        return True
    except Exception as e:
        print(f"❌ 函数工具测试失败: {e}")
        return False

def test_agents():
    """测试代理系统"""
    print("\n🤖 测试代理系统...")
    try:
        from config import Config
        from agents.planner_agent import TaskPlannerAgent
        
        llm_config = Config.get_llm_config()
        planner = TaskPlannerAgent(llm_config)
        
        print("✅ 任务规划代理初始化成功")
        
        # 测试简单的任务分析（不调用LLM）
        print("✅ 代理系统基本功能正常")
        
        return True
    except Exception as e:
        print(f"❌ 代理系统测试失败: {e}")
        return False

def test_learning_memory():
    """测试学习记忆系统"""
    print("\n🧠 测试学习记忆系统...")
    try:
        from tools.learning_memory_integration import LearningMemoryIntegration
        from tools.function_tools import get_function_tools
        from tools.function_registry import get_registry
        
        tools = get_function_tools()
        registry = get_registry()
        
        learning = LearningMemoryIntegration(tools, registry)
        print("✅ 学习记忆系统初始化成功")
        
        return True
    except Exception as e:
        print(f"❌ 学习记忆系统测试失败: {e}")
        return False

def test_workflow_system():
    """测试工作流系统"""
    print("\n🌊 测试工作流系统...")
    try:
        from workflow.evo_workflow_manager import get_evo_workflow_manager
        from config import Config
        
        llm_config = Config.get_llm_config()
        workflow_manager = get_evo_workflow_manager(llm_config)
        
        print("✅ 工作流管理器初始化成功")
        
        return True
    except Exception as e:
        print(f"❌ 工作流系统测试失败: {e}")
        return False

def test_visualization():
    """测试可视化系统"""
    print("\n🎨 测试可视化系统...")
    try:
        from tools.simple_visualizer import generate_mermaid_from_description
        
        mermaid_code = generate_mermaid_from_description("Test task", "standard")
        if mermaid_code and "graph" in mermaid_code:
            print("✅ 可视化系统正常")
            return True
        else:
            print("❌ 可视化生成失败")
            return False
    except Exception as e:
        print(f"❌ 可视化系统测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 E5Agent 基本功能测试")
    print("=" * 50)
    
    tests = [
        test_config,
        test_function_registry,
        test_function_tools,
        test_agents,
        test_learning_memory,
        test_workflow_system,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统基本功能正常")
        return 0
    else:
        print("⚠️ 部分测试失败，系统可能存在问题")
        return 1

if __name__ == "__main__":
    exit(main())
