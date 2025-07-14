#!/usr/bin/env python3
"""
测试改进后的函数搜索准确性
"""

import sys
import os
sys.path.append('autogen_godel_agent')

def test_enhanced_search():
    """测试增强的搜索功能"""
    print("🔍 测试增强的函数搜索功能...")
    
    try:
        from tools.function_tools import get_function_tools
        from config import Config
        from agents.planner_agent import TaskPlannerAgent
        
        # 初始化组件
        tools = get_function_tools()
        llm_config = Config.get_llm_config()
        planner = TaskPlannerAgent(llm_config)
        
        # 测试用例
        test_queries = [
            "email validation",
            "validate email", 
            "email",
            "reverse string",
            "string reverse",
            "fibonacci",
            "calculate fibonacci",
            "password generator",
            "generate password",
            "nonexistent function"
        ]
        
        print("\n=== 底层搜索测试 ===")
        for query in test_queries:
            print(f"\n🔍 搜索: '{query}'")
            results = tools.search_functions(query)
            if results:
                print(f"✅ 找到 {len(results)} 个结果:")
                for result in results[:3]:  # 只显示前3个
                    print(f"  - {result['name']} (分数: {result.get('score', 'N/A')})")
                    print(f"    匹配类型: {result.get('match_type', [])}")
            else:
                print("❌ 未找到匹配的函数")
        
        print("\n=== TaskPlanner搜索测试 ===")
        for query in test_queries[:5]:  # 测试前5个查询
            print(f"\n🤖 TaskPlanner搜索: '{query}'")
            result = planner._search_functions(query)
            print(result[:200] + "..." if len(result) > 200 else result)
        
        print("\n=== 函数验证测试 ===")
        test_functions = [
            "validate_email_address_b8f3e2a1",
            "reverse_string",
            "fibonacci_calculator_a1b2c3d4",
            "nonexistent_function"
        ]
        
        for func_name in test_functions:
            print(f"\n🔍 验证函数: '{func_name}'")
            result = planner._verify_function_exists(func_name)
            print(result)
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_search_accuracy():
    """测试搜索准确性"""
    print("\n📊 测试搜索准确性...")
    
    try:
        from tools.function_tools import get_function_tools
        
        tools = get_function_tools()
        
        # 获取所有函数
        all_functions = tools.list_functions()
        print(f"📚 当前注册的函数总数: {len(all_functions)}")
        
        if all_functions:
            print("\n📋 已注册的函数:")
            for i, func in enumerate(all_functions[:10]):  # 显示前10个
                if isinstance(func, dict):
                    print(f"  {i+1}. {func.get('name', 'Unknown')}: {func.get('description', 'No description')}")
                else:
                    print(f"  {i+1}. {func}")
        
        # 测试精确匹配
        if all_functions:
            first_func = all_functions[0]
            func_name = first_func['name'] if isinstance(first_func, dict) else first_func
            
            print(f"\n🎯 测试精确匹配: '{func_name}'")
            results = tools.search_functions(func_name)
            if results and results[0]['name'] == func_name:
                print("✅ 精确匹配成功")
            else:
                print("❌ 精确匹配失败")
        
        return True
        
    except Exception as e:
        print(f"❌ 准确性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 函数搜索准确性改进测试")
    print("=" * 60)
    
    tests = [
        test_enhanced_search,
        test_search_accuracy
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！搜索功能改进成功")
        return 0
    else:
        print("⚠️ 部分测试失败，需要进一步改进")
        return 1

if __name__ == "__main__":
    exit(main())
