#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池材料元素筛选器
作业：输入原子序数，输出是否为锂/钠/钾
"""

def battery_element_filter(atomic_number):
    """
    电池材料元素筛选器函数
    
    参数:
        atomic_number (int): 原子序数
    
    返回:
        str: 元素名称，如果不是锂/钠/钾则返回"不是电池材料元素"
    """
    # 定义电池材料元素的原子序数
    battery_elements = {
        3: "锂 (Li)",
        11: "钠 (Na)", 
        19: "钾 (K)"
    }
    
    # 检查输入的原子序数是否为电池材料元素
    if atomic_number in battery_elements:
        return f"是电池材料元素：{battery_elements[atomic_number]}"
    else:
        return "不是电池材料元素"

def main():
    """
    主函数 - 处理用户输入和输出
    """
    print("=== 电池材料元素筛选器 ===")
    print("请输入原子序数，程序将判断是否为锂/钠/钾元素")
    print("输入 'q' 或 'quit' 退出程序\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("请输入原子序数: ").strip()
            
            # 检查是否要退出
            if user_input.lower() in ['q', 'quit', '退出']:
                print("程序结束，再见！")
                break
            
            # 转换为整数
            atomic_number = int(user_input)
            
            # 验证原子序数范围（1-118）
            if atomic_number < 1 or atomic_number > 118:
                print("错误：原子序数应该在1-118之间\n")
                continue
            
            # 调用筛选器函数
            result = battery_element_filter(atomic_number)
            print(f"结果：{result}\n")
            
        except ValueError:
            print("错误：请输入有效的数字\n")
        except KeyboardInterrupt:
            print("\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"发生错误：{e}\n")

def test_battery_element_filter():
    """
    测试函数 - 验证筛选器功能
    """
    print("=== 测试电池材料元素筛选器 ===")
    
    test_cases = [
        (3, "锂 (Li)"),
        (11, "钠 (Na)"),
        (19, "钾 (K)"),
        (1, "氢"),
        (6, "碳"),
        (8, "氧")
    ]
    
    for atomic_number, expected_element in test_cases:
        result = battery_element_filter(atomic_number)
        print(f"原子序数 {atomic_number} ({expected_element}): {result}")
    
    print()

if __name__ == "__main__":
    # 运行测试
    test_battery_element_filter()
    
    # 运行主程序
    main()