import os
import asyncio
import sys
import dotenv

# 从.env文件加载环境变量
dotenv.load_dotenv()

# 将父目录添加到路径中，以便我们可以导入模块
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app.llm.factory import LLMFactory

async def test_simple_response():
    """测试简单响应生成"""
    # 使用工厂创建LLM实例
    llm = LLMFactory.create_llm("gemini")
    
    # 测试消息
    message = "什么是人工智能？"
    
    # 生成响应
    response = await llm.generate_response(message)
    
    print("\n=== 简单响应测试 ===")
    print(f"问题: {message}")
    print(f"回答: {response}")
    print("===========================\n")

async def test_streaming_response():
    """测试流式响应生成"""
    # 使用工厂创建LLM实例
    llm = LLMFactory.create_llm("gemini")
    
    # 测试消息
    message = "写一首关于人工智能的短诗。"
    
    print("\n=== 流式响应测试 ===")
    print(f"提示: {message}")
    print("响应:")
    
    # 生成流式响应
    async for chunk in llm.generate_stream(message):
        print(chunk, end="", flush=True)
    
    print("\n==============================\n")

async def test_conversation():
    """测试带历史记录的对话"""
    # 使用工厂创建LLM实例
    llm = LLMFactory.create_llm("gemini")
    
    # 对话历史
    conversation_history = [
        {"role": "user", "content": "你好，你能帮我学习Python编程吗？"},
        {"role": "assistant", "content": "当然！我很乐意帮助你学习Python编程。你想了解什么？"}
    ]
    
    # 新消息
    message = "我如何使用列表推导式？"
    
    # 系统提示
    system_prompt = "你是一个专注于Python的有帮助的编程助手。给出清晰、简洁的解释和示例。"
    
    # 生成响应
    response = await llm.generate_response(
        message, 
        conversation_history=conversation_history,
        system_prompt=system_prompt
    )
    
    print("\n=== 对话测试 ===")
    print("系统提示:", system_prompt)
    print("对话历史:")
    for msg in conversation_history:
        print(f"  {msg['role']}: {msg['content']}")
    print(f"当前消息: {message}")
    print(f"响应: {response}")
    print("=========================\n")

async def test_model_info():
    """测试获取模型信息"""
    # 使用工厂创建LLM实例
    llm = LLMFactory.create_llm("gemini")
    
    # 获取模型信息
    model_info = llm.get_model_info()
    
    print("\n=== 模型信息测试 ===")
    print("模型信息:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print("=======================\n")

async def run_all_tests():
    """运行所有测试"""
    print("开始LLM测试...")
    
    # 运行测试
    await test_simple_response()
    await test_streaming_response()
    await test_conversation()
    await test_model_info()
    
    print("所有测试完成。")

if __name__ == "__main__":
    asyncio.run(run_all_tests())