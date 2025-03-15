import pytest
import os
import uuid
import json
import dotenv
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

# 加载环境变量
dotenv.load_dotenv()

# 添加项目根目录到路径
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
from app.llm.base import BaseLLM
from app.llm.factory import LLMFactory
from app.models.chat import Conversation, Message
from app.core.dependencies import get_llm, get_conversation_storage, get_system_prompt

# 确定是否使用真实API
USE_REAL_API = os.getenv("USE_REAL_API", "False").lower() in ["true", "1", "yes"]
print(f"\n{'='*80}\n🧪 测试模式: {'真实API调用' if USE_REAL_API else '模拟API调用'}\n{'='*80}")

# 创建测试客户端
client = TestClient(app)

# 模拟LLM实例
class MockLLM(BaseLLM):
    def __init__(self):
        self.calls = []  # 跟踪调用历史
    
    async def generate_response(self, message, conversation_history=None, system_prompt=None, **kwargs):
        # 记录调用
        self.calls.append({
            "method": "generate_response",
            "message": message,
            "conversation_history": conversation_history,
            "system_prompt": system_prompt,
            "kwargs": kwargs
        })
        print(f"\n📝 模拟LLM收到消息: '{message}'")
        return f"这是对'{message}'的测试回复"
    
    async def generate_stream(self, message, conversation_history=None, system_prompt=None, **kwargs):
        # 记录调用
        self.calls.append({
            "method": "generate_stream",
            "message": message,
            "conversation_history": conversation_history,
            "system_prompt": system_prompt,
            "kwargs": kwargs
        })
        print(f"\n📝 模拟LLM流式收到消息: '{message}'")
        yield "这是对"
        yield "'"
        yield message
        yield "'"
        yield "的测试回复"
    
    def get_model_info(self):
        # 记录调用
        self.calls.append({
            "method": "get_model_info"
        })
        return {
            "provider": "Mock",
            "model": "mock-model",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 64,
                "max_tokens": 8192,
            }
        }

# 模拟对话存储
mock_conversations = {}

# 全局变量，用于存储模拟LLM实例
mock_llm = MockLLM()

# 如果使用真实API，使用真实的LLM工厂创建LLM
def get_real_llm():
    # 使用工厂创建真实的LLM实例
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("\n⚠️ 警告: 未找到GEMINI_API_KEY环境变量，将使用模拟LLM")
            return mock_llm
        
        # 创建真实的LLM实例
        print("\n🔑 使用真实的Gemini API密钥")
        llm_config = {
            "api_key": gemini_api_key,
            "model": "gemini-2.0-pro-exp-02-05", 
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 2048,
        }
        return LLMFactory.create_llm("gemini", llm_config)
    except Exception as e:
        print(f"\n⚠️ 创建真实LLM时出错: {str(e)}")
        return mock_llm

# 配置模拟依赖
@pytest.fixture(autouse=True)
def override_dependencies():
    # 决定使用真实还是模拟LLM
    if USE_REAL_API:
        app.dependency_overrides[get_llm] = get_real_llm
    else:
        # 重置模拟LLM的调用历史
        mock_llm.calls = []
        app.dependency_overrides[get_llm] = lambda: mock_llm
    
    # 模拟对话存储依赖
    app.dependency_overrides[get_conversation_storage] = lambda: mock_conversations
    
    # 模拟系统提示依赖
    app.dependency_overrides[get_system_prompt] = lambda: "你是一个测试助手"
    
    # 测试后清理
    yield
    mock_conversations.clear()
    app.dependency_overrides.clear()

# 测试API根路径
def test_root():
    print("\n🧪 测试: API根路径")
    response = client.get("/")
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    print("✅ 测试通过: API根路径正常运行")

# 测试健康检查路径
def test_health_check():
    print("\n🧪 测试: 健康检查")
    response = client.get("/api/health")
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["services"]["llm_api"] == "healthy"
    print("✅ 测试通过: 健康检查正常运行")

# 测试聊天API
def test_chat():
    print("\n🧪 测试: 聊天API")
    test_message = "你好，这是一条测试消息"
    
    # 发送聊天请求
    print(f"📤 发送请求: {test_message}")
    response = client.post(
        "/api/chat",
        json={"message": test_message}
    )
    
    # 检查响应
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    if USE_REAL_API:
        print(f"🤖 真实API回复: {data['data']['response']}")
        assert data["data"]["response"] != "", "API响应不应为空"
    else:
        assert data["data"]["response"] == f"这是对'{test_message}'的测试回复"
        print("📝 验证模拟LLM调用:")
        assert len(mock_llm.calls) > 0, "模拟LLM应该被调用"
        print(f"   - 调用方法: {mock_llm.calls[0]['method']}")
        print(f"   - 接收消息: {mock_llm.calls[0]['message']}")
        assert mock_llm.calls[0]['message'] == test_message
    
    assert "conversation_id" in data["data"]
    print(f"🆔 会话ID: {data['data']['conversation_id']}")
    
    # 保存会话ID以供后续测试使用
    conversation_id = data["data"]["conversation_id"]
    
    # 测试继续对话
    follow_up_message = "请继续我们的对话"
    print(f"\n📤 发送后续请求: {follow_up_message} (会话ID: {conversation_id})")
    response = client.post(
        "/api/chat",
        json={
            "message": follow_up_message,
            "conversation_id": conversation_id
        }
    )
    
    # 检查响应
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    if USE_REAL_API:
        print(f"🤖 真实API回复: {data['data']['response']}")
        assert data["data"]["response"] != "", "API响应不应为空"
    else:
        assert data["data"]["response"] == f"这是对'{follow_up_message}'的测试回复"
        assert len(mock_llm.calls) > 1, "模拟LLM应该被再次调用"
        print("📝 验证模拟LLM第二次调用:")
        print(f"   - 调用方法: {mock_llm.calls[1]['method']}")
        print(f"   - 接收消息: {mock_llm.calls[1]['message']}")
        print(f"   - 历史记录长度: {len(mock_llm.calls[1]['conversation_history'])}")
        assert mock_llm.calls[1]['message'] == follow_up_message
    
    assert data["data"]["conversation_id"] == conversation_id
    print("✅ 测试通过: 聊天API正常工作")

# 测试流式聊天API
def test_chat_stream():
    if not USE_REAL_API:
        print("\n🧪 测试: 流式聊天API (模拟版)")
        # 对于模拟版本，我们只验证API调用是否正确传递
        response = client.post(
            "/api/chat/stream",
            json={"message": "流式消息测试"}
        )
        assert response.status_code in [200, 206]  # 流式响应可能返回200或206
        print("✅ 测试通过: 流式API接口正常响应")
        
        # 检查模拟LLM是否被调用了generate_stream方法
        stream_calls = [call for call in mock_llm.calls if call["method"] == "generate_stream"]
        assert len(stream_calls) > 0, "模拟LLM的生成流方法应该被调用"
        print(f"📝 验证模拟LLM流式调用:")
        print(f"   - 调用方法: {stream_calls[0]['method']}")
        print(f"   - 接收消息: {stream_calls[0]['message']}")
    else:
        print("\n🧪 测试: 流式聊天API (真实版)")
        print("⏩ 跳过: 流式响应难以在同步测试中测试实际内容")

# 测试创建对话
def test_create_conversation():
    print("\n🧪 测试: 创建对话")
    # 创建对话
    test_title = "API测试对话"
    test_system_prompt = "你是一个API测试助手"
    
    print(f"📤 请求创建对话: 标题='{test_title}', 系统提示='{test_system_prompt}'")
    response = client.post(
        "/api/conversations",
        json={
            "title": test_title,
            "system_prompt": test_system_prompt
        }
    )
    
    # 检查响应
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["title"] == test_title
    assert "id" in data["data"]
    print(f"🆔 创建的会话ID: {data['data']['id']}")
    
    # 保存会话ID
    conversation_id = data["data"]["id"]
    
    # 验证对话已创建
    assert UUID(conversation_id) in mock_conversations
    print("✅ 测试通过: 对话创建成功并已存储")

# 测试获取对话列表
def test_get_conversations():
    print("\n🧪 测试: 获取对话列表")
    # 创建几个测试对话
    created_ids = []
    for i in range(3):
        print(f"📤 创建测试对话 #{i+1}")
        response = client.post(
            "/api/conversations",
            json={"title": f"测试对话 {i+1}"}
        )
        created_ids.append(response.json()["data"]["id"])
    
    print(f"🆔 创建了3个测试对话: {', '.join(created_ids)}")
    
    # 获取对话列表
    print("📤 请求获取所有对话")
    response = client.get("/api/conversations")
    
    # 检查响应
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) >= 3
    print(f"📋 返回了 {len(data['data'])} 个对话")
    
    # 测试分页
    print("\n📤 测试分页: limit=2, offset=0")
    response = client.get("/api/conversations?limit=2&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 2
    print(f"📋 返回了 {len(data['data'])} 个对话 (已分页)")
    print("✅ 测试通过: 对话列表获取和分页功能正常")

# 测试获取对话详情
def test_get_conversation_detail():
    print("\n🧪 测试: 获取对话详情")
    # 创建测试对话
    test_title = "详情测试对话"
    print(f"📤 创建测试对话: '{test_title}'")
    response = client.post(
        "/api/conversations",
        json={"title": test_title}
    )
    conversation_id = response.json()["data"]["id"]
    print(f"🆔 创建的会话ID: {conversation_id}")
    
    # 获取对话详情
    print(f"📤 获取对话详情: ID={conversation_id}")
    response = client.get(f"/api/conversations/{conversation_id}")
    
    # 检查响应
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["id"] == conversation_id
    assert data["data"]["title"] == test_title
    assert "messages" in data["data"]
    print(f"📝 对话包含 {len(data['data']['messages'])} 条消息")
    
    # 测试获取不存在的对话
    random_id = str(uuid.uuid4())
    print(f"\n📤 测试获取不存在的对话: ID={random_id}")
    response = client.get(f"/api/conversations/{random_id}")
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 错误响应: {response.text[:200]}...")
    assert response.status_code == 404
    print("✅ 测试通过: 对话详情获取功能正常")

# 测试删除对话
def test_delete_conversation():
    print("\n🧪 测试: 删除对话")
    # 创建测试对话
    test_title = "将被删除的对话"
    print(f"📤 创建测试对话: '{test_title}'")
    response = client.post(
        "/api/conversations",
        json={"title": test_title}
    )
    conversation_id = response.json()["data"]["id"]
    print(f"🆔 创建的会话ID: {conversation_id}")
    
    # 删除对话
    print(f"📤 删除对话: ID={conversation_id}")
    response = client.delete(f"/api/conversations/{conversation_id}")
    
    # 检查响应
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # 验证对话已删除
    assert UUID(conversation_id) not in mock_conversations
    print("✓ 验证对话已从存储中删除")
    
    # 测试删除不存在的对话
    random_id = str(uuid.uuid4())
    print(f"\n📤 测试删除不存在的对话: ID={random_id}")
    response = client.delete(f"/api/conversations/{random_id}")
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 错误响应: {response.text[:200]}...")
    assert response.status_code == 404
    print("✅ 测试通过: 对话删除功能正常")

# 测试设置/获取系统提示
def test_system_prompt():
    print("\n🧪 测试: 系统提示设置和获取")
    # 设置系统提示
    test_prompt = "这是一个测试系统提示，你应该按照这个提示行事"
    print(f"📤 设置系统提示: '{test_prompt}'")
    response = client.post(
        "/api/system-prompt",
        json={"system_prompt": test_prompt}
    )
    
    # 检查响应
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"] == test_prompt
    
    # 获取系统提示
    print(f"📤 获取系统提示")
    response = client.get("/api/system-prompt")
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["data"] == test_prompt
    print("✅ 测试通过: 系统提示设置和获取功能正常")

# 测试获取模型列表
def test_get_models():
    print("\n🧪 测试: 获取模型列表")
    response = client.get("/api/models")
    
    # 检查响应
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "models" in data["data"]
    assert "default_model" in data["data"]
    assert len(data["data"]["models"]) > 0
    print(f"📋 返回了 {len(data['data']['models'])} 个模型")
    print(f"⭐ 默认模型: {data['data']['default_model']}")
    print("✅ 测试通过: 模型列表获取功能正常")

# 测试更新模型配置
def test_update_model_config():
    print("\n🧪 测试: 更新模型配置")
    test_config = {
        "model": "gemini-2.0-pro-exp-02-05",
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 4000
    }
    print(f"📤 更新模型配置: {json.dumps(test_config, indent=2)}")
    response = client.post(
        "/api/models/config",
        json=test_config
    )
    
    # 检查响应
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["model"] == "gemini-2.0-pro-exp-02-05"
    assert "config" in data["data"]
    print("✅ 测试通过: 模型配置更新功能正常")

# 测试无效请求
def test_invalid_requests():
    print("\n🧪 测试: 无效请求处理")
    # 测试缺少必需字段
    print("📤 测试缺少必需字段 (空JSON请求)")
    response = client.post(
        "/api/chat",
        json={}  # 缺少message字段
    )
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 错误响应: {response.text[:200]}...")
    assert response.status_code == 422
    
    # 测试无效UUID
    invalid_uuid = "invalid-uuid"
    print(f"📤 测试无效UUID: '{invalid_uuid}'")
    response = client.get(f"/api/conversations/{invalid_uuid}")
    print(f"📊 状态码: {response.status_code}")
    print(f"📄 错误响应: {response.text[:200]}...")
    assert response.status_code == 422
    print("✅ 测试通过: 无效请求处理正确")

# 测试特定场景：真实API调用
@pytest.mark.skipif(not USE_REAL_API, reason="只在使用真实API时运行")
def test_real_api_call():
    print("\n🧪 测试: 真实API调用验证")
    test_message = "请用中文解释什么是大型语言模型？"
    
    print(f"📤 发送请求到真实API: '{test_message}'")
    response = client.post(
        "/api/chat", 
        json={"message": test_message}
    )
    
    print(f"📊 状态码: {response.status_code}")
    data = response.json()
    
    # 打印部分响应以验证真实内容
    api_response = data["data"]["response"]
    print(f"🤖 API回复 (前200字符): {api_response[:200]}...")
    
    # 检查响应是否包含预期内容
    assert "语言模型" in api_response.lower() or "llm" in api_response.lower(), "响应应该包含关于语言模型的信息"
    assert len(api_response) > 100, "真实API响应应该相当详细"
    print("✅ 测试通过: 成功验证了真实API调用")

if __name__ == "__main__":
    print("\n🚀 开始运行API测试...\n")
    
    # 指定要运行的测试文件
    test_args = ["-xvs", __file__]
    
    # 如果想只运行特定测试，取消下面的注释并设置测试名称
    # test_args.append("-k test_chat")
    
    pytest.main(test_args)
    
    print("\n🏁 API测试完成！\n")