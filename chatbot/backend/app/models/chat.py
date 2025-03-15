from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4

class Message(BaseModel):
    """单条聊天消息模型"""
    role: str = Field(..., description="消息发送者角色（user或assistant）")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "你好，请介绍一下你自己",
                "timestamp": "2024-03-15T12:34:56.789Z"
            }
        }
    )

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息内容")
    conversation_id: Optional[UUID] = Field(None, description="对话ID，用于继续现有对话")
    model: Optional[str] = Field(None, description="要使用的LLM模型名称")
    temperature: Optional[float] = Field(None, description="模型温度参数 (0.0-1.0)")
    top_p: Optional[float] = Field(None, description="模型top_p参数 (0.0-1.0)")
    max_tokens: Optional[int] = Field(None, description="最大生成标记数")
    stream: Optional[bool] = Field(False, description="是否启用流式响应")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "你好，请介绍一下你自己",
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "model": "gemini-2.0-pro-exp-02-05",
                "temperature": 0.7,
                "stream": False
            }
        }
    )

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="助手回复内容")
    conversation_id: UUID = Field(..., description="对话ID")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "你好！我是一个AI助手，可以帮助回答问题、提供信息，或者与你进行有趣的对话。我被设计为友好、有礼貌且乐于助人。有什么我能帮到你的吗？",
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    )

class StreamResponse(BaseModel):
    """流式响应模型（用于SSE）"""
    content: str = Field(..., description="响应内容片段")
    done: bool = Field(False, description="是否是最后一个片段")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "你好！",
                "done": False
            }
        }
    )

class SystemPromptRequest(BaseModel):
    """系统提示请求模型"""
    system_prompt: str = Field(..., description="系统提示内容")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "system_prompt": "你是一个专业的技术顾问，擅长回答编程和技术问题。"
            }
        }
    )

class Conversation(BaseModel):
    """对话模型"""
    id: UUID = Field(default_factory=uuid4, description="对话唯一ID")
    title: Optional[str] = Field(None, description="对话标题")
    system_prompt: Optional[str] = Field(None, description="对话的系统提示")
    messages: List[Message] = Field(default_factory=list, description="对话消息历史")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最后更新时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="对话元数据")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "技术咨询对话",
                "system_prompt": "你是一名专业的技术顾问。",
                "messages": [
                    {
                        "role": "user",
                        "content": "你好，我需要帮助解决一个Python问题。",
                        "timestamp": "2024-03-15T12:30:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "你好！我很乐意帮助你解决Python问题。请告诉我具体是什么问题？",
                        "timestamp": "2024-03-15T12:30:05Z"
                    }
                ],
                "created_at": "2024-03-15T12:30:00Z",
                "updated_at": "2024-03-15T12:30:05Z",
                "metadata": {
                    "user_locale": "zh-CN",
                    "client_version": "1.0.0"
                }
            }
        }
    )

class ConversationCreateRequest(BaseModel):
    """创建对话请求模型"""
    title: Optional[str] = Field(None, description="对话标题")
    system_prompt: Optional[str] = Field(None, description="对话的系统提示")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "技术咨询对话",
                "system_prompt": "你是一名专业的技术顾问。"
            }
        }
    )

class ConversationResponse(BaseModel):
    """对话信息响应模型"""
    id: UUID = Field(..., description="对话唯一ID")
    title: Optional[str] = Field(None, description="对话标题")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="最后更新时间")
    message_count: int = Field(..., description="消息数量")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "技术咨询对话",
                "created_at": "2024-03-15T12:30:00Z",
                "updated_at": "2024-03-15T12:35:10Z",
                "message_count": 5
            }
        }
    )

class ConversationDetailResponse(BaseModel):
    """详细对话响应模型"""
    id: UUID = Field(..., description="对话唯一ID")
    title: Optional[str] = Field(None, description="对话标题")
    system_prompt: Optional[str] = Field(None, description="对话的系统提示")
    messages: List[Message] = Field(..., description="对话消息历史")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="最后更新时间")
    metadata: Dict[str, Any] = Field(..., description="对话元数据")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "技术咨询对话",
                "system_prompt": "你是一名专业的技术顾问。",
                "messages": [
                    {
                        "role": "user",
                        "content": "你好，我需要帮助解决一个Python问题。",
                        "timestamp": "2024-03-15T12:30:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "你好！我很乐意帮助你解决Python问题。请告诉我具体是什么问题？",
                        "timestamp": "2024-03-15T12:30:05Z"
                    }
                ],
                "created_at": "2024-03-15T12:30:00Z",
                "updated_at": "2024-03-15T12:30:05Z",
                "metadata": {
                    "user_locale": "zh-CN",
                    "client_version": "1.0.0"
                }
            }
        }
    )