
from pydantic import BaseModel, Field, UUID4
from typing import List, Optional, Literal
from datetime import datetime
import uuid


class Message(BaseModel):
    """
    表示单条消息的模型
    """
    role: Literal["user", "assistant", "system"] = Field(
        ..., 
        description="消息角色：user(用户)、assistant(助手)或system(系统)"
    )
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息创建时间戳")


class Conversation(BaseModel):
    """
    表示一个完整对话的模型
    """
    id: UUID4 = Field(default_factory=uuid.uuid4, description="对话唯一标识符")
    title: str = Field(..., description="对话标题")
    messages: List[Message] = Field(default_factory=list, description="消息列表")
    system_prompt: str = Field(default="", description="系统提示词")
    model: str = Field(..., description="使用的模型名称")
    created_at: datetime = Field(default_factory=datetime.now, description="对话创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="对话最后更新时间")

    class Config:
        json_encoders = {
            uuid.UUID: str,
            datetime: lambda v: v.isoformat()
        }


class ChatRequest(BaseModel):
    """
    聊天请求模型
    """
    conversation_id: Optional[UUID4] = Field(None, description="对话ID，如果为空则创建新对话")
    message: str = Field(..., description="用户消息内容")
    model: Optional[str] = Field(None, description="模型名称，如果不提供则使用对话关联的模型")
    system_prompt: Optional[str] = Field(None, description="系统提示词，如果不提供则使用对话关联的系统提示词")


class ChatResponse(BaseModel):
    """
    聊天响应模型
    """
    conversation_id: UUID4 = Field(..., description="对话ID")
    message: Message = Field(..., description="助手消息")
    model: str = Field(..., description="使用的模型名称")


class ModelInfo(BaseModel):
    """
    模型信息模型
    """
    id: str = Field(..., description="模型唯一标识符")
    name: str = Field(..., description="模型名称")
    provider: str = Field(..., description="提供者（如OpenAI、Google等）")
    description: str = Field(default="", description="模型描述")


class CreateConversationRequest(BaseModel):
    """
    创建对话请求模型
    """
    title: str = Field(..., description="对话标题")
    system_prompt: str = Field(default="", description="系统提示词")
    model: str = Field(..., description="使用的模型名称")
