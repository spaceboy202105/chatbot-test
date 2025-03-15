from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
from fastapi import HTTPException, status

from models.schemas import (
    Conversation, Message, ChatRequest, 
    ChatResponse, CreateConversationRequest
)
from services.llm_service import LLMService

class ChatService:
    def __init__(self):
        """
        初始化聊天服务
        使用内存字典存储对话，实际生产环境应替换为数据库存储
        """
        # 使用内存字典存储对话
        self.conversations: Dict[str, Conversation] = {}
        self.llm_service = LLMService()
    
    async def get_all_conversations(self) -> List[Conversation]:
        """
        获取所有对话列表
        
        Returns:
            List[Conversation]: 所有对话的列表
        """
        return list(self.conversations.values())
    
    async def create_conversation(self, conversation_data: CreateConversationRequest) -> Conversation:
        """
        创建新对话
        
        Args:
            conversation_data (CreateConversationRequest): 创建对话所需的数据
            
        Returns:
            Conversation: 创建的对话对象
        """
        # 创建新的对话对象
        conversation = Conversation(
            id=uuid.uuid4(),
            title=conversation_data.title,
            system_prompt=conversation_data.system_prompt,
            model=conversation_data.model,
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 如果有系统提示词，添加系统消息
        if conversation_data.system_prompt:
            conversation.messages.append(
                Message(
                    role="system",
                    content=conversation_data.system_prompt,
                    timestamp=datetime.now()
                )
            )
        
        # 存储对话
        self.conversations[str(conversation.id)] = conversation
        return conversation
    
    async def get_conversation(self, conversation_id: uuid.UUID) -> Optional[Conversation]:
        """
        获取特定对话详情
        
        Args:
            conversation_id (uuid.UUID): 对话ID
            
        Returns:
            Optional[Conversation]: 如果找到对话则返回对话对象，否则返回None
        """
        return self.conversations.get(str(conversation_id))
    
    async def update_conversation(self, conversation_id: uuid.UUID, update_data: Dict[str, Any]) -> Conversation:
        """
        更新对话信息
        
        Args:
            conversation_id (uuid.UUID): 对话ID
            update_data (Dict[str, Any]): 要更新的字段和值
            
        Returns:
            Conversation: 更新后的对话对象
            
        Raises:
            HTTPException: 如果对话不存在
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"对话 {conversation_id} 不存在"
            )
        
        # 更新对话字段
        for key, value in update_data.items():
            if hasattr(conversation, key):
                setattr(conversation, key, value)
        
        # 更新最后修改时间
        conversation.updated_at = datetime.now()
        
        # 更新存储
        self.conversations[str(conversation_id)] = conversation
        return conversation
    
    async def delete_conversation(self, conversation_id: uuid.UUID) -> None:
        """
        删除特定对话
        
        Args:
            conversation_id (uuid.UUID): 对话ID
            
        Raises:
            HTTPException: 如果对话不存在
        """
        if str(conversation_id) not in self.conversations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"对话 {conversation_id} 不存在"
            )
        
        # 从存储中删除对话
        del self.conversations[str(conversation_id)]
    
    async def send_message(self, chat_request: ChatRequest) -> ChatResponse:
        """
        发送消息并获取模型响应
        
        Args:
            chat_request (ChatRequest): 聊天请求数据
            
        Returns:
            ChatResponse: 包含模型响应的聊天响应对象
            
        Raises:
            HTTPException: 如果指定的对话不存在或模型调用失败
        """
        conversation_id = chat_request.conversation_id
        user_message = chat_request.message
        model = chat_request.model
        system_prompt = chat_request.system_prompt
        
        # 如果没有提供对话ID，创建新对话
        if not conversation_id:
            # 确保提供了模型名称
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="创建新对话时必须提供模型名称"
                )
            
            # 创建新对话
            conversation = await self.create_conversation(
                CreateConversationRequest(
                    title=user_message[:30] + "..." if len(user_message) > 30 else user_message,
                    system_prompt=system_prompt or "",
                    model=model
                )
            )
            conversation_id = conversation.id
        else:
            # 获取现有对话
            conversation = await self.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"对话 {conversation_id} 不存在"
                )
            
            # 如果没有提供模型，使用对话中的模型
            if not model:
                model = conversation.model
        
        # 创建用户消息
        user_message_obj = Message(
            role="user",
            content=user_message,
            timestamp=datetime.now()
        )
        
        # 将用户消息添加到对话历史
        conversation.messages.append(user_message_obj)
        
        try:
            # 调用LLM服务获取响应
            # 传递完整的消息历史以保持上下文
            llm_response = await self.llm_service.generate_response(
                messages=conversation.messages,
                model=model
            )
            
            # 创建助手消息
            assistant_message = Message(
                role="assistant",
                content=llm_response,
                timestamp=datetime.now()
            )
            
            # 将助手消息添加到对话历史
            conversation.messages.append(assistant_message)
            
            # 更新对话的最后修改时间
            conversation.updated_at = datetime.now()
            
            # 更新存储
            self.conversations[str(conversation_id)] = conversation
            
            # 返回聊天响应
            return ChatResponse(
                conversation_id=conversation_id,
                message=assistant_message,
                model=model
            )
        except Exception as e:
            # 处理LLM调用过程中的错误
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"模型调用失败: {str(e)}"
            )

# 创建聊天服务实例
chat_service = ChatService()
