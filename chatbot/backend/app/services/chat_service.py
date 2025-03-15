from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from uuid import UUID, uuid4
from datetime import datetime
from fastapi import HTTPException, status

from app.llm.base import BaseLLM
from app.models.chat import (
    Conversation, Message, ChatRequest, 
    ConversationCreateRequest, ConversationResponse,
    ConversationDetailResponse
)

class ChatService:
    """聊天服务类，管理对话并与LLM交互"""
    
    def __init__(self, llm: BaseLLM, conversations: Dict[UUID, Conversation]):
        """
        初始化聊天服务
        
        Args:
            llm: 大语言模型实例
            conversations: 对话存储字典
        """
        self.llm = llm
        self.conversations = conversations
    
    async def create_conversation(
        self, request: ConversationCreateRequest
    ) -> Conversation:
        """
        创建新对话
        
        Args:
            request: 创建对话请求
            
        Returns:
            新创建的对话
        """
        conversation = Conversation(
            title=request.title,
            system_prompt=request.system_prompt,
        )
        
        # 存储对话
        self.conversations[conversation.id] = conversation
        
        return conversation
    
    def get_conversation(self, conversation_id: UUID) -> Conversation:
        """
        获取对话
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            对话对象
            
        Raises:
            HTTPException: 如果对话不存在
        """
        if conversation_id not in self.conversations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"对话ID '{conversation_id}'不存在"
            )
        return self.conversations[conversation_id]
    
    def get_all_conversations(self, limit: int = 100, offset: int = 0) -> List[ConversationResponse]:
        """
        获取所有对话
        
        Args:
            limit: 返回的最大对话数
            offset: 起始偏移量
            
        Returns:
            对话响应列表
        """
        # 从所有对话中获取分页结果
        conversations = list(self.conversations.values())
        # 按更新时间排序，最新的排在前面
        conversations.sort(key=lambda c: c.updated_at, reverse=True)
        
        # 应用分页
        paginated = conversations[offset:offset + limit]
        
        # 转换为响应模型
        return [
            ConversationResponse(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=len(conv.messages)
            )
            for conv in paginated
        ]
    
    def get_conversation_detail(self, conversation_id: UUID) -> ConversationDetailResponse:
        """
        获取对话详情
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            对话详情响应
            
        Raises:
            HTTPException: 如果对话不存在
        """
        conversation = self.get_conversation(conversation_id)
        
        return ConversationDetailResponse(
            id=conversation.id,
            title=conversation.title,
            system_prompt=conversation.system_prompt,
            messages=conversation.messages,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            metadata=conversation.metadata
        )
    
    def delete_conversation(self, conversation_id: UUID) -> bool:
        """
        删除对话
        
        Args:
            conversation_id: 要删除的对话ID
            
        Returns:
            是否成功删除
            
        Raises:
            HTTPException: 如果对话不存在
        """
        # 检查对话是否存在
        self.get_conversation(conversation_id)
        
        # 删除对话
        del self.conversations[conversation_id]
        return True
    
    async def process_message(
        self, request: ChatRequest, default_system_prompt: str
    ) -> Tuple[str, Conversation]:
        """
        处理聊天消息并获取回复
        
        Args:
            request: 聊天请求
            default_system_prompt: 默认系统提示
            
        Returns:
            (LLM回复, 对话对象)
        """
        # 获取或创建对话
        conversation = None
        if request.conversation_id:
            conversation = self.get_conversation(request.conversation_id)
        else:
            # 创建新对话
            create_request = ConversationCreateRequest(system_prompt=default_system_prompt)
            conversation = await self.create_conversation(create_request)
        
        # 添加用户消息到对话
        user_message = Message(role="user", content=request.message)
        conversation.messages.append(user_message)
        
        # 准备对话历史
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in conversation.messages[:-1]  # 不包括刚刚添加的消息
        ]
        
        # 获取系统提示
        system_prompt = conversation.system_prompt or default_system_prompt
        
        # 获取生成参数
        gen_params = {}
        if request.temperature is not None:
            gen_params["temperature"] = request.temperature
        if request.top_p is not None:
            gen_params["top_p"] = request.top_p
        if request.max_tokens is not None:
            gen_params["max_tokens"] = request.max_tokens
        
        # 调用LLM
        llm_response = await self.llm.generate_response(
            message=request.message,
            conversation_history=history,
            system_prompt=system_prompt,
            **gen_params
        )
        
        # 添加助手回复到对话
        assistant_message = Message(role="assistant", content=llm_response)
        conversation.messages.append(assistant_message)
        
        # 更新对话
        conversation.updated_at = datetime.now()
        if not conversation.title and len(conversation.messages) == 2:
            # 如果是新对话且没有标题，将第一个用户消息作为标题
            # 截取前30个字符作为标题
            conversation.title = user_message.content[:30] + ("..." if len(user_message.content) > 30 else "")
        
        # 更新对话存储
        self.conversations[conversation.id] = conversation
        
        return llm_response, conversation
    
    async def stream_message(
        self, request: ChatRequest, default_system_prompt: str
    ) -> AsyncGenerator[Tuple[str, Optional[Conversation]], None]:
        """
        流式处理消息并获取回复
        
        Args:
            request: 聊天请求
            default_system_prompt: 默认系统提示
            
        Yields:
            (响应片段, 对话对象) 对话对象仅在最后一个片段中返回
        """
        # 获取或创建对话
        conversation = None
        if request.conversation_id:
            conversation = self.get_conversation(request.conversation_id)
        else:
            # 创建新对话
            create_request = ConversationCreateRequest(system_prompt=default_system_prompt)
            conversation = await self.create_conversation(create_request)
        
        # 添加用户消息到对话
        user_message = Message(role="user", content=request.message)
        conversation.messages.append(user_message)
        
        # 准备对话历史
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in conversation.messages[:-1]  # 不包括刚刚添加的消息
        ]
        
        # 获取系统提示
        system_prompt = conversation.system_prompt or default_system_prompt
        
        # 获取生成参数
        gen_params = {}
        if request.temperature is not None:
            gen_params["temperature"] = request.temperature
        if request.top_p is not None:
            gen_params["top_p"] = request.top_p
        if request.max_tokens is not None:
            gen_params["max_tokens"] = request.max_tokens
        
        # 收集完整响应以便更新对话
        full_response = ""
        
        # 流式调用LLM
        async for chunk in self.llm.generate_stream(
            message=request.message,
            conversation_history=history,
            system_prompt=system_prompt,
            **gen_params
        ):
            full_response += chunk
            yield chunk, None  # 返回片段，但暂不返回对话
        
        # 添加助手回复到对话
        assistant_message = Message(role="assistant", content=full_response)
        conversation.messages.append(assistant_message)
        
        # 更新对话
        conversation.updated_at = datetime.now()
        if not conversation.title and len(conversation.messages) == 2:
            # 如果是新对话且没有标题，将第一个用户消息作为标题
            conversation.title = user_message.content[:30] + ("..." if len(user_message.content) > 30 else "")
        
        # 更新对话存储
        self.conversations[conversation.id] = conversation
        
        # 返回最后一个空块，带有对话
        yield "", conversation