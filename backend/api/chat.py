from fastapi import APIRouter, HTTPException, Depends, Path, Query, status
from typing import List, Optional
import uuid
from uuid import UUID

from models.schemas import (
    Conversation, ChatRequest, ChatResponse, 
    CreateConversationRequest, Message
)
from services.chat_service import ChatService

# 创建路由器对象
router = APIRouter(prefix="/api/chat", tags=["chat"])

# 创建聊天服务实例
chat_service = ChatService()

@router.get("/conversations", response_model=List[Conversation])
async def get_conversations():
    """
    获取所有对话列表
    
    Returns:
        List[Conversation]: 所有对话的列表
    """
    try:
        return await chat_service.get_all_conversations()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取对话列表失败: {str(e)}"
        )

@router.post("/conversations", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_conversation(request: CreateConversationRequest):
    """
    创建新对话
    
    Args:
        request (CreateConversationRequest): 创建对话的请求数据
        
    Returns:
        Conversation: 新创建的对话
    """
    try:
        return await chat_service.create_conversation(
            title=request.title,
            system_prompt=request.system_prompt,
            model=request.model
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建对话失败: {str(e)}"
        )

@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: UUID = Path(..., description="对话ID")
):
    """
    获取特定对话详情
    
    Args:
        conversation_id (UUID): 对话的唯一标识符
        
    Returns:
        Conversation: 包含完整消息历史的对话
        
    Raises:
        HTTPException: 如果对话不存在
    """
    try:
        conversation = await chat_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"对话 {conversation_id} 不存在"
            )
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取对话失败: {str(e)}"
        )

@router.post("/conversations/{conversation_id}/messages", response_model=ChatResponse)
async def send_message(
    chat_request: ChatRequest,
    conversation_id: UUID = Path(..., description="对话ID")
):
    """
    发送消息并获取模型响应
    
    Args:
        chat_request (ChatRequest): 聊天请求数据
        conversation_id (UUID): 对话的唯一标识符
        
    Returns:
        ChatResponse: 包含模型响应的聊天响应
        
    Raises:
        HTTPException: 如果对话不存在或消息发送失败
    """
    try:
        # 确保请求中的conversation_id与路径参数一致
        if chat_request.conversation_id and chat_request.conversation_id != conversation_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="请求体中的conversation_id与URL中的不匹配"
            )
        
        # 设置请求中的conversation_id
        chat_request.conversation_id = conversation_id
        
        # 调用服务发送消息
        response = await chat_service.send_message(chat_request)
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"发送消息失败: {str(e)}"
        )

@router.put("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation(
    update_data: dict,
    conversation_id: UUID = Path(..., description="对话ID")
):
    """
    更新对话信息（标题、系统提示词或模型）
    
    Args:
        update_data (dict): 要更新的对话数据
        conversation_id (UUID): 对话的唯一标识符
        
    Returns:
        Conversation: 更新后的对话
        
    Raises:
        HTTPException: 如果对话不存在或更新失败
    """
    try:
        # 检查对话是否存在
        conversation = await chat_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"对话 {conversation_id} 不存在"
            )
        
        # 更新对话
        updated_conversation = await chat_service.update_conversation(conversation_id, update_data)
        return updated_conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新对话失败: {str(e)}"
        )

@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID = Path(..., description="对话ID")
):
    """
    删除特定对话
    
    Args:
        conversation_id (UUID): 对话的唯一标识符
        
    Returns:
        None
        
    Raises:
        HTTPException: 如果对话不存在或删除失败
    """
    try:
        # 检查对话是否存在
        conversation = await chat_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"对话 {conversation_id} 不存在"
            )
        
        # 删除对话
        await chat_service.delete_conversation(conversation_id)
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除对话失败: {str(e)}"
        )
