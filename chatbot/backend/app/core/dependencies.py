from typing import Dict, Optional, AsyncGenerator
from fastapi import Depends, HTTPException, status
from uuid import UUID

from app.core.config import get_settings, Settings
from app.llm.factory import LLMFactory
from app.llm.base import BaseLLM
from app.models.chat import Conversation

# 内存存储
_conversations: Dict[UUID, Conversation] = {}

# 内存中的系统提示
_system_prompt: Optional[str] = None


async def get_llm() -> BaseLLM:
    """
    获取LLM实例的依赖
    
    Returns:
        初始化的LLM实例
    """
    settings = get_settings()
    
    # 从配置创建LLM
    llm_config = {
        "api_key": settings.GEMINI_API_KEY,
        "model": settings.DEFAULT_MODEL,
        "temperature": settings.DEFAULT_TEMPERATURE,
        "top_p": settings.DEFAULT_TOP_P, 
        "top_k": settings.DEFAULT_TOP_K,
        "max_tokens": settings.DEFAULT_MAX_TOKENS,
    }
    
    # 创建LLM实例
    try:
        llm = LLMFactory.create_llm("gemini", llm_config)
        return llm
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"无法连接到LLM服务: {str(e)}"
        )


def get_conversation_storage() -> Dict[UUID, Conversation]:
    """
    获取对话存储的依赖
    
    Returns:
        对话字典
    """
    return _conversations


async def get_conversation(
    conversation_id: UUID,
    conversations: Dict[UUID, Conversation] = Depends(get_conversation_storage)
) -> Conversation:
    """
    通过ID获取对话
    
    Args:
        conversation_id: 对话ID
        conversations: 对话存储
        
    Returns:
        匹配的对话，如果找不到则抛出异常
    """
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"对话ID '{conversation_id}'不存在"
        )
    return conversations[conversation_id]


def get_system_prompt() -> str:
    """
    获取系统提示的依赖
    
    Returns:
        当前系统提示或默认系统提示
    """
    settings = get_settings()
    return _system_prompt or settings.DEFAULT_SYSTEM_PROMPT


def update_system_prompt(new_prompt: str) -> str:
    """
    更新系统提示
    
    Args:
        new_prompt: 新的系统提示
        
    Returns:
        更新后的系统提示
    """
    global _system_prompt
    _system_prompt = new_prompt
    return _system_prompt


async def get_settings_dependency() -> Settings:
    """
    获取设置的依赖
    
    Returns:
        应用设置实例
    """
    return get_settings()