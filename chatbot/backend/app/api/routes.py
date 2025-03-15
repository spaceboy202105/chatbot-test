from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from typing import List, Dict, Optional, Any
from uuid import UUID
import json
import asyncio

from app.core.config import get_settings, Settings
from app.core.dependencies import (
    get_llm, get_conversation_storage, get_conversation, 
    get_system_prompt, update_system_prompt, get_settings_dependency
)
from app.llm.base import BaseLLM
from app.models.chat import (
    ChatRequest, ChatResponse, StreamResponse,
    ConversationCreateRequest, ConversationResponse,
    ConversationDetailResponse, SystemPromptRequest,
    Conversation
)
from app.models.response import (
    APIResponse, ErrorResponse, HealthResponse,
    ModelInfo, ModelsResponse, ModelConfigRequest,
    ModelConfigResponse, StatusEnum
)
from app.services.chat_service import ChatService

# 创建路由器
router = APIRouter()

# 聊天相关路由
@router.post("/chat", response_model=APIResponse[ChatResponse])
async def chat(
    request: ChatRequest,
    llm: BaseLLM = Depends(get_llm),
    conversations: Dict[UUID, Conversation] = Depends(get_conversation_storage),
    system_prompt: str = Depends(get_system_prompt)
):
    """
    发送消息到聊天机器人并获取回复
    """
    try:
        # 创建聊天服务
        chat_service = ChatService(llm, conversations)
        
        # 处理消息
        response, conversation = await chat_service.process_message(request, system_prompt)
        
        # 返回响应
        return APIResponse[ChatResponse](
            status=StatusEnum.SUCCESS,
            data=ChatResponse(
                response=response,
                conversation_id=conversation.id
            ),
            message="消息处理成功"
        )
    except HTTPException as e:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理消息时出错: {str(e)}"
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    llm: BaseLLM = Depends(get_llm),
    conversations: Dict[UUID, Conversation] = Depends(get_conversation_storage),
    system_prompt: str = Depends(get_system_prompt)
):
    """
    流式发送消息到聊天机器人并获取回复
    """
    try:
        # 创建聊天服务
        chat_service = ChatService(llm, conversations)
        
        # 定义事件生成器
        async def event_generator():
            conversation_id = None
            
            # 获取流式响应
            async for chunk, conversation in chat_service.stream_message(request, system_prompt):
                if conversation:  # 最后一个响应
                    # 发送最终事件，包含对话ID
                    conversation_id = str(conversation.id)
                    yield {
                        "event": "done",
                        "data": json.dumps({"conversation_id": conversation_id})
                    }
                else:
                    # 发送内容片段
                    yield {
                        "event": "message",
                        "data": chunk
                    }
        
        # 返回SSE响应
        return EventSourceResponse(event_generator())
    
    except HTTPException as e:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理流式消息时出错: {str(e)}"
        )


# 对话管理路由
@router.post("/conversations", response_model=APIResponse[ConversationResponse])
async def create_conversation(
    request: ConversationCreateRequest,
    llm: BaseLLM = Depends(get_llm),
    conversations: Dict[UUID, Conversation] = Depends(get_conversation_storage),
    system_prompt: str = Depends(get_system_prompt)
):
    """
    创建新对话
    """
    try:
        # 如果未提供系统提示，使用默认系统提示
        if not request.system_prompt:
            request.system_prompt = system_prompt
        
        # 创建聊天服务
        chat_service = ChatService(llm, conversations)
        
        # 创建对话
        conversation = await chat_service.create_conversation(request)
        
        # 返回响应
        return APIResponse[ConversationResponse](
            status=StatusEnum.SUCCESS,
            data=ConversationResponse(
                id=conversation.id,
                title=conversation.title,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                message_count=len(conversation.messages)
            ),
            message="对话创建成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建对话时出错: {str(e)}"
        )


@router.get("/conversations", response_model=APIResponse[List[ConversationResponse]])
async def get_conversations(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    llm: BaseLLM = Depends(get_llm),
    conversations: Dict[UUID, Conversation] = Depends(get_conversation_storage)
):
    """
    获取所有对话的列表
    """
    try:
        # 创建聊天服务
        chat_service = ChatService(llm, conversations)
        
        # 获取对话列表
        conversation_list = chat_service.get_all_conversations(limit, offset)
        
        # 返回响应
        return APIResponse[List[ConversationResponse]](
            status=StatusEnum.SUCCESS,
            data=conversation_list,
            message=f"获取了{len(conversation_list)}个对话"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取对话列表时出错: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", response_model=APIResponse[ConversationDetailResponse])
async def get_conversation_detail(
    conversation_id: UUID = Path(..., description="对话ID"),
    llm: BaseLLM = Depends(get_llm),
    conversations: Dict[UUID, Conversation] = Depends(get_conversation_storage)
):
    """
    获取特定对话的详细信息
    """
    try:
        # 创建聊天服务
        chat_service = ChatService(llm, conversations)
        
        # 获取对话详情
        conversation_detail = chat_service.get_conversation_detail(conversation_id)
        
        # 返回响应
        return APIResponse[ConversationDetailResponse](
            status=StatusEnum.SUCCESS,
            data=conversation_detail,
            message="获取对话详情成功"
        )
    except HTTPException as e:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取对话详情时出错: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}", response_model=APIResponse)
async def delete_conversation(
    conversation_id: UUID = Path(..., description="要删除的对话ID"),
    llm: BaseLLM = Depends(get_llm),
    conversations: Dict[UUID, Conversation] = Depends(get_conversation_storage)
):
    """
    删除特定对话
    """
    try:
        # 创建聊天服务
        chat_service = ChatService(llm, conversations)
        
        # 删除对话
        chat_service.delete_conversation(conversation_id)
        
        # 返回响应
        return APIResponse(
            status=StatusEnum.SUCCESS,
            message=f"成功删除对话ID: {conversation_id}"
        )
    except HTTPException as e:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除对话时出错: {str(e)}"
        )


# 系统提示管理路由
@router.post("/system-prompt", response_model=APIResponse[str])
async def set_system_prompt(request: SystemPromptRequest):
    """
    设置系统提示
    """
    try:
        # 更新系统提示
        new_prompt = update_system_prompt(request.system_prompt)
        
        # 返回响应
        return APIResponse[str](
            status=StatusEnum.SUCCESS,
            data=new_prompt,
            message="系统提示已更新"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新系统提示时出错: {str(e)}"
        )


@router.get("/system-prompt", response_model=APIResponse[str])
async def get_current_system_prompt(
    system_prompt: str = Depends(get_system_prompt)
):
    """
    获取当前系统提示
    """
    return APIResponse[str](
        status=StatusEnum.SUCCESS,
        data=system_prompt,
        message="获取系统提示成功"
    )


# 模型配置路由
@router.get("/models", response_model=APIResponse[ModelsResponse])
async def get_models(
    settings: Settings = Depends(get_settings_dependency)
):
    """
    获取支持的模型列表
    """
    try:
        # 获取支持的模型
        models_dict = settings.get_supported_models()
        
        # 转换为ModelsResponse格式
        models = [
            ModelInfo(
                id=model_id,
                name=info["name"],
                provider=info["provider"],
                description=info.get("description", "")
            )
            for model_id, info in models_dict.items()
        ]
        
        # 返回响应
        return APIResponse[ModelsResponse](
            status=StatusEnum.SUCCESS,
            data=ModelsResponse(
                models=models,
                default_model=settings.DEFAULT_MODEL
            ),
            message="获取模型列表成功"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型列表时出错: {str(e)}"
        )


@router.post("/models/config", response_model=APIResponse[ModelConfigResponse])
async def update_model_config(
    request: ModelConfigRequest,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    更新模型配置（目前只是返回当前配置，需要扩展以实际更新配置）
    
    注意: 该实现暂时不会实际更改配置，因为设置是不可变的。
    真实实现需要使用配置存储或环境变量管理。
    """
    try:
        # 获取当前模型配置
        model_config = settings.get_llm_config()
        
        # 返回响应 (注意: 在实际实现中应该更新配置)
        return APIResponse[ModelConfigResponse](
            status=StatusEnum.SUCCESS,
            data=ModelConfigResponse(
                model=request.model,
                config=model_config
            ),
            message="此为当前模型配置（实际配置更新功能尚未实现）"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理模型配置时出错: {str(e)}"
        )


# 健康检查路由
@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings_dependency),
    llm: BaseLLM = Depends(get_llm)
):
    """
    API健康状态检查
    """
    from datetime import datetime
    
    # 检查LLM状态
    llm_status = "healthy"
    try:
        _ = llm.get_model_info()
    except Exception:
        llm_status = "unhealthy"
    
    # 返回健康状态
    return HealthResponse(
        status=StatusEnum.SUCCESS,
        version=settings.API_VERSION,
        timestamp=datetime.now().isoformat(),
        services={
            "llm_api": llm_status
        }
    )