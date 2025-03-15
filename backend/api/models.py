from fastapi import APIRouter, HTTPException, Path, status
from typing import List, Dict
from uuid import UUID

from models.schemas import ModelInfo
from services.llm_service import LLMService

# 创建路由器对象，设置前缀和标签
router = APIRouter(prefix="/api/models", tags=["models"])
llm_service = LLMService()

@router.get("/", response_model=List[ModelInfo])
async def get_models():
    """
    获取所有支持的LLM模型列表
    
    Returns:
        List[ModelInfo]: 支持的模型列表
    
    Raises:
        HTTPException: 如果获取模型列表失败
    """
    try:
        # 获取所有支持的模型
        models = await llm_service.get_all_models()
        return models
    except Exception as e:
        # 处理异常情况
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型列表失败: {str(e)}"
        )

@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str = Path(..., description="模型ID")
):
    """
    获取特定模型的详细信息
    
    Args:
        model_id (str): 模型的唯一标识符
        
    Returns:
        ModelInfo: 模型详细信息
        
    Raises:
        HTTPException: 如果模型不存在或获取失败
    """
    try:
        # 获取特定模型信息
        model = await llm_service.get_model(model_id)
        if not model:
            # 如果模型不存在，返回404错误
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型 {model_id} 不存在"
            )
        return model
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 处理其他异常情况
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型信息失败: {str(e)}"
        )

@router.get("/provider/{provider}", response_model=List[ModelInfo])
async def get_models_by_provider(
    provider: str = Path(..., description="提供商名称")
):
    """
    获取特定提供商的所有模型
    
    Args:
        provider (str): 模型提供商名称（如OpenAI、Google等）
        
    Returns:
        List[ModelInfo]: 该提供商的模型列表
        
    Raises:
        HTTPException: 如果获取失败
    """
    try:
        # 获取特定提供商的模型
        models = await llm_service.get_models_by_provider(provider)
        return models
    except Exception as e:
        # 处理异常情况
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取提供商 {provider} 的模型列表失败: {str(e)}"
        )
