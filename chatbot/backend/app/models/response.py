from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, TypeVar, Generic
from enum import Enum

T = TypeVar('T')

class StatusEnum(str, Enum):
    """API响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"

class APIResponse(BaseModel, Generic[T]):
    """通用API响应模型"""
    status: StatusEnum = Field(StatusEnum.SUCCESS, description="响应状态")
    data: Optional[T] = Field(None, description="响应数据")
    message: Optional[str] = Field(None, description="响应消息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "data": {},
                "message": "操作成功"
            }
        }

class ErrorResponse(BaseModel):
    """错误响应模型"""
    status: StatusEnum = Field(StatusEnum.ERROR, description="错误状态")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(None, description="错误代码")
    details: Optional[Dict[str, Any]] = Field(None, description="详细错误信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "请求参数无效",
                "error_code": "INVALID_REQUEST",
                "details": {
                    "message": ["字段不能为空"]
                }
            }
        }

class ValidationError(BaseModel):
    """验证错误模型"""
    loc: List[str] = Field(..., description="错误位置")
    msg: str = Field(..., description="错误消息")
    type: str = Field(..., description="错误类型")
    
    class Config:
        json_schema_extra = {
            "example": {
                "loc": ["body", "message"],
                "msg": "字段不能为空",
                "type": "value_error.missing"
            }
        }

class ValidationErrorResponse(BaseModel):
    """验证错误响应模型"""
    status: StatusEnum = Field(StatusEnum.ERROR, description="错误状态")
    message: str = Field("请求验证失败", description="错误消息")
    error_code: str = Field("VALIDATION_ERROR", description="错误代码")
    errors: List[ValidationError] = Field(..., description="验证错误列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "请求验证失败",
                "error_code": "VALIDATION_ERROR",
                "errors": [
                    {
                        "loc": ["body", "message"],
                        "msg": "字段不能为空",
                        "type": "value_error.missing"
                    }
                ]
            }
        }

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: StatusEnum = Field(StatusEnum.SUCCESS, description="服务状态")
    version: str = Field(..., description="API版本")
    timestamp: str = Field(..., description="服务器当前时间")
    services: Dict[str, str] = Field(..., description="服务状态")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "version": "1.0.0",
                "timestamp": "2024-03-15T12:34:56.789Z",
                "services": {
                    "database": "healthy",
                    "llm_api": "healthy"
                }
            }
        }

class ModelInfo(BaseModel):
    """模型信息模型"""
    id: str = Field(..., description="模型ID")
    name: str = Field(..., description="模型名称")
    provider: str = Field(..., description="提供商")
    description: Optional[str] = Field(None, description="模型描述")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "gemini-2.0-pro-exp-02-05",
                "name": "Gemini 2.0 Pro",
                "provider": "Google",
                "description": "Google的大型语言模型"
            }
        }

class ModelsResponse(BaseModel):
    """模型列表响应模型"""
    models: List[ModelInfo] = Field(..., description="可用模型列表")
    default_model: str = Field(..., description="默认模型ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "id": "gemini-2.0-pro-exp-02-05",
                        "name": "Gemini 2.0 Pro",
                        "provider": "Google",
                        "description": "Google的大型语言模型"
                    }
                ],
                "default_model": "gemini-2.0-pro-exp-02-05"
            }
        }

class ModelConfigRequest(BaseModel):
    """模型配置请求模型"""
    model: str = Field(..., description="模型ID")
    temperature: Optional[float] = Field(None, description="生成温度")
    top_p: Optional[float] = Field(None, description="Top-p采样参数")
    top_k: Optional[int] = Field(None, description="Top-k采样参数")
    max_tokens: Optional[int] = Field(None, description="最大生成标记数")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "gemini-2.0-pro-exp-02-05",
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 4096
            }
        }

class ModelConfigResponse(BaseModel):
    """模型配置响应模型"""
    model: str = Field(..., description="模型ID")
    config: Dict[str, Any] = Field(..., description="模型配置")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "gemini-2.0-pro-exp-02-05",
                "config": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_tokens": 4096
                }
            }
        }