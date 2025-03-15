import os
from typing import Dict, Any, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ConfigDict
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

class Settings(BaseSettings):
    """应用配置设置类"""
    
    # API设置
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)
    
    # LLM配置
    GEMINI_API_KEY: str = Field(default="")
    DEFAULT_MODEL: str = Field(default="gemini-2.0-pro-exp-02-05")
    DEFAULT_TEMPERATURE: float = Field(default=0.7)
    DEFAULT_TOP_P: float = Field(default=0.95)
    DEFAULT_TOP_K: int = Field(default=64)
    DEFAULT_MAX_TOKENS: int = Field(default=8192)
    
    # 系统提示
    DEFAULT_SYSTEM_PROMPT: str = Field(
        default="你是一个有帮助的AI助手。"
    )
    
    # 数据存储（内存模式）
    ENABLE_PERSISTENCE: bool = Field(default=False)
    
    # 使用SettingsConfigDict而不是Config类
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

    def get_llm_config(self) -> Dict[str, Any]:
        """
        获取当前LLM配置
        
        Returns:
            包含LLM配置的字典
        """
        return {
            "model": self.DEFAULT_MODEL,
            "temperature": self.DEFAULT_TEMPERATURE,
            "top_p": self.DEFAULT_TOP_P,
            "top_k": self.DEFAULT_TOP_K,
            "max_tokens": self.DEFAULT_MAX_TOKENS,
        }

    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """
        获取支持的LLM模型列表
        
        Returns:
            支持的模型配置字典
        """
        return {
            self.DEFAULT_MODEL: {
                "provider": "gemini",
                "name": "Gemini 2.0 Pro Experimental",
                "description": "Google的大型语言模型",
                "config": self.get_llm_config()
            }
        }


@lru_cache()
def get_settings() -> Settings:
    """
    获取应用设置单例
    
    Returns:
        Settings实例
    """
    return Settings()