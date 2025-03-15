import os
import logging
from typing import Dict, Optional, Any, List
from dotenv import load_dotenv
from pydantic import BaseSettings, validator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

class Settings(BaseSettings):
    """配置设置类，使用Pydantic进行验证"""
    
    # API密钥配置
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    QWEN_API_KEY: Optional[str] = None
    CLAUDE_API_KEY: Optional[str] = None
    
    # API调用配置
    API_TIMEOUT: int = 60  # 默认超时时间为60秒
    API_MAX_RETRIES: int = 3  # 默认最大重试次数
    API_RETRY_DELAY: float = 1.0  # 默认重试延迟时间(秒)
    
    # 应用配置
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    @validator('API_TIMEOUT')
    def validate_timeout(cls, v):
        """验证超时时间是否合理"""
        if v <= 0:
            raise ValueError("API_TIMEOUT必须大于0")
        return v
    
    @validator('API_MAX_RETRIES')
    def validate_retries(cls, v):
        """验证重试次数是否合理"""
        if v < 0:
            raise ValueError("API_MAX_RETRIES不能为负数")
        return v


class Config:
    """配置管理类"""
    
    def __init__(self):
        """初始化配置管理类"""
        self.settings = Settings()
        self._provider_key_mapping = {
            "openai": "OPENAI_API_KEY",
            "google": "GEMINI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen": "QWEN_API_KEY",
            "anthropic": "CLAUDE_API_KEY",
        }
        # 验证配置完整性
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置完整性，记录缺失的API密钥"""
        missing_keys = []
        for provider, env_var in self._provider_key_mapping.items():
            if not getattr(self.settings, env_var):
                missing_keys.append(f"{env_var} ({provider})")
                logger.warning(f"未设置 {env_var} 环境变量，{provider} 相关功能将不可用")
        
        if missing_keys:
            logger.warning(f"以下API密钥未配置: {', '.join(missing_keys)}")
    
    def get_api_key(self, provider: str) -> str:
        """
        获取指定提供商的API密钥
        
        Args:
            provider: 提供商名称，如openai, google, deepseek等
            
        Returns:
            str: API密钥
            
        Raises:
            ValueError: 如果提供商不支持或API密钥未设置
        """
        env_var = self._provider_key_mapping.get(provider.lower())
        if not env_var:
            raise ValueError(f"不支持的提供商: {provider}")
            
        api_key = getattr(self.settings, env_var)
        if not api_key:
            raise ValueError(f"未设置 {env_var} 环境变量，无法使用 {provider} 服务")
            
        return api_key
    
    def get_api_timeout(self) -> int:
        """获取API调用超时时间"""
        return self.settings.API_TIMEOUT
    
    def get_api_retry_config(self) -> Dict[str, Any]:
        """获取API重试配置"""
        return {
            "max_retries": self.settings.API_MAX_RETRIES,
            "retry_delay": self.settings.API_RETRY_DELAY
        }
    
    def is_provider_configured(self, provider: str) -> bool:
        """
        检查提供商是否已配置API密钥
        
        Args:
            provider: 提供商名称
            
        Returns:
            bool: 如果已配置返回True，否则返回False
        """
        env_var = self._provider_key_mapping.get(provider.lower())
        if not env_var:
            return False
        
        return bool(getattr(self.settings, env_var))
    
    def get_configured_providers(self) -> List[str]:
        """
        获取所有已配置API密钥的提供商列表
        
        Returns:
            List[str]: 已配置的提供商列表
        """
        return [
            provider for provider in self._provider_key_mapping.keys()
            if self.is_provider_configured(provider)
        ]


# 创建全局配置实例
config = Config()
