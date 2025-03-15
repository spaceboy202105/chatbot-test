from typing import Dict, Any, Optional
from .base import BaseLLM
from .gemini import GeminiLLM

class LLMFactory:
    """创建LLM实例的工厂"""
    
    @staticmethod
    def create_llm(provider: str, config: Optional[Dict[str, Any]] = None) -> BaseLLM:
        """
        基于提供商名称和配置创建LLM实例
        
        Args:
            provider: LLM提供商名称（例如，"gemini"）
            config: LLM的配置参数
            
        Returns:
            BaseLLM的实例
            
        Raises:
            ValueError: 如果不支持该提供商
        """
        config = config or {}
        
        if provider.lower() == "gemini":
            return GeminiLLM(
                api_key=config.get("api_key"),
                model=config.get("model", "gemini-2.0-pro-exp-02-05"),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 0.95),
                top_k=config.get("top_k", 64),
                max_tokens=config.get("max_tokens", 8192),
            )
        # 未来: 在此处添加其他LLM提供商
        # elif provider.lower() == "openai":
        #     return OpenAILLM(**config)
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")