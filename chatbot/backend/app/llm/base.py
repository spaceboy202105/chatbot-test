from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional

class BaseLLM(ABC):
    """LLM提供商的抽象基类"""
    
    @abstractmethod
    async def generate_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        基于提供的消息和对话历史生成LLM响应
        
        Args:
            message: 用户消息
            conversation_history: 以下格式的之前消息列表
                                 [{"role": "user|assistant", "content": "message"}]
            system_prompt: 可选的系统提示，用于设置模型行为
            **kwargs: 额外的模型特定参数
            
        Returns:
            LLM的响应字符串
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        生成LLM的流式响应
        
        Args:
            message: 用户消息
            conversation_history: 之前的消息列表
            system_prompt: 可选的系统提示，用于设置模型行为
            **kwargs: 额外的模型特定参数
            
        Returns:
            异步生成器，产生可用的响应块
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        pass