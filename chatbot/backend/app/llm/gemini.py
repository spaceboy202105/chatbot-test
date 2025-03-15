import os
from typing import List, Dict, Any, AsyncGenerator, Optional
import google.generativeai as genai

from .base import BaseLLM

class GeminiLLM(BaseLLM):
    """Gemini LLM实现"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-pro-exp-02-05",
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 64,
        max_tokens: int = 8192,
    ):
        """
        初始化Gemini LLM
        
        Args:
            api_key: Gemini API密钥（如果为None，从环境变量GEMINI_API_KEY获取）
            model: 模型标识符
            temperature: 采样温度（0.0到1.0）
            top_p: 核采样参数（0.0到1.0）
            top_k: 保留的最高概率标记数量
            max_tokens: 生成的最大标记数
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("未提供Gemini API密钥且环境中未找到")
        
        # 配置API客户端
        genai.configure(api_key=self.api_key)
        
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        
        # 获取模型
        self.model_client = genai.GenerativeModel(model_name=self.model)
    
    def _prepare_contents(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        准备对话内容
        
        Args:
            message: 当前用户消息
            conversation_history: 之前的对话消息
            system_prompt: 模型的系统提示
            
        Returns:
            准备好的消息列表
        """
        contents = []
        
        # 如果提供了系统提示，则添加
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System: {system_prompt}"}]
            })
        
        # 添加对话历史
        if conversation_history:
            for msg in conversation_history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
        
        # 添加当前消息
        contents.append({
            "role": "user",
            "parts": [{"text": message}]
        })
        
        return contents
    
    async def generate_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        基于提供的消息和对话历史从Gemini生成响应
        
        Args:
            message: 用户消息
            conversation_history: 之前的消息列表
            system_prompt: 可选的系统提示
            **kwargs: 覆盖默认值的额外参数
            
        Returns:
            LLM的响应字符串
        """
        # 如果提供了参数，则覆盖默认参数
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # 准备会话历史
        contents = self._prepare_contents(message, conversation_history, system_prompt)
        
        # 创建虚拟会话
        chat = self.model_client.start_chat(history=[])
        
        # 配置生成参数
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_tokens,
        }
        
        # 使用最后一条消息内容作为用户输入
        last_message = contents[-1]["parts"][0]["text"]
        
        # 生成响应
        response = chat.send_message(
            last_message,
            generation_config=generation_config
        )
        
        return response.text
    
    async def generate_stream(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        从Gemini生成流式响应
        
        Args:
            message: 用户消息
            conversation_history: 之前的消息列表
            system_prompt: 可选的系统提示
            **kwargs: 覆盖默认值的额外参数
            
        Yields:
            可用的响应块
        """
        # 如果提供了参数，则覆盖默认参数
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # 准备会话历史
        contents = self._prepare_contents(message, conversation_history, system_prompt)
        
        # 创建虚拟会话
        chat = self.model_client.start_chat(history=[])
        
        # 配置生成参数
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_tokens,
        }
        
        # 使用最后一条消息内容作为用户输入
        last_message = contents[-1]["parts"][0]["text"]
        
        # 生成流式响应
        stream = chat.send_message(
            last_message,
            generation_config=generation_config,
            stream=True
        )
        
        # 产生响应块
        for chunk in stream:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取关于Gemini模型的信息
        
        Returns:
            包含模型信息的字典
        """
        return {
            "provider": "Google",
            "model": self.model,
            "parameters": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_tokens": self.max_tokens,
            }
        }