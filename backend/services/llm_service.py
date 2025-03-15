from typing import List, Dict, Optional, Any
import os
from abc import ABC, abstractmethod
import logging
import httpx
import json
import asyncio
from datetime import datetime

import openai
from google import generativeai as genai
import qianfan
try:
    import anthropic
except ImportError:
    anthropic = None

from models.schemas import ModelInfo, Message

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAdapter(ABC):
    """LLM适配器基类"""
    
    @abstractmethod
    async def generate_response(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """
        生成响应抽象方法
        
        Args:
            messages: 消息历史列表
            system_prompt: 可选的系统提示词
            
        Returns:
            str: 模型生成的响应文本
        """
        pass

class OpenAIAdapter(LLMAdapter):
    """OpenAI适配器，用于处理OpenAI API调用"""
    
    def __init__(self):
        """初始化OpenAI适配器，从环境变量加载API密钥"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("未设置OPENAI_API_KEY环境变量")
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        
    async def generate_response(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """
        调用OpenAI API生成响应
        
        Args:
            messages: 消息历史列表
            system_prompt: 可选的系统提示词
            
        Returns:
            str: 模型生成的响应文本
        """
        try:
            # 转换消息格式为OpenAI API所需格式
            formatted_messages = []
            
            # 添加系统提示词（如果有）
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            # 添加历史消息
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # 调用API
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # 默认使用gpt-3.5-turbo，可根据model_id动态设置
                messages=formatted_messages,
                temperature=0.7,
                max_tokens=1000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # 提取并返回生成的文本
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {str(e)}")
            raise

class GeminiAdapter(LLMAdapter):
    """Google Gemini适配器，用于处理Google Gemini API调用"""
    
    def __init__(self):
        """初始化Gemini适配器，从环境变量加载API密钥"""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("未设置GOOGLE_API_KEY环境变量")
        genai.configure(api_key=self.api_key)
        
    async def generate_response(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """
        调用Google Gemini API生成响应
        
        Args:
            messages: 消息历史列表
            system_prompt: 可选的系统提示词
            
        Returns:
            str: 模型生成的响应文本
        """
        try:
            # 创建Gemini模型实例
            model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')
            
            # 创建聊天会话
            chat = model.start_chat(history=[])
            
            # 添加系统提示词（如果有）
            if system_prompt:
                # Gemini没有直接的系统提示词概念，将其作为第一条消息
                chat.send_message(system_prompt)
            
            # 添加历史消息
            for msg in messages:
                chat.send_message(msg.content)
            
            # 使用asyncio.to_thread将同步API调用转换为异步
            response = await asyncio.to_thread(
                lambda: chat.last.text
            )
            
            return response
        except Exception as e:
            logger.error(f"Google Gemini API调用失败: {str(e)}")
            raise

class DeepseekAdapter(LLMAdapter):
    """Deepseek适配器，用于处理Deepseek API调用"""
    
    def __init__(self):
        """初始化Deepseek适配器，从环境变量加载API密钥"""
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("未设置DEEPSEEK_API_KEY环境变量")
        self.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        
    async def generate_response(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """
        调用Deepseek API生成响应
        
        Args:
            messages: 消息历史列表
            system_prompt: 可选的系统提示词
            
        Returns:
            str: 模型生成的响应文本
        """
        try:
            # 转换消息格式
            formatted_messages = []
            
            # 添加系统提示词（如果有）
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            # 添加历史消息
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # 准备请求数据
            payload = {
                "model": "deepseek-chat",
                "messages": formatted_messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 发送异步请求
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Deepseek API返回错误: {response.text}")
                    raise Exception(f"Deepseek API返回状态码 {response.status_code}: {response.text}")
                
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Deepseek API调用失败: {str(e)}")
            raise

class QwenAdapter(LLMAdapter):
    """Qwen适配器，用于处理阿里云Qwen API调用"""
    
    def __init__(self):
        """初始化Qwen适配器，从环境变量加载API密钥"""
        self.access_key = os.getenv("QIANFAN_ACCESS_KEY")
        self.secret_key = os.getenv("QIANFAN_SECRET_KEY")
        if not self.access_key or not self.secret_key:
            logger.warning("未设置QIANFAN_ACCESS_KEY或QIANFAN_SECRET_KEY环境变量")
        
    async def generate_response(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """
        调用Qwen API生成响应
        
        Args:
            messages: 消息历史列表
            system_prompt: 可选的系统提示词
            
        Returns:
            str: 模型生成的响应文本
        """
        try:
            # 创建千帆客户端
            client = qianfan.ChatCompletion(
                ak=self.access_key,
                sk=self.secret_key
            )
            
            # 转换消息格式
            formatted_messages = []
            
            # 添加系统提示词（如果有）
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            # 添加历史消息
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # 使用asyncio.to_thread将同步API调用转换为异步
            response = await asyncio.to_thread(
                lambda: client.do(
                    messages=formatted_messages,
                    model="qwen-turbo",  # 默认使用qwen-turbo，可根据model_id动态设置
                    temperature=0.7,
                    max_tokens=1000
                )
            )
            
            if "result" not in response:
                raise Exception(f"Qwen API返回异常: {json.dumps(response)}")
            
            return response["result"]
        except Exception as e:
            logger.error(f"Qwen API调用失败: {str(e)}")
            raise

class ClaudeAdapter(LLMAdapter):
    """Claude适配器，用于处理Anthropic Claude API调用"""
    
    def __init__(self):
        """初始化Claude适配器，从环境变量加载API密钥"""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("未设置ANTHROPIC_API_KEY环境变量")
        
        if anthropic is None:
            logger.warning("未安装anthropic库，Claude适配器将无法使用")
        else:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
    async def generate_response(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """
        调用Claude API生成响应
        
        Args:
            messages: 消息历史列表
            system_prompt: 可选的系统提示词
            
        Returns:
            str: 模型生成的响应文本
        """
        if anthropic is None:
            raise Exception("未安装anthropic库，无法使用Claude适配器")
        
        try:
            # 转换消息格式为Claude API所需格式
            formatted_messages = []
            
            # 添加历史消息
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # 调用API
            response = await self.client.messages.create(
                model="claude-2",  # 默认使用claude-2，可根据model_id动态设置
                messages=formatted_messages,
                system=system_prompt,  # Claude直接支持system参数
                max_tokens=1000,
                temperature=0.7
            )
            
            # 提取并返回生成的文本
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API调用失败: {str(e)}")
            raise

class LLMService:
    """LLM服务类，提供统一的接口访问不同的LLM模型"""
    
    def __init__(self):
        """初始化LLM服务，设置支持的模型列表和适配器映射"""
        # 初始化支持的模型列表
        self.models: List[ModelInfo] = [
            # OpenAI模型
            ModelInfo(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                provider="openai",
                description="OpenAI的GPT-3.5 Turbo模型，适合一般对话和创意写作"
            ),
            ModelInfo(
                id="gpt-4",
                name="GPT-4",
                provider="openai",
                description="OpenAI的GPT-4模型，更强大的推理和知识能力"
            ),
            ModelInfo(
                id="gpt-4-turbo",
                name="GPT-4 Turbo",
                provider="openai",
                description="OpenAI的GPT-4 Turbo模型，更快速的响应和更新的知识"
            ),
            
            # Google Gemini模型
            ModelInfo(
                id="gemini-pro",
                name="Gemini Pro",
                provider="google",
                description="Google的Gemini Pro模型，强大的多模态理解能力"
            ),
            
            # Deepseek模型
            ModelInfo(
                id="deepseek-chat",
                name="Deepseek Chat",
                provider="deepseek",
                description="Deepseek的对话模型，擅长中英文对话"
            ),
            
            # 阿里云Qwen模型
            ModelInfo(
                id="qwen-turbo",
                name="Qwen Turbo",
                provider="qwen",
                description="阿里云的通义千问Turbo模型，高效的中文理解能力"
            ),
            ModelInfo(
                id="qwen-plus",
                name="Qwen Plus",
                provider="qwen",
                description="阿里云的通义千问Plus模型，更强的推理和创作能力"
            ),
            
            # Claude模型
            ModelInfo(
                id="claude-2",
                name="Claude 2",
                provider="anthropic",
                description="Anthropic的Claude 2模型，擅长长文本理解和创作"
            ),
            ModelInfo(
                id="claude-instant",
                name="Claude Instant",
                provider="anthropic",
                description="Anthropic的Claude Instant模型，更快速的响应"
            )
        ]
        
        # 创建模型ID到模型信息的映射，方便快速查找
        self.model_map = {model.id: model for model in self.models}
        
        # 创建提供商到模型列表的映射
        self.provider_models: Dict[str, List[ModelInfo]] = {}
        for model in self.models:
            if model.provider not in self.provider_models:
                self.provider_models[model.provider] = []
            self.provider_models[model.provider].append(model)
        
        # 初始化各适配器
        self.adapters = {
            # "openai": OpenAIAdapter(),
            "google": GeminiAdapter(),
            # "deepseek": DeepseekAdapter(),
            # "qwen": QwenAdapter(),
            # "anthropic": ClaudeAdapter() if anthropic else None
        }
    
    async def get_all_models(self) -> List[ModelInfo]:
        """获取所有支持的模型列表"""
        return self.models
    
    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """获取特定模型信息"""
        return self.model_map.get(model_id)

    async def get_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """获取特定提供商的所有模型"""
        return self.provider_models.get(provider, [])
    
    async def generate_response(self, model_id: str, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """生成响应"""
        adapter = self.adapters.get(model_id.split("-")[0])
        if not adapter:
            raise ValueError(f"不支持的模型: {model_id}")
        return await adapter.generate_response(messages, system_prompt)
