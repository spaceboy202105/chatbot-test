from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# 导入路由模块
# 注意：这里假设api目录下有chat.py和models.py模块，分别包含相应的路由
# 如果目录结构不同，请相应调整导入路径
from api import chat, models

# 加载环境变量
# 这将从.env文件中加载环境变量到os.environ中
load_dotenv()

# 创建FastAPI应用实例
# 设置API文档的基本信息
app = FastAPI(
    title="LLM Chatbot API",
    description="支持多种大型语言模型(LLM)的聊天机器人API",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI路径
    redoc_url="/redoc",  # ReDoc路径
)

# 配置CORS(跨源资源共享)
# 允许前端应用从不同的域访问API
app.add_middleware(
    CORSMiddleware,
    # 在开发环境中允许所有来源，生产环境应限制为特定域名
    allow_origins=["*"],  # 例如: ["http://localhost:3000", "https://yourdomain.com"]
    allow_credentials=True,  # 允许发送cookies等凭证
    allow_methods=["*"],    # 允许所有HTTP方法
    allow_headers=["*"],    # 允许所有HTTP头
)

# 注册API路由
# 将chat和models模块中定义的路由添加到主应用
app.include_router(chat.router)
app.include_router(models.router)

# 健康检查端点
# 用于监控系统确认API服务正常运行
@app.get("/health", tags=["Health"])
async def health_check():
    """
    健康检查接口，返回API服务状态
    
    Returns:
        dict: 包含状态信息的字典
    """
    return {
        "status": "ok",
        "service": "LLM Chatbot API",
        "version": app.version
    }

# 应用启动入口
if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量获取端口，默认为8000
    port = int(os.getenv("PORT", "8000"))
    
    # 启动uvicorn服务器
    # host="0.0.0.0"表示监听所有网络接口
    # reload=True在开发环境中启用热重载
    uvicorn.run(
        "main:app",  # 应用的导入路径
        host="0.0.0.0", 
        port=port,
        reload=True  # 开发环境中启用，生产环境应设为False
    )
