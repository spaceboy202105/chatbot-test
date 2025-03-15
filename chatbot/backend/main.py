import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.models.response import ErrorResponse, ValidationErrorResponse, ValidationError

# 获取应用设置
settings = get_settings()

# 创建FastAPI应用
app = FastAPI(
    title="AI聊天机器人API",
    description="基于Python后端的AI聊天机器人API，支持多种LLM模型",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该更具体
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加API路由
app.include_router(api_router, prefix=settings.API_PREFIX)

# 异常处理
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """处理通用异常"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            status="error",
            message="服务器内部错误",
            error_code="INTERNAL_SERVER_ERROR",
            details={"error": str(exc)}
        ).dict(),
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    # 转换验证错误为更友好的格式
    errors = []
    for error in exc.errors():
        errors.append(
            ValidationError(
                loc=error["loc"],
                msg=error["msg"],
                type=error["type"]
            )
        )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ValidationErrorResponse(
            status="error",
            message="请求验证失败",
            error_code="VALIDATION_ERROR",
            errors=errors
        ).dict(),
    )

# 健康检查路由
@app.get("/")
async def root():
    """API根路径，用于快速验证API是否运行"""
    return {
        "message": "AI聊天机器人API正在运行",
        "status": "online",
        "version": settings.API_VERSION,
        "docs": "/docs"
    }

# 启动应用
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )