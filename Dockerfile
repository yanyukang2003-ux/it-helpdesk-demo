# ==================================================
# IT Helpdesk Agent - Docker 镜像
# ==================================================
# 构建: docker build -t it-helpdesk-agent .
# 运行: docker run -p 8000:8000 --env-file .env it-helpdesk-agent

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（chromadb 需要）
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 启动服务（兼容 Render 的 PORT 环境变量）
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
