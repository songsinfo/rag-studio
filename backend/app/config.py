import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENVIRONMENT: str = "demo"
    DB_NAME: str = "rag_demo"
    DB_USER: str = "rag"
    DB_PASSWORD: str = ""
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3307
    DB_URL: str = ""
    RAG_CSRF_SECRET: str = ""
    RAG_COOKIE_SECURE: bool = False
    PORT: int = 8080
    MILVUS_HOST: str = "127.0.0.1"
    MILVUS_PORT: int = 19530
    MILVUS_URI: str = ""
    LITELLM_BASE_URL: str = ""
    LITELLM_PORT: int = 4000
    LITELLM_MASTER_KEY: str = ""
    DOCLING_URL: str = "http://127.0.0.1:5001"
    DOCLING_API_KEY: str = ""
    DOCLING_CHUNK_MAX_TOKENS: int = 400
    DOCLING_PARENT_MAX_TOKENS: int = 1200
    DOCLING_CHUNK_TOKENIZER: str = "sentence-transformers/all-MiniLM-L6-v2"
    DOCLING_CHUNK_MERGE_PEERS: bool = True
    RAG_EMBEDDED_WORKER: bool = True
    RAG_QA_WORKERS: int = 4
    RAG_INGEST_WORKERS: int = 2
    RAG_EVAL_WORKERS: int = 1
    RAG_ARTIFACT_STORE: str = "local"
    S3_ENDPOINT_URL: str = ""
    S3_BUCKET: str = "rag-artifacts"
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: str = ""
    S3_REGION: str = "us-east-1"

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore"
    )

    @property
    def milvus_effective_uri(self) -> str:
        if self.MILVUS_URI:
            return self.MILVUS_URI
        return f"http://{self.MILVUS_HOST}:{self.MILVUS_PORT}"

    @property
    def litellm_effective_url(self) -> str:
        if self.LITELLM_BASE_URL:
            return self.LITELLM_BASE_URL
        return f"http://127.0.0.1:{self.LITELLM_PORT}"

    @property
    def sqlalchemy_database_uri(self) -> str:
        if self.DB_URL and self.DB_URL.startswith("mysql+pymysql://"):
            return self.DB_URL
        if self.DB_URL and self.DB_URL.startswith("jdbc:mysql://"):
            # 将 Java/JDBC 风格的 MySQL URL 转为 SQLAlchemy + PyMySQL URL；仅提取主机、端口和库名，原查询参数不保留，固定添加 charset=utf8mb4。
            # 例如 jdbc:mysql://localhost:3307/rag_demo?connectionTimeZone=UTC... 转换后不会保留 connectionTimeZone，只会附加 charset=utf8mb4。
            import re
            m = re.match(r"jdbc:mysql://([^:/]+):(\d+)/([^?]+)", self.DB_URL)
            if m:
                host, port, db = m.groups()
                return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{host}:{port}/{db}?charset=utf8mb4"
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"


def load_settings() -> Settings:
    # 优先读取 .runtime/<ENVIRONMENT>.env，让不同运行环境覆盖默认配置，同时避免把本地运行参数写进仓库根目录。
    rag_root = Path(__file__).resolve().parent.parent.parent
    env_name = os.getenv("RAG_ENVIRONMENT", "demo")
    runtime_env = rag_root / f".runtime/{env_name}.env"
    env_file = runtime_env if runtime_env.exists() else rag_root / ".env"
    
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    s = Settings(
        ENVIRONMENT=os.getenv("RAG_ENVIRONMENT", "demo"),
        DB_NAME=os.getenv("DB_NAME", "rag_demo"),
        DB_USER=os.getenv("DB_USER", "rag"),
        DB_PASSWORD=os.getenv("DB_PASSWORD", ""),
        DB_HOST=os.getenv("DB_HOST", "127.0.0.1"),
        DB_PORT=int(os.getenv("DB_PORT", "3307")),
        DB_URL=os.getenv("DB_URL", ""),
        RAG_CSRF_SECRET=os.getenv("RAG_CSRF_SECRET", ""),
        RAG_COOKIE_SECURE=os.getenv("RAG_COOKIE_SECURE", "false").lower() in ("true", "1", "yes"),
        PORT=int(os.getenv("PORT", "8080")),
        MILVUS_HOST=os.getenv("MILVUS_HOST", "127.0.0.1"),
        MILVUS_PORT=int(os.getenv("MILVUS_PORT", "19530")),
        MILVUS_URI=os.getenv("MILVUS_URI", ""),
        LITELLM_BASE_URL=os.getenv("LITELLM_BASE_URL", ""),
        LITELLM_PORT=int(os.getenv("LITELLM_PORT", "4000")),
        LITELLM_MASTER_KEY=os.getenv("LITELLM_MASTER_KEY", ""),
        DOCLING_URL=os.getenv("DOCLING_URL", "http://127.0.0.1:5001"),
        DOCLING_API_KEY=os.getenv("DOCLING_API_KEY", ""),
        DOCLING_CHUNK_MAX_TOKENS=int(os.getenv("DOCLING_CHUNK_MAX_TOKENS", "400")),
        DOCLING_PARENT_MAX_TOKENS=int(os.getenv("DOCLING_PARENT_MAX_TOKENS", "1200")),
        DOCLING_CHUNK_TOKENIZER=os.getenv(
            "DOCLING_CHUNK_TOKENIZER",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        DOCLING_CHUNK_MERGE_PEERS=os.getenv(
            "DOCLING_CHUNK_MERGE_PEERS",
            "true",
        ).lower() in ("true", "1", "yes"),
        RAG_EMBEDDED_WORKER=os.getenv("RAG_EMBEDDED_WORKER", "true").lower() in ("true", "1", "yes"),
        RAG_QA_WORKERS=int(os.getenv("RAG_QA_WORKERS", "4")),
        RAG_INGEST_WORKERS=int(os.getenv("RAG_INGEST_WORKERS", "2")),
        RAG_EVAL_WORKERS=int(os.getenv("RAG_EVAL_WORKERS", "1")),
        RAG_ARTIFACT_STORE=os.getenv("RAG_ARTIFACT_STORE", "local"),
        S3_ENDPOINT_URL=os.getenv("S3_ENDPOINT_URL", ""),
        S3_BUCKET=os.getenv("S3_BUCKET", "rag-artifacts"),
        S3_ACCESS_KEY_ID=os.getenv("S3_ACCESS_KEY_ID", ""),
        S3_SECRET_ACCESS_KEY=os.getenv("S3_SECRET_ACCESS_KEY", ""),
        S3_REGION=os.getenv("S3_REGION", "us-east-1"),
    )
    if s.ENVIRONMENT != "test":
        if not os.getenv("MILVUS_URI"):
            os.environ["MILVUS_URI"] = s.milvus_effective_uri
        if not os.getenv("LITELLM_BASE_URL") and s.LITELLM_BASE_URL:
            os.environ["LITELLM_BASE_URL"] = s.LITELLM_BASE_URL
    return s


settings = load_settings()


def is_test_env() -> bool:
    return (
        os.getenv("RAG_ENVIRONMENT") == "test"
        or getattr(settings, "ENVIRONMENT", "") == "test"
        or bool(os.getenv("PYTEST_CURRENT_TEST"))
    )
