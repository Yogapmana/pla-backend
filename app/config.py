from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database PostgreSQL
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://pla_user:pla_password@localhost:5433/pla_db"
    )

    # Vector DB & Redis
    QDRANT_URL: str = Field(default="http://localhost:6333")
    REDIS_URL: str = Field(default="redis://localhost:6380/0")

    # CORS — comma-separated list of allowed origins.
    # Set in .env as: CORS_ORIGINS=http://localhost:5173,https://pla.example.com
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://localhost:5174"
    )

    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS into a list of allowed origin strings."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # LLM Providers
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    GROQ_API_KEY: str = Field(default="")
    TAVILY_API_KEY: str = Field(default="")
    JINA_API_KEY: str = Field(default="")
    POLLINATIONS_API_KEY: str = Field(default="")

    # Model Assignment
    ORCHESTRATOR_MODEL: str = Field(default="llama-3.3-70b-versatile")
    PLANNER_MODEL: str = Field(default="llama-3.3-70b-versatile")
    RESEARCHER_MODEL: str = Field(default="gemma4:e4b")
    COMPOSER_MODEL: str = Field(default="llama-3.3-70b-versatile")
    TUTOR_MODEL: str = Field(default="gemma4:e4b")
    GENERAL_CHAT_MODEL: str = Field(default="llama-3.3-70b-versatile")
    RAGAS_MODEL: str = Field(default="llama-3.1-8b-instant")
    EMBEDDING_MODEL: str = Field(default="nomic-embed-text")

    # Image Generation
    IMAGE_MODEL: str = Field(default="qwen-image")
    IMAGE_WIDTH: int = Field(default=800)
    IMAGE_HEIGHT: int = Field(default=400)

    # Keamanan & Auth
    SECRET_KEY: str = Field(default="bikin_rahasia_bebas_contoh_1234567890")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=1440)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
