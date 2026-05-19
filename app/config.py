from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Database PostgreSQL
    DATABASE_URL: str = Field(default="postgresql+asyncpg://pla_user:pla_password@localhost:5433/pla_db")

    # Vector DB & Redis
    QDRANT_URL: str = Field(default="http://localhost:6333")
    REDIS_URL: str = Field(default="redis://localhost:6380/0")

    # LLM Providers
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    GROQ_API_KEY: str = Field(default="")
    TAVILY_API_KEY: str = Field(default="")
    JINA_API_KEY: str = Field(default="")

    # Model Assignment
    ORCHESTRATOR_MODEL: str = Field(default="llama-3.3-70b-versatile")
    PLANNER_MODEL: str = Field(default="llama-3.3-70b-versatile")
    RESEARCHER_MODEL: str = Field(default="gemma4:e4b")
    COMPOSER_MODEL: str = Field(default="llama-3.3-70b-versatile")
    TUTOR_MODEL: str = Field(default="gemma4:e4b")
    EMBEDDING_MODEL: str = Field(default="nomic-embed-text")

    # Keamanan & Auth
    SECRET_KEY: str = Field(default="bikin_rahasia_bebas_contoh_1234567890")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=1440)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
