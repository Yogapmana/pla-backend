from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MIN_SECRET_LEN = 32


class Settings(BaseSettings):
    # Database PostgreSQL
    DATABASE_URL: str = Field(default="")

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
        return [
            origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()
        ]

    # LLM Providers
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    # Legacy (ignored by llm_factory; kept so old .env still loads)
    GROQ_API_KEY: str = Field(default="")
    TAVILY_API_KEY: str = Field(default="")
    JINA_API_KEY: str = Field(default="")
    POLLINATIONS_API_KEY: str = Field(default="")

    # Model Assignment — OpenRouter IDs (or Ollama name:tag for local)
    ORCHESTRATOR_MODEL: str = Field(default="meta-llama/llama-3.3-70b-instruct")
    PLANNER_MODEL: str = Field(default="meta-llama/llama-3.3-70b-instruct")
    RESEARCHER_MODEL: str = Field(default="gemma4:e4b")
    COMPOSER_MODEL: str = Field(default="meta-llama/llama-3.3-70b-instruct")
    TUTOR_MODEL: str = Field(default="gemma4:e4b")
    GENERAL_CHAT_MODEL: str = Field(default="meta-llama/llama-3.3-70b-instruct")
    RAGAS_MODEL: str = Field(default="meta-llama/llama-3.1-8b-instruct")
    MINDMAP_MODEL: str = Field(default="meta-llama/llama-3.3-70b-instruct")
    EMBEDDING_MODEL: str = Field(default="nomic-embed-text")

    # RPS ↔ Curriculum verification (post-generation coverage audit)
    RPS_VERIFY_ENABLED: bool = Field(default=True)
    RPS_VERIFY_MODEL: str = Field(default="meta-llama/llama-3.3-70b-instruct")

    # LangSmith Tracing
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_API_KEY: str = Field(default="")
    LANGSMITH_PROJECT: str = Field(default="synapsa-pla")

    # Reranker (FlashRank model name)
    RERANKER_MODEL: str = Field(default="ms-marco-MiniLM-L-12-v2")

    # Image Generation
    IMAGE_MODEL: str = Field(default="qwen-image")
    IMAGE_WIDTH: int = Field(default=800)
    IMAGE_HEIGHT: int = Field(default=400)

    # Keamanan & Auth — set SECRET_KEY in .env (min 32 chars)
    SECRET_KEY: str = Field(default="")
    # Access JWT is short-lived; refresh cookie renews it (rotation).
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=14)
    # Empty = auto (Secure only if all CORS origins are https, no localhost).
    # Set COOKIE_SECURE=true in pure HTTPS production if needed.
    COOKIE_SECURE: str = Field(default="")
    # Cross-site needs "none" (requires Secure). Same-site can stay "lax".
    # Set COOKIE_SAMESITE=none in docker-compose (production) for vercel.app origin.
    COOKIE_SAMESITE: str = Field(default="lax")
    GOOGLE_CLIENT_ID: str = Field(default="")

    # Email (Resend)
    RESEND_API_KEY: str = Field(default="")
    MAIL_FROM: str = Field(default="Acme <onboarding@resend.dev>")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_key_required(cls, v: str) -> str:
        key = (v or "").strip()
        if len(key) < _MIN_SECRET_LEN:
            raise ValueError(
                f"SECRET_KEY must be set in .env and at least {_MIN_SECRET_LEN} characters "
                f"(got {len(key)}). Generate with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        return key


settings = Settings()
