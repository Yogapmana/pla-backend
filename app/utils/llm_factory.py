from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from app.config import settings


def uses_json_mode(llm) -> bool:
    """OpenAI-compatible providers (OpenRouter) need json_mode for structured output."""
    return isinstance(llm, ChatOpenAI)


def _is_ollama_model(model_name: str) -> bool:
    # Local Ollama tags look like "gemma4:e4b"; cloud IDs use "provider/model".
    if model_name.startswith("ollama/"):
        return True
    return ":" in model_name and "/" not in model_name


def get_llm(model_name: str, temperature: float = 0.2, max_tokens: int | None = None):
    """
    Return LLM for model name.
    - Ollama: local models (name:tag)
    - OpenRouter: everything else (model ID from .env)
    """
    kwargs: dict = {"temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if _is_ollama_model(model_name):
        ollama_name = model_name.removeprefix("ollama/")
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=ollama_name,
            num_ctx=32768,
            **kwargs,
        )

    api_key = (settings.OPENROUTER_API_KEY or "").strip()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY kosong. Isi di .env (https://openrouter.ai/keys)."
        )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=settings.OPENROUTER_BASE_URL,
        max_retries=5,
        timeout=180,
        default_headers={
            "HTTP-Referer": "https://github.com/synapsa",
            "X-Title": "Synapsa PLA",
        },
        **kwargs,
    )
