from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from app.config import settings

def get_llm(model_name: str, temperature: float = 0.2, max_tokens: int | None = None):
    """
    Factory function to return the correct LLM instance based on the model name.
    If the model name contains 'llama' or 'gemma', we might still use Groq if it's a known Groq model,
    or we default to Groq if the model matches Groq's typical names.
    Since we are testing, we'll try to map it appropriately.

    `max_tokens`: optional cap on the LLM response. Required for some metrics
    (e.g. RAGAS, long answer generation) to avoid LLMDidNotFinishException.
    """
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
        "openai/gpt-oss-120b"
    ]

    kwargs = {"temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # Check if the requested model is likely a Groq model
    if model_name in groq_models or "llama-3" in model_name or model_name.startswith("openai/"):
        return ChatGroq(
            model_name=model_name,
            api_key=settings.GROQ_API_KEY,
            **kwargs
        )
    else:
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=model_name,
            **kwargs
        )
