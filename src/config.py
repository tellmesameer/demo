import os
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

LLMProvider = Literal["google", "offline", "openai", "anthropic"]


@dataclass
class LLMConfig:
    provider: LLMProvider = "google"
    model: str = "gemini-2.0-flash"
    temperature: float = 0.2


@dataclass
class ProjectPaths:
    ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_PDF: str = os.path.join(ROOT, "data", "raw", "IPC-to-BNS-Conversion-Guide.pdf")
    PROCESSED_CHUNKS: str = os.path.join(
        ROOT, "data", "processed", "ipcbns_chunks.json"
    )
    SQLITE_DB: str = os.path.join(ROOT, "data", "db", "ipcbns_mapping.db")
    EVAL_LOG: str = os.path.join(ROOT, "data", "eval_log.jsonl")


paths = ProjectPaths()


def get_llm(config: Optional[LLMConfig] = None):
    """
    Return a LangChain-compatible chat model.

    Gemini is default; fall back to offline dummy or other providers as configured.
    """
    from langchain_core.language_models import BaseChatModel  # type: ignore

    if config is None:
        config = LLMConfig()

    if config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in environment.")
        llm: BaseChatModel = ChatGoogleGenerativeAI(
            model=config.model,
            api_key=api_key,
            temperature=config.temperature,
        )
        return llm

    if config.provider == "offline":
        from langchain_core.runnables import RunnableLambda  # type: ignore

        def _dummy(messages):
            last = messages[-1].content if messages else ""
            return {"content": f"OFFLINE DUMMY ANSWER (echo): {last[:200]}"}

        return RunnableLambda(_dummy)

    if config.provider == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore

        return ChatOpenAI(
            model=config.model or "gpt-4o-mini",
            temperature=config.temperature,
        )

    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore

        return ChatAnthropic(
            model=config.model or "claude-3-haiku-20240307",
            temperature=config.temperature,
        )

    raise ValueError(f"Unknown provider: {config.provider}")
