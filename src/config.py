import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

LLMProvider = Literal["google", "offline", "openai", "anthropic"]

# ── Settings loader ──────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETTINGS_PATH = os.path.join(_PROJECT_ROOT, "settings.yaml")

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "llm": {
        "provider": "google",
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
        "fallback_models": [
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash-lite",
        ],
    },
    "embedding": {"model": "models/gemini-embedding-001"},
    "vectorstore": {"persist_dir": "data/chroma_ipcbns"},
    "verification": {"human_review_confidence_threshold": 0.7},
    "logging": {"level": "INFO"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(path: str = _SETTINGS_PATH) -> Dict[str, Any]:
    """Load settings from YAML, falling back to defaults for any missing keys."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_settings = yaml.safe_load(f) or {}
        return _deep_merge(_DEFAULT_SETTINGS, user_settings)

    # File missing → write defaults so the user has a template.
    logger.info("settings.yaml not found – creating default at %s", path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(_DEFAULT_SETTINGS, f, default_flow_style=False, sort_keys=False)
    return _DEFAULT_SETTINGS.copy()


settings: Dict[str, Any] = load_settings()

# Convenience aliases used throughout the codebase.
GOOGLE_FALLBACK_MODELS: List[str] = settings["llm"]["fallback_models"]
EMBEDDING_MODEL: str = settings["embedding"]["model"]
HUMAN_REVIEW_THRESHOLD: float = settings["verification"]["human_review_confidence_threshold"]

# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class LLMConfig:
    provider: LLMProvider = field(default_factory=lambda: settings["llm"]["provider"])
    model: str = field(default_factory=lambda: settings["llm"]["model"])
    temperature: float = field(default_factory=lambda: settings["llm"]["temperature"])


@dataclass
class ProjectPaths:
    ROOT: str = _PROJECT_ROOT
    RAW_PDF: str = str(Path(_PROJECT_ROOT) / "data" / "raw" / "IPC-to-BNS-Conversion-Guide.pdf")
    PROCESSED_CHUNKS: str = str(Path(_PROJECT_ROOT) / "data" / "processed" / "ipcbns_chunks.json")
    SQLITE_DB: str = str(Path(_PROJECT_ROOT) / "data" / "db" / "ipcbns_mapping.db")
    EVAL_LOG: str = str(Path(_PROJECT_ROOT) / "data" / "eval_log.jsonl")
    HUMAN_REVIEW_QUEUE: str = str(Path(_PROJECT_ROOT) / "data" / "human_review_queue.jsonl")
    CHROMA_DIR: str = str(Path(_PROJECT_ROOT) / settings["vectorstore"]["persist_dir"])


paths = ProjectPaths()


# ── Data-directory bootstrap ─────────────────────────────────────────────────


def init_data_dirs() -> None:
    """Create required data directories and seed empty files if they don't exist."""
    # Directories
    for directory in (
        os.path.join(paths.ROOT, "data", "db"),
        os.path.join(paths.ROOT, "data", "processed"),
        os.path.join(paths.ROOT, "data", "raw"),
    ):
        os.makedirs(directory, exist_ok=True)

    # Empty JSONL files
    for jsonl_path in (paths.EVAL_LOG, paths.HUMAN_REVIEW_QUEUE):
        if not os.path.exists(jsonl_path):
            Path(jsonl_path).touch()
            logger.info("Created empty file: %s", jsonl_path)

    # SQLite database
    if not os.path.exists(paths.SQLITE_DB):
        conn = sqlite3.connect(paths.SQLITE_DB)
        conn.close()
        logger.info("Created empty SQLite database: %s", paths.SQLITE_DB)


# ── LLM factory ──────────────────────────────────────────────────────────────


def get_llm(config: Optional[LLMConfig] = None):
    """
    Return a LangChain-compatible chat model.

    For the Google provider, the returned model is wrapped with fallbacks so
    that if the primary model's quota is exhausted (429 RESOURCE_EXHAUSTED),
    the next model in GOOGLE_FALLBACK_MODELS is tried automatically.
    """
    from langchain_core.language_models import BaseChatModel  # type: ignore

    if config is None:
        config = LLMConfig()

    if config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in environment.")

        def _make_google_llm(model_name: str) -> ChatGoogleGenerativeAI:
            return ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                temperature=config.temperature,
                max_retries=1,  # fail fast → let fallback handle it
                timeout=30,
                max_output_tokens=2048,
            )

        primary: BaseChatModel = _make_google_llm(config.model)

        # Build fallback LLMs from every model in the list except the primary.
        fallbacks = [
            _make_google_llm(m)
            for m in GOOGLE_FALLBACK_MODELS
            if m != config.model
        ]

        if fallbacks:
            logger.info(
                "Google LLM: primary=%s, fallbacks=%s",
                config.model,
                [m for m in GOOGLE_FALLBACK_MODELS if m != config.model],
            )
            return primary.with_fallbacks(fallbacks)

        return primary

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
