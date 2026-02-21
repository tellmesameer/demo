import logging

from langchain_core.prompts import ChatPromptTemplate

from src.config import LLMConfig, get_llm, settings
from src.graph.state import VerificationState
from src.agents.utils import extract_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a legal assistant answering questions about \
Indian Penal Code (IPC) and Bharatiya Nyaya Sanhita (BNS).

Guidelines:
- Be concise and precise.
- Do NOT invent IPC/BNS section numbers.
- If you are unsure, explicitly say so.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "{question}"),
    ]
)


def primary_llm_node(state: VerificationState) -> dict:
    provider = state.get("llm_provider", settings["llm"]["provider"])
    model = state.get("llm_model", settings["llm"]["model"])

    try:
        llm = get_llm(LLMConfig(provider=provider, model=model))
        chain = prompt | llm  # type: ignore[operator]
        result = chain.invoke({"question": state["question"]})
        return {"llm_answer": extract_text(result)}
    except Exception as e:
        logger.error("Primary LLM failed: %s", e, exc_info=True)
        return {
            "llm_answer": f"⚠️ All models failed to generate an answer. Error: {e}"
        }
