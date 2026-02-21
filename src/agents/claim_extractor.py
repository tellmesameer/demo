import logging
import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from src.config import LLMConfig, get_llm, settings
from src.graph.state import VerificationState
from src.agents.utils import extract_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You extract atomic factual claims from a legal answer.

Rules:
- Focus on verifiable factual statements (sections, punishments, changes).
- Each claim must be self-contained and not depend on previous sentences.
- Ignore opinions, hedging, or generic statements.

Return a numbered list of claims.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "Answer:\n{answer}\n\nExtract atomic claims."),
    ]
)


def _parse_claims(text: str) -> List[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    claims: List[str] = []
    for line in lines:
        # Strip leading numbers / bullets.
        line = re.sub(r"^[\d.\-)\s]+", "", line)
        if len(line) > 5:
            claims.append(line)
    return claims


def claim_extractor_node(state: VerificationState) -> dict:
    if state.get("route", "verify") == "direct":
        return {"claims": []}

    provider = state.get("llm_provider", settings["llm"]["provider"])
    model = state.get("llm_model", settings["llm"]["model"])

    try:
        llm = get_llm(LLMConfig(provider=provider, model=model))
        chain = prompt | llm  # type: ignore[operator]
        result = chain.invoke({"answer": state["llm_answer"]})
        content = extract_text(result)
        return {"claims": _parse_claims(content)}
    except Exception as e:
        logger.error("Claim extraction failed: %s", e, exc_info=True)
        return {"claims": []}
