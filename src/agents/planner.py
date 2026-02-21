import json
import logging

from langchain_core.prompts import ChatPromptTemplate

from src.config import LLMConfig, get_llm, settings
from src.graph.state import VerificationState
from src.agents.utils import extract_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a planner for a hallucination-guardrail system \
verifying Indian criminal law questions (IPC ↔ BNS).

Decide:
- Whether strict verification against a trusted knowledge base is required.
- A brief natural-language plan.

Routes:
- "verify": run full pipeline (answer + claim extraction + verification).
- "direct": answer only; used for clearly non-legal or opinion questions.

Respond ONLY in JSON with keys:
- plan: string
- route: "verify" or "direct"
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "user",
            "User question: {question}\nReturn JSON.",
        ),
    ]
)


def planner_node(state: VerificationState) -> dict:
    provider = state.get("llm_provider", settings["llm"]["provider"])
    model = state.get("llm_model", settings["llm"]["model"])

    try:
        llm = get_llm(LLMConfig(provider=provider, model=model))
        # Try structured output first (works with primary model).
        try:
            chain = prompt | llm.with_structured_output(
                schema={"plan": "string", "route": "string"}
            )  # type: ignore[arg-type]
            result = chain.invoke({"question": state["question"]})
        except (AttributeError, NotImplementedError):
            # Fallback: some models/wrappers don't support with_structured_output.
            chain = prompt | llm  # type: ignore[operator]
            raw = chain.invoke({"question": state["question"]})
            text = extract_text(raw)
            # Try to parse JSON from the text response.
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                # Last resort: extract from markdown code block.
                import re
                m = re.search(r"\{[^}]+\}", text, re.DOTALL)
                result = json.loads(m.group()) if m else {"plan": text, "route": "verify"}

        route = result.get("route", "verify").lower()
        if route not in {"verify", "direct"}:
            route = "verify"

        return {"plan": result.get("plan", ""), "route": route}

    except Exception as e:
        logger.error("Planner failed: %s", e, exc_info=True)
        return {"plan": f"Planner error: {e}", "route": "verify"}
