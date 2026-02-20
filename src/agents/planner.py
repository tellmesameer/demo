from langchain_core.prompts import ChatPromptTemplate

from src.config import LLMConfig, get_llm
from src.graph.state import VerificationState

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


def planner_node(state: VerificationState) -> VerificationState:
    llm = get_llm(LLMConfig())
    chain = prompt | llm.with_structured_output(schema={"plan": "string", "route": "string"})  # type: ignore[arg-type]
    result = chain.invoke({"question": state["question"]})

    route = result.get("route", "verify").lower()
    if route not in {"verify", "direct"}:
        route = "verify"

    state["plan"] = result.get("plan", "")
    state["route"] = route  # type: ignore[assignment]
    return state
