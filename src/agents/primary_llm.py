from langchain_core.prompts import ChatPromptTemplate

from src.config import LLMConfig, get_llm
from src.graph.state import VerificationState

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


def primary_llm_node(state: VerificationState) -> VerificationState:
    provider = state.get("llm_provider", "google")
    model = state.get("llm_model", "gemini-1.5-flash")
    llm = get_llm(LLMConfig(provider=provider, model=model))

    chain = prompt | llm  # type: ignore[operator]
    result = chain.invoke({"question": state["question"]})
    content = getattr(result, "content", None) or str(result)
    state["llm_answer"] = content
    return state
