from langgraph.graph import END, StateGraph

from src.agents.claim_extractor import claim_extractor_node
from src.agents.evaluation import evaluation_node
from src.agents.human_validation import human_validation_node
from src.agents.planner import planner_node
from src.agents.primary_llm import primary_llm_node
from src.agents.verifier import verifier_node
from src.graph.state import VerificationState


def create_workflow():
    g = StateGraph(VerificationState)

    g.add_node("planner", planner_node)
    g.add_node("primary_llm", primary_llm_node)
    g.add_node("claim_extractor", claim_extractor_node)
    g.add_node("verifier", verifier_node)
    g.add_node("human_validation", human_validation_node)
    g.add_node("evaluation", evaluation_node)

    g.set_entry_point("planner")

    # Planner → Primary LLM
    g.add_edge("planner", "primary_llm")

    def _should_verify(state: VerificationState) -> bool:
        return state.get("route", "verify") == "verify"

    # Conditional branch: verify or direct.
    g.add_conditional_edges(
        "primary_llm",
        _should_verify,
        {
            True: "claim_extractor",
            False: "evaluation",
        },
    )

    g.add_edge("claim_extractor", "verifier")
    g.add_edge("verifier", "human_validation")
    g.add_edge("human_validation", "evaluation")
    g.add_edge("evaluation", END)

    return g.compile()


def run_workflow(
    question: str,
    llm_provider: str = "google",
    llm_model: str = "gemini-2.0-flash",
):
    app = create_workflow()
    initial: VerificationState = {
        "question": question,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "metadata": {},
    }
    final_state = app.invoke(initial)
    return final_state


if __name__ == "__main__":
    q = "What is the BNS equivalent of IPC Section 302?"
    out = run_workflow(q)
    print("Question:", out["question"])
    print("Answer:", out["llm_answer"])
    print("Final:", out["final_result"])
