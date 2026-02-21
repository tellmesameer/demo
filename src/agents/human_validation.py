import json
from datetime import datetime
from pathlib import Path

from src.config import paths, HUMAN_REVIEW_THRESHOLD
from src.graph.state import VerificationState


def human_validation_node(state: VerificationState) -> VerificationState:
    """
    Decide whether a human should review and, if so, queue the case.

    For the assignment this is simulated by writing to a JSONL queue.
    """
    final = state.get("final_result", {})
    overall = final.get("overall_status", "unknown")
    avg_conf = float(final.get("average_confidence", 0.0))

    needs = overall in {"unreliable", "uncertain"} or avg_conf < HUMAN_REVIEW_THRESHOLD
    state["needs_human"] = needs

    if not needs:
        state["human_feedback"] = "auto-approved"
        return state

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": state.get("question"),
        "llm_answer": state.get("llm_answer"),
        "verifications": state.get("verifications", []),
        "final_result": final,
    }

    queue_path = Path(paths.HUMAN_REVIEW_QUEUE)
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    state["human_feedback"] = "queued_for_review"
    return state
