import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import paths, HUMAN_REVIEW_THRESHOLD
from src.graph.state import VerificationState


def human_validation_node(state: VerificationState) -> dict:
    """
    Decide whether a human should review and, if so, queue the case.

    For the assignment this is simulated by writing to a JSONL queue.
    """
    final = state.get("final_result", {})
    overall = final.get("overall_status", "unknown")
    avg_conf = float(final.get("average_confidence", 0.0))

    needs = overall in {"unreliable", "uncertain"} or avg_conf < HUMAN_REVIEW_THRESHOLD

    if not needs:
        return {"needs_human": False, "human_feedback": "auto-approved"}

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": state.get("question"),
        "llm_answer": state.get("llm_answer"),
        "verifications": state.get("verifications", []),
        "final_result": final,
    }

    queue_path = Path(paths.HUMAN_REVIEW_QUEUE)
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"needs_human": True, "human_feedback": "queued_for_review"}
