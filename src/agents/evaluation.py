import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import paths
from src.graph.state import VerificationState


def evaluation_node(state: VerificationState) -> dict:
    """
    Log run metadata for offline evaluation (precision/recall, latency, etc.).
    """
    final = state.get("final_result", {})
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": state.get("question"),
        "plan": state.get("plan"),
        "route": state.get("route"),
        "overall_status": final.get("overall_status"),
        "average_confidence": final.get("average_confidence"),
        "counts": {
            "supported": final.get("supported_claims"),
            "contradicted": final.get("contradicted_claims"),
            "uncertain": final.get("uncertain_claims"),
            "total": final.get("total_claims"),
        },
    }

    p = Path(paths.EVAL_LOG)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

    return {"evaluation": log}
