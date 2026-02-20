import re
from typing import Dict, List

from src.config import paths
from src.graph.state import VerificationRecord, VerificationState
from src.rag.vectorstore import IPCBNSRelationalStore, IPCBNSVectorStore

SECTION_RE = re.compile(r"(?:IPC|BNS)?\s*Section\s*(\d+[A-Z]?)", re.IGNORECASE)


def _extract_sections(text: str) -> List[str]:
    return list({m.group(1) for m in SECTION_RE.finditer(text)})


def _score_relational(claim: str, rel: IPCBNSRelationalStore) -> Dict[str, object]:
    secs = _extract_sections(claim)
    if not secs:
        return {
            "status": "uncertain",
            "confidence": 0.0,
            "evidence": "",
            "source": "relational",
        }

    ipc = secs[0]
    record = rel.get_by_ipc(ipc)
    if not record:
        return {
            "status": "uncertain",
            "confidence": 0.4,
            "evidence": "No mapping found in IPC↔BNS table.",
            "source": "relational",
        }

    bns = record["bns_section"]
    notes = record.get("notes", "")
    ok = (bns in claim) or (f"BNS {bns}" in claim)
    status = "supported" if ok else "contradicted"
    conf = 0.9 if ok else 0.7
    evidence = f"IPC {ipc} → BNS {bns}. {notes}"
    return {
        "status": status,
        "confidence": conf,
        "evidence": evidence,
        "source": "relational",
    }


def _score_vector(claim: str, vec: IPCBNSVectorStore) -> Dict[str, object]:
    results = vec.query(claim, k=3)
    if not results:
        return {
            "status": "uncertain",
            "confidence": 0.0,
            "evidence": "No semantic evidence.",
            "source": "vector",
        }

    best_text, meta, dist = results[0]
    # Convert distance to a crude similarity.
    sim = max(0.0, min(1.0, 1.0 - dist))

    if sim >= 0.75:
        status = "supported"
    elif sim <= 0.45:
        status = "contradicted"
    else:
        status = "uncertain"

    return {
        "status": status,
        "confidence": sim,
        "evidence": best_text[:500],
        "source": "vector",
    }


def _fuse(rel: Dict[str, object], vec: Dict[str, object]) -> VerificationRecord:
    if rel["status"] != "uncertain":
        base_status = str(rel["status"])
        base_conf = float(rel["confidence"])
        vec_status = str(vec["status"])
        vec_conf = float(vec["confidence"])

        if base_status == "supported" and vec_status == "contradicted" and vec_conf > 0.7:
            status = "uncertain"
            conf = (base_conf + vec_conf) / 2
        else:
            status = base_status
            conf = max(base_conf, vec_conf)

        evidence = f"{rel['evidence']}\n\nVector evidence:\n{vec['evidence']}"
        source = "mixed"
    else:
        status = str(vec["status"])
        conf = float(vec["confidence"])
        evidence = str(vec["evidence"])
        source = "vector"

    return {
        "claim": "",
        "status": status,  # type: ignore[assignment]
        "confidence": float(round(conf, 3)),
        "evidence": evidence,
        "source": source,
    }


def verifier_node(state: VerificationState) -> VerificationState:
    if state.get("route", "verify") == "direct":
        state["verifications"] = []
        state["final_result"] = {
            "overall_status": "direct_answer",
            "average_confidence": 1.0,
            "supported_claims": 0,
            "contradicted_claims": 0,
            "uncertain_claims": 0,
            "total_claims": 0,
        }
        return state

    claims = state.get("claims", [])
    rel = IPCBNSRelationalStore(paths.SQLITE_DB)
    vec = IPCBNSVectorStore()

    verifications: List[VerificationRecord] = []

    for claim in claims:
        rel_score = _score_relational(claim, rel)
        vec_score = _score_vector(claim, vec)
        fused = _fuse(rel_score, vec_score)
        fused["claim"] = claim
        verifications.append(fused)

    state["verifications"] = verifications

    supported = sum(1 for v in verifications if v["status"] == "supported")
    contradicted = sum(1 for v in verifications if v["status"] == "contradicted")
    uncertain = sum(1 for v in verifications if v["status"] == "uncertain")
    total = len(verifications)
    avg_conf = sum(v["confidence"] for v in verifications) / total if total else 0.0

    if total == 0:
        overall = "no_claims"
    elif contradicted > 0:
        overall = "unreliable"
    elif supported / max(total, 1) >= 0.7 and avg_conf >= 0.75:
        overall = "reliable"
    else:
        overall = "uncertain"

    state["final_result"] = {
        "overall_status": overall,
        "average_confidence": float(round(avg_conf, 3)),
        "supported_claims": supported,
        "contradicted_claims": contradicted,
        "uncertain_claims": uncertain,
        "total_claims": total,
    }
    return state
