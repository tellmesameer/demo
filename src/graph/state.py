from typing import Any, Dict, List, Literal, Optional, TypedDict

StatusLabel = Literal["supported", "contradicted", "uncertain"]


class VerificationRecord(TypedDict):
    claim: str
    status: StatusLabel
    confidence: float
    evidence: str
    source: str  # "relational", "vector", or "mixed"


class VerificationState(TypedDict, total=False):
    # Input
    question: str
    llm_provider: str
    llm_model: str

    # Planner output
    plan: str
    route: Literal["direct", "verify"]

    # Primary LLM output
    llm_answer: str

    # Claims
    claims: List[str]

    # Verification
    verifications: List[VerificationRecord]
    final_result: Dict[str, Any]

    # Human validation
    needs_human: bool
    human_feedback: Optional[str]

    # Evaluation
    evaluation: Dict[str, Any]

    # Misc
    metadata: Dict[str, Any]
