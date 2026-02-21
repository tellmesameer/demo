from dotenv import load_dotenv

import sys
from pathlib import Path

import streamlit as st
import logging
from datetime import datetime

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.config import init_data_dirs, settings  # noqa: E402
from src.graph.workflow import run_workflow  # noqa: E402

# Ensure data directories and seed files exist before anything else.
init_data_dirs()

st.set_page_config(
    page_title="Hallucination Guardrail – IPC/BNS",
    page_icon="🛡️",
    layout="wide",
)

st.title("Hallucination Guardrail Meta‑Agent (IPC ↔ BNS)")
st.markdown(
    "Meta‑agent that breaks LLM responses into claims and verifies them against "
    "a trusted IPC–BNS knowledge base (relational + vector)."
)

# Setup logging
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, settings.get("logging", {}).get("level", "INFO")),
    format=log_format,
    handlers=[
        logging.FileHandler(LOG_DIR / "user_queries.log"),
        logging.StreamHandler(),  # also print to terminal
    ],
)

with st.sidebar:
    st.header("Configuration")

    provider = st.selectbox(
        "LLM Provider",
        ["google", "offline", "openai", "anthropic"],
        index=["google", "offline", "openai", "anthropic"].index(
            settings["llm"]["provider"]
        ),
        help="Gemini (google) is recommended and free with Jio.",
    )

    default_model = settings["llm"]["model"] if provider == "google" else ""
    model = st.text_input(
        "Model name",
        value=default_model,
        help="Leave blank for provider default (except Gemini).",
    )

    st.markdown("---")
    st.markdown(
        "**Workflow**\n\n"
        "1. Planner decides whether verification is needed.\n"
        "2. Primary LLM answers the question.\n"
        "3. Claims are extracted and verified via relational + vector stores.\n"
        "4. Low‑confidence outputs are queued for human review; all runs are logged."
    )

st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "Ask a question",
        "What is the BNS equivalent of IPC Section 302?",
        height=120,
    )
    run_btn = st.button("Run Guardrail", type="primary", use_container_width=True)

with col2:
    st.subheader("Run Status")
    status_placeholder = st.empty()
    progress = st.progress(0)

if run_btn and question.strip():
    logging.info(f"User Question: {question.strip()} | Provider: {provider} | Model: {model or default_model}")
    try:
        status_placeholder.info("Running planner and primary LLM...")
        progress.progress(25)
        result = run_workflow(
            question.strip(),
            llm_provider=provider,
            llm_model=model or default_model,
        )
        progress.progress(100)
        st.session_state["last_result"] = result
        status_placeholder.success("Verification complete.")
    except Exception as e:
        status_placeholder.error(f"Error: {e}")
        st.exception(e)
        result = None
else:
    result = st.session_state.get("last_result")

if result:
    final = result.get("final_result", {})

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Primary Answer", "Verification", "Evidence / Evaluation", "Debug"]
    )

    with tab1:
        st.subheader("Primary LLM Response")
        st.markdown(result.get("llm_answer", ""))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Status", final.get("overall_status", "").upper())
        c2.metric(
            "Avg Confidence", f"{final.get('average_confidence', 0.0):.2f}"
        )
        c3.metric("Supported", final.get("supported_claims", 0))
        c4.metric("Contradicted", final.get("contradicted_claims", 0))

    with tab2:
        st.subheader("Claim‑by‑Claim Verification")
        verifications = result.get("verifications", [])
        if not verifications:
            st.info("No claims extracted (planner may have chosen DIRECT route).")
        else:
            for i, v in enumerate(verifications, start=1):
                status = v["status"]
                icon = (
                    "✅"
                    if status == "supported"
                    else "❌"
                    if status == "contradicted"
                    else "⚠️"
                )
                with st.container():
                    c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
                    c1.markdown(f"**Claim {i}:** {v['claim']}")
                    c2.markdown(f"{icon} {status.upper()}")
                    c3.write(f"{v['confidence']:.2f}")
                    c4.caption(v.get("source", ""))
                    with st.expander("View evidence"):
                        st.write(v["evidence"])
                st.divider()

    with tab3:
        st.subheader("Planner, Human Validation & Evaluation")
        st.json(
            {
                "plan": result.get("plan"),
                "route": result.get("route"),
                "needs_human": result.get("needs_human"),
                "human_feedback": result.get("human_feedback"),
                "evaluation": result.get("evaluation"),
            }
        )

    with tab4:
        st.subheader("Raw State (Debug)")
        st.json(result)
else:
    st.info("Enter a question and click **Run Guardrail** to start.")
