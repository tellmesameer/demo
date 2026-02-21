"""
Hallucination Guardrail Meta-Agent (IPC ↔ BNS)
Streamlit entry-point — orchestrates UI components and workflow execution.
"""

from dotenv import load_dotenv

import sys
import time
import logging
from pathlib import Path

import streamlit as st

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.config import init_data_dirs, settings  # noqa: E402
from src.graph.workflow import run_workflow       # noqa: E402
from ui_components import (                       # noqa: E402
    inject_custom_css,
    render_header,
    render_config,
    render_predefined_prompts,
    render_query_input,
    render_status,
    render_loading_skeleton,
    render_metrics,
    render_tabs,
)

# ── Initialisation ──────────────────────────────────────────────────────────
init_data_dirs()

st.set_page_config(
    page_title="Hallucination Guardrail – IPC/BNS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
_log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, settings.get("logging", {}).get("level", "INFO")),
        format=_log_fmt,
        handlers=[
            logging.FileHandler(LOG_DIR / "user_queries.log"),
            logging.StreamHandler(),
        ],
    )

# ── Session state defaults ──────────────────────────────────────────────────
if "run_status" not in st.session_state:
    st.session_state["run_status"] = "idle"
if "run_step" not in st.session_state:
    st.session_state["run_step"] = 0
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_elapsed" not in st.session_state:
    st.session_state["last_elapsed"] = 0.0
if "query_input" not in st.session_state:
    st.session_state["query_input"] = "What is the BNS equivalent of IPC Section 302?"
if "selected_prompt" not in st.session_state:
    st.session_state["selected_prompt"] = None

# ── Render UI ───────────────────────────────────────────────────────────────
inject_custom_css()
render_header()

provider, model = render_config(settings)
render_predefined_prompts()
question, run_pressed = render_query_input()

# Status area (inline, no container wrapper)
render_status(None)

# ── Execute workflow (start on button press, run on subsequent rerun) ─────
if run_pressed and question.strip():
    logging.info(
        f"User Question: {question.strip()} | Provider: {provider} | Model: {model}"
    )
    # Mark running and request a rerun so the header/status updates immediately
    st.session_state["run_status"] = "running"
    st.session_state["run_step"] = 0
    st.session_state["last_result"] = None
    st.session_state["last_elapsed"] = 0.0
    st.session_state["_start_run"] = True
    st.rerun()

# Actual workflow execution happens when `_start_run` is set. This lets the
# UI render the "running" state first, then perform the long-running LLM call
# on the next script run so the status indicator is visible during execution.
if st.session_state.get("_start_run", False):
    # clear the flag immediately to avoid repeated execution
    st.session_state["_start_run"] = False
    try:
        t0 = time.perf_counter()
        result = run_workflow(
            question.strip(),
            llm_provider=provider,
            llm_model=model,
        )
        elapsed = time.perf_counter() - t0

        st.session_state["run_status"] = "success"
        st.session_state["run_step"] = 5
        st.session_state["last_result"] = result
        st.session_state["last_elapsed"] = elapsed

        logging.info(f"Response time: {elapsed:.2f}s")
        st.rerun()

    except Exception as e:
        st.session_state["run_status"] = "failed"
        logging.error(f"Workflow error: {e}")
        st.error(f"Error: {e}")
        st.exception(e)

# ── Display results ─────────────────────────────────────────────────────────
result = st.session_state.get("last_result")

if result:
    final = result.get("final_result", {})
    render_metrics(final)
    render_tabs(result)
elif st.session_state.get("run_status") == "running":
    render_loading_skeleton()
else:
    st.markdown("""
    <div class="ui-card" style="text-align:center; padding:40px;">
        <div style="font-size:1.1rem; color:#5a5e70; font-weight:500;">
            Enter a question above and click <b style="color:#4A90D9;">⏵ Run</b> to start verification
        </div>
        <div style="font-size:0.75rem; color:#3a3d4a; margin-top:8px;">
            The meta-agent will break the LLM response into claims and verify each against the IPC–BNS knowledge base.
        </div>
    </div>
    """, unsafe_allow_html=True)
