"""Streamlit UI for the HDB RAG chatbot."""
from __future__ import annotations

import os

import streamlit as st

# Streamlit Cloud puts deploy-time secrets into st.secrets (TOML), not os.environ.
# Bridge them into os.environ before importing config, which reads via os.getenv.
# setdefault keeps local dev (.env via python-dotenv) untouched.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # No secrets.toml locally — fall back to .env

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from hdb_rag import config
from hdb_rag.chain import build_chain, dedupe_sources
from hdb_rag.retrieval.factory import build_retriever
from hdb_rag.stores import build_embedder, build_vector_store


# --- Page config ---
st.set_page_config(page_title="HDB Rules Assistant", page_icon="🏠", layout="wide")


# --- Cached resource builders ---
@st.cache_resource
def _store():
    return build_vector_store(build_embedder())


@st.cache_resource
def _llms():
    answer = ChatGoogleGenerativeAI(model=config.ANSWER_MODEL, google_api_key=config.GOOGLE_API_KEY)
    fast = ChatGoogleGenerativeAI(model=config.FAST_MODEL, google_api_key=config.GOOGLE_API_KEY)
    return answer, fast


@st.cache_resource
def _chain(mode: str):
    answer_llm, fast_llm = _llms()
    retriever = build_retriever(mode=mode, store=_store(), fast_llm=fast_llm)
    return build_chain(retriever=retriever, answer_llm=answer_llm, fast_llm=fast_llm)


# --- Sidebar ---
chunk_cache_available = config.CHUNKS_JSONL.exists()
retrieval_modes = ["best", "fast"] if chunk_cache_available else ["fast"]

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Retrieval mode",
        retrieval_modes,
        index=0,
        help="best = hybrid + multi-query + rerank. fast = vector only.",
    )
    if not chunk_cache_available:
        st.warning(
            "Best mode unavailable because `data/chunks.jsonl` is missing. "
            "Run `make ingest` and deploy the generated cache."
        )
    st.divider()
    st.caption(f"Answer LLM: `{config.ANSWER_MODEL}`")
    st.caption(f"Fast LLM:   `{config.FAST_MODEL}`")
    st.caption(f"Embedder:   `{config.EMBEDDING_MODEL}`")
    st.caption(f"Vector DB:  `{config.VECTOR_STORE}`")
    st.divider()
    st.markdown(
        "[GitHub](https://github.com/MoeMoePanda/lang_chain_rag) · "
        "[LangSmith](https://smith.langchain.com/)"
    )


# --- Disclaimer banner ---
ingested_at = (
    config.INGESTED_AT.read_text().strip() if config.INGESTED_AT.exists() else "unknown"
)
st.warning(
    f"⚠️ **Informational only.** Verify on [hdb.gov.sg](https://www.hdb.gov.sg) "
    f"before any decisions. Knowledge last indexed: `{ingested_at}` · "
    f"Scope: Buying + Selling + Living rules."
)


# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources?}

# Empty-state seed prompts
if not st.session_state.messages:
    st.info(
        "Try asking: *Am I eligible for a BTO if I'm 32 and single?* · "
        "*How long is the MOP?* · *Can I rent out my flat after MOP?*"
    )

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})")


# --- Input ---
prompt = st.chat_input("Ask about HDB buying, selling, or living rules…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat_history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        try:
            chain = _chain(mode)
            placeholder = st.empty()
            full_text = ""
            final_state: dict | None = None
            for chunk in chain.stream({"question": prompt, "chat_history": chat_history}):
                if "answer" in chunk and isinstance(chunk["answer"], str):
                    full_text += chunk["answer"]
                    placeholder.markdown(full_text + "▌")
                final_state = {**(final_state or {}), **chunk}
            placeholder.markdown(full_text)

            sources = (
                dedupe_sources(final_state.get("context", [])) if final_state else []
            )
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for s in sources:
                        st.markdown(f"- [{s['title']}]({s['url']})")

            st.session_state.messages.append(
                {"role": "assistant", "content": full_text, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
