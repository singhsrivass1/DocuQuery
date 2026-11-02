"""Streamlit interface for DocuQuery: upload technical PDFs and ask cited questions."""

import html
import logging
from typing import List, Optional, Tuple, TypedDict

import streamlit as st
from langchain_core.embeddings import Embeddings
from streamlit.runtime.uploaded_file_manager import UploadedFile

import config
from rag_engine import DocumentProcessingError, QueryResult, RAGEngine, SourceCitation, create_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("docuquery.app")

AVATARS = {"user": ":material/person:", "assistant": ":material/description:"}

PAGE_STYLE = """
<style>
.dq-card {
    border: 1px solid rgb(229, 229, 229);
    border-radius: 8px;
    background: rgb(255, 255, 255);
    color: rgb(17, 17, 17);
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
}
.dq-card-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.35rem;
}
.dq-card p, .dq-card li {
    font-size: 0.9rem;
    line-height: 1.55;
    margin: 0.2rem 0;
}
.dq-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    padding: 0.35rem 0;
    font-size: 0.875rem;
}
.dq-row + .dq-row {
    border-top: 1px solid rgb(240, 240, 240);
}
.dq-label {
    color: rgb(85, 85, 85);
}
.dq-value {
    font-weight: 600;
    text-align: right;
}
.dq-dot {
    display: inline-block;
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    border: 1.5px solid rgb(17, 17, 17);
    margin-right: 0.4rem;
    vertical-align: middle;
}
.dq-dot-on {
    background: rgb(17, 17, 17);
}
.dq-badge {
    display: inline-block;
    white-space: nowrap;
    border: 1px solid rgb(214, 214, 214);
    border-radius: 999px;
    background: rgb(247, 247, 247);
    color: rgb(17, 17, 17);
    padding: 0.15rem 0.65rem;
    font-size: 0.75rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote {
    color: rgb(17, 17, 17);
    opacity: 1;
    border-left: 3px solid rgb(214, 214, 214);
}
.dq-source-meta {
    font-size: 0.8rem;
    color: rgb(85, 85, 85);
    margin-bottom: 0.45rem;
}
.dq-source-meta strong {
    color: rgb(17, 17, 17);
}
.dq-snippet {
    font-size: 0.875rem;
    line-height: 1.6;
    color: rgb(17, 17, 17);
}
</style>
"""


class ChatMessage(TypedDict):
    role: str
    content: str
    sources: List[SourceCitation]


@st.cache_resource(show_spinner="Loading the embedding model. The first run downloads it once.")
def load_embeddings() -> Embeddings:
    return create_embeddings(config.EMBEDDING_MODEL_NAME)


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "engine": None,
        "index_signature": "",
        "indexed_files": [],
        "index_errors": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_conversation() -> None:
    st.session_state.messages = []


def card(body_html: str, title: Optional[str] = None) -> None:
    title_html = f'<div class="dq-card-title">{html.escape(title)}</div>' if title else ""
    st.markdown(f'<div class="dq-card">{title_html}{body_html}</div>', unsafe_allow_html=True)


def as_html_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def build_index(files: List[UploadedFile]) -> None:
    engine = RAGEngine(embeddings=load_embeddings())
    documents = []
    indexed_files: List[str] = []
    errors: List[str] = []
    for uploaded in files:
        try:
            documents.extend(engine.load_pdf(uploaded.getvalue(), source_name=uploaded.name))
            indexed_files.append(uploaded.name)
        except DocumentProcessingError as error:
            errors.append(str(error))
    if documents:
        engine.build_vector_store(engine.chunk_text(documents))
    st.session_state.engine = engine if engine.is_ready else None
    st.session_state.indexed_files = indexed_files
    st.session_state.index_errors = errors


def sync_index(files: List[UploadedFile]) -> None:
    signature = "|".join(sorted(uploaded.file_id for uploaded in files))
    if signature == st.session_state.index_signature:
        return
    st.session_state.index_signature = signature
    st.session_state.engine = None
    st.session_state.indexed_files = []
    st.session_state.index_errors = []
    if not files:
        return
    with st.spinner(f"Indexing {len(files)} PDF file(s)..."):
        try:
            build_index(files)
        except Exception as error:
            logger.exception("Indexing failed")
            st.session_state.index_errors = [f"Indexing failed: {error}"]


def resolve_api_key(typed_key: str) -> str:
    return typed_key.strip() or config.GROQ_API_KEY


def render_status(api_key: str) -> None:
    engine: Optional[RAGEngine] = st.session_state.engine
    ready = engine is not None
    status = (
        '<span class="dq-dot dq-dot-on"></span>Ready'
        if ready
        else '<span class="dq-dot"></span>Not built'
    )
    rows: List[Tuple[str, str]] = [
        ("Index status", status),
        ("Documents", str(len(st.session_state.indexed_files))),
        ("Indexed chunks", str(engine.chunk_count if engine else 0)),
        ("Passages per answer", str(config.TOP_K)),
        ("Answer mode", html.escape(config.DEFAULT_LLM_MODEL) if api_key else "Local fallback"),
    ]
    rows_html = "".join(
        f'<div class="dq-row"><span class="dq-label">{label}</span><span class="dq-value">{value}</span></div>'
        for label, value in rows
    )
    model_name = html.escape(config.EMBEDDING_MODEL_NAME)
    badge = (
        f'<span class="dq-badge" title="{model_name}">'
        f'{html.escape(config.EMBEDDING_MODEL_NAME.split("/")[-1])} · {config.EMBEDDING_DIMENSION} dim</span>'
    )
    card(rows_html, title="Status")
    card(badge, title="Embedding model")


def render_sidebar() -> str:
    with st.sidebar:
        st.subheader("Documents")
        files = st.file_uploader(
            "Upload technical PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="C++ standard drafts, API references, university syllabi, or any PDF with selectable text.",
        )
        sync_index(files or [])
        for error in st.session_state.index_errors:
            card(f"<p>{as_html_text(error)}</p>", title="Could not index a file")

        st.subheader("Answer generation")
        typed_key = st.text_input(
            "Groq API key (optional)",
            type="password",
            placeholder="gsk_...",
            help="The key stays in this browser session and is only sent to Groq when you ask a question.",
        )
        api_key = resolve_api_key(typed_key)
        if typed_key.strip():
            st.caption(f"Answers are written by {config.DEFAULT_LLM_MODEL} from the retrieved passages.")
        elif api_key:
            st.caption(f"Using the GROQ_API_KEY from your environment with {config.DEFAULT_LLM_MODEL}.")
        else:
            st.caption("Without a key, DocuQuery shows the most relevant passage verbatim.")

        render_status(api_key)
        st.button("Clear conversation", on_click=clear_conversation, width="stretch")
    return api_key


def render_sources(sources: List[SourceCitation]) -> None:
    with st.expander("📌 Source Passages"):
        for position, source in enumerate(sources, start=1):
            meta = (
                f"<strong>Source {position}</strong> · Page {source['page_number']} · "
                f"Chunk {source['chunk_id']} · Relevance {source['score']:.2f}"
            )
            if source["source"]:
                meta += f" · {html.escape(source['source'])}"
            st.markdown(
                f'<div class="dq-card"><div class="dq-source-meta">{meta}</div>'
                f'<div class="dq-snippet">{as_html_text(source["snippet"])}</div></div>',
                unsafe_allow_html=True,
            )


def render_message(message: ChatMessage) -> None:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])
        if message["sources"]:
            render_sources(message["sources"])


def render_empty_state() -> None:
    card(
        "<p>1. Upload one or more PDFs with selectable text in the sidebar.</p>"
        f"<p>2. DocuQuery splits them into {config.CHUNK_SIZE} character passages and embeds them "
        f"with {html.escape(config.EMBEDDING_MODEL_NAME.split('/')[-1])}.</p>"
        "<p>3. Ask a question. Every answer lists the page numbers and passages it was drawn from.</p>",
        title="Get started",
    )


def answer_question(engine: RAGEngine, question: str, api_key: str) -> QueryResult:
    try:
        return engine.query(question, api_key=api_key)
    except Exception as error:
        logger.exception("Query failed")
        return QueryResult(
            answer=f"Something went wrong while searching your documents ({type(error).__name__}). Please try again.",
            sources=[],
        )


def main() -> None:
    st.set_page_config(page_title="DocuQuery", page_icon=":material/description:", layout="centered")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    init_session_state()

    api_key = render_sidebar()
    engine: Optional[RAGEngine] = st.session_state.engine

    st.title("DocuQuery")
    st.caption("Transformer based RAG assistant for technical documents, with a cited source for every answer.")

    if engine is None and not st.session_state.messages:
        render_empty_state()

    for message in st.session_state.messages:
        render_message(message)

    submitted = st.chat_input(
        "Ask a question about your documents" if engine else "Upload a PDF to start asking questions",
        disabled=engine is None,
    )
    question = (submitted or "").strip()
    if question and engine is not None:
        user_message = ChatMessage(role="user", content=question, sources=[])
        st.session_state.messages.append(user_message)
        render_message(user_message)

        with st.spinner("Searching your documents..."):
            result = answer_question(engine, question, api_key)
        assistant_message = ChatMessage(role="assistant", content=result["answer"], sources=result["sources"])
        st.session_state.messages.append(assistant_message)
        render_message(assistant_message)


if __name__ == "__main__":
    main()
