"""Streamlit Chat with PDF RAG application.

This single-file app parses one or more uploaded PDFs, indexes their text in a
shared FAISS vector store, and answers questions using a conversational RAG
chain backed by OpenAI.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional, Union

import fitz
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()


APP_TITLE = "Chat with PDF"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVER_K = 5
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o-mini"

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

QA_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly based on the 
provided document context. 

Rules:
- Only use information from the context below. 
- If the answer is not in the context, respond with: 
  "I couldn't find that information in the uploaded documents."
- Be concise and factual. Cite which document your answer comes from 
  when possible.
- Do not make up or infer information not present in the context.

Context: {context}
Question: {question}
Answer:"""


# ---------------------------------------------------------------------------
# Cached model resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_embeddings(api_key: str) -> OpenAIEmbeddings:
    """Initialize and cache the OpenAI embeddings model."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def initialize_session_state() -> None:
    """Create all Streamlit session-state keys used by the app."""
    defaults: dict[str, Any] = {
        "indexed_files": {},
        "removed_file_ids": set(),
        "vector_store": None,
        "chain": None,
        "chain_signature": None,
        "chat_history": [],
        "total_chunks": 0,
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_chat_history() -> None:
    """Remove all chat messages from session state."""
    st.session_state.chat_history = []


def reset_all() -> None:
    """Clear PDFs, vector store, chain, chat history, and reset the uploader."""
    st.session_state.indexed_files = {}
    st.session_state.removed_file_ids = set()
    st.session_state.vector_store = None
    st.session_state.chain = None
    st.session_state.chain_signature = None
    st.session_state.chat_history = []
    st.session_state.total_chunks = 0
    st.session_state.uploader_key += 1


# ---------------------------------------------------------------------------
# API key and document processing
# ---------------------------------------------------------------------------


def get_openai_api_key(sidebar_key: Optional[str]) -> Optional[str]:
    """Resolve the OpenAI API key from Streamlit secrets, env vars, or input."""
    def clean_key(value: Optional[str]) -> Optional[str]:
        """Return a usable API key, ignoring empty or example placeholder values."""
        if not value:
            return None

        key = value.strip()
        placeholder_fragments = (
            "your-openai-api-key",
            "your-real-key",
            "your_real_openai_api_key",
            "your-api-key",
            "sk-your",
        )
        if any(fragment in key for fragment in placeholder_fragments):
            return None

        return key

    secret_key = None
    secrets_paths = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]

    if any(path.exists() for path in secrets_paths):
        try:
            secret_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            secret_key = None

    return clean_key(secret_key) or clean_key(os.getenv("OPENAI_API_KEY")) or clean_key(sidebar_key)


def get_file_id(file_name: str, file_bytes: bytes) -> str:
    """Create a stable ID for an uploaded file from its name and contents."""
    digest = hashlib.sha256()
    digest.update(file_name.encode("utf-8"))
    digest.update(file_bytes)
    return digest.hexdigest()


def extract_pdf_pages(file_name: str, file_bytes: bytes) -> tuple[list[Document], int]:
    """Extract text page-by-page from a PDF and return LangChain documents."""
    documents: list[Document] = []

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        page_count = pdf.page_count
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "page": page_index,
                    },
                )
            )

    return documents, page_count


def split_documents(documents: list[Document]) -> list[Document]:
    """Split extracted PDF page documents into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = chunk_index

    return chunks


def rebuild_vector_store(api_key: str) -> None:
    """Rebuild the FAISS index from all currently indexed document chunks."""
    all_chunks: list[Document] = []
    for file_record in st.session_state.indexed_files.values():
        all_chunks.extend(file_record["chunks"])

    st.session_state.total_chunks = len(all_chunks)
    st.session_state.chain = None
    st.session_state.chain_signature = None

    if not all_chunks:
        st.session_state.vector_store = None
        return

    embeddings = get_embeddings(api_key)
    st.session_state.vector_store = FAISS.from_documents(all_chunks, embeddings)


def add_chunks_to_vector_store(chunks: list[Document], api_key: str) -> None:
    """Add new chunks to FAISS without rebuilding already indexed files."""
    if not chunks:
        st.session_state.total_chunks = sum(
            len(file_record["chunks"])
            for file_record in st.session_state.indexed_files.values()
        )
        return

    embeddings = get_embeddings(api_key)
    if st.session_state.vector_store is None:
        st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        st.session_state.vector_store.add_documents(chunks)

    st.session_state.total_chunks += len(chunks)
    st.session_state.chain = None
    st.session_state.chain_signature = None


def process_uploaded_files(uploaded_files: list[Any], api_key: str) -> None:
    """Parse and index newly uploaded PDFs into the shared vector store."""
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_id = get_file_id(uploaded_file.name, file_bytes)

        if (
            file_id in st.session_state.indexed_files
            or file_id in st.session_state.removed_file_ids
        ):
            continue

        try:
            page_documents, page_count = extract_pdf_pages(uploaded_file.name, file_bytes)
            chunks = split_documents(page_documents) if page_documents else []
            st.session_state.indexed_files[file_id] = {
                "name": uploaded_file.name,
                "page_count": page_count,
                "chunks": chunks,
            }
            add_chunks_to_vector_store(chunks, api_key)

            if not chunks:
                st.warning(
                    f"`{uploaded_file.name}` was uploaded, but no readable text "
                    "was found. Scanned PDFs may need OCR before indexing."
                )
        except Exception as exc:
            st.error(
                f"Could not process `{uploaded_file.name}`. The file may be "
                f"corrupted, encrypted, or unsupported. Details: {exc}"
            )


def has_indexable_chunks() -> bool:
    """Return True when at least one indexed file has chunks to embed."""
    return any(
        file_record["chunks"]
        for file_record in st.session_state.indexed_files.values()
    )


# ---------------------------------------------------------------------------
# RAG chain and chat helpers
# ---------------------------------------------------------------------------


def get_chain(api_key: str) -> Optional[ConversationalRetrievalChain]:
    """Create or reuse the conversational retrieval chain for the current index."""
    vector_store = st.session_state.vector_store
    if vector_store is None:
        return None

    signature = (api_key, id(vector_store), st.session_state.total_chunks)
    if st.session_state.chain is not None and st.session_state.chain_signature == signature:
        return st.session_state.chain

    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, openai_api_key=api_key)

    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    st.session_state.chain_signature = signature
    return st.session_state.chain


def get_chain_chat_history() -> list[tuple[str, str]]:
    """Convert displayed messages into the tuple format expected by the chain."""
    pairs: list[tuple[str, str]] = []
    pending_user_message: Optional[str] = None

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            pending_user_message = message["content"]
        elif message["role"] == "assistant" and pending_user_message is not None:
            pairs.append((pending_user_message, message["content"]))
            pending_user_message = None

    return pairs


def format_sources(source_documents: list[Document]) -> list[dict[str, Union[str, int]]]:
    """Format retrieved source documents for display under an answer."""
    sources = []
    for doc in source_documents:
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown document"),
                "page": doc.metadata.get("page", "Unknown"),
                "chunk_text": doc.page_content,
            }
        )
    return sources


def ask_question(
    question: str, api_key: str
) -> tuple[str, list[dict[str, Union[str, int]]]]:
    """Run a user question through the RAG chain and return answer plus sources."""
    chain = get_chain(api_key)
    if chain is None:
        return "Please upload at least one text-based PDF before asking a question.", []

    result = chain.invoke(
        {
            "question": question,
            "chat_history": get_chain_chat_history(),
        }
    )

    answer = result.get("answer", "").strip()
    source_documents = result.get("source_documents", [])
    return answer, format_sources(source_documents)


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------


def render_sources(sources: list[dict[str, Union[str, int]]]) -> None:
    """Render the retrieved sources for an assistant response."""
    with st.expander("View sources", expanded=False):
        if not sources:
            st.caption("No source chunks were returned for this answer.")
            return

        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**{index}. {source['source']}**  \n"
                f"Page: `{source['page']}`"
            )
            st.caption(str(source["chunk_text"]))


def render_chat_history() -> None:
    """Display all chat messages from session state."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))


def render_sidebar() -> tuple[Optional[str], list[Any]]:
    """Render sidebar controls and return the API key input plus uploads."""
    with st.sidebar:
        st.title(APP_TITLE)
        st.caption("Ask grounded questions across one or more PDF documents.")

        sidebar_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Used only if OPENAI_API_KEY is not set in Streamlit secrets or .env.",
        )

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_uploader_{st.session_state.uploader_key}",
        )

        st.divider()
        st.subheader("Uploaded PDFs")

        if st.session_state.indexed_files:
            for file_id, file_record in list(st.session_state.indexed_files.items()):
                left, right = st.columns([0.74, 0.26], vertical_alignment="center")
                with left:
                    st.markdown(
                        f"**{file_record['name']}**  \n"
                        f"{file_record['page_count']} page(s) | "
                        f"{len(file_record['chunks'])} chunk(s)"
                    )
                with right:
                    if st.button("Remove", key=f"remove_{file_id}"):
                        del st.session_state.indexed_files[file_id]
                        st.session_state.removed_file_ids.add(file_id)
                        resolved_api_key = get_openai_api_key(sidebar_api_key)
                        if resolved_api_key:
                            try:
                                rebuild_vector_store(resolved_api_key)
                            except Exception as exc:
                                st.session_state.vector_store = None
                                st.session_state.chain = None
                                st.error(f"Could not rebuild the index after removal: {exc}")
                        else:
                            st.session_state.vector_store = None
                            st.session_state.chain = None
                        st.rerun()
        else:
            st.caption("No PDFs indexed yet.")

        st.metric("Total chunks indexed", st.session_state.total_chunks)

        st.divider()
        if st.button("Clear chat history", use_container_width=True):
            clear_chat_history()
            st.rerun()

        if st.button("Reset all (clear PDFs + chat)", use_container_width=True):
            reset_all()
            st.rerun()

    return sidebar_api_key, uploaded_files or []


def main() -> None:
    """Run the Streamlit application."""
    initialize_session_state()

    sidebar_api_key, uploaded_files = render_sidebar()
    api_key = get_openai_api_key(sidebar_api_key)

    st.title(APP_TITLE)
    st.write("Upload PDFs, index their text, and ask questions grounded in the documents.")

    if not api_key:
        st.warning(
            "Add your OpenAI API key in Streamlit secrets, an OPENAI_API_KEY "
            "environment variable, or the sidebar input to use the app."
        )
        st.stop()

    if uploaded_files:
        with st.spinner("Reading and indexing uploaded PDFs..."):
            process_uploaded_files(uploaded_files, api_key)

    if st.session_state.vector_store is None and has_indexable_chunks():
        with st.spinner("Restoring the PDF index..."):
            try:
                rebuild_vector_store(api_key)
            except Exception as exc:
                st.error(f"Could not restore the PDF index: {exc}")

    if not st.session_state.indexed_files:
        st.info("Upload one or more PDFs from the sidebar to start chatting.")

    render_chat_history()

    question = st.chat_input(
        "Ask a question about your PDFs",
        disabled=st.session_state.vector_store is None,
    )

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the documents and generating an answer..."):
                try:
                    answer, sources = ask_question(question, api_key)
                except Exception as exc:
                    answer = (
                        "Sorry, I couldn't generate an answer. Please check your "
                        "API key, model access, and network connection."
                    )
                    sources = []
                    st.error(f"OpenAI or retrieval error: {exc}")

            st.markdown(answer)
            render_sources(sources)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()
