"""
Document Intelligence RAG Assistant: Streamlit interface.

Workflow
  Upload -> load_document -> split_pages -> embed_texts -> vector_store.add_chunks
  Ask    -> retrieve (embed question + similarity search) -> generate_answer (Groq) -> sources
"""

from dotenv import load_dotenv

# Load .env BEFORE importing our modules' settings are read (they read at call time anyway).
load_dotenv()

import streamlit as st  # noqa: E402

from src import vector_store  # noqa: E402
from src.document_loader import DocumentLoadError, load_document  # noqa: E402
from src.embeddings import EmbeddingError, embed_texts  # noqa: E402
from src.llm import (  # noqa: E402
    LLMError,
    find_citations,
    generate_answer,
    is_insufficient,
    missing_config,
)
from src.retriever import RetrievedChunk, default_top_k, retrieve  # noqa: E402
from src.text_splitter import get_chunk_settings, split_pages  # noqa: E402
from src.vector_store import VectorStoreError, document_exists, file_hash  # noqa: E402

st.set_page_config(page_title="Document Intelligence RAG Assistant", page_icon="📄", layout="wide")

# ---------------------------------------------------------------- session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history (display only; each question is independent)
if "index_results" not in st.session_state:
    st.session_state.index_results = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # changing this resets the file uploader widget


# ---------------------------------------------------------------- helpers
def index_files(uploaded_files) -> list[dict]:
    """Run the indexing pipeline for each uploaded file and report per-file status."""
    results = []
    progress = st.progress(0.0, text="Starting...")
    for i, uploaded in enumerate(uploaded_files):
        name = uploaded.name
        progress.progress(i / len(uploaded_files), text=f"Processing {name}...")
        try:
            data = uploaded.getvalue()
            doc_hash = file_hash(data)
            if document_exists(doc_hash):  # same content already indexed
                results.append({"File": name, "Status": "Skipped", "Details": "Already indexed"})
                continue

            pages = load_document(name, data)
            chunk_size, overlap = get_chunk_settings()
            chunks = split_pages(pages, chunk_size, overlap)
            if not chunks:
                raise DocumentLoadError(f"'{name}' produced no text chunks.")
            vectors = embed_texts([c.text for c in chunks])
            vector_store.add_chunks(chunks, vectors, doc_hash)
            results.append(
                {
                    "File": name,
                    "Status": "Indexed",
                    "Details": f"{len(pages)} page(s)/section(s), {len(chunks)} chunks",
                }
            )
        except (DocumentLoadError, EmbeddingError, VectorStoreError, ValueError) as exc:
            results.append({"File": name, "Status": "Failed", "Details": str(exc)})
        except Exception as exc:  # last resort so one bad file never crashes the app
            results.append({"File": name, "Status": "Failed", "Details": f"Unexpected error: {exc}"})
    progress.progress(1.0, text="Done")
    return results


def render_sources(chunks: list[RetrievedChunk], answer: str) -> None:
    """Show citations (mapped to real retrieved chunks) and the full retrieved context."""
    if not chunks:
        return
    cited, invalid = find_citations(answer, len(chunks))

    if cited and not is_insufficient(answer):
        st.markdown("**Sources cited in the answer**")
        for number in cited:
            c = chunks[number - 1]
            st.markdown(f"- [{number}] **{c.file_name}**, {c.page_label}")
        if invalid:
            st.warning(
                f"The model cited source number(s) {invalid} that were not retrieved. "
                "Ignore those citations."
            )
        if any(c.page_label.startswith("Section") for c in chunks):
            st.caption("DOCX files have no fixed pages, so they are cited by section.")
    elif not is_insufficient(answer):
        st.caption("The model did not cite specific passages. See the retrieved context below.")

    with st.expander(f"Retrieved context ({len(chunks)} passages)"):
        cited_set = set(cited)
        for number, c in enumerate(chunks, start=1):
            tag = " (cited)" if number in cited_set else ""
            st.markdown(
                f"**[{number}] {c.file_name}, {c.page_label}**{tag}  \n"
                f"similarity: `{c.similarity:.3f}`"
            )
            st.text(c.text)
            if number < len(chunks):
                st.divider()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Documents")

    missing = missing_config()
    if missing:
        st.warning(f"Missing in .env: {', '.join(missing)}. Indexing works, but answers need these.")

    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if st.button("Index documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Indexing... the first run downloads the embedding model."):
            st.session_state.index_results = index_files(uploaded_files)

    if st.session_state.index_results:
        st.subheader("Indexing status")
        st.table(st.session_state.index_results)

    st.subheader("Indexed documents")
    try:
        documents = vector_store.list_documents()
    except VectorStoreError as exc:
        documents = []
        st.error(str(exc))
    if documents:
        for doc in documents:
            st.markdown(f"- **{doc['file_name']}** ({doc['chunks']} chunk(s))")
    else:
        st.caption("No documents indexed yet.")

    st.subheader("Settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, min(max(default_top_k(), 1), 10))

    st.divider()
    if st.button("Clear all documents", disabled=not documents):
        try:
            vector_store.clear_all()
            st.session_state.messages = []
            st.session_state.index_results = []
            st.session_state.uploader_key += 1
            st.rerun()
        except VectorStoreError as exc:
            st.error(str(exc))

# ---------------------------------------------------------------- main page
st.title("📄 Document Intelligence RAG Assistant")
st.write(
    "Upload PDF and DOCX files, then ask questions. Answers are generated only from "
    "your documents, with the source passages shown so you can verify them."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("chunks", []), message["content"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chunks: list[RetrievedChunk] = []
        if not documents:
            answer = "Please upload and index at least one document first."
        else:
            try:
                with st.spinner("Searching documents and generating answer..."):
                    chunks = retrieve(question, top_k)
                    answer = generate_answer(question, chunks)
            except (LLMError, EmbeddingError, VectorStoreError, ValueError) as exc:
                answer = f"Error: {exc}"
            except Exception as exc:
                answer = f"Unexpected error: {exc}"
        st.markdown(answer)
        render_sources(chunks, answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})
