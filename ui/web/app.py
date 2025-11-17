"""Streamlit stub demonstrating ingest and search flows."""
from __future__ import annotations

import streamlit as st

from application.use_cases.ingest_documents import ingest_documents
from application.use_cases.search import search
from infrastructure.config import build_default_container

container = build_default_container()
st.set_page_config(page_title="ContextSearch Demo")
st.title("ContextSearch Demo")

st.header("Ingest")
ingest_form = st.form("ingest")
document_id = ingest_form.text_input("Document ID", value="doc-1")
document_content = ingest_form.text_area("Content", value="Type your document here...")
ingest_submit = ingest_form.form_submit_button("Ingest document")
if ingest_submit:
    ingest_documents(
        [(document_id, document_content)],
        extractor=container.extractor,
        splitter=container.splitter,
        embedder=container.embedder,
        embedding_store=container.embedding_store,
        document_repository=container.document_repository,
    )
    st.success(f"Document {document_id} ingested")

st.header("Search")
search_query = st.text_input("Query", value="context search")
if st.button("Search"):
    results = search(
        search_query,
        embedder=container.embedder,
        embedding_store=container.embedding_store,
        document_repository=container.document_repository,
        query_rewriter=container.query_rewriter,
        reranker=container.reranker,
    )
    for result in results:
        st.write(
            {
                "document_id": result.chunk.document_id,
                "score": round(result.score, 3),
                "text": result.chunk.text,
            }
        )
