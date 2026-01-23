"""Streamlit-заглушка для демонстрации индексации и поиска."""
from __future__ import annotations

import streamlit as st

from application.use_cases.ingest_documents import ingest_documents
from application.use_cases.search import search
from infrastructure.config import build_default_container

container = build_default_container()
st.set_page_config(page_title="ContextSearch Демо")
st.title("ContextSearch Демо")

st.header("Индексация")
ingest_form = st.form("ingest")
document_content = ingest_form.text_area("Содержимое", value="Введите текст документа...")
ingest_submit = ingest_form.form_submit_button("Индексировать документ")
if ingest_submit:
    ingest_documents(
        [(None, document_content)],
        extractor=container.extractor,
        splitter=container.splitter,
        embedder=container.embedder,
        embedding_store=container.embedding_store,
        document_repository=container.document_repository,
    )
    st.success("Документ проиндексирован")

st.header("Поиск")
search_query = st.text_input("Запрос", value="контекстный поиск")
if st.button("Найти"):
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
                "документ": result.chunk.document_id,
                "оценка": round(result.score, 3),
                "текст": result.chunk.text,
            }
        )
