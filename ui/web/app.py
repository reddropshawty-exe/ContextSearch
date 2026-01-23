"""Streamlit-заглушка для демонстрации индексации и поиска."""
from __future__ import annotations

import streamlit as st

from application.use_cases.ingest_documents import ingest_documents
from application.use_cases.search import search
from infrastructure.config import build_default_container
from ui.logging_utils import setup_logging

setup_logging()


@st.cache_resource
def get_container():
    return build_default_container()


container = get_container()

if "documents" not in st.session_state:
    st.session_state["documents"] = []
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

st.header("Индексированные документы")
if st.button("Обновить список документов"):
    st.session_state["documents"] = container.document_repository.list()

documents = st.session_state.get("documents", [])
if documents:
    for doc in documents:
        st.write(
            {
                "id": doc.id,
                "название": doc.metadata.get("display_name"),
                "источник": doc.metadata.get("source_uri"),
            }
        )
else:
    st.caption("Список пока пуст. Нажмите «Обновить список документов» после индексации.")
