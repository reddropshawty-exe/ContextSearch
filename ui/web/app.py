"""Простой Streamlit UI для демонстрации сценариев ContextSearch."""
from __future__ import annotations

import streamlit as st

from application.use_cases.ingest_documents import ingest_documents
from application.use_cases.search import search
from infrastructure.config import build_default_container
from ui.logging_utils import setup_logging

setup_logging()

st.set_page_config(page_title="ContextSearch Demo", layout="wide")
st.title("ContextSearch (демо)")

container = build_default_container()

st.header("Индексация")
ingest_id = st.text_input("ID документа", value="doc-1")
ingest_text = st.text_area("Текст документа", value="Это тестовый документ про контекстный поиск.")
if st.button("Индексировать документ"):
    ingest_documents(
        [(ingest_id, ingest_text)],
        extractor=container.extractor,
        splitter=container.splitter,
        embedders=container.embedders,
        embedding_store=container.embedding_store,
        document_repository=container.document_repository,
        chunk_repository=container.chunk_repository,
        embedding_specs=container.embedding_specs,
        bm25_index=container.bm25_index,
    )
    st.success("Документ проиндексирован")

st.header("Поиск")
search_query = st.text_input("Запрос", value="контекстный поиск")
ranking_mode = st.selectbox("Метод ранжирования", ["rrf", "vector", "bm25", "combsum", "combmnz"], index=0)
top_k = st.number_input("Количество документов (k)", min_value=1, max_value=100, value=10, step=1)
vector_weight = st.number_input("Вес vector", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
bm25_weight = st.number_input("Вес BM25", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

if st.button("Найти"):
    results = search(
        search_query,
        embedders=container.embedders,
        embedding_store=container.embedding_store,
        document_repository=container.document_repository,
        chunk_repository=container.chunk_repository,
        embedding_record_repository=container.embedding_record_repository,
        embedding_specs=container.embedding_specs,
        bm25_index=container.bm25_index,
        query_rewriter=container.query_rewriter,
        reranker=container.reranker,
        top_k=int(top_k),
        ranking_mode=ranking_mode,
        use_bm25=ranking_mode in {"bm25", "rrf", "combsum", "combmnz"},
        vector_weight=float(vector_weight),
        bm25_weight=float(bm25_weight),
    )
    for result in results:
        chunk_score = result.metadata.get("chunk_score", result.score)
        document_score = result.metadata.get("document_vector_score")
        bm25_score = result.metadata.get("bm25_score")
        st.write(
            {
                "документ": result.document_id,
                "rank_score": round(result.score, 3),
                "chunk_score": round(chunk_score, 3),
                "document_score": round(document_score, 3) if document_score is not None else None,
                "bm25_score": round(bm25_score, 3) if bm25_score is not None else None,
                "document_weight": result.metadata.get("document_weight"),
                "текст": result.chunk.text if result.chunk else None,
            }
        )

st.header("Индексированные документы")
if st.button("Обновить список документов"):
    st.session_state["documents"] = container.document_repository.list()

documents = st.session_state.get("documents", [])
if documents:
    document_spec = next((spec for spec in container.embedding_specs if spec.level == "document"), None)
    if document_spec is not None:
        indexed_ids = set(
            container.embedding_record_repository.list_object_ids(document_spec.id, "document")
        )
        documents = [doc for doc in documents if doc.id in indexed_ids]
    for doc in documents:
        st.write(
            {
                "id": doc.id,
                "название": doc.title or doc.metadata.get("display_name"),
                "источник": doc.path or doc.metadata.get("source_uri"),
            }
        )
else:
    st.caption("Список пока пуст. Нажмите «Обновить список документов» после индексации.")
