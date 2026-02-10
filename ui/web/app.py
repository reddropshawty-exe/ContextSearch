"""Streamlit-заглушка для демонстрации индексации и поиска."""
from __future__ import annotations

import streamlit as st

from application.use_cases.ingest_documents import ingest_documents
from application.use_cases.search import search
from infrastructure.config import ContainerConfig, build_default_container
from ui.logging_utils import setup_logging

setup_logging()


@st.cache_resource
def get_container(safe_mode: bool, embedding_store: str):
    return build_default_container(ContainerConfig(safe_mode=safe_mode, embedding_store=embedding_store))


if "documents" not in st.session_state:
    st.session_state["documents"] = []
safe_mode = st.toggle("Безопасный режим (без HNSW/torch)", value=False)
embedding_store = st.selectbox("Хранилище эмбеддингов", ["sqlite", "in_memory", "hnsw"], index=0)
container = get_container(safe_mode, embedding_store)
store_label = {"sqlite": "sqlite", "in_memory": "in_memory", "hnsw": "hnsw"}.get(embedding_store, embedding_store)
st.set_page_config(page_title="ContextSearch Демо")
st.title("ContextSearch Демо")
st.caption(
    "Активная конфигурация: "
    f"эмбеддер={container.embedder.model_id}, хранилище={store_label}, "
    f"specs={len(container.embedding_specs)}"
)

st.header("Индексация")
ingest_form = st.form("ingest")
document_content = ingest_form.text_area("Содержимое", value="Введите текст документа...")
ingest_submit = ingest_form.form_submit_button("Индексировать документ")
if ingest_submit:
    ingest_documents(
        [(None, document_content)],
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
    )
    for result in results:
        chunk_score = result.metadata.get("chunk_score", result.score)
        document_score = result.metadata.get("document_vector_score")
        bm25_score = result.metadata.get("bm25_score")
        st.write(
            {
                "документ": result.document_id,
                "chunk_score": round(chunk_score, 3),
                "document_score": round(document_score, 3) if document_score is not None else None,
                "bm25_score": round(bm25_score, 3) if bm25_score is not None else None,
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
