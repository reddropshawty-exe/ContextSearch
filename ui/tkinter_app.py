"""Tkinter-интерфейс для локальной индексации и поиска."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tkinter import (
    END,
    LEFT,
    BooleanVar,
    Button,
    Checkbutton,
    Entry,
    Frame,
    Label,
    Listbox,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)
from tkinter.ttk import Combobox

from application.use_cases.ingest_paths import ingest_paths
from application.use_cases.search import search
from domain.entities import Document, RetrievalResult
from infrastructure.config import ContainerConfig, build_default_container
from ui.logging_utils import setup_logging


class ContextSearchApp:
    def __init__(self) -> None:
        setup_logging()
        self.root = Tk()
        self.root.title("ContextSearch")
        self.paths: list[Path] = []
        self.results: list[dict[str, str]] = []
        self._container_cache: tuple[tuple[str, str, str], object] | None = None
        self._documents_cache: list[Document] = []
        self._search_results_cache: list[RetrievalResult] = []
        self._result_mode = "documents"

        self.embedder_var = StringVar(value="all-minilm")
        self.hnsw_var = BooleanVar(value=False)
        self.llm_rewrite_var = BooleanVar(value=False)
        self.bm25_var = BooleanVar(value=True)
        self.status_var = StringVar(value="Конфигурация: не выбрана")
        self.spec_meta_var = StringVar(value="Размерность: -, Метрика: -, Проиндексировано: 0")

        self._build_layout()

    def _build_layout(self) -> None:
        settings = Frame(self.root)
        settings.pack(padx=10, pady=10)

        Label(settings, text="Модель").pack(side=LEFT)
        Combobox(
            settings,
            textvariable=self.embedder_var,
            values=["all-minilm", "all-mpnet", "multilingual-e5-base", "embedding-gemma"],
            width=22,
        ).pack(side=LEFT, padx=5)
        Checkbutton(settings, text="HNSW", variable=self.hnsw_var).pack(side=LEFT, padx=5)
        Button(settings, text="Обновить документы", command=self.refresh_documents).pack(side=LEFT, padx=5)

        Label(self.root, textvariable=self.spec_meta_var).pack(pady=3)

        controls = Frame(self.root)
        controls.pack(padx=10, pady=5)
        Button(controls, text="Добавить файлы", command=self.choose_files).pack(side=LEFT, padx=5)
        Button(controls, text="Добавить папку", command=self.choose_folder).pack(side=LEFT, padx=5)
        Button(controls, text="Удалить выбранное", command=self.remove_selected_path).pack(side=LEFT, padx=5)
        Button(controls, text="Индексировать", command=self.index_documents).pack(side=LEFT, padx=5)

        self.paths_list = Listbox(self.root, width=100, height=6)
        self.paths_list.pack(padx=10, pady=5)

        search_frame = Frame(self.root)
        search_frame.pack(padx=10, pady=5)
        Label(search_frame, text="Какой документ вы ищете?").pack(side=LEFT)
        self.query_entry = Entry(search_frame, width=50)
        self.query_entry.pack(side=LEFT, padx=5)
        Button(search_frame, text="Поиск", command=self.search_query).pack(side=LEFT, padx=5)

        toggles = Frame(self.root)
        toggles.pack(padx=10, pady=3)
        Checkbutton(toggles, text="BM25", variable=self.bm25_var).pack(side=LEFT, padx=5)
        Checkbutton(toggles, text="LLM улучшение запроса", variable=self.llm_rewrite_var).pack(side=LEFT, padx=5)

        results_controls = Frame(self.root)
        results_controls.pack(padx=10, pady=3)
        Button(results_controls, text="Показать документы", command=self.show_document_results).pack(side=LEFT, padx=5)
        Button(results_controls, text="Показать фрагменты", command=self.show_chunk_results).pack(side=LEFT, padx=5)

        self.results_list = Listbox(self.root, width=120, height=10)
        self.results_list.pack(padx=10, pady=5)
        Button(self.root, text="Открыть выбранный", command=self.open_selected).pack(pady=5)

        documents_frame = Frame(self.root)
        documents_frame.pack(padx=10, pady=5)
        Label(documents_frame, text="Индексированные документы по активной спеке").pack(side=LEFT, padx=5)
        self.documents_list = Listbox(self.root, width=120, height=6)
        self.documents_list.pack(padx=10, pady=5)
        Button(self.root, text="Открыть выбранный документ", command=self.open_selected_document).pack(pady=5)

        Label(self.root, textvariable=self.status_var).pack(pady=5)

    def build_container(self):
        config = ContainerConfig(
            embedder=self.embedder_var.get(),
            rewriter="llm" if self.llm_rewrite_var.get() else "simple",
            embedding_store="hnsw" if self.hnsw_var.get() else "in_memory",
            safe_mode=False,
        )
        cache_key = (
            config.embedder,
            config.rewriter,
            config.embedding_store,
        )
        if self._container_cache and self._container_cache[0] == cache_key:
            return self._container_cache[1]
        container = build_default_container(config)
        self._container_cache = (cache_key, container)
        self.status_var.set(
            f"Конфигурация: эмбеддер={container.embedder.model_id}, "
            f"хранилище={config.embedding_store}, specs={len(container.embedding_specs)}"
        )
        return container

    def _active_document_spec(self, container):
        return next(
            (
                spec
                for spec in container.embedding_specs
                if spec.level == "document" and spec.model_name == self.embedder_var.get()
            ),
            None,
        )

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.paths.append(Path(folder))
            self.paths_list.insert(END, folder)

    def choose_files(self) -> None:
        files = filedialog.askopenfilenames(
            filetypes=[
                ("Документы", "*.pdf *.docx *.txt *.md *.html *.htm"),
                ("Все файлы", "*.*"),
            ]
        )
        for file_path in files:
            self.paths.append(Path(file_path))
            self.paths_list.insert(END, file_path)

    def remove_selected_path(self) -> None:
        selection = self.paths_list.curselection()
        if not selection:
            return
        for idx in sorted(selection, reverse=True):
            if idx < len(self.paths):
                self.paths.pop(idx)
            self.paths_list.delete(idx)

    def index_documents(self) -> None:
        if not self.paths:
            messagebox.showinfo("ContextSearch", "Сначала выберите файлы или папку.")
            return
        container = self.build_container()
        report = ingest_paths(
            self.paths,
            splitter=container.splitter,
            embedders=container.embedders,
            embedding_store=container.embedding_store,
            document_repository=container.document_repository,
            chunk_repository=container.chunk_repository,
            embedding_specs=container.embedding_specs,
        )
        self.paths.clear()
        self.paths_list.delete(0, END)
        message = f"Проиндексировано {report.indexed}/{report.total} документов."
        if report.errors:
            message += f" Ошибок: {len(report.errors)}"
        messagebox.showinfo("ContextSearch", message)
        self.refresh_documents()

    def search_query(self) -> None:
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showinfo("ContextSearch", "Введите поисковый запрос.")
            return
        container = self.build_container()
        results = search(
            query,
            embedders=container.embedders,
            embedding_store=container.embedding_store,
            document_repository=container.document_repository,
            chunk_repository=container.chunk_repository,
            embedding_record_repository=container.embedding_record_repository,
            embedding_specs=container.embedding_specs,
            query_rewriter=container.query_rewriter,
            reranker=container.reranker,
            use_bm25=self.bm25_var.get(),
        )
        self._search_results_cache = results
        self.show_document_results()

    def show_document_results(self) -> None:
        self._result_mode = "documents"
        self.results_list.delete(0, END)
        self.results = []
        grouped: dict[str, dict[str, object]] = {}
        for result in self._search_results_cache:
            if result.chunk is None:
                continue
            key = result.document_id
            chunk_score = float(result.metadata.get("chunk_score", result.score))
            doc_vector = result.metadata.get("document_vector_score")
            bm25_score = result.metadata.get("bm25_score")
            current = grouped.get(key)
            if current is None:
                grouped[key] = {
                    "result": result,
                    "best_chunk": chunk_score,
                    "doc_vector": doc_vector,
                    "bm25": bm25_score,
                    "chunks": 1,
                }
            else:
                current["best_chunk"] = max(float(current["best_chunk"]), chunk_score)
                current["chunks"] = int(current["chunks"]) + 1
        ranked = sorted(grouped.values(), key=lambda item: float(item["best_chunk"]), reverse=True)
        for item in ranked:
            result = item["result"]
            assert isinstance(result, RetrievalResult)
            source_uri = result.chunk.metadata.get("source_uri", "") if result.chunk else ""
            display_name = result.document.metadata.get("display_name") if result.document else result.document_id
            doc_vector = item["doc_vector"]
            bm25_score = item["bm25"]
            line = (
                f"{display_name} | chunks={item['chunks']} | doc={self._score_label(doc_vector)} | "
                f"bm25={self._score_label(bm25_score)} | best_chunk={float(item['best_chunk']):.3f}"
            )
            self.results.append({"source_uri": source_uri})
            self.results_list.insert(END, line)

    def show_chunk_results(self) -> None:
        self._result_mode = "chunks"
        self.results_list.delete(0, END)
        self.results = []
        for result in self._search_results_cache:
            if result.chunk is None:
                continue
            source_uri = result.chunk.metadata.get("source_uri", "")
            display_name = result.document.metadata.get("display_name") if result.document else None
            snippet = result.chunk.text.replace("\n", " ")[:120]
            chunk_score = result.metadata.get("chunk_score", result.score)
            document_score = result.metadata.get("document_vector_score")
            bm25_score = result.metadata.get("bm25_score")
            line = (
                f"{display_name or result.document_id} | chunk={float(chunk_score):.3f} | "
                f"doc={self._score_label(document_score)} | bm25={self._score_label(bm25_score)} | "
                f"{snippet}"
            )
            self.results.append({"source_uri": source_uri})
            self.results_list.insert(END, line)

    @staticmethod
    def _score_label(score: object) -> str:
        if score is None:
            return "n/a"
        return f"{float(score):.3f}"

    def refresh_documents(self) -> None:
        container = self.build_container()
        document_spec = self._active_document_spec(container)
        if document_spec is None:
            documents = container.document_repository.list()
            self.spec_meta_var.set("Размерность: -, Метрика: -, Проиндексировано: 0")
        else:
            document_ids = set(container.embedding_record_repository.list_object_ids(document_spec.id, "document"))
            documents = [doc for doc in container.document_repository.list() if doc.id in document_ids]
            self.spec_meta_var.set(
                f"Размерность: {document_spec.dimension}, Метрика: {document_spec.metric}, "
                f"Проиндексировано: {len(documents)}"
            )
        self._documents_cache = documents
        self.documents_list.delete(0, END)
        for doc in documents:
            display_name = doc.title or doc.metadata.get("display_name") or doc.id
            source_uri = doc.path or doc.metadata.get("source_uri", "")
            self.documents_list.insert(END, f"{display_name} | {source_uri}")

    def open_selected_document(self) -> None:
        selection = self.documents_list.curselection()
        if not selection:
            messagebox.showinfo("ContextSearch", "Выберите документ для открытия.")
            return
        index = selection[0]
        documents = getattr(self, "_documents_cache", [])
        if index >= len(documents):
            messagebox.showwarning("ContextSearch", "Документ не найден.")
            return
        source_uri = documents[index].path or documents[index].metadata.get("source_uri", "")
        if not source_uri or not Path(source_uri).exists():
            messagebox.showwarning("ContextSearch", "Файл не найден.")
            return
        self._open_path(source_uri)

    def open_selected(self) -> None:
        selection = self.results_list.curselection()
        if not selection:
            messagebox.showinfo("ContextSearch", "Выберите результат для открытия.")
            return
        index = selection[0]
        if index >= len(self.results):
            return
        path = self.results[index]["source_uri"]
        if not path or not Path(path).exists():
            messagebox.showwarning("ContextSearch", "Файл не найден.")
            return
        self._open_path(path)

    def _open_path(self, path: str) -> None:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", path], check=False)
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", path], check=False)


def main() -> None:
    app = ContextSearchApp()
    app.root.mainloop()


if __name__ == "__main__":
    main()
