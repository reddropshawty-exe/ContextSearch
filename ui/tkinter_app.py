"""Tkinter-интерфейс для локальной индексации и поиска."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tkinter import (
    END,
    LEFT,
    RIGHT,
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
from domain.entities import Document
from infrastructure.config import ContainerConfig, build_default_container
from ui.logging_utils import setup_logging


class ContextSearchApp:
    def __init__(self) -> None:
        setup_logging()
        self.root = Tk()
        self.root.title("ContextSearch")
        self.paths: list[Path] = []
        self.results: list[dict[str, str]] = []
        self._container_cache: tuple[tuple[str, str, str, str | None], object] | None = None
        self._documents_cache: list[Document] = []

        self.profile_var = StringVar(value="stable")
        self.embedder_var = StringVar(value="all-minilm")
        self.rewriter_var = StringVar(value="simple")
        self.collection_var = StringVar(value="experimental")
        self.safe_mode_var = BooleanVar(value=False)

        self._build_layout()

    def _build_layout(self) -> None:
        settings = Frame(self.root)
        settings.pack(padx=10, pady=10)

        Label(settings, text="Профиль").pack(side=LEFT)
        Combobox(settings, textvariable=self.profile_var, values=["stable", "experimental"], width=12).pack(
            side=LEFT, padx=5
        )
        Label(settings, text="Эмбеддер").pack(side=LEFT)
        Combobox(
            settings,
            textvariable=self.embedder_var,
            values=["all-minilm", "all-mpnet", "multilingual-e5-base", "embedding-gemma"],
            width=22,
        ).pack(side=LEFT, padx=5)
        Label(settings, text="Переписыватель").pack(side=LEFT)
        Combobox(settings, textvariable=self.rewriter_var, values=["simple", "llm"], width=12).pack(
            side=LEFT, padx=5
        )
        Label(settings, text="Коллекция").pack(side=LEFT)
        Entry(settings, textvariable=self.collection_var, width=18).pack(side=LEFT, padx=5)
        Checkbutton(settings, text="Безопасный режим", variable=self.safe_mode_var).pack(side=LEFT, padx=5)

        controls = Frame(self.root)
        controls.pack(padx=10, pady=5)
        Button(controls, text="Выбрать папку", command=self.choose_folder).pack(side=LEFT, padx=5)
        Button(controls, text="Выбрать файлы", command=self.choose_files).pack(side=LEFT, padx=5)
        Button(controls, text="Индексировать", command=self.index_documents).pack(side=LEFT, padx=5)

        self.paths_list = Listbox(self.root, width=80, height=6)
        self.paths_list.pack(padx=10, pady=5)

        search_frame = Frame(self.root)
        search_frame.pack(padx=10, pady=5)
        Label(search_frame, text="Запрос").pack(side=LEFT)
        self.query_entry = Entry(search_frame, width=50)
        self.query_entry.pack(side=LEFT, padx=5)
        Button(search_frame, text="Найти", command=self.search_query).pack(side=LEFT, padx=5)

        self.results_list = Listbox(self.root, width=100, height=10)
        self.results_list.pack(padx=10, pady=5)

        Button(self.root, text="Открыть выбранный", command=self.open_selected).pack(pady=5)

        documents_frame = Frame(self.root)
        documents_frame.pack(padx=10, pady=5)
        Button(documents_frame, text="Обновить документы", command=self.refresh_documents).pack(side=LEFT, padx=5)
        self.documents_list = Listbox(self.root, width=100, height=6)
        self.documents_list.pack(padx=10, pady=5)
        Button(self.root, text="Открыть выбранный документ", command=self.open_selected_document).pack(pady=5)

    def build_container(self):
        profile = self.profile_var.get()
        config = ContainerConfig(
            profile=profile,
            embedder=self.embedder_var.get(),
            rewriter=self.rewriter_var.get(),
            collection_id=self.collection_var.get() or None if profile == "experimental" else None,
            embedding_store="faiss",
            safe_mode=self.safe_mode_var.get(),
        )
        cache_key = (config.profile, config.embedder, config.rewriter, config.collection_id, config.safe_mode)
        if self._container_cache and self._container_cache[0] == cache_key:
            return self._container_cache[1]
        container = build_default_container(config)
        self._container_cache = (cache_key, container)
        return container

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

    def index_documents(self) -> None:
        if not self.paths:
            messagebox.showinfo("ContextSearch", "Сначала выберите файлы или папку.")
            return
        container = self.build_container()
        report = ingest_paths(
            self.paths,
            splitter=container.splitter,
            embedder=container.embedder,
            embedding_store=container.embedding_store,
            document_repository=container.document_repository,
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
            embedder=container.embedder,
            embedding_store=container.embedding_store,
            document_repository=container.document_repository,
            query_rewriter=container.query_rewriter,
            reranker=container.reranker,
        )
        self.results_list.delete(0, END)
        self.results = []
        for result in results:
            source_uri = result.chunk.metadata.get("source_uri", "")
            display_name = result.document.metadata.get("display_name") if result.document else None
            snippet = result.chunk.text.replace("\n", " ")[:120]
            line = (
                f"{display_name or result.chunk.document_id} | "
                f"оценка={result.score:.3f} | {snippet} | {source_uri}"
            )
            self.results.append({"source_uri": source_uri})
            self.results_list.insert(END, line)

    def refresh_documents(self) -> None:
        container = self.build_container()
        documents = container.document_repository.list()
        self._documents_cache = documents
        self.documents_list.delete(0, END)
        for doc in documents:
            display_name = doc.metadata.get("display_name") or doc.id
            source_uri = doc.metadata.get("source_uri", "")
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
        source_uri = documents[index].metadata.get("source_uri", "")
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
