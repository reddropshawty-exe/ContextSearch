"""Tkinter-интерфейс для локальной индексации и поиска."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tkinter import (
    END,
    BOTH,
    LEFT,
    X,
    Y,
    BooleanVar,
    Button,
    Entry,
    Frame,
    Label,
    Listbox,
    Radiobutton,
    Scrollbar,
    StringVar,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
)
from tkinter.ttk import Combobox, Progressbar

from application.use_cases.ingest_paths import ingest_paths
from application.use_cases.search import search
from domain.entities import Document, RetrievalResult
from infrastructure.config import ContainerConfig, build_default_container
from ui.logging_utils import setup_logging

MODEL_CHOICES: list[tuple[str, str]] = [
    ("MiniLM", "all-minilm"),
    ("E5", "multilingual-e5-base"),
    ("Gemma", "embedding-gemma"),
]


class ContextSearchApp:
    def __init__(self) -> None:
        setup_logging()
        self.root = Tk()
        self.root.title("Контекстный поиск")
        self.root.geometry("1180x760")

        self.query_var = StringVar(value="")
        self.bm25_var = BooleanVar(value=False)
        self.llm_rewrite_var = BooleanVar(value=False)
        self.embedder_label_var = StringVar(value="MiniLM")
        self.embedder_key_var = StringVar(value="all-minilm")
        self.storage_var = StringVar(value="in_memory")

        self.status_var = StringVar(value="Готово")
        self.selected_docs_var = StringVar(value="0 документов выбрано")
        self.config_count_var = StringVar(value="0 документов проиндексировано")

        self._container_cache: tuple[tuple[str, str, str], object] | None = None
        self._documents_cache: list[Document] = []
        self._selected_paths: list[Path] = []
        self._search_results_cache: list[RetrievalResult] = []

        self._build_layout()
        self.refresh_documents()

    def _build_layout(self) -> None:
        main = Frame(self.root)
        main.pack(fill=BOTH, expand=True, padx=12, pady=12)

        top = Frame(main)
        top.pack(fill=X, expand=False)

        self._build_search_zone(top)
        self._build_config_zone(top)
        self._build_bottom_zone(main)

        Label(main, textvariable=self.status_var).pack(anchor="w", pady=6)

    def _build_search_zone(self, parent: Frame) -> None:
        zone = Frame(parent, bd=2, relief="groove")
        zone.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

        Label(zone, text="ПОИСК ДОКУМЕНТА", font=("Arial", 18, "bold")).pack(pady=10)
        self.query_entry = Entry(zone, textvariable=self.query_var, width=65)
        self.query_entry.pack(padx=10, pady=6)
        self.query_entry.insert(0, "Какой документ вы ищете?..")
        self.query_entry.bind("<FocusIn>", self._on_query_focus_in)
        self.query_entry.bind("<FocusOut>", self._on_query_focus_out)
        self.query_entry.bind("<KeyRelease>", lambda _e: self._sync_search_button_state())

        Frame(zone, height=4).pack()
        Button(zone, text="☐ BM25", command=self._toggle_bm25, width=30).pack(anchor="w", padx=16, pady=3)
        Button(zone, text="☐ LLM улучшение запроса", command=self._toggle_llm, width=30).pack(anchor="w", padx=16, pady=3)

        self.search_btn = Button(zone, text="ПОИСК", command=self.search_query, state="disabled", width=28)
        self.search_btn.pack(pady=12)

        self._bm25_btn = zone.winfo_children()[-3]
        self._llm_btn = zone.winfo_children()[-2]
        self._update_toggle_labels()

    def _build_config_zone(self, parent: Frame) -> None:
        panel = Frame(parent, width=320, bd=2, relief="groove")
        panel.pack(side=LEFT, fill=Y, expand=False)
        panel.pack_propagate(False)

        Label(panel, text="Текущая конфигурация", font=("Arial", 14, "bold")).pack(pady=8)

        card = Frame(panel, bd=2, relief="ridge")
        card.pack(fill=X, padx=12, pady=8)

        self.config_name_btn = Button(
            card,
            text="MiniLM - In-memory",
            command=lambda: self.open_config_modal(mode="view"),
            relief="flat",
            fg="blue",
        )
        self.config_name_btn.pack(fill=X, padx=8, pady=8)

        Label(card, textvariable=self.config_count_var).pack(anchor="w", padx=10, pady=4)

        btns = Frame(card)
        btns.pack(fill=X, padx=8, pady=8)
        Button(btns, text="Показать", command=lambda: self.open_config_modal(mode="view")).pack(fill=X, pady=3)
        Button(btns, text="Сменить конфиг", command=lambda: self.open_config_modal(mode="change")).pack(fill=X, pady=3)

    def _build_bottom_zone(self, parent: Frame) -> None:
        zone = Frame(parent, bd=2, relief="groove")
        zone.pack(fill=X, pady=(12, 0))

        top = Frame(zone)
        top.pack(fill=X, padx=10, pady=8)

        Button(top, text="+", width=4, command=self.open_add_documents_modal).pack(side=LEFT, padx=6)
        Label(top, textvariable=self.selected_docs_var).pack(side=LEFT, padx=12)

        Button(top, text="Посмотреть", command=lambda: self.open_config_modal(mode="view")).pack(side=LEFT, padx=6)
        Button(top, text="Индексировать", command=self.index_documents).pack(side=LEFT, padx=6)

        self.documents_list = Listbox(zone, width=140, height=7)
        self.documents_list.pack(fill=X, padx=10, pady=(0, 10))

    def _on_query_focus_in(self, _event) -> None:
        if self.query_var.get().strip() == "Какой документ вы ищете?..":
            self.query_var.set("")
            self._sync_search_button_state()

    def _on_query_focus_out(self, _event) -> None:
        if not self.query_var.get().strip():
            self.query_var.set("Какой документ вы ищете?..")
            self._sync_search_button_state()

    def _toggle_bm25(self) -> None:
        self.bm25_var.set(not self.bm25_var.get())
        self._update_toggle_labels()

    def _toggle_llm(self) -> None:
        self.llm_rewrite_var.set(not self.llm_rewrite_var.get())
        self._update_toggle_labels()

    def _update_toggle_labels(self) -> None:
        self._bm25_btn.configure(text=("☑ BM25" if self.bm25_var.get() else "☐ BM25"))
        self._llm_btn.configure(
            text=("☑ LLM улучшение запроса" if self.llm_rewrite_var.get() else "☐ LLM улучшение запроса")
        )

    def _sync_search_button_state(self) -> None:
        value = self.query_var.get().strip()
        if value and value != "Какой документ вы ищете?..":
            self.search_btn.configure(state="normal")
        else:
            self.search_btn.configure(state="disabled")

    def build_container(self):
        config = ContainerConfig(
            embedder=self.embedder_key_var.get(),
            rewriter="llm" if self.llm_rewrite_var.get() else "simple",
            embedding_store=self.storage_var.get(),
        )
        cache_key = (config.embedder, config.embedding_store, config.rewriter)
        if self._container_cache and self._container_cache[0] == cache_key:
            return self._container_cache[1]

        self.status_var.set("Загрузка конфигурации...")
        self.root.update_idletasks()
        container = build_default_container(config)
        self._container_cache = (cache_key, container)
        self.status_var.set("Готово")
        return container

    def _active_document_spec(self, container):
        return next(
            (
                spec
                for spec in container.embedding_specs
                if spec.level == "document" and spec.model_name == self.embedder_key_var.get()
            ),
            None,
        )

    def refresh_documents(self) -> None:
        container = self.build_container()
        spec = self._active_document_spec(container)
        if spec is None:
            docs = container.document_repository.list()
        else:
            ids = set(container.embedding_record_repository.list_object_ids(spec.id, "document"))
            docs = [doc for doc in container.document_repository.list() if doc.id in ids]

        self._documents_cache = docs
        self.documents_list.delete(0, END)
        for doc in docs:
            self.documents_list.insert(END, doc.title or doc.metadata.get("display_name") or doc.id)

        self.config_count_var.set(f"{len(docs)} документов проиндексировано")
        store_label = "HNSW" if self.storage_var.get() == "hnsw" else "In-memory"
        self.config_name_btn.configure(text=f"{self.embedder_label_var.get()} - {store_label}")

    def open_add_documents_modal(self) -> None:
        modal = Toplevel(self.root)
        modal.title("Добавление документов")
        modal.transient(self.root)
        modal.grab_set()
        modal.geometry("560x540")

        Button(modal, text="✕", command=modal.destroy).pack(anchor="ne", padx=6, pady=6)

        controls = Frame(modal)
        controls.pack(fill=X, padx=12, pady=8)
        Button(controls, text="Добавить файл", command=lambda: self._modal_add_files(list_frame)).pack(side=LEFT, padx=5)
        Button(controls, text="Добавить папку", command=lambda: self._modal_add_folder(list_frame)).pack(side=LEFT, padx=5)

        list_wrap = Frame(modal, bd=1, relief="sunken")
        list_wrap.pack(fill=BOTH, expand=True, padx=12, pady=8)

        canvas_holder = Frame(list_wrap)
        canvas_holder.pack(fill=BOTH, expand=True)
        scrollbar = Scrollbar(canvas_holder)
        scrollbar.pack(side="right", fill=Y)
        list_frame = Frame(canvas_holder)
        list_frame.pack(fill=BOTH, expand=True)

        Button(modal, text="ГОТОВО", command=lambda: self._close_add_docs_modal(modal), width=20).pack(pady=12)
        self._refresh_modal_file_list(list_frame)

    def _modal_add_files(self, list_frame: Frame) -> None:
        files = filedialog.askopenfilenames(
            filetypes=[
                ("Документы", "*.pdf *.docx *.txt *.md *.html *.htm"),
                ("Все файлы", "*.*"),
            ]
        )
        for file_path in files:
            path = Path(file_path)
            if path not in self._selected_paths:
                self._selected_paths.append(path)
        self._refresh_modal_file_list(list_frame)

    def _modal_add_folder(self, list_frame: Frame) -> None:
        folder = filedialog.askdirectory()
        if not folder:
            return
        root = Path(folder)
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}:
                if file_path not in self._selected_paths:
                    self._selected_paths.append(file_path)
        self._refresh_modal_file_list(list_frame)

    def _refresh_modal_file_list(self, list_frame: Frame) -> None:
        for child in list_frame.winfo_children():
            child.destroy()
        for path in self._selected_paths:
            row = Frame(list_frame)
            row.pack(fill=X, padx=6, pady=4)
            Label(row, text=path.name, anchor="w").pack(side=LEFT, fill=X, expand=True)
            Button(row, text="❌", command=lambda p=path: self._remove_selected_path(p, list_frame)).pack(side=LEFT)

    def _remove_selected_path(self, path: Path, list_frame: Frame) -> None:
        self._selected_paths = [item for item in self._selected_paths if item != path]
        self._refresh_modal_file_list(list_frame)

    def _close_add_docs_modal(self, modal: Toplevel) -> None:
        self.selected_docs_var.set(f"{len(self._selected_paths)} документов выбрано")
        modal.destroy()

    def index_documents(self) -> None:
        if not self._selected_paths:
            messagebox.showinfo("Контекстный поиск", "Сначала добавьте документы через '+'")
            return

        container = self.build_container()
        progress = self._open_progress("Индексация документов...")
        try:
            report = ingest_paths(
                self._selected_paths,
                splitter=container.splitter,
                embedders=container.embedders,
                embedding_store=container.embedding_store,
                document_repository=container.document_repository,
                chunk_repository=container.chunk_repository,
                embedding_specs=container.embedding_specs,
            )
        finally:
            progress.destroy()

        self._selected_paths = []
        self.selected_docs_var.set("0 документов выбрано")
        message = f"Проиндексировано {report.indexed}/{report.total} документов"
        if report.errors:
            message += f". Ошибок: {len(report.errors)}"
        messagebox.showinfo("Контекстный поиск", message)
        self.refresh_documents()

    def search_query(self) -> None:
        query = self.query_var.get().strip()
        if not query or query == "Какой документ вы ищете?..":
            return

        container = self.build_container()
        progress = self._open_progress("Выполняется поиск...")
        try:
            self._search_results_cache = search(
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
        finally:
            progress.destroy()

        self.open_search_results_modal(mode="documents")

    def open_search_results_modal(self, mode: str = "documents") -> None:
        modal = Toplevel(self.root)
        modal.title("Итоги поиска")
        modal.transient(self.root)
        modal.grab_set()
        modal.geometry("980x620")

        Button(modal, text="✕", command=modal.destroy).pack(anchor="ne", padx=6, pady=6)
        Label(modal, text="Итоги поиска", font=("Arial", 18, "bold")).pack(pady=4)

        listbox = Listbox(modal, width=130, height=20)
        listbox.pack(fill=BOTH, expand=True, padx=10, pady=8)

        actions = Frame(modal)
        actions.pack(fill=X, padx=10, pady=6)

        Button(actions, text="Показать все", command=lambda: self._fill_results_list(listbox, "documents", show_all=True)).pack(
            side=LEFT, padx=4
        )
        Button(
            actions,
            text="Показать результаты по фрагментам",
            command=lambda: self._fill_results_list(listbox, "chunks", show_all=True),
        ).pack(side=LEFT, padx=4)
        Button(actions, text="Открыть выбранный", command=lambda: self._open_selected_result_from_listbox(listbox)).pack(
            side=LEFT, padx=4
        )

        Button(actions, text="См. фрагменты документа", command=lambda: self._show_doc_chunks_from_listbox(listbox)).pack(
            side=LEFT, padx=4
        )

        self._fill_results_list(listbox, mode=mode, show_all=False)

    def _fill_results_list(self, listbox: Listbox, mode: str, show_all: bool) -> None:
        listbox.delete(0, END)
        self.results = []
        if mode == "chunks":
            rows = self._search_results_cache if show_all else self._search_results_cache[:20]
            for result in rows:
                if result.chunk is None:
                    continue
                display_name = result.document.metadata.get("display_name") if result.document else result.document_id
                chunk_score = result.metadata.get("chunk_score", result.score)
                doc_score = result.metadata.get("document_vector_score")
                bm25_score = result.metadata.get("bm25_score")
                line = (
                    f"{display_name} | chunk={float(chunk_score):.3f} | "
                    f"doc={self._fmt(doc_score)} | bm25={self._fmt(bm25_score)} | "
                    f"{result.chunk.text[:80].replace(chr(10), ' ')}"
                )
                listbox.insert(END, line)
                self.results.append({"source_uri": result.chunk.metadata.get("source_uri", ""), "document_id": result.document_id})
            return

        grouped: dict[str, dict[str, object]] = {}
        for result in self._search_results_cache:
            if result.chunk is None:
                continue
            key = result.document_id
            chunk_score = float(result.metadata.get("chunk_score", result.score))
            current = grouped.get(key)
            if current is None:
                grouped[key] = {
                    "result": result,
                    "best_chunk": chunk_score,
                    "chunks": 1,
                    "doc": result.metadata.get("document_vector_score"),
                    "bm25": result.metadata.get("bm25_score"),
                }
            else:
                current["best_chunk"] = max(float(current["best_chunk"]), chunk_score)
                current["chunks"] = int(current["chunks"]) + 1

        ranked = sorted(grouped.values(), key=lambda item: float(item["best_chunk"]), reverse=True)
        if not show_all:
            ranked = ranked[:10]

        for item in ranked:
            result = item["result"]
            assert isinstance(result, RetrievalResult)
            display_name = result.document.metadata.get("display_name") if result.document else result.document_id
            line = (
                f"{display_name} | bm25={self._fmt(item['bm25'])} | doc={self._fmt(item['doc'])} | "
                f"фрагментов={item['chunks']}"
            )
            listbox.insert(END, line)
            self.results.append(
                {
                    "source_uri": result.chunk.metadata.get("source_uri", "") if result.chunk else "",
                    "document_id": result.document_id,
                }
            )

    @staticmethod
    def _fmt(value: object) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}"

    def _show_doc_chunks_from_listbox(self, listbox: Listbox) -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo("Итоги поиска", "Выберите документ")
            return
        idx = selection[0]
        if idx >= len(self.results):
            return
        doc_id = self.results[idx]["document_id"]
        lines: list[str] = []
        for result in self._search_results_cache:
            if result.document_id != doc_id or result.chunk is None:
                continue
            lines.append(
                f"chunk={self._fmt(result.metadata.get('chunk_score', result.score))} | "
                f"{result.chunk.text[:120].replace(chr(10), ' ')}"
            )
        if not lines:
            messagebox.showinfo("Итоги поиска", "Фрагменты не найдены")
            return
        messagebox.showinfo("Фрагменты документа", "\n".join(lines[:20]))

    def _open_selected_result_from_listbox(self, listbox: Listbox) -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo("Итоги поиска", "Выберите строку")
            return
        idx = selection[0]
        if idx >= len(self.results):
            return
        path = self.results[idx]["source_uri"]
        if not path or not Path(path).exists():
            messagebox.showwarning("Итоги поиска", "Файл недоступен")
            return
        self._open_path(path)

    def open_config_modal(self, mode: str = "view") -> None:
        modal = Toplevel(self.root)
        modal.title("Выбор конфигурации")
        modal.transient(self.root)
        modal.grab_set()
        modal.geometry("560x640")

        Button(modal, text="✕", command=modal.destroy).pack(anchor="ne", padx=6, pady=6)
        Label(modal, text="Выбор конфигурации", font=("Arial", 16, "bold")).pack(pady=4)

        model_var = StringVar(value=self.embedder_label_var.get())
        Label(modal, text="Модель").pack(anchor="w", padx=10)
        dropdown = Combobox(modal, textvariable=model_var, values=[name for name, _ in MODEL_CHOICES], width=20)
        dropdown.pack(anchor="w", padx=10, pady=4)
        dropdown.configure(state="readonly")

        Label(modal, text="Тип хранилища").pack(anchor="w", padx=10, pady=(8, 0))
        storage_var = StringVar(value=self.storage_var.get())
        Radiobutton(modal, text="HNSW", variable=storage_var, value="hnsw").pack(anchor="w", padx=14)
        Radiobutton(modal, text="In-memory", variable=storage_var, value="in_memory").pack(anchor="w", padx=14)

        meta_label = Label(modal, text="")
        meta_label.pack(anchor="w", padx=10, pady=8)

        docs_list = Listbox(modal, width=70, height=16)
        docs_list.pack(fill=BOTH, expand=True, padx=10, pady=8)

        def refresh_preview(*_args) -> None:
            embedder_name = model_var.get()
            embedder_key = next((key for name, key in MODEL_CHOICES if name == embedder_name), self.embedder_key_var.get())
            config = ContainerConfig(
                embedder=embedder_key,
                embedding_store=storage_var.get(),
                rewriter="llm" if self.llm_rewrite_var.get() else "simple",
            )
            container = build_default_container(config)
            spec = next((s for s in container.embedding_specs if s.level == "document" and s.model_name == embedder_key), None)

            docs_list.delete(0, END)
            if spec is None:
                docs = container.document_repository.list()
                meta_label.configure(text="Размерность вектора: -\nМетрика сходства: -\nТип индекса: -")
            else:
                ids = set(container.embedding_record_repository.list_object_ids(spec.id, "document"))
                docs = [d for d in container.document_repository.list() if d.id in ids]
                meta_label.configure(
                    text=(
                        f"Размерность вектора: {spec.dimension}\n"
                        f"Метрика сходства: {spec.metric}\n"
                        f"Тип индекса: {storage_var.get()}\n"
                        f"Проиндексировано: {len(docs)} документа"
                    )
                )
            for doc in docs:
                docs_list.insert(END, doc.title or doc.metadata.get("display_name") or doc.id)

        dropdown.bind("<<ComboboxSelected>>", refresh_preview)
        storage_var.trace_add("write", refresh_preview)
        refresh_preview()

        if mode == "change":
            def apply_config() -> None:
                name = model_var.get()
                key = next((k for n, k in MODEL_CHOICES if n == name), self.embedder_key_var.get())
                self.embedder_label_var.set(name)
                self.embedder_key_var.set(key)
                self.storage_var.set(storage_var.get())
                self._container_cache = None
                self.refresh_documents()
                modal.destroy()

            Button(modal, text="Применить", command=apply_config).pack(pady=8)

    def _open_progress(self, text: str) -> Toplevel:
        modal = Toplevel(self.root)
        modal.title("Выполнение")
        modal.transient(self.root)
        modal.grab_set()
        modal.geometry("320x120")
        Label(modal, text=text).pack(pady=10)
        bar = Progressbar(modal, mode="indeterminate")
        bar.pack(fill=X, padx=14, pady=6)
        bar.start(10)
        self.root.update_idletasks()
        return modal

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
