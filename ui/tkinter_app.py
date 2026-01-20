"""Tkinter UI for local indexing and search."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tkinter import END, LEFT, RIGHT, Button, Entry, Frame, Label, Listbox, StringVar, Tk, filedialog, messagebox
from tkinter.ttk import Combobox

from application.use_cases.ingest_paths import ingest_paths
from application.use_cases.search import search
from infrastructure.config import ContainerConfig, build_default_container


class ContextSearchApp:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("ContextSearch")
        self.paths: list[Path] = []
        self.results: list[dict[str, str]] = []
        self._container_cache: tuple[tuple[str, str, str, str | None], object] | None = None

        self.profile_var = StringVar(value="stable")
        self.embedder_var = StringVar(value="all-minilm")
        self.rewriter_var = StringVar(value="simple")
        self.collection_var = StringVar(value="experimental")

        self._build_layout()

    def _build_layout(self) -> None:
        settings = Frame(self.root)
        settings.pack(padx=10, pady=10)

        Label(settings, text="Profile").pack(side=LEFT)
        Combobox(settings, textvariable=self.profile_var, values=["stable", "experimental"], width=12).pack(
            side=LEFT, padx=5
        )
        Label(settings, text="Embedder").pack(side=LEFT)
        Combobox(
            settings,
            textvariable=self.embedder_var,
            values=["all-minilm", "all-mpnet", "multilingual-e5-base", "embedding-gemma"],
            width=22,
        ).pack(side=LEFT, padx=5)
        Label(settings, text="Rewriter").pack(side=LEFT)
        Combobox(settings, textvariable=self.rewriter_var, values=["simple", "llm"], width=10).pack(
            side=LEFT, padx=5
        )
        Label(settings, text="Collection").pack(side=LEFT)
        Entry(settings, textvariable=self.collection_var, width=18).pack(side=LEFT, padx=5)

        controls = Frame(self.root)
        controls.pack(padx=10, pady=5)
        Button(controls, text="Choose Folder", command=self.choose_folder).pack(side=LEFT, padx=5)
        Button(controls, text="Choose Files", command=self.choose_files).pack(side=LEFT, padx=5)
        Button(controls, text="Index", command=self.index_documents).pack(side=LEFT, padx=5)

        self.paths_list = Listbox(self.root, width=80, height=6)
        self.paths_list.pack(padx=10, pady=5)

        search_frame = Frame(self.root)
        search_frame.pack(padx=10, pady=5)
        Label(search_frame, text="Query").pack(side=LEFT)
        self.query_entry = Entry(search_frame, width=50)
        self.query_entry.pack(side=LEFT, padx=5)
        Button(search_frame, text="Search", command=self.search_query).pack(side=LEFT, padx=5)

        self.results_list = Listbox(self.root, width=100, height=10)
        self.results_list.pack(padx=10, pady=5)

        Button(self.root, text="Open Selected", command=self.open_selected).pack(pady=5)

    def build_container(self):
        profile = self.profile_var.get()
        config = ContainerConfig(
            profile=profile,
            embedder=self.embedder_var.get(),
            rewriter=self.rewriter_var.get(),
            collection_id=self.collection_var.get() or None if profile == "experimental" else None,
            embedding_store="faiss",
        )
        cache_key = (config.profile, config.embedder, config.rewriter, config.collection_id)
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
                ("Documents", "*.pdf *.docx *.txt *.md *.html *.htm"),
                ("All files", "*.*"),
            ]
        )
        for file_path in files:
            self.paths.append(Path(file_path))
            self.paths_list.insert(END, file_path)

    def index_documents(self) -> None:
        if not self.paths:
            messagebox.showinfo("ContextSearch", "Please select files or a folder first.")
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
        message = f"Indexed {report.indexed}/{report.total} documents."
        if report.errors:
            message += f" Errors: {len(report.errors)}"
        messagebox.showinfo("ContextSearch", message)

    def search_query(self) -> None:
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showinfo("ContextSearch", "Enter a search query.")
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
                f"score={result.score:.3f} | {snippet} | {source_uri}"
            )
            self.results.append({"source_uri": source_uri})
            self.results_list.insert(END, line)

    def open_selected(self) -> None:
        selection = self.results_list.curselection()
        if not selection:
            messagebox.showinfo("ContextSearch", "Select a result to open.")
            return
        index = selection[0]
        path = self.results[index]["source_uri"]
        if not path or not Path(path).exists():
            messagebox.showwarning("ContextSearch", "File not found.")
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
