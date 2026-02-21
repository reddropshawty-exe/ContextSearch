# ContextSearch
Решение для удобного контекстного поиска документов на компьютере

## Установка зависимостей
Для локального запуска установите зависимости:

```bash
python -m pip install -r requirements.txt
```

## Запуск приложения
### Локально
Основной UI (Tkinter, macOS first):

```bash
python -m ui.tkinter_app
```

API (FastAPI):

```bash
uvicorn ui.api.main:app --host 0.0.0.0 --port 8000
```

Web-интерфейс (Streamlit):

```bash
streamlit run ui/web/app.py --server.address 0.0.0.0 --server.port 8501
```

### Через Docker (быстрый старт)
Соберите и запустите API и Streamlit-демо одной командой:

```bash
docker compose up --build
```

После запуска:
- API доступен на `http://localhost:8000`
- Streamlit UI доступен на `http://localhost:8501`

## Артефакты
После индексации на диске появятся:

- SQLite база: `contextsearch.db`
- Индексы ANN (HNSW): `indexes/<spec_id>.bin`

## Диагностика падений
Для локализации проблем можно запустить диагностические тесты:

```bash
python -m unittest tests.test_diagnostics
```

Тесты для sentence-transformers и HNSW запускаются только при наличии
переменных окружения:

```bash
CONTEXTSEARCH_ENABLE_ST=1 python -m unittest tests.test_diagnostics
CONTEXTSEARCH_ENABLE_HNSW=1 python -m unittest tests.test_diagnostics
```

Логи приложения пишутся в файл `contextsearch.log` в корне проекта. Можно
переопределить путь и уровень логирования через переменные окружения:

```bash
CONTEXTSEARCH_LOG_FILE=logs/contextsearch.log CONTEXTSEARCH_LOG_LEVEL=DEBUG \
  python -m ui.tkinter_app
```

Если приложение падает из-за нативных библиотек, можно включить безопасный
режим (in-memory хранилище и хэш-эмбеддеры без HNSW/torch):

```bash
CONTEXTSEARCH_SAFE_MODE=1 python -m ui.tkinter_app
CONTEXTSEARCH_SAFE_MODE=1 streamlit run ui/web/app.py
```

Если нужно оставить реальные эмбеддеры, но отключить HNSW, выберите
`in_memory` в интерфейсе (Streamlit/Tkinter) или выставьте
`embedding_store="in_memory"` в конфигурации.


## Offline-модели (без повторной загрузки из сети)

Чтобы приложение работало полностью локально и не тянуло веса из интернета,
можно заранее скачать модели в папку проекта:

```bash
python scripts/prefetch_models.py --output-dir models
```

Далее укажите путь к каталогу моделей:

```bash
CONTEXTSEARCH_MODELS_DIR=models python -m ui.tkinter_app
CONTEXTSEARCH_MODELS_DIR=models streamlit run ui/web/app.py
```

Логика резолвинга такая:
- если в `CONTEXTSEARCH_MODELS_DIR/<model_id>` есть папка модели — используется она;
- иначе используется исходный `model_id` из Hugging Face (стандартное поведение).

Это работает и для эмбеддеров (`all-minilm`, `all-mpnet`, `multilingual-e5-base`,
`embedding-gemma`), и для LLM-query-rewriter.

## Инфраструктурные модели
В проекте доступно несколько вариантов экстракторов и эмбеддеров, которые можно
подобрать под разные источники данных:

- **Экстракторы:**
  - `PdfExtractor` — извлекает текст через pdfplumber.
  - `DocxExtractor` — извлекает текст из DOCX (параграфы и таблицы).
  - `PlainTextExtractor` — для уже очищенных текстов/логов.
  - `HtmlExtractor` — удаляет HTML‑теги и игнорирует `<script>/<style>`.
- **Эмбеддеры:**
  - `SentenceTransformersEmbedder` — реальные модели (MiniLM, MPNet, E5, EmbeddingGemma).
  - `MiniLMEmbedder` — хешевый аналог sentence-transformer’а.
  - `MeanWordHashEmbedder` — усреднение словарных векторов.
  - `CharacterNgramEmbedder` — нграммный вариант, устойчивый к опечаткам.

Нужные реализации выбираются через `ContainerConfig` в модуле
`infrastructure.config`, что позволяет быстро переключать пайплайн под текущий
датасет или эксперимент.

## Что уже готово
- **Домен и use cases.** В `domain.entities` лежат датаклассы `Document`,
  `Chunk`, `Query`, `RetrievalResult`, а в `domain.interfaces` — абстракции для
  экстракции, нарезки, эмбеддинга, хранилищ и переписывания/переранжирования.
  Оркестраторы `application/use_cases/ingest_documents.py`,
  `search.py` и `run_experiment.py` работают только с этими интерфейсами.
- **Инфраструктурные реализации.** Модули в `infrastructure/` покрывают весь
  жизненный цикл: текстовые экстракторы (`text_extraction/`), нарезчик
  `FixedWindowSplitter`, три эмбеддера (`embedding/`), in-memory хранилище,
  SQLite‑репозиторий документов и простые query‑модули. Параметры пайплайна
  задаются через `ContainerConfig`, поэтому можно комбинировать разные
  экстракторы и эмбеддеры под задачу.
- **UI‑слои.** В `ui/api/main.py` развёрнут FastAPI с эндпоинтами `/ingest`,
  `/documents`, `/search`, которые напрямую вызывают use cases. В
  `ui/web/app.py` есть Streamlit‑заглушка для ручной демонстрации ingest/search
  цепочек.

## Что можно улучшить дальше
- **Более глубокие экстракторы.** Расширить `PdfExtractor` поддержкой PyPDF2 или
  pdfminer, добавить обработку DOCX/HTML с извлечением метаданных и языков.
- **Настоящие эмбеддинги и storage.** Интегрировать sentence-transformers,
  fastText или open-source LLM, заменить in-memory store на HNSW/Qdrant и
  прописать конфигурацию подключения.
- **Сложные пайплайны запросов.** Добавить LLM‑переписывание (локальные модели),
  BM25‑retrieval и cross-encoder reranking, а также гибкие параметры top‑K и агрегации.
- **Наблюдаемость и тесты.** Ввести unit-тесты для use cases, structured logging
  в API, сбор метрик и сохранение результатов экспериментов.

## Retrieval evaluation

- Requirements: `docs/retrieval_evaluation_requirements.md`
- Class diagram (PlantUML): `docs/retrieval_evaluation_class_diagram.puml`

### Evaluation subsystem (implemented)

- API endpoints:
  - `POST /evaluation/suites`, `GET /evaluation/suites`
  - `POST /evaluation/configs`, `GET /evaluation/configs`
  - `POST /evaluation/runs`, `GET /evaluation/runs`
- Tkinter: кнопка `Тестирование` открывает режим разметки/прогона и историю запусков.
- SQLite storage tables: `eval_test_suites`, `eval_test_cases`, `eval_experiment_configs`, `eval_experiment_runs`, `eval_case_results`.


## UML-диаграммы системы

В `docs/diagrams` добавлены PlantUML-диаграммы для архитектурного описания:

- ER-диаграмма: `docs/diagrams/contextsearch_er_diagram.puml`
- Sequence (индексация): `docs/diagrams/sequence_indexing.puml`
- Sequence (поиск): `docs/diagrams/sequence_search.puml`
- Sequence (тестирование качества): `docs/diagrams/sequence_evaluation_testing.puml`
- Диаграмма классов: `docs/diagrams/contextsearch_class_diagram.puml`

Каждая sequence-диаграмма разделена на общий пайплайн и отдельную страницу
с детализацией вариативной логики (через `newpage`).
