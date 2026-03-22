## Embedding Web Service (FastAPI)

Русский | English

### Описание / Description

- Простой веб‑сервис на FastAPI с POST `/embed`, который принимает текст и параметр `role` (`query` или `passage`) и возвращает эмбеддинг.
- Дополнительно предоставляет POST `/rerank`, который принимает поисковый запрос и список документов и возвращает релевантности документов к запросу.
- Для эмбеддингов используется библиотека `sentence-transformers` с моделью `intfloat/multilingual-e5-large-instruct` (E5 instruct, подходит для русского и английского; для модели автоматически добавляются префиксы `query:` и `passage:`).
- Для rerank используется модель `BAAI/bge-reranker-v2-m3` из библиотеки `FlagEmbedding`.

### Требования / Requirements

- Python 3.9+
- Интернет для первого скачивания модели (кэшируется локально)

### Установка / Setup

```bash
python -m venv .venv
# Убедитесь, что пакеты ставятся в venv проекта (используйте python -m pip):
# Make sure packages install into the project venv (use python -m pip):
./.venv/Scripts/python -m pip install -r requirements.txt  # Windows
# или / or
source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
```

### Запуск / Run

```bash
./.venv/Scripts/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload  # Windows
# или / or
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload  # активированный venv
```

### Проверка / Healthcheck

```bash
curl http://127.0.0.1:8000/health
```

### Использование / Usage

POST `/embed`

Тело запроса / Request body: `text` (строка) и `role` — один из `"query"` (поисковый запрос) или `"passage"` (документ/пассаж для индексации). Сервис сам добавляет префиксы `query:` и `passage:` к тексту перед кодированием.

```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Пример поискового запроса", "role": "query"}'
```

Для пассажа / For passage:

```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Фрагмент документа для индексации", "role": "passage"}'
```

Пример ответа / Example response:

```json
{
  "embedding": [0.0123, -0.0456, ...],
  "model": "intfloat/multilingual-e5-large-instruct",
  "dim": 1024
}
```

### Использование reranker / Reranker usage

POST `/rerank`

Тело запроса / Request body:

```json
{
  "query": "Пример поискового запроса",
  "documents": [
    "Краткий текст документа 1",
    "Краткий текст документа 2"
  ]
}
```

Пример запроса / Example request:

```bash
curl -X POST http://127.0.0.1:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Пример поискового запроса",
    "documents": [
      "Краткий текст документа 1",
      "Краткий текст документа 2"
    ]
  }'
```

Пример ответа / Example response:

```json
{
  "scores": [0.95, 0.12],
  "model": "BAAI/bge-reranker-v2-m3"
}
```

### Примечания / Notes

- Первая генерация может занять время из‑за загрузки модели. Модель кэшируется.
- Эндпоинт `/docs` содержит авто‑документацию Swagger UI.

