## Embedding Web Service (FastAPI)

Русский | English

### Описание / Description

- Простой веб‑сервис на FastAPI с POST `/embed`, который принимает текст и параметр `role` (`query` или `passage`) и возвращает эмбеддинг.
- Использует библиотеку `sentence-transformers` с моделью `intfloat/multilingual-e5-large-instruct` (E5 instruct, подходит для русского и английского; для модели автоматически добавляются префиксы `query:` и `passage:`).

### Требования / Requirements

- Python 3.9+
- Интернет для первого скачивания модели (кэшируется локально)

### Установка / Setup

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt  # Windows PowerShell
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

### Примечания / Notes

- Первая генерация может занять время из‑за загрузки модели. Модель кэшируется.
- Эндпоинт `/docs` содержит авто‑документацию Swagger UI.
