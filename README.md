# API для анализа тональности отзывов

FastAPI приложение для хранения и анализа тональности отзывов с использованием SQLite.

## Возможности

- ✅ Автоматический анализ тональности (позитивный/нейтральный/негативный)
- ✅ Создание отзывов и фильтрация по тональности
- ✅ Хранение данных в SQLite
- ✅ Современные инструменты Python с UV

## Технологии

- Python 3.11+
- FastAPI
- SQLAlchemy 2.0
- Pydantic 2.0
- Uvicorn
- UV (современная альтернатива pip)

## Быстрый старт

### Требования
- Python 3.13
- Установленный UV (`pip install uv`)

## Установка
### Создание виртуального окружения
```bash
uv venv venv
```

### Активация окружения
```bash
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate   # Windows
```

### Установка зависимостей
```bash
uv sync
```

### Запуск приложения
```bash
uvicorn main:app --reload
```

API будет доступно по адресу: http://localhost:80

# Документация API

## Конечные точки

### Создать отзыв
- POST /reviews
Content-Type: application/json

`{
  "text": "Мне нравится этот продукт!"
`}

### Получить отзывы по тональности
- GET /reviews?sentiment=positive

## Интерактивная документация
- Swagger UI: http://localhost:80/docs



