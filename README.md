# LLM Model Requirements Calculator

Микросервис для оценки, поместится ли LLM на вашей машине.

## Что теперь умеет

- Web UI + API (`FastAPI`)
- Автоопределение системы через `GET /api/system_info`
- Расчет в двух режимах: `real` и `ollama`
- Настраиваемый контекст `context_len` (256..131072)
- Локализация ответа API: `ru` и `en`
- Валидация входных данных
- Сравнение моделей и история проверок в UI (localStorage)
- Наблюдаемость: `GET /api/metrics`
- Защита API: in-memory rate limit
- Кэширование результатов `/api/check_model`
- Docker / docker-compose

## Архитектура

- `server.py`: API, расчет, rate-limit, metrics, кеш, статика
- `static/`: фронтенд (HTML/CSS/JS)

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Локальный запуск

```bash
venv/bin/python -m uvicorn server:app --reload
```

Откройте:

- UI: `http://localhost:8000`
- Metrics: `http://localhost:8000/api/metrics`

## Docker

```bash
docker compose up --build
```

Сервис будет доступен на `http://localhost:8000`.

## Web UI

В интерфейсе доступны:

- выбор платформы `Mac/PC`
- выбор языка `Русский/English`
- настройка контекста
- автоопределение параметров системы
- сравнение до 3 моделей
- история последних проверок

## API

### `POST /api/check_model`

Пример запроса:

```json
{
  "model_name": "qwen3.5:9b",
  "mode": "real",
  "context_len": 8192,
  "language": "ru",
  "system_info": {
    "ram_gb": 32,
    "vram_gb": 24,
    "has_gpu": true,
    "gpu_name": "RTX 4090",
    "os": "Linux",
    "cpu_cores": 16
  }
}
```

Пример ответа:

```json
{
  "model_parsed": {
    "family": "qwen",
    "version": "3.5",
    "param_count_b": 9.0,
    "is_moe": false,
    "specialization": "base",
    "quantization": "Q4"
  },
  "memory_requirements": {
    "weights_gb": 4.2,
    "kv_cache_gb": 2.8,
    "total_vram_gb": 7.0
  },
  "recommendation": "...",
  "possible_quantizations": {
    "FP16": 20.1,
    "BF16": 20.1,
    "Q8": 11.7,
    "Q4": 7.0,
    "Q3": 5.9,
    "Q2": 4.9
  },
  "context_len": 8192,
  "language": "ru"
}
```

### `GET /api/system_info`

Возвращает RAM/VRAM/GPU/OS/CPU для машины, где запущен сервер.

### `GET /api/metrics`

Возвращает базовые счетчики и среднюю задержку:

- `requests_total`
- `errors_total`
- `check_model_total`
- `system_info_total`
- `avg_latency_ms`
- `cache_items`
- `rate_limit_per_minute`

## Ограничения

- Rate limit и cache сейчас in-memory (сбрасываются при рестарте).
- Для production лучше вынести это в Redis.

## Лицензия

MIT
