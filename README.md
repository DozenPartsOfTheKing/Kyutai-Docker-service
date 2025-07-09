# 🗣️🎙️ SpeakerMan — Форк Kyutai STT / TTS с готовым GPU-сервером

![GPU](https://img.shields.io/badge/GPU-ready-brightgreen)
![Docker](https://img.shields.io/badge/Docker-image-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-turquoise)
![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-lightgrey)

> Проект основан на оригинальном [Kyutai Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) и представляет собой удобную обёртку для быстрого деплоя STT- и TTS-моделей Kyutai. Мы добавили:
>
> 1. **FastAPI-сервер** с REST-эндпоинтами `/stt` и `/tts`.
> 2. **GPU-образ** (`Dockerfile.gpu`) и пример **docker-compose** для запуска на RTX A5000/4090, L40S и т.п.
> 3. Поддержку **FP16**, выбор **формата вывода** (`wav`/`pcm`) и регулировку **температуры** (интонации) голоса.
> 4. Swagger UI с подсказками на русском языке.

---

## 🔎 Содержание
1. [Быстрый старт](#быстрый-старт)
2. [API](#api)
   * [/health](#health)
   * [/stt](#stt)
   * [/tts](#tts)
3. [Конфиг Docker/GPU](#конфиг-dockergpu)
4. [Примеры](#примеры)
5. [Разработка](#разработка)
6. [Лицензия](#лицензия)

---

## 🚀 Быстрый старт

```bash
# 1. Клонируем репозиторий
git clone https://github.com/DozenPartsOfTheKing/Kyutai-Docker-service
cd Kyutai-Docker-service

# 2. Стартуем GPU-контейнер (нужен NVIDIA Container Toolkit)
#    Используется файл docker-compose.gpu.yml

docker compose -f docker-compose.gpu.yml up --build

# После запуска сервер доступен на http://localhost:8000
# Swagger UI:  http://localhost:8000/docs
```

> 💡 **Совет:** переменная `NVIDIA_VISIBLE_DEVICES` в `docker-compose.gpu.yml` задаёт номер GPU. Если у вас несколько видеокарт — запустите два сервиса с разными GPU для балансировки.

---

## 📖 API

### <a id="health"></a>GET `/health`
Простой probe для Kubernetes/NGINX. Возвращает `ok`.

### <a id="stt"></a>POST `/stt`
Потоковое **Speech-to-Text**. Принимает аудио-файл, выдаёт транскрипт.

| Параметр | Тип | По-умолчанию | Описание |
|----------|-----|--------------|----------|
| `audio` | `file` | — | Аудио в любом формате, поддерживаемом `sphn` (mp3, wav, flac…). |
| `hf_repo` | `string` | — | HF-репозиторий с весами STT-модели. |
| `vad` | `bool` | `false` | Включить semantic VAD. |
| `device` | `string` | `cuda` | `cuda`, `cpu` или `mps`.

Ответ: `text/plain` с тайм-стемпами слов.

### <a id="tts"></a>POST `/tts`
Генерация речи **Text-to-Speech**.

| Параметр | Тип | По-умолчанию | Описание |
|----------|-----|--------------|----------|
| `text` | `string` | — | Текст для синтеза. |
| `hf_repo` | `string` | `kyutai/tts-1.6b-en_fr` | Репозиторий весов TTS. |
| `voice_repo` | `string` | — | Репозиторий голосовых эмбеддингов. |
| `voice` | `string` | — | Конкретный голос внутри `voice_repo`. |
| `device` | `string` | `cuda` | Устройство инференса. |
| `format` | `string` | `wav` | `wav` или `pcm`. |
| `temp` | `float` | `0.6` | Температура сэмплирования (интонация). 0 — монотонно, 1 — эмоционально. |

Ответ: файл `audio/wav` или `application/octet-stream` (PCM).

> 🎧 Если указать `out == "-"` внутри скриптов, аудио будет стримиться прямо на устройство (см. `scripts/tts_pytorch.py`). Для REST-endpoint’а стриминг пока не реализован — можно добавить WebSocket, если понадобится.

---

## 🐳 Конфиг Docker/GPU

`Dockerfile.gpu` наследуется от официального `pytorch/pytorch:2.2.2-cuda12.1` и устанавливает:

* `fastapi`, `uvicorn[standard]`, `moshi`, `sphn` и др.
* ENV-тюнинги:
  * `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:128`
  * `CUDA_LAUNCH_BLOCKING=0`
  * `TF_FORCE_GPU_ALLOW_GROWTH=true`

Запуск Uvicorn:
```bash
uvicorn app.main:app \
  --host 0.0.0.0 --port $PORT \
  --workers $WORKERS \
  --loop uvloop \
  --timeout-keep-alive $UVICORN_TIMEOUT
```

`docker-compose.gpu.yml` монтирует кеш HuggingFace на SSD:
```yaml
environment:
  - WORKERS=4
  - UVICORN_TIMEOUT=120
volumes:
  - /mnt/ssd/huggingface_cache:/root/.cache/huggingface
```

---

## 🎤 Примеры

### STT (транскрипция файла)
```bash
curl -F audio=@audio/bria.mp3 http://localhost:8000/stt
```

### TTS → WAV
```bash
curl -F text="Привет, как дела?" \
     -F temp=0.4 \
     http://localhost:8000/tts --output speech.wav
```

### TTS → PCM (сырое int16)
```bash
curl -F text="Hello" -F format=pcm http://localhost:8000/tts --output speech.pcm
```

---

## 🛠️ Разработка

```bash
# Установка хуков
pip install pre-commit
pre-commit install

# Локальный запуск без Docker (только CPU):
uvicorn app.main:app --reload
```

* Все изменения проходят `ruff`, `black`, `mypy`.
* Папки `scripts/` содержат примеры запуска моделей напрямую.

---

## 📜 Лицензия

Код Python и Docker-часть — MIT, Rust-бэкенд — Apache 2.0. Веса моделей распространяются под CC-BY 4.0 от Kyutai.

---

### 💌 Благодарности и связь

Проект основан на работе команды [Kyutai Labs](https://kyutai.org/). Большое спасибо авторам оригинального DSM!
