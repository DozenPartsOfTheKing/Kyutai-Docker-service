# Kyutai TTS Russian Fine-Tuning

Этот README описывает весь процесс дообучения (fine-tuning) модели TTS для русского языка на базе проекта [Kyutai-STT-TTS-service](https://github.com/DozenPartsOfTheKing/Kyutai-STT-TTS-service) и репозитория обучения [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling).

---

## Содержание

1. [Требования](#требования)
2. [Аппаратные ресурсы](#аппаратные-ресурсы)
3. [Установка окружения](#установка-окружения)
4. [Клонирование репозиториев](#клонирование-репозиториев)
5. [Подготовка датасета](#подготовка-датасета)
6. [Конфигурация обучения](#конфигурация-обучения)
7. [Dry-run (тестовый прогон)](#dry-run-тестовый-прогон)
8. [Запуск полного обучения](#запуск-полного-обучения)
9. [Промежуточная оценка качества](#промежуточная-оценка-качества)
10. [Экспорт и публикация модели](#экспорт-и-публикация-модели)
11. [Интеграция в сервис](#интеграция-в-сервис)
12. [Советы и лучшие практики](#советы-и-лучшие-практики)

---

## Требования

* **ОС**: Ubuntu 20.04/22.04 или аналогичная Linux-среда
* **Python**: 3.8+
* **CUDA**: Toolkit ≥ 11.7, драйвер NVIDIA ≥ 525
* **Git**, **sox**, **ffmpeg**, **wget**
* **Disk space**: ≥ 100 GB свободного места
* **RAM**: ≥ 32 GB (для обработки датасета)

---

## Аппаратные ресурсы

| Устройство              | VRAM    | Примечание                                  |
| ----------------------- | ------- | ------------------------------------------- |
| RTX 5070                | \~12 GB | Для отладки и dry-run                       |
| 2× NVIDIA A5000         | 48 GB   | Основное обучение (рекомендуется)           |
| Mac (CPU/Apple Silicon) | —       | Только для экспериментов без GPU (медленно) |

Используем прод‑сервер с двумя A5000 для основного обучения, локально — 5070 для отладочных прогоны.

---

## Установка окружения

```bash
# Обновление и установка утилит
sudo apt update && sudo apt install -y git wget sox ffmpeg

# Клонируем репозитории (см. ниже)
# Создаем виртуальное окружение
python3 -m venv kyutai-venv && source kyutai-venv/bin/activate

# Обновляем pip и устанавливаем зависимости тренировки
pip install --upgrade pip
cd delayed-streams-modeling
pip install -r requirements.txt
pip install torchaudio librosa montreal-forced-aligner transformers datasets g2p_ru
```

---

## Клонирование репозиториев

```bash
# Инференс-сервис
git clone https://github.com/DozenPartsOfTheKing/Kyutai-STT-TTS-service.git

# Код для обучения
git clone https://github.com/kyutai-labs/delayed-streams-modeling.git
```

---

## Подготовка датасета

1. **Скачать M‑AILABS Russian**

   ```bash
   git clone https://github.com/imdatceleste/m-ailabs-dataset.git
   cd m-ailabs-dataset/Russian
   # Переместить wav/ и txt/ в ~/datasets/ru/
   ```
2. **Структура**:

   ```
   ~/datasets/ru/
     wav/          # .wav, 22050 Hz, моно
     txt/          # .txt с транскриптом (lowercase, без лишней пунктуации)
   ```
3. **Создание metadata.csv**

   ```bash
   cd ~/datasets/ru
   printf "wav_filename|transcript
   ```

" > metadata.csv
for f in wav/\*.wav; do
name=\$(basename "\$f")
text=\$(sed 's/|/ /g' txt/\${name%.wav}.txt)
echo "\${name}|\${text}" >> metadata.csv
done

````
4. **Обрезка тишины**
```bash
mkdir wav_trimmed
for f in wav/*.wav; do
  sox "$f" "wav_trimmed/$(basename "$f")" silence 1 0.1 1% -1 0.1 1%
done
mv wav_trimmed wav
````

5. *(Опционально)* **Forced alignment**:

   ```bash
   mfa align ~/datasets/ru/wav ~/datasets/ru/txt russian_model ~/datasets/ru/aligned
   ```

---

## Конфигурация обучения

1. Создаем `configs/tts_russian.yaml` скопировав `configs/tts_base.yaml`:

   ```bash
   cd delayed-streams-modeling/configs
   cp tts_base.yaml tts_russian.yaml
   ```
2. Редактируем ключи в `tts_russian.yaml`:

   ```yaml
   model_name_or_path: kyutai/tts-1.6b-en_fr
   output_dir: ../checkpoints/tts-ru
   train_dataset:
     path: /home/you/datasets/ru/metadata.csv
     audio_dir: /home/you/datasets/ru/wav
     lang: ru
   phoneme_converter: russian_g2p
   tokenizer:
     type: phoneme
   num_train_steps: 20000
   learning_rate: 3e-5
   batch_size: 12
   warmup_steps: 500
   grad_accumulation_steps: 1
   ```

---

## Dry-run (тестовый прогон)

Проверяем загрузку данных и модели:

```bash
CUDA_VISIBLE_DEVICES=0 python train_tts.py \
  --config configs/tts_russian.yaml \
  --max_steps 100 \
  --dry_run
```

* Модель должна загрузиться
* Датасет читаться корректно
* G2P-преобразование кириллицы без ошибок

---

## Запуск полного обучения

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_tts.py \
  --config configs/tts_russian.yaml
```

### Мониторинг

* Запустите TensorBoard:

  ```bash
  tensorboard --logdir ./checkpoints/tts-ru/logs
  ```
* Следите за `loss/train` и `loss/val`
* Пример синтеза каждые 1000 шагов

---

## Промежуточная оценка качества

1. Подготовьте `eval_sentences.txt` (10–20 фраз разных типов).
2. Синтез для контрольных чекпоинтов:

   ```bash
   python synthesize.py \
     --model_dir checkpoints/tts-ru/checkpoint-5000 \
     --input_file eval_sentences.txt \
     --output_dir samples/5000
   ```
3. Прослушайте и оцените:

   * Чёткость фонем и ударений
   * Интонацию и паузы
   * Артефакты

---

## Экспорт и публикация модели

```bash
transformers-cli login
transformers-cli repo create tts-ru-custom --type model
git clone https://huggingface.co/your-org/tts-ru-custom
cp -r checkpoints/tts-ru/* tts-ru-custom/
cd tts-ru-custom
git add .
git commit -m "Fine-tuned on Russian"
git push
```

---

## Интеграция в сервис

1. **Dockerfile**:

   ```dockerfile
   RUN python - <<EOF
   from huggingface_hub import snapshot_download
   snapshot_download("your-org/tts-ru-custom", "/opt/models/tts-ru")
   EOF
   ```
2. **app/main.py**: изменить дефолтный репозиторий:

   ```python
   @app.post("/tts")
   async def tts(..., hf_repo: str = "your-org/tts-ru-custom"):
       model = TTSModel.from_pretrained(hf_repo)
       ...
   ```
3. **docker-compose.gpu.yml**:

   ```yaml
   ```

environment:

* TTS\_RU\_REPO=your-org/tts-ru-custom

````
4. Пересборка и запуск:
```bash
docker compose -f docker-compose.gpu.yml up --build -d
````

---

## Советы и лучшие практики

* **Версионирование**: храните датасет, конфиги и результаты
* **Логи**: включайте метки и потоки логов с `timestamps`
* **Чекпоинты**: сохраняйте каждые 2000 шагов для отката
* **Документация**: фиксируйте гиперпараметры и результаты в отдельном `logs.md`
* **Контроль качества**: используйте разноплановый набор eval-фраз

