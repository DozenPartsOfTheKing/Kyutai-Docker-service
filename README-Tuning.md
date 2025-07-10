# 📦 OpenTTS Mini-Pipeline

В этом документе собраны инструкции, как:

1. Скачать уменьшенную выборку (`1 %` или любую другую) из датасета `ishine/open_tts` с помощью скрипта `download_open_tts_subset.sh`;
2. Подготовить манифесты и (опционально) конвертировать аудио в WAV 24 кГц;
3. Запустить QLoRA-дообучение Kyutai TTS на двух GPU RTX A5000 (или любой другой поддерживаемой конфигурации);
4. Быстро проверить получившуюся модель.

Скорость: < 1 час на скачивание 1 % (≈ 6 ч аудио) при 40 Мбит/с, ещё ~30 минут на препроцессинг и несколько часов на первые эпохи обучения.

---

## 0. Предварительные зависимости

```bash
# Установим параллельный скачиватель и базовые утилиты
sudo apt-get update && sudo apt-get install -y aria2 ffmpeg sox git
```

---

# если DNS до DO недоступен —
bash ./download_open_tts_subset.sh \
     --fraction 0.01 \
     --outdir  open_tts_1pct \
     --mirror  azure

58 минут - 5% данных 4mb/s

или просто без флага: скрипт сначала попробует DigitalOcean, при ошибке
автоматически переключится на Azure.
При первой же возможности поставьте aria2c:

sudo apt-get update && sudo apt-get install -y aria2



Установив датасет, его нужно распаковать и соранить - формат 

(base) root@cv4881639:~/Dev/SpeakerMan/scripts/open_tts_full# ls
public_speech  radio-v4  radio_v4_add
(base) root@cv4881639:~/Dev/SpeakerMan/scripts/open_tts_full# cd public_speech/
(base) root@cv4881639:~/Dev/SpeakerMan/scripts/open_tts_full/public_speech# la
0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
(base) root@cv4881639:~/Dev/SpeakerMan/scripts/open_tts_full/public_speech# ls
0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
(base) root@cv4881639:~/Dev/SpeakerMan/scripts/open_tts_full/public_speech# cd 0
(base) root@cv4881639:~/Dev/SpeakerMan/scripts/open_tts_full/public_speech/0# ls
00  07  0e  15  1c  23  2a  31  38  3f  46  4d  54  5b  62  69  70  77  7e  85  8c  93  9a  a1  a8  af  b6  bd  c4  cb  d2  d9  e0  e7  ee  f5  fc
01  08  0f  16  1d  24  2b  32  39  40  47  4e  55  5c  63  6a  71  78  7f  86  8d  94  9b  a2  a9  b0  b7  be  c5  cc  d3  da  e1  e8  ef  f6  fd
02  09  10  17  1e  25  2c  33  3a  41  48  4f  56  5d  64  6b  72  79  80  87  8e  95  9c  a3  aa  b1  b8  bf  c6  cd  d4  db  e2  e9  f0  f7  fe
03  0a  11  18  1f  26  2d  34  3b  42  49  50  57  5e  65  6c  73  7a  81  88  8f  96  9d  a4  ab  b2  b9  c0  c7  ce  d5  dc  e3  ea  f1  f8  ff
04  0b  12  19  20  27  2e  35  3c  43  4a  51  58  5f  66  6d  74  7b  82  89  90  97  9e  a5  ac  b3  ba  c1  c8  cf  d6  dd  e4  eb  f2  f9
05  0c  13  1a  21  28  2f  36  3d  44  4b  52  59  60  67  6e  75  7c  83  8a  91  98  9f  a6  ad  b4  bb  c2  c9  d0  d7  de  e5  ec  f3  fa
06  0d  14  1b  22  29  30  37  3e  45  4c  53  5a  61  68  6f  76  7d  84  8b  92  99  a0  a7  ae  b5  bc  c3  ca  d1  d8  df  e6  ed  f4  fb
(base) root@cv4881639:~/Dev/SpeakerMan/scripts/open_tts_full/public_speech/0# cd 00
(base) root@cv4881639:~/Dev/SpeakerMan/scripts/open_tts_full/public_speech/0/00# ls
0b2a451f96e5.opus  1283e6c1794a.txt   1de5f9ae358f.opus  2ff7dc2f1f6c.txt   45ddbc3dcbdb.opus  7ea60067d14c.txt   bde49bb67525.opus  e43f1e215355.txt
0b2a451f96e5.txt   12bfc3fcf442.opus  1de5f9ae358f.txt   328511d52ed0.opus  45ddbc3dcbdb.txt   89ad282cae4c.opus  bde49bb67525.txt   e6a7951da39f.opus
0cfbac8b350b.opus  12bfc3fcf442.txt   295d2d981238.opus  328511d52ed0.txt   4e9f9d587caa.opus  89ad282cae4c.txt   c451900bb5b6.opus  e6a7951da39f.txt
0cfbac8b350b.txt   13da1b3bce27.opus  295d2d981238.txt   43d1d1e874d3.opus  4e9f9d587caa.txt   a4aa3246348d.opus  c451900bb5b6.txt   f580c6fd3f3c.opus
1283e6c1794a.opus  13da1b3bce27.txt   2ff7dc2f1f6c.opus  43d1d1e874d3.txt   7ea60067d14c.opus  a4aa3246348d.txt   e43f1e215355.opus  f580c6fd3f3c.txt
(base) root@cv4881639:~/Dev/SpeakerMan/scripts/open_tts_full/public_speech/0/00# 