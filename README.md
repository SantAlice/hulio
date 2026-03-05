# Хулио — Голосовой бот для Discord

Голосовой бот который слушает что говорят в войсе, отвечает голосом, запоминает разговоры и имеет настраиваемую личность.

**Полностью бесплатный** — никаких платных API:
- **STT**: Vosk (локально, офлайн)
- **LLM**: Google Gemini Flash (бесплатный tier: 15 запросов/мин, 1500/день)
- **TTS**: Edge TTS (бесплатный, голоса Microsoft)

## Возможности

- Слушает голосовой канал и распознаёт речь каждого участника
- Знает кто говорит по никнейму в Discord
- Отвечает голосом с минимальной задержкой
- Запоминает контекст разговора
- Настраиваемая личность (дерзкий, анимешный, токсичный геймер, или своя)
- Выбор голоса из 300+ голосов Microsoft
- Настройка скорости речи

## Быстрый старт

### 1. Системные зависимости

```bash
# Ubuntu/Debian
sudo apt install ffmpeg python3-pip

# macOS
brew install ffmpeg
```

### 2. Установка

```bash
git clone <repo-url> && cd hulio
pip install -r requirements.txt
```

### 3. Скачать модель Vosk

```bash
mkdir -p models && cd models

# Русская модель (45 MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip

# ИЛИ английская модель (40 MB)
# wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
# unzip vosk-model-small-en-us-0.15.zip
```

### 4. Настройка

```bash
cp .env.example .env
```

Заполни `.env`:
- `DISCORD_TOKEN` — токен бота ([Discord Developer Portal](https://discord.com/developers/applications))
- `GEMINI_API_KEY` — ключ API ([Google AI Studio](https://aistudio.google.com/apikey), бесплатно)

### 5. Создание Discord бота

1. Зайди на [Discord Developer Portal](https://discord.com/developers/applications)
2. New Application → назови бота
3. Bot → Reset Token → скопируй токен в `.env`
4. Bot → включи **Message Content Intent**, **Server Members Intent**
5. OAuth2 → URL Generator:
   - Scopes: `bot`, `applications.commands`
   - Bot Permissions: `Connect`, `Speak`, `Use Voice Activity`
6. Скопируй ссылку → пригласи бота на сервер

### 6. Запуск

```bash
python bot.py
```

## Команды

| Команда | Описание |
|---------|----------|
| `/join` | Зайти в твой голосовой канал |
| `/leave` | Выйти из войса |
| `/voice <name>` | Сменить голос (например `ru-RU-SvetlanaNeural`) |
| `/voices [lang]` | Показать доступные голоса |
| `/personality <name>` | Сменить личность (`default`, `anime_girl`, `toxic_gamer`) |
| `/say <text>` | Сказать текст в войс |
| `/rate <speed>` | Скорость речи (`+20%`, `-10%`) |
| `/clear` | Очистить историю разговоров |

## Личности

Готовые личности в папке `personalities/`:

- **default** — Хулио, дерзкий и остроумный чувак
- **anime_girl** — Мику, кавайная анимешница
- **toxic_gamer** — ТоксикМастер, токсичный геймер

### Создать свою личность

Создай файл `personalities/my_personality.txt` с описанием характера, затем:
```
/personality my_personality
```

## Популярные голоса

### Русские
- `ru-RU-DmitryNeural` — мужской
- `ru-RU-SvetlanaNeural` — женский

### Английские
- `en-US-GuyNeural` — мужской
- `en-US-JennyNeural` — женский
- `en-US-AriaNeural` — женский

Полный список: `edge-tts --list-voices`

## Архитектура

```
Голос в Discord
    ↓
PCM Audio (48kHz stereo)
    ↓
Конвертация → 16kHz mono
    ↓
Vosk STT → текст
    ↓
Gemini Flash → ответ
    ↓
Edge TTS → MP3
    ↓
FFmpeg → голос в Discord
```
