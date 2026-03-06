import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")

# LLM настройки (OpenAI-совместимый API: DeepSeek, OpenAI, и т.д.)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324")
LLM_API_KEYS = [k.strip() for k in os.getenv("LLM_API_KEY", "").split(",") if k.strip()]
LLM_API_KEY = LLM_API_KEYS[0] if LLM_API_KEYS else ""

# Обратная совместимость с GEMINI_API_KEY
if not LLM_API_KEYS:
    _gemini_keys = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
    if _gemini_keys:
        LLM_API_KEYS = _gemini_keys
        LLM_API_KEY = _gemini_keys[0]
        LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        LLM_MODEL = "gemini-2.5-flash"
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-ru-0.54")
TTS_ENGINE = os.getenv("TTS_ENGINE", "silero")  # "silero" or "edge"
TTS_VOICE = os.getenv("TTS_VOICE", "eugene")  # silero: aidar/baya/kseniya/xenia/eugene/random
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_PITCH_SEMITONES = float(os.getenv("TTS_PITCH_SEMITONES", "-4"))  # сдвиг тона в полутонах (-4 = басистый)
BOT_NAME = os.getenv("BOT_NAME", "Хулио")
PERSONALITY_FILE = os.getenv("PERSONALITY_FILE", "personalities/default.txt")

# Discord audio settings
SAMPLE_RATE = 48000  # Discord uses 48kHz
CHANNELS = 2  # Discord stereo
VOSK_SAMPLE_RATE = 16000  # Vosk needs 16kHz mono

# Silence detection
SILENCE_THRESHOLD = 500  # RMS threshold for silence
SILENCE_DURATION = 1.5  # Seconds of silence to trigger processing
MAX_RECORD_DURATION = 30  # Max seconds of recording per utterance

# Conversation memory
MAX_HISTORY_PER_USER = 20  # Max messages to remember per user
