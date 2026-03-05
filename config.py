import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-small-ru-0.22")
TTS_VOICE = os.getenv("TTS_VOICE", "ru-RU-DmitryNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
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
