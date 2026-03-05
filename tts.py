"""Text-to-Speech модуль на основе Edge TTS (бесплатный)."""

import asyncio
import io
import logging
import tempfile
import edge_tts
import config

log = logging.getLogger(__name__)


async def synthesize(text: str, voice: str | None = None, rate: str | None = None) -> str:
    """
    Синтезировать речь и сохранить в временный MP3 файл.

    Args:
        text: текст для озвучки
        voice: голос (по умолчанию из config)
        rate: скорость речи (по умолчанию из config)

    Returns:
        Путь к MP3 файлу
    """
    voice = voice or config.TTS_VOICE
    rate = rate or config.TTS_RATE

    communicate = edge_tts.Communicate(text, voice, rate=rate)

    # Сохраняем в временный файл
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_path = tmp.name
    tmp.close()

    await communicate.save(tmp_path)
    log.debug("TTS: сохранено в %s (%d символов, голос: %s)", tmp_path, len(text), voice)
    return tmp_path


async def list_voices(language: str = "ru") -> list[dict]:
    """Получить список доступных голосов для языка."""
    voices = await edge_tts.list_voices()
    return [v for v in voices if v["Locale"].startswith(language)]
