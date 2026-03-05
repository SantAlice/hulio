"""Speech-to-Text модуль на основе Vosk (бесплатный, офлайн)."""

import json
import logging
import numpy as np
from vosk import Model, KaldiRecognizer
import config

log = logging.getLogger(__name__)

_model = None


def get_model() -> Model:
    """Загрузить модель Vosk (ленивая загрузка)."""
    global _model
    if _model is None:
        log.info("Загрузка модели Vosk из %s ...", config.VOSK_MODEL_PATH)
        _model = Model(config.VOSK_MODEL_PATH)
        log.info("Модель Vosk загружена.")
    return _model


def recognize(audio_data: bytes, sample_rate: int = 16000) -> str:
    """
    Распознать речь из PCM аудио (16-bit signed, mono).

    Args:
        audio_data: сырые PCM данные
        sample_rate: частота дискретизации (по умолчанию 16000)

    Returns:
        Распознанный текст или пустая строка
    """
    model = get_model()
    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(False)

    # Отправляем аудио на распознавание
    recognizer.AcceptWaveform(audio_data)
    result = json.loads(recognizer.FinalResult())
    text = result.get("text", "").strip()

    if text:
        log.debug("Распознано: %s", text)
    return text


def pcm_stereo_48k_to_mono_16k(pcm_data: bytes) -> bytes:
    """
    Конвертировать PCM из формата Discord (48kHz, stereo, 16-bit)
    в формат Vosk (16kHz, mono, 16-bit).
    """
    # Декодируем 16-bit signed samples
    samples = np.frombuffer(pcm_data, dtype=np.int16)

    if len(samples) == 0:
        return b""

    # Stereo -> Mono (среднее двух каналов)
    if len(samples) % 2 == 0:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # Downsample 48kHz -> 16kHz (берём каждый 3-й сэмпл)
    samples = samples[::3]

    return samples.tobytes()


def calculate_rms(pcm_data: bytes) -> float:
    """Вычислить RMS (громкость) PCM аудио."""
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
