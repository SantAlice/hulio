"""Speech-to-Text модуль с поддержкой Vosk (офлайн) и Google SR (онлайн)."""

import io
import json
import logging
import wave

import numpy as np
import speech_recognition as sr

import config

log = logging.getLogger(__name__)

# Vosk model (lazy loaded, optional)
_vosk_model = None
_use_vosk = None  # None = не проверяли ещё


def _try_init_vosk():
    """Попытаться загрузить Vosk. Возвращает True если удалось."""
    global _vosk_model, _use_vosk
    if _use_vosk is not None:
        return _use_vosk
    try:
        import os
        from vosk import Model
        path = config.VOSK_MODEL_PATH
        if os.path.exists(path) and any(
            f.endswith('.conf') or f == 'mfcc.conf' or f == 'final.mdl'
            for f in os.listdir(path)
        ):
            log.info("Загрузка модели Vosk из %s ...", path)
            _vosk_model = Model(path)
            _use_vosk = True
            log.info("Модель Vosk загружена.")
            return True
        else:
            log.warning("Папка модели Vosk пустая или не содержит файлов модели: %s", path)
            _use_vosk = False
            return False
    except Exception as e:
        log.warning("Vosk недоступен: %s. Используем Google Speech Recognition.", e)
        _use_vosk = False
        return False


def _recognize_vosk(audio_data: bytes, sample_rate: int) -> str:
    """Распознавание через Vosk (офлайн)."""
    from vosk import KaldiRecognizer
    recognizer = KaldiRecognizer(_vosk_model, sample_rate)
    recognizer.SetWords(False)
    recognizer.AcceptWaveform(audio_data)
    result = json.loads(recognizer.FinalResult())
    return result.get("text", "").strip()


def _recognize_google(audio_data: bytes, sample_rate: int) -> str:
    """Распознавание через Google Speech Recognition (онлайн, бесплатно)."""
    recognizer = sr.Recognizer()

    # Конвертируем raw PCM в WAV в памяти
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    wav_buffer.seek(0)

    with sr.AudioFile(wav_buffer) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        return text.strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        log.error("Ошибка Google SR: %s", e)
        return ""


def recognize(audio_data: bytes, sample_rate: int = 16000) -> str:
    """
    Распознать речь из PCM аудио (16-bit signed, mono).

    Использует Vosk если модель доступна, иначе Google Speech Recognition.
    """
    _try_init_vosk()

    if _use_vosk:
        text = _recognize_vosk(audio_data, sample_rate)
    else:
        text = _recognize_google(audio_data, sample_rate)

    if text:
        log.debug("Распознано: %s", text)
    return text


def pcm_stereo_48k_to_mono_16k(pcm_data: bytes) -> bytes:
    """
    Конвертировать PCM из формата Discord (48kHz, stereo, 16-bit)
    в формат распознавания (16kHz, mono, 16-bit).
    """
    samples = np.frombuffer(pcm_data, dtype=np.int16)

    if len(samples) == 0:
        return b""

    # Stereo -> Mono
    if len(samples) % 2 == 0:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # Downsample 48kHz -> 16kHz
    samples = samples[::3]

    return samples.tobytes()


def calculate_rms(pcm_data: bytes) -> float:
    """Вычислить RMS (громкость) PCM аудио."""
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
