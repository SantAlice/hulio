"""Text-to-Speech модуль с поддержкой Silero TTS и Edge TTS + pitch shift."""

import asyncio
import logging
import os
import subprocess
import tempfile
import torch
import config

log = logging.getLogger(__name__)

# Silero model (lazy loaded)
_silero_model = None
_silero_sample_rate = 48000


def _get_silero_model():
    """Загрузить модель Silero TTS (ленивая загрузка)."""
    global _silero_model
    if _silero_model is None:
        log.info("Загрузка Silero TTS модели...")
        _silero_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v4_ru',
        )
        log.info("Silero TTS загружен.")
    return _silero_model


def _apply_pitch_shift(input_path: str, semitones: float) -> str:
    """Сдвинуть тон аудио через ffmpeg. Возвращает путь к новому файлу."""
    if semitones == 0:
        return input_path

    # Формула: множитель частоты = 2^(semitones/12)
    factor = 2 ** (semitones / 12.0)

    output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = output.name
    output.close()

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-af", f"asetrate={int(_silero_sample_rate * factor)},aresample={_silero_sample_rate}",
                "-loglevel", "error",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        os.unlink(input_path)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warning("Pitch shift не удался (ffmpeg): %s, используем оригинал", e)
        try:
            os.unlink(output_path)
        except OSError:
            pass
        return input_path


async def synthesize(text: str, voice: str | None = None, rate: str | None = None) -> str:
    """
    Синтезировать речь и сохранить в временный файл.

    Args:
        text: текст для озвучки
        voice: голос (по умолчанию из config)
        rate: скорость речи (только для Edge TTS)

    Returns:
        Путь к аудио файлу
    """
    if config.TTS_ENGINE == "silero":
        return await _synthesize_silero(text, voice)
    else:
        return await _synthesize_edge(text, voice, rate)


async def _synthesize_silero(text: str, voice: str | None = None) -> str:
    """Синтез через Silero TTS + pitch shift."""
    voice = voice or config.TTS_VOICE
    valid_speakers = ["aidar", "baya", "kseniya", "xenia", "eugene", "random"]
    if voice not in valid_speakers:
        log.warning("Голос '%s' не поддерживается Silero, использую 'xenia'", voice)
        voice = "xenia"

    def _generate():
        model = _get_silero_model()
        audio = model.apply_tts(
            text=text,
            speaker=voice,
            sample_rate=_silero_sample_rate,
        )
        # Сохраняем как WAV
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        import torchaudio
        torchaudio.save(tmp_path, audio.unsqueeze(0), _silero_sample_rate)

        # Применяем pitch shift
        tmp_path = _apply_pitch_shift(tmp_path, config.TTS_PITCH_SEMITONES)
        return tmp_path

    loop = asyncio.get_event_loop()
    tmp_path = await loop.run_in_executor(None, _generate)
    log.debug("Silero TTS: сохранено в %s (%d символов, голос: %s, pitch: %+.0f)",
              tmp_path, len(text), voice, config.TTS_PITCH_SEMITONES)
    return tmp_path


async def _synthesize_edge(text: str, voice: str | None = None, rate: str | None = None) -> str:
    """Синтез через Edge TTS (fallback)."""
    import edge_tts

    voice = voice or config.TTS_VOICE
    rate = rate or config.TTS_RATE

    communicate = edge_tts.Communicate(text, voice, rate=rate)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_path = tmp.name
    tmp.close()

    await communicate.save(tmp_path)
    log.debug("Edge TTS: сохранено в %s (%d символов, голос: %s)", tmp_path, len(text), voice)
    return tmp_path


async def list_voices(language: str = "ru") -> list[dict]:
    """Получить список доступных голосов."""
    voices = []

    # Silero voices
    silero_speakers = ["aidar", "baya", "kseniya", "xenia", "eugene", "random"]
    for s in silero_speakers:
        voices.append({
            "ShortName": f"silero-{s}",
            "Gender": "Male" if s in ("aidar", "eugene") else "Female" if s != "random" else "Random",
            "Locale": "ru-RU",
            "Engine": "silero",
        })

    # Edge TTS voices
    if language:
        import edge_tts
        edge_voices = await edge_tts.list_voices()
        for v in edge_voices:
            if v["Locale"].startswith(language):
                v["Engine"] = "edge"
                voices.append(v)

    return voices
