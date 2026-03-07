"""Text-to-Speech модуль с поддержкой Silero TTS и Edge TTS + голосовые пресеты."""

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

# Голосовые пресеты с FFmpeg эффектами
VOICE_PRESETS = {
    "shrek": {
        "pitch": -7,
        "effects": "lowshelf=g=6:f=300,highshelf=g=-3:f=3000",
        "speaker": "eugene",
        "description": "Шрек — глубокий басистый голос огра",
    },
    "donkey": {
        "pitch": +5,
        "effects": "highshelf=g=5:f=2000,atempo=1.15",
        "speaker": "eugene",
        "description": "Осёл — высокий энергичный болтливый голос",
    },
    "bass": {
        "pitch": -4,
        "effects": "lowshelf=g=4:f=300",
        "speaker": "eugene",
        "description": "Басистый мужской голос",
    },
    "robot": {
        "pitch": -2,
        "effects": "afftfilt=real='hypot(re,im)*cos((random(0)*2-1)*2*3.14)':imag='hypot(re,im)*sin((random(0)*2-1)*2*3.14)':win_size=512:overlap=0.75",
        "speaker": "eugene",
        "description": "Роботизированный голос",
    },
    "demon": {
        "pitch": -10,
        "effects": "lowshelf=g=8:f=200,vibrato=f=5:d=0.5",
        "speaker": "eugene",
        "description": "Демонический голос из ада",
    },
    "chipmunk": {
        "pitch": +8,
        "effects": "highshelf=g=6:f=3000,atempo=1.2",
        "speaker": "xenia",
        "description": "Бурундучий писклявый голос",
    },
    "normal": {
        "pitch": 0,
        "effects": "",
        "speaker": None,
        "description": "Обычный голос без эффектов",
    },
}

# Текущий пресет
_current_preset: str | None = None


def preload_model():
    """Предзагрузить Silero модель (вызывать при старте бота)."""
    if config.TTS_ENGINE == "silero":
        _get_silero_model()


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


def set_preset(name: str) -> bool:
    """Установить голосовой пресет. Возвращает True если пресет найден."""
    global _current_preset
    if name in VOICE_PRESETS:
        _current_preset = name
        preset = VOICE_PRESETS[name]
        config.TTS_PITCH_SEMITONES = preset["pitch"]
        if preset["speaker"]:
            config.TTS_VOICE = preset["speaker"]
        log.info("Пресет голоса: %s (%s)", name, preset["description"])
        return True
    return False


def get_preset() -> str | None:
    """Получить текущий пресет."""
    return _current_preset


def list_presets() -> dict:
    """Получить список пресетов."""
    return VOICE_PRESETS


def _apply_audio_effects(input_path: str, semitones: float, extra_effects: str = "") -> str:
    """Применить pitch shift и дополнительные эффекты через ffmpeg."""
    if semitones == 0 and not extra_effects:
        return input_path

    # Собираем цепочку фильтров
    filters = []

    if semitones != 0:
        factor = 2 ** (semitones / 12.0)
        filters.append(f"asetrate={int(_silero_sample_rate * factor)}")
        filters.append(f"aresample={_silero_sample_rate}")

    if extra_effects:
        filters.append(extra_effects)

    if not filters:
        return input_path

    filter_chain = ",".join(filters)

    output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = output.name
    output.close()

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-af", filter_chain,
                "-loglevel", "error",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        os.unlink(input_path)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warning("Audio effects не удались (ffmpeg): %s, используем оригинал", e)
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
    """Синтез через Silero TTS + audio effects."""
    voice = voice or config.TTS_VOICE
    valid_speakers = ["aidar", "baya", "kseniya", "xenia", "eugene", "random"]
    if voice not in valid_speakers:
        log.warning("Голос '%s' не поддерживается Silero, использую 'xenia'", voice)
        voice = "xenia"

    # Определяем эффекты из пресета или дефолт
    extra_effects = ""
    if _current_preset and _current_preset in VOICE_PRESETS:
        extra_effects = VOICE_PRESETS[_current_preset].get("effects", "")

    pitch = config.TTS_PITCH_SEMITONES

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

        import scipy.io.wavfile
        scipy.io.wavfile.write(tmp_path, _silero_sample_rate, audio.numpy())

        # Применяем эффекты
        tmp_path = _apply_audio_effects(tmp_path, pitch, extra_effects)
        return tmp_path

    loop = asyncio.get_event_loop()
    tmp_path = await loop.run_in_executor(None, _generate)
    log.debug("Silero TTS: сохранено в %s (%d символов, голос: %s, pitch: %+.0f)",
              tmp_path, len(text), voice, pitch)
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
