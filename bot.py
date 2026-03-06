#!/usr/bin/env python3
"""
Хулио — голосовой бот для Discord.

Слушает голосовой канал, распознаёт речь, отвечает голосом.
Бесплатные технологии: Vosk (STT), Edge TTS (TTS), Gemini (LLM).
"""

import asyncio
import logging
import os
import sys

import discord
from discord import app_commands
from discord.ext import commands

import config
import llm
import tts
from voice_handler import VoiceHandler

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")

# Создаём бота
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)
voice_handler: VoiceHandler | None = None
_synced = False


def load_personality(path: str) -> str:
    """Загрузить файл личности."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    log.warning("Файл личности не найден: %s, используем дефолт", path)
    return "Ты — дружелюбный голосовой помощник. Отвечай коротко и по делу."


@bot.event
async def on_ready():
    global voice_handler, _synced
    log.info("Бот запущен как %s (ID: %s)", bot.user.name, bot.user.id)

    # Инициализируем LLM
    personality = load_personality(config.PERSONALITY_FILE)
    llm.init(personality)

    # Создаём обработчик голоса
    voice_handler = VoiceHandler(bot)

    # Синхронизируем slash-команды (только один раз)
    if not _synced:
        try:
            synced = await bot.tree.sync()
            _synced = True
            log.info("Синхронизировано %d команд", len(synced))
        except Exception as e:
            log.error("Ошибка синхронизации команд: %s", e)

    log.info("Готов к работе! Используй /join чтобы пригласить в войс.")


# === Slash Commands ===

@bot.tree.command(name="join", description="Присоединиться к твоему голосовому каналу")
async def join_cmd(interaction: discord.Interaction):
    member = interaction.guild.get_member(interaction.user.id)
    if not member or not member.voice:
        await interaction.response.send_message(
            "Ты не в голосовом канале! Зайди в войс сначала.", ephemeral=True
        )
        return

    channel = member.voice.channel
    await interaction.response.defer()

    try:
        await voice_handler.join(channel)
        await interaction.followup.send(f"Я в **{channel.name}**! Говорите — я слушаю.")
    except Exception as e:
        log.error("Ошибка подключения: %s", e)
        await interaction.followup.send(f"Не могу подключиться: {e}")


@bot.tree.command(name="leave", description="Выйти из голосового канала")
async def leave_cmd(interaction: discord.Interaction):
    if not voice_handler or not voice_handler.voice_client:
        await interaction.response.send_message("Я и так не в войсе.", ephemeral=True)
        return

    await voice_handler.leave()
    await interaction.response.send_message("Всё, я вышел. Пока!")


@bot.tree.command(name="voice", description="Сменить голос бота")
@app_commands.describe(voice_name="Название голоса (например ru-RU-DmitryNeural)")
async def voice_cmd(interaction: discord.Interaction, voice_name: str):
    config.TTS_VOICE = voice_name
    await interaction.response.send_message(f"Голос изменён на **{voice_name}**")


@bot.tree.command(name="voices", description="Показать доступные голоса")
@app_commands.describe(language="Язык (ru, en, ja, ...)")
async def voices_cmd(interaction: discord.Interaction, language: str = "ru"):
    await interaction.response.defer()
    voices = await tts.list_voices(language)
    if not voices:
        await interaction.followup.send(f"Голоса для языка '{language}' не найдены.")
        return

    lines = []
    for v in voices[:20]:
        gender = "М" if v["Gender"] == "Male" else "Ж" if v["Gender"] == "Female" else "?"
        name = v.get('FriendlyName', v['ShortName'])
        engine = v.get('Engine', 'edge')
        lines.append(f"`{v['ShortName']}` — {name} ({gender}, {engine})")

    text = f"**Голоса для '{language}':**\n" + "\n".join(lines)
    if len(voices) > 20:
        text += f"\n... и ещё {len(voices) - 20}"
    await interaction.followup.send(text)


@bot.tree.command(name="personality", description="Сменить личность бота")
@app_commands.describe(name="Имя личности (default, anime_girl, toxic_gamer) или путь к файлу")
async def personality_cmd(interaction: discord.Interaction, name: str):
    path = name
    if not os.path.exists(path):
        path = f"personalities/{name}.txt"
    if not os.path.exists(path):
        await interaction.response.send_message(
            f"Файл личности не найден: `{name}`\n"
            f"Доступные: {', '.join(list_personalities())}",
            ephemeral=True,
        )
        return

    personality_text = load_personality(path)
    llm.set_personality(personality_text)
    config.PERSONALITY_FILE = path
    await interaction.response.send_message(
        f"Личность изменена на **{name}**! История очищена."
    )


@bot.tree.command(name="clear", description="Очистить историю разговоров")
async def clear_cmd(interaction: discord.Interaction):
    llm.clear_all_history()
    await interaction.response.send_message("История очищена! Начинаем с чистого листа.")


@bot.tree.command(name="rate", description="Скорость речи бота")
@app_commands.describe(speed="Скорость (например +20%, -10%, +0%)")
async def rate_cmd(interaction: discord.Interaction, speed: str):
    config.TTS_RATE = speed
    await interaction.response.send_message(f"Скорость речи: **{speed}**")


@bot.tree.command(name="pitch", description="Сдвиг тона голоса в полутонах")
@app_commands.describe(semitones="Полутоны (-8 = очень басистый, -4 = басистый, 0 = норма, +4 = мультяшный)")
async def pitch_cmd(interaction: discord.Interaction, semitones: float):
    config.TTS_PITCH_SEMITONES = semitones
    if semitones < 0:
        desc = "басистый"
    elif semitones > 0:
        desc = "мультяшный"
    else:
        desc = "нормальный"
    await interaction.response.send_message(f"Тон голоса: **{semitones:+.0f}** полутонов ({desc})")


@bot.tree.command(name="say", description="Заставить бота сказать что-то в войс")
@app_commands.describe(text="Текст для озвучки")
async def say_cmd(interaction: discord.Interaction, text: str):
    if not voice_handler or not voice_handler.voice_client:
        await interaction.response.send_message(
            "Я не в войсе! Используй /join", ephemeral=True
        )
        return

    await interaction.response.defer()
    audio_file = await tts.synthesize(text)
    await voice_handler._play_audio(audio_file)
    try:
        os.unlink(audio_file)
    except OSError:
        pass
    await interaction.followup.send(f"Сказал: *{text}*")


def list_personalities() -> list[str]:
    """Получить список доступных личностей."""
    if not os.path.exists("personalities"):
        return []
    return [
        f.replace(".txt", "")
        for f in os.listdir("personalities")
        if f.endswith(".txt")
    ]


def main():
    if not config.DISCORD_TOKEN:
        print("ОШИБКА: Установи DISCORD_TOKEN в .env файле!")
        print("Скопируй .env.example в .env и заполни токены.")
        sys.exit(1)

    if not config.LLM_API_KEY:
        print("ОШИБКА: Установи LLM_API_KEY в .env файле!")
        print("DeepSeek: https://platform.deepseek.com/api_keys")
        sys.exit(1)

    # Проверяем модель Vosk (опционально — есть фоллбэк на Google SR)
    if not os.path.exists(config.VOSK_MODEL_PATH):
        log.warning("Модель Vosk не найдена: %s — будет использован Google Speech Recognition (онлайн)", config.VOSK_MODEL_PATH)

    log.info("Запуск бота %s...", config.BOT_NAME)
    bot.run(config.DISCORD_TOKEN)


if __name__ == "__main__":
    main()
