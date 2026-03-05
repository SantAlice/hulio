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

bot = discord.Bot(intents=intents)
voice_handler: VoiceHandler | None = None


def load_personality(path: str) -> str:
    """Загрузить файл личности."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    log.warning("Файл личности не найден: %s, используем дефолт", path)
    return "Ты — дружелюбный голосовой помощник. Отвечай коротко и по делу."


@bot.event
async def on_ready():
    global voice_handler
    log.info("Бот запущен как %s (ID: %s)", bot.user.name, bot.user.id)

    # Инициализируем LLM
    personality = load_personality(config.PERSONALITY_FILE)
    llm.init(personality)

    # Создаём обработчик голоса
    voice_handler = VoiceHandler(bot)

    log.info("Готов к работе! Используй /join чтобы пригласить в войс.")


# === Slash Commands ===

@bot.slash_command(name="join", description="Присоединиться к твоему голосовому каналу")
async def join_cmd(ctx: discord.ApplicationContext):
    # Получаем участника из кэша гильдии (ctx.author может не содержать voice state)
    member = ctx.guild.get_member(ctx.author.id)
    if not member or not member.voice:
        await ctx.respond("Ты не в голосовом канале! Зайди в войс сначала.", ephemeral=True)
        return

    channel = member.voice.channel
    await ctx.respond(f"Захожу в **{channel.name}**...")

    try:
        await voice_handler.join(channel)
        await ctx.edit(content=f"Я в **{channel.name}**! Говорите — я слушаю.")
    except Exception as e:
        log.error("Ошибка подключения: %s", e)
        await ctx.edit(content=f"Не могу подключиться: {e}")


@bot.slash_command(name="leave", description="Выйти из голосового канала")
async def leave_cmd(ctx: discord.ApplicationContext):
    if not voice_handler or not voice_handler.voice_client:
        await ctx.respond("Я и так не в войсе.", ephemeral=True)
        return

    await voice_handler.leave()
    await ctx.respond("Всё, я вышел. Пока!")


@bot.slash_command(name="voice", description="Сменить голос бота")
async def voice_cmd(
    ctx: discord.ApplicationContext,
    voice_name: discord.Option(
        str,
        "Название голоса (например ru-RU-DmitryNeural)",
        required=True,
    ),
):
    config.TTS_VOICE = voice_name
    await ctx.respond(f"Голос изменён на **{voice_name}**")


@bot.slash_command(name="voices", description="Показать доступные голоса")
async def voices_cmd(
    ctx: discord.ApplicationContext,
    language: discord.Option(str, "Язык (ru, en, ja, ...)", default="ru"),
):
    await ctx.defer()
    voices = await tts.list_voices(language)
    if not voices:
        await ctx.respond(f"Голоса для языка '{language}' не найдены.")
        return

    lines = []
    for v in voices[:20]:
        gender = "М" if v["Gender"] == "Male" else "Ж"
        lines.append(f"`{v['ShortName']}` — {v['FriendlyName']} ({gender})")

    text = f"**Голоса для '{language}':**\n" + "\n".join(lines)
    if len(voices) > 20:
        text += f"\n... и ещё {len(voices) - 20}"
    await ctx.respond(text)


@bot.slash_command(name="personality", description="Сменить личность бота")
async def personality_cmd(
    ctx: discord.ApplicationContext,
    name: discord.Option(
        str,
        "Имя личности (default, anime_girl, toxic_gamer) или путь к файлу",
        required=True,
    ),
):
    # Пробуем найти файл
    path = name
    if not os.path.exists(path):
        path = f"personalities/{name}.txt"
    if not os.path.exists(path):
        await ctx.respond(
            f"Файл личности не найден: `{name}`\n"
            f"Доступные: {', '.join(list_personalities())}",
            ephemeral=True,
        )
        return

    personality_text = load_personality(path)
    llm.set_personality(personality_text)
    config.PERSONALITY_FILE = path
    await ctx.respond(f"Личность изменена на **{name}**! История очищена.")


@bot.slash_command(name="clear", description="Очистить историю разговоров")
async def clear_cmd(ctx: discord.ApplicationContext):
    llm.clear_all_history()
    await ctx.respond("История очищена! Начинаем с чистого листа.")


@bot.slash_command(name="rate", description="Скорость речи бота")
async def rate_cmd(
    ctx: discord.ApplicationContext,
    speed: discord.Option(str, "Скорость (например +20%, -10%, +0%)", required=True),
):
    config.TTS_RATE = speed
    await ctx.respond(f"Скорость речи: **{speed}**")


@bot.slash_command(name="say", description="Заставить бота сказать что-то в войс")
async def say_cmd(
    ctx: discord.ApplicationContext,
    text: discord.Option(str, "Текст для озвучки", required=True),
):
    if not voice_handler or not voice_handler.voice_client:
        await ctx.respond("Я не в войсе! Используй /join", ephemeral=True)
        return

    await ctx.defer()
    audio_file = await tts.synthesize(text)
    await voice_handler._play_audio(audio_file)
    try:
        os.unlink(audio_file)
    except OSError:
        pass
    await ctx.respond(f"Сказал: *{text}*")


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

    if not config.GEMINI_API_KEY:
        print("ОШИБКА: Установи GEMINI_API_KEY в .env файле!")
        print("Получи бесплатный ключ: https://aistudio.google.com/apikey")
        sys.exit(1)

    if not os.path.exists(config.VOSK_MODEL_PATH):
        print(f"ОШИБКА: Модель Vosk не найдена: {config.VOSK_MODEL_PATH}")
        print("Скачай модель:")
        print("  mkdir -p models && cd models")
        print("  wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip")
        print("  unzip vosk-model-small-ru-0.22.zip")
        sys.exit(1)

    log.info("Запуск бота %s...", config.BOT_NAME)
    bot.run(config.DISCORD_TOKEN)


if __name__ == "__main__":
    main()
