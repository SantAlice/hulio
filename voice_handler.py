"""Обработчик голоса — приём аудио из Discord, распознавание, ответ."""

import asyncio
import logging
import os
import time
from collections import defaultdict

import discord

import config
import stt
import tts
import llm

log = logging.getLogger(__name__)


class UserAudioBuffer:
    """Буфер аудио для одного пользователя."""

    def __init__(self):
        self.chunks: list[bytes] = []
        self.last_voice_time: float = 0.0
        self.is_speaking: bool = False
        self.total_bytes: int = 0

    def add_chunk(self, data: bytes):
        self.chunks.append(data)
        self.total_bytes += len(data)
        self.last_voice_time = time.time()
        self.is_speaking = True

    def get_audio(self) -> bytes:
        return b"".join(self.chunks)

    def clear(self):
        self.chunks.clear()
        self.total_bytes = 0
        self.is_speaking = False

    @property
    def duration_seconds(self) -> float:
        # 16-bit stereo 48kHz = 192000 bytes/sec
        return self.total_bytes / 192000.0


class VoiceHandler:
    """Управляет приёмом и отправкой голоса в Discord канале."""

    def __init__(self, bot: discord.Bot):
        self.bot = bot
        self.buffers: dict[int, UserAudioBuffer] = defaultdict(UserAudioBuffer)
        self.processing: set[int] = set()  # user_ids currently being processed
        self.voice_client: discord.VoiceClient | None = None
        self._monitor_task: asyncio.Task | None = None
        self._active = False

    async def join(self, channel: discord.VoiceChannel) -> discord.VoiceClient:
        """Подключиться к голосовому каналу."""
        if self.voice_client and self.voice_client.is_connected():
            await self.voice_client.move_to(channel)
        else:
            self.voice_client = await channel.connect(cls=discord.VoiceClient)

        # Начинаем слушать
        self.voice_client.start_recording(
            discord.sinks.PCMSink(),
            self._recording_finished,
            channel,
        )

        self._active = True
        self._monitor_task = asyncio.create_task(self._silence_monitor())
        log.info("Подключен к каналу: %s", channel.name)
        return self.voice_client

    async def leave(self):
        """Отключиться от голосового канала."""
        self._active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

        if self.voice_client:
            if self.voice_client.recording:
                self.voice_client.stop_recording()
            await self.voice_client.disconnect()
            self.voice_client = None

        self.buffers.clear()
        self.processing.clear()
        log.info("Отключен от голосового канала")

    async def _recording_finished(self, sink: discord.sinks.PCMSink, channel: discord.VoiceChannel):
        """Колбэк когда запись останавливается — обрабатываем всё накопленное аудио."""
        for user_id, audio_data in sink.audio_data.items():
            audio_bytes = audio_data.file.read()
            if len(audio_bytes) < 3200:  # слишком короткий фрагмент
                continue
            await self._process_user_audio(user_id, audio_bytes, channel)

    async def _silence_monitor(self):
        """
        Периодически останавливаем и перезапускаем запись,
        чтобы обработать накопленное аудио.
        """
        while self._active:
            await asyncio.sleep(config.SILENCE_DURATION)

            if not self.voice_client or not self.voice_client.is_connected():
                break

            if not self.voice_client.recording:
                continue

            try:
                # Останавливаем запись — вызовется _recording_finished
                self.voice_client.stop_recording()
                # Небольшая пауза и перезапуск
                await asyncio.sleep(0.3)
                if self._active and self.voice_client and self.voice_client.is_connected():
                    self.voice_client.start_recording(
                        discord.sinks.PCMSink(),
                        self._recording_finished,
                        self.voice_client.channel,
                    )
            except Exception as e:
                log.error("Ошибка в мониторе тишины: %s", e)
                await asyncio.sleep(1)

    async def _process_user_audio(
        self, user_id: int, audio_bytes: bytes, channel: discord.VoiceChannel
    ):
        """Обработать аудио одного пользователя: STT -> LLM -> TTS -> Play."""
        if user_id in self.processing:
            return

        self.processing.add(user_id)
        try:
            # Конвертируем аудио для Vosk
            converted = stt.pcm_stereo_48k_to_mono_16k(audio_bytes)
            if len(converted) < 1600:
                return

            # Распознаём речь
            text = stt.recognize(converted, config.VOSK_SAMPLE_RATE)
            if not text or len(text) < 2:
                return

            # Определяем имя пользователя
            member = channel.guild.get_member(user_id)
            username = member.display_name if member else f"User#{user_id}"
            log.info("[%s] сказал: %s", username, text)

            # Получаем ответ от LLM
            reply = await llm.chat(user_id, username, text)
            log.info("[%s] ответ: %s", config.BOT_NAME, reply)

            # Синтезируем речь
            audio_file = await tts.synthesize(reply)

            # Проигрываем в канале
            await self._play_audio(audio_file)

            # Удаляем временный файл
            try:
                os.unlink(audio_file)
            except OSError:
                pass

        except Exception as e:
            log.error("Ошибка обработки аудио от %s: %s", user_id, e)
        finally:
            self.processing.discard(user_id)

    async def _play_audio(self, file_path: str):
        """Проиграть аудиофайл в голосовом канале."""
        if not self.voice_client or not self.voice_client.is_connected():
            return

        # Ждём пока текущее аудио доиграет
        while self.voice_client.is_playing():
            await asyncio.sleep(0.1)

        source = discord.FFmpegPCMAudio(file_path)
        self.voice_client.play(source)

        # Ждём окончания воспроизведения
        while self.voice_client.is_playing():
            await asyncio.sleep(0.1)
