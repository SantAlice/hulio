"""Обработчик голоса — приём аудио из Discord, распознавание, ответ."""

import asyncio
import logging
import os
import time
from collections import defaultdict

import discord
from discord.ext import commands

try:
    from discord.ext import voice_recv
    HAS_VOICE_RECV = True
except ImportError:
    HAS_VOICE_RECV = False

try:
    from davey import MediaType
    HAS_DAVEY = True
except ImportError:
    HAS_DAVEY = False

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


if HAS_VOICE_RECV:
    class DaveAudioSink(voice_recv.AudioSink):
        """
        Кастомный sink, расшифровывающий DAVE E2EE перед декодированием Opus.

        С марта 2026 Discord шифрует все голосовые пакеты через DAVE.
        BasicSink получает зашифрованные данные и декодирует мусор.
        Этот sink: DAVE decrypt → Opus decode → PCM.
        """

        def __init__(self, callback, voice_client):
            self.callback = callback
            self.vc = voice_client
            self.decoder = discord.opus.Decoder()
            self._use_dave = HAS_DAVEY and hasattr(self.vc, '_connection') and hasattr(getattr(self.vc, '_connection', None), 'dave_session')

            if self._use_dave:
                try:
                    self.vc._connection.dave_session.set_passthrough_mode(True, 10)
                    log.info("DAVE passthrough режим включён — ручная расшифровка")
                except Exception as e:
                    log.warning("Не удалось включить DAVE passthrough: %s", e)
                    self._use_dave = False

        @voice_recv.AudioSink.listener()
        def on_voice_member_speaking_start(self, member):
            pass

        @voice_recv.AudioSink.listener()
        def on_voice_member_speaking_stop(self, member):
            pass

        def wants_opus(self) -> bool:
            return True

        def write(self, user, data):
            if user is None:
                return

            try:
                if self._use_dave:
                    opus_data = self.vc._connection.dave_session.decrypt(
                        user.id, MediaType.audio, bytes(data.opus)
                    )
                    if not opus_data:
                        return
                    pcm = self.decoder.decode(opus_data, fec=False)
                else:
                    pcm = data.pcm
            except Exception as e:
                log.debug("Ошибка расшифровки DAVE для %s: %s", user, e)
                return

            self.callback(user, pcm)

        def cleanup(self):
            pass


class VoiceHandler:
    """Управляет приёмом и отправкой голоса в Discord канале."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.buffers: dict[int, UserAudioBuffer] = defaultdict(UserAudioBuffer)
        self.processing: set[int] = set()
        self.voice_client: discord.VoiceClient | None = None
        self._monitor_task: asyncio.Task | None = None
        self._active = False

    async def join(self, channel: discord.VoiceChannel):
        """Подключиться к голосовому каналу."""
        if self.voice_client and self.voice_client.is_connected():
            await self.voice_client.move_to(channel)
        else:
            if HAS_VOICE_RECV:
                self.voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
            else:
                self.voice_client = await channel.connect()
                log.warning("voice_recv не установлен — приём аудио недоступен")

        # Начинаем слушать если есть voice_recv
        if HAS_VOICE_RECV and isinstance(self.voice_client, voice_recv.VoiceRecvClient):
            sink = DaveAudioSink(self._on_audio_data, self.voice_client)
            self.voice_client.listen(sink)

        self._active = True
        self._monitor_task = asyncio.create_task(self._silence_monitor())
        log.info("Подключен к каналу: %s", channel.name)

    async def leave(self):
        """Отключиться от голосового канала."""
        self._active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

        if self.voice_client:
            if HAS_VOICE_RECV and isinstance(self.voice_client, voice_recv.VoiceRecvClient):
                self.voice_client.stop_listening()
            await self.voice_client.disconnect()
            self.voice_client = None

        self.buffers.clear()
        self.processing.clear()
        log.info("Отключен от голосового канала")

    def _on_audio_data(self, user, pcm_data):
        """Колбэк для расшифрованного PCM аудио."""
        if user is None:
            return
        buf = self.buffers[user.id]
        buf.add_chunk(pcm_data)

    async def _silence_monitor(self):
        """
        Периодически проверяем буферы пользователей.
        Если пользователь замолчал (нет данных > SILENCE_DURATION), обрабатываем.
        """
        while self._active:
            await asyncio.sleep(0.5)

            if not self.voice_client or not self.voice_client.is_connected():
                break

            now = time.time()
            for user_id, buf in list(self.buffers.items()):
                if not buf.is_speaking:
                    continue
                if buf.total_bytes < 3200:
                    continue

                silence_elapsed = now - buf.last_voice_time
                if silence_elapsed >= config.SILENCE_DURATION or buf.duration_seconds >= config.MAX_RECORD_DURATION:
                    audio_bytes = buf.get_audio()
                    buf.clear()
                    if len(audio_bytes) > 3200:
                        guild = self.voice_client.guild if self.voice_client else None
                        channel = self.voice_client.channel if self.voice_client else None
                        asyncio.create_task(
                            self._process_user_audio(user_id, audio_bytes, guild)
                        )

    async def _process_user_audio(
        self, user_id: int, audio_bytes: bytes, guild: discord.Guild | None
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
            username = f"User#{user_id}"
            if guild:
                member = guild.get_member(user_id)
                if member:
                    username = member.display_name
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
