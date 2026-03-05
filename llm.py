"""LLM модуль на основе Google Gemini (бесплатный tier)."""

import logging
from collections import defaultdict
import google.generativeai as genai
import config

log = logging.getLogger(__name__)

# История разговоров: {user_id: [{"role": ..., "parts": ...}]}
_conversations: dict[int, list[dict]] = defaultdict(list)

_model = None
_personality = ""


def init(personality_text: str):
    """Инициализировать Gemini с заданной личностью."""
    global _model, _personality
    genai.configure(api_key=config.GEMINI_API_KEY)
    _personality = personality_text
    _model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=_personality,
    )
    log.info("Gemini инициализирован (модель: gemini-2.0-flash)")


def _trim_history(user_id: int):
    """Обрезать историю до лимита."""
    history = _conversations[user_id]
    if len(history) > config.MAX_HISTORY_PER_USER * 2:
        _conversations[user_id] = history[-(config.MAX_HISTORY_PER_USER * 2):]


async def chat(user_id: int, username: str, text: str) -> str:
    """
    Отправить сообщение и получить ответ.

    Args:
        user_id: Discord user ID
        username: отображаемое имя пользователя
        text: что сказал пользователь

    Returns:
        Ответ бота
    """
    if _model is None:
        return "Я ещё не готов, подожди секунду."

    # Добавляем контекст кто говорит
    user_message = f"[{username}]: {text}"

    history = _conversations[user_id]
    history.append({"role": "user", "parts": [user_message]})
    _trim_history(user_id)

    try:
        chat_session = _model.start_chat(history=history[:-1])
        response = await chat_session.send_message_async(user_message)
        reply = response.text.strip()

        history.append({"role": "model", "parts": [reply]})
        log.debug("LLM ответ для %s: %s", username, reply[:100])
        return reply

    except Exception as e:
        log.error("Ошибка Gemini: %s", e)
        # Убираем неудачное сообщение из истории
        if history and history[-1]["role"] == "user":
            history.pop()
        return "Чёт я затупил, повтори ещё раз."


def clear_history(user_id: int):
    """Очистить историю для пользователя."""
    _conversations.pop(user_id, None)


def clear_all_history():
    """Очистить всю историю."""
    _conversations.clear()


def set_personality(text: str):
    """Сменить личность на лету."""
    global _model, _personality
    _personality = text
    _model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=_personality,
    )
    clear_all_history()
    log.info("Личность обновлена")
