"""LLM модуль на основе Google Gemini (бесплатный tier)."""

import asyncio
import logging
from collections import defaultdict
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
import config

log = logging.getLogger(__name__)

# История разговоров: {user_id: [{"role": ..., "parts": ...}]}
_conversations: dict[int, list[dict]] = defaultdict(list)

_model = None
_personality = ""

# Ротация ключей
_api_keys: list[str] = []
_current_key_index: int = 0

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # секунды


def init(personality_text: str):
    """Инициализировать Gemini с заданной личностью."""
    global _model, _personality, _api_keys, _current_key_index
    _api_keys = config.GEMINI_API_KEYS.copy()
    _current_key_index = 0

    if not _api_keys:
        log.error("GEMINI_API_KEY не задан!")
        return

    genai.configure(api_key=_api_keys[_current_key_index])
    _personality = personality_text
    _model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=_personality,
    )
    log.info("Gemini инициализирован (модель: gemini-2.0-flash, ключей: %d)", len(_api_keys))


def _rotate_key() -> bool:
    """Переключиться на следующий API ключ. Возвращает True если удалось."""
    global _model, _current_key_index
    if len(_api_keys) <= 1:
        return False

    _current_key_index = (_current_key_index + 1) % len(_api_keys)
    genai.configure(api_key=_api_keys[_current_key_index])
    _model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=_personality,
    )
    log.info("Переключен на API ключ #%d", _current_key_index + 1)
    return True


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

    last_error = None
    keys_tried = 0

    while keys_tried <= len(_api_keys):
        for attempt in range(MAX_RETRIES):
            try:
                chat_session = _model.start_chat(history=history[:-1])
                response = await chat_session.send_message_async(user_message)
                reply = response.text.strip()

                history.append({"role": "model", "parts": [reply]})
                log.debug("LLM ответ для %s: %s", username, reply[:100])
                return reply

            except ResourceExhausted as e:
                last_error = e
                log.warning("Квота Gemini исчерпана (ключ #%d, попытка %d/%d): %s",
                            _current_key_index + 1, attempt + 1, MAX_RETRIES, e)

                # Если есть retry_delay в ответе и он маленький — ждём
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    if delay <= 10:
                        log.info("Повтор через %.1f сек...", delay)
                        await asyncio.sleep(delay)
                    else:
                        break  # Слишком долго ждать, пробуем другой ключ

            except GoogleAPIError as e:
                last_error = e
                log.error("Ошибка Gemini API: %s", e)
                break  # Не retry-able

            except Exception as e:
                last_error = e
                log.error("Ошибка Gemini: %s", e)
                break

        # Retry исчерпаны — пробуем следующий ключ
        if _rotate_key():
            keys_tried += 1
            log.info("Пробую следующий API ключ...")
            continue
        else:
            break

    # Все попытки исчерпаны
    if history and history[-1]["role"] == "user":
        history.pop()

    if isinstance(last_error, ResourceExhausted):
        log.error("Все ключи исчерпали квоту Gemini")
        return "У меня закончился лимит запросов. Попробуйте позже или добавьте новый API ключ."

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
