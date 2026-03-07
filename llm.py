"""LLM модуль с поддержкой DeepSeek и Gemini (OpenAI-совместимый API) + веб-поиск."""

import asyncio
import json
import logging
from collections import defaultdict
from openai import AsyncOpenAI, RateLimitError, APIError
import config

log = logging.getLogger(__name__)

# История разговоров: {user_id: [{"role": ..., "content": ...}]}
_conversations: dict[int, list[dict]] = defaultdict(list)

_client: AsyncOpenAI | None = None
_personality = ""
_model_name = ""

# Ротация ключей
_api_keys: list[str] = []
_current_key_index: int = 0

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

# Флаг: модель поддерживает tools
_tools_supported = True

# Определение инструментов для LLM (веб-поиск)
_tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Поиск в интернете. Используй когда нужна актуальная информация, факты, новости, цены, погода, или когда не уверен в ответе.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на русском или английском"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def init(personality_text: str):
    """Инициализировать LLM с заданной личностью."""
    global _client, _personality, _model_name, _api_keys, _current_key_index, _tools_supported
    _api_keys = config.LLM_API_KEYS.copy()
    _current_key_index = 0
    _personality = personality_text
    _model_name = config.LLM_MODEL
    _tools_supported = True  # Сбрасываем при смене модели

    if not _api_keys:
        log.error("LLM_API_KEY не задан!")
        return

    _client = AsyncOpenAI(
        api_key=_api_keys[_current_key_index],
        base_url=config.LLM_BASE_URL,
    )
    log.info("LLM инициализирован (модель: %s, base_url: %s, ключей: %d)",
             _model_name, config.LLM_BASE_URL, len(_api_keys))


def _rotate_key() -> bool:
    """Переключиться на следующий API ключ."""
    global _client, _current_key_index
    if len(_api_keys) <= 1:
        return False

    _current_key_index = (_current_key_index + 1) % len(_api_keys)
    _client = AsyncOpenAI(
        api_key=_api_keys[_current_key_index],
        base_url=config.LLM_BASE_URL,
    )
    log.info("Переключен на API ключ #%d", _current_key_index + 1)
    return True


def _trim_history(user_id: int):
    """Обрезать историю до лимита."""
    history = _conversations[user_id]
    if len(history) > config.MAX_HISTORY_PER_USER * 2:
        _conversations[user_id] = history[-(config.MAX_HISTORY_PER_USER * 2):]


async def _execute_tool_call(tool_call) -> str:
    """Выполнить вызов инструмента и вернуть результат."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return "Ошибка парсинга аргументов"

    if name == "web_search":
        query = args.get("query", "")
        log.info("Веб-поиск: %s", query)
        try:
            import web_search
            return await web_search.search(query)
        except Exception as e:
            log.error("Ошибка веб-поиска: %s", e)
            return f"Поиск недоступен: {e}"

    return f"Неизвестный инструмент: {name}"


async def chat(user_id: int, username: str, text: str) -> str:
    """Отправить сообщение и получить ответ (с поддержкой tool calling)."""
    global _tools_supported

    if _client is None:
        return "Я ещё не готов, подожди секунду."

    user_message = f"[{username}]: {text}"

    history = _conversations[user_id]
    history.append({"role": "user", "content": user_message})
    _trim_history(user_id)

    # Собираем сообщения для API
    messages = [{"role": "system", "content": _personality}] + history

    last_error = None
    keys_tried = 0

    while keys_tried <= len(_api_keys):
        for attempt in range(MAX_RETRIES):
            try:
                # Формируем запрос
                kwargs = {
                    "model": _model_name,
                    "messages": messages,
                    "max_tokens": config.LLM_MAX_TOKENS,
                    "temperature": 0.9,
                }
                # Добавляем tools только если модель их поддерживает
                if _tools_supported:
                    kwargs["tools"] = _tools
                    kwargs["tool_choice"] = "auto"

                response = await _client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                msg = choice.message

                # Проверяем, хочет ли LLM вызвать инструмент
                if msg.tool_calls and _tools_supported:
                    # Добавляем ответ ассистента с tool_calls в контекст
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                            }
                            for tc in msg.tool_calls
                        ]
                    })

                    # Выполняем все tool calls
                    for tc in msg.tool_calls:
                        result = await _execute_tool_call(tc)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })

                    # Второй вызов LLM с результатами поиска (без tools)
                    response = await _client.chat.completions.create(
                        model=_model_name,
                        messages=messages,
                        max_tokens=config.LLM_MAX_TOKENS,
                        temperature=0.9,
                    )
                    reply = response.choices[0].message.content.strip()
                else:
                    reply = msg.content.strip()

                history.append({"role": "assistant", "content": reply})
                log.debug("LLM ответ для %s: %s", username, reply[:100])
                return reply

            except RateLimitError as e:
                last_error = e
                log.warning("Квота LLM исчерпана (ключ #%d, попытка %d/%d): %s",
                            _current_key_index + 1, attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    if delay <= 10:
                        log.info("Повтор через %.1f сек...", delay)
                        await asyncio.sleep(delay)
                    else:
                        break

            except (APIError, Exception) as e:
                last_error = e
                error_str = str(e).lower()
                # Если модель не поддерживает tools — отключаем и пробуем без них
                if _tools_supported and ("tool" in error_str or "function" in error_str
                                          or "unsupported" in error_str or "invalid" in error_str):
                    log.warning("Модель не поддерживает tools, отключаю: %s", e)
                    _tools_supported = False
                    # Пробуем заново без tools
                    try:
                        response = await _client.chat.completions.create(
                            model=_model_name,
                            messages=messages,
                            max_tokens=config.LLM_MAX_TOKENS,
                            temperature=0.9,
                        )
                        reply = response.choices[0].message.content.strip()
                        history.append({"role": "assistant", "content": reply})
                        return reply
                    except Exception as e2:
                        log.error("Ошибка LLM без tools: %s", e2)
                        last_error = e2
                        break

                log.error("Ошибка LLM: %s", e)
                break

        if _rotate_key():
            keys_tried += 1
            log.info("Пробую следующий API ключ...")
            continue
        else:
            break

    # Все попытки исчерпаны
    if history and history[-1]["role"] == "user":
        history.pop()

    if isinstance(last_error, RateLimitError):
        log.error("Все ключи исчерпали квоту LLM")
        return "У меня закончился лимит запросов. Попробуйте позже."

    return "Чёт я затупил, повтори ещё раз."


def clear_history(user_id: int):
    """Очистить историю для пользователя."""
    _conversations.pop(user_id, None)


def clear_all_history():
    """Очистить всю историю."""
    _conversations.clear()


def set_personality(text: str):
    """Сменить личность на лету."""
    global _personality
    _personality = text
    clear_all_history()
    log.info("Личность обновлена")
