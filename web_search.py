"""Модуль веб-поиска через DuckDuckGo (бесплатно, без API ключа)."""

import asyncio
import logging

log = logging.getLogger(__name__)


async def search(query: str, max_results: int = 3) -> str:
    """
    Поиск в интернете через DuckDuckGo.

    Returns:
        Строка с результатами поиска для вставки в контекст LLM.
    """
    def _search():
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "Ничего не найдено."
            lines = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                lines.append(f"- {title}: {body}")
            return "\n".join(lines)
        except Exception as e:
            log.error("Ошибка веб-поиска: %s", e)
            return f"Ошибка поиска: {e}"

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _search)
