import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROMPTS_PATH = Path(__file__).with_name("prompts.json")
PROMPT_DOCUMENTS_TABLE = "ai_prompt_documents"
BACKEND_PROMPT_FILE_KEY = "backend_prompts"


@lru_cache(maxsize=1)
def _load_prompts() -> dict[str, Any]:
    try:
        import main
        if main.supabase:
            response = (
                main.supabase
                .table(PROMPT_DOCUMENTS_TABLE)
                .select("content")
                .eq("file_key", BACKEND_PROMPT_FILE_KEY)
                .limit(1)
                .execute()
            )
            rows = response.data or []
            content = rows[0].get("content") if rows else None
            if isinstance(content, dict):
                return content
    except Exception:
        pass

    with PROMPTS_PATH.open("r", encoding="utf-8") as prompt_file:
        return json.load(prompt_file)


def clear_prompt_cache() -> None:
    _load_prompts.cache_clear()


def get_prompt(key: str) -> str:
    prompt = _load_prompts().get(key)
    if not isinstance(prompt, str) or not prompt.strip():
        raise KeyError(f"Prompt '{key}' not found in {PROMPTS_PATH.name}")
    return prompt
