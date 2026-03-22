import copy
import json
from pathlib import Path
from typing import Any, Dict

from .config import DEFAULT_USER_PROFILE, PROFILE_STORE_PATH


def _store_path() -> Path:
    return Path(PROFILE_STORE_PATH)


def _ensure_store_exists() -> None:
    p = _store_path()
    if not p.exists():
        p.write_text("{}", encoding="utf-8")


def _read_store() -> Dict[str, Any]:
    _ensure_store_exists()
    p = _store_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_store(data: Dict[str, Any]) -> None:
    p = _store_path()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_user_profile(user_id: str) -> Dict[str, Any]:
    data = _read_store()
    if user_id not in data:
        data[user_id] = copy.deepcopy(DEFAULT_USER_PROFILE)
        _write_store(data)
    return data[user_id]


def update_user_profile(user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    data = _read_store()
    current = data.get(user_id, copy.deepcopy(DEFAULT_USER_PROFILE))
    merged = _deep_merge(current, patch)
    data[user_id] = merged
    _write_store(data)
    return merged


def profile_to_prompt_text(profile: Dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=False)
