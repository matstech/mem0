import threading
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Dict, Optional


_ACTIVE_SCOPE: ContextVar[tuple[str, ...]] = ContextVar("mem0_llm_usage_active_scope", default=())


class LLMUsageCollector:
    """Collect prompt/completion usage reported by LLM providers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._totals = self._empty_bucket()
        self._scopes: Dict[str, Dict[str, int]] = {}

    @staticmethod
    def _empty_bucket() -> Dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    @staticmethod
    def _normalize_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _get_usage_field(usage: Any, field_name: str) -> Optional[int]:
        if usage is None:
            return None
        if isinstance(usage, dict):
            return LLMUsageCollector._normalize_int(usage.get(field_name))
        return LLMUsageCollector._normalize_int(getattr(usage, field_name, None))

    @classmethod
    def _extract_usage(cls, response: Any) -> Optional[Dict[str, int]]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return None

        prompt_tokens = cls._get_usage_field(usage, "prompt_tokens") or 0
        completion_tokens = cls._get_usage_field(usage, "completion_tokens") or 0
        total_tokens = cls._get_usage_field(usage, "total_tokens")
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "calls": 1,
        }

    @staticmethod
    def _increment(bucket: Dict[str, int], usage: Dict[str, int]) -> None:
        bucket["prompt_tokens"] += usage["prompt_tokens"]
        bucket["completion_tokens"] += usage["completion_tokens"]
        bucket["total_tokens"] += usage["total_tokens"]
        bucket["calls"] += usage["calls"]

    def _current_scope_name(self) -> Optional[str]:
        active_scope = _ACTIVE_SCOPE.get()
        if not active_scope:
            return None
        return active_scope[-1]

    @contextmanager
    def scope(self, scope_name: str):
        token = _ACTIVE_SCOPE.set(_ACTIVE_SCOPE.get() + (scope_name,))
        try:
            yield
        finally:
            _ACTIVE_SCOPE.reset(token)

    def record_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        calls: int = 1,
    ) -> None:
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        usage = {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
            "calls": int(calls),
        }
        scope_name = self._current_scope_name()

        with self._lock:
            self._increment(self._totals, usage)
            if scope_name:
                scoped_bucket = self._scopes.setdefault(scope_name, self._empty_bucket())
                self._increment(scoped_bucket, usage)

    def record_response(self, response: Any) -> None:
        usage = self._extract_usage(response)
        if usage is None:
            return
        self.record_usage(**usage)

    def callback(self, _llm: Any, response: Any, _params: Dict[str, Any]) -> None:
        self.record_response(response)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            totals = deepcopy(self._totals)
            scopes = deepcopy(self._scopes)
        return {**totals, "scopes": scopes}

    def reset(self) -> None:
        with self._lock:
            self._totals = self._empty_bucket()
            self._scopes = {}
