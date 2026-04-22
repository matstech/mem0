import logging
import random
import time

from openai import RateLimitError


logger = logging.getLogger(__name__)


def _should_retry_rate_limit_error(error):
    error_text = str(error).lower()
    if any(
        marker in error_text
        for marker in (
            "insufficient_quota",
            "insufficient quota",
            "exceeded your current quota",
            "billing_hard_limit_reached",
        )
    ):
        return False
    return (
        isinstance(error, RateLimitError)
        or getattr(error, "status_code", None) == 429
        or "rate limit" in error_text
        or "too many requests" in error_text
    )


def call_with_rate_limit_retry(func, *args, request_name="OpenAI request", **kwargs):
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as error:
            if not _should_retry_rate_limit_error(error):
                raise

            delay = min(2**attempt, 60) + random.uniform(0, 0.5)
            logger.warning("%s hit a rate limit, retrying in %.2fs", request_name, delay)
            time.sleep(delay)
            attempt += 1
