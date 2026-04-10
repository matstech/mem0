import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_RUNTIME_CONFIG_PATH = Path("config/benchmark_runtime.toml")


def _load_toml(path):
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def load_runtime_config(path=None):
    config_path = Path(path or DEFAULT_RUNTIME_CONFIG_PATH)
    data = _load_toml(config_path)
    data["_config_path"] = config_path.resolve().as_posix()
    return data


def _non_empty(value):
    return value if isinstance(value, str) and value.strip() else None


def apply_runtime_env(runtime_config):
    api_cfg = runtime_config.get("api", {})
    env_updates = {
        "OPENAI_API_KEY": _non_empty(api_cfg.get("openai_api_key")),
        "OPENAI_BASE_URL": _non_empty(api_cfg.get("openai_base_url")),
        "HF_TOKEN": _non_empty(api_cfg.get("hf_token")),
    }
    for key, value in env_updates.items():
        if value is not None:
            os.environ[key] = value


def get_answer_model(runtime_config):
    return runtime_config.get("models", {}).get("answer_model", "gpt-4o-mini")


def get_embedding_settings(runtime_config):
    models_cfg = runtime_config.get("models", {})
    return {
        "provider": models_cfg.get("embedding_provider", "huggingface"),
        "model": models_cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"),
        "embedding_dims": models_cfg.get("embedding_dimensions", 384),
    }
