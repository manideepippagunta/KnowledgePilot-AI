"""Environment-based configuration for KnowledgePilot AI.

Secrets are never hard-coded. All sensitive values are read from
environment variables. A real API key is never created or committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Environment = Literal["development", "staging", "production", "test"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
GeneratorProvider = Literal["mock", "ollama", "openai"]

_VALID_ENVIRONMENTS: frozenset[str] = frozenset(
    {"development", "staging", "production", "test"}
)
_VALID_LOG_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)
_VALID_GENERATOR_PROVIDERS: frozenset[str] = frozenset({"mock", "ollama", "openai"})


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings loaded from the environment.

    Attributes:
        env: Runtime environment name.
        log_level: Logging verbosity.
        host: Bind host for the future HTTP API.
        port: Bind port for the future HTTP API.
        openai_api_key: Optional LLM API key if present in the environment.
        ollama_base_url: Base HTTP URL for local Ollama daemon.
        ollama_model: Default local LLM model name for Ollama.
        generator_provider: Active generation backend ('mock' or 'ollama').
        storage_path: Optional file path for atomic disk persistence of index state.
    """

    env: Environment = "development"
    log_level: LogLevel = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    generator_provider: GeneratorProvider = "mock"
    storage_path: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables.

        Recognised variables:
            KNOWLEDGEPILOT_ENV, KNOWLEDGEPILOT_LOG_LEVEL,
            KNOWLEDGEPILOT_HOST, KNOWLEDGEPILOT_PORT,
            OPENAI_API_KEY, KNOWLEDGEPILOT_OPENAI_API_KEY,
            KNOWLEDGEPILOT_OLLAMA_BASE_URL, KNOWLEDGEPILOT_OLLAMA_MODEL,
            KNOWLEDGEPILOT_GENERATOR_PROVIDER, KNOWLEDGEPILOT_STORAGE_PATH

        Raises:
            ValueError: If any supplied value fails validation.
        """
        raw_env = _get_env("KNOWLEDGEPILOT_ENV", "development")
        assert raw_env is not None  # default guarantees non-None
        env_value = raw_env.lower()
        if env_value not in _VALID_ENVIRONMENTS:
            raise ValueError(
                f"Invalid KNOWLEDGEPILOT_ENV={raw_env!r}. "
                f"Expected one of {sorted(_VALID_ENVIRONMENTS)}."
            )

        raw_level = _get_env("KNOWLEDGEPILOT_LOG_LEVEL", "INFO")
        assert raw_level is not None
        level_value = raw_level.upper()
        if level_value not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid KNOWLEDGEPILOT_LOG_LEVEL={raw_level!r}. "
                f"Expected one of {sorted(_VALID_LOG_LEVELS)}."
            )

        raw_host = _get_env("KNOWLEDGEPILOT_HOST", "127.0.0.1")
        assert raw_host is not None
        host_value = raw_host.strip() or "127.0.0.1"

        raw_port = _get_env("KNOWLEDGEPILOT_PORT", "8000")
        assert raw_port is not None
        try:
            port_value = int(raw_port.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid KNOWLEDGEPILOT_PORT={raw_port!r}: must be an integer."
            ) from exc
        if not 1 <= port_value <= 65535:
            raise ValueError(
                f"Invalid KNOWLEDGEPILOT_PORT={port_value}: "
                "must be between 1 and 65535."
            )

        # Secrets — prefer OPENAI_API_KEY, fall back to namespaced variant.
        # Empty / whitespace-only values are treated as absent.
        api_key = _get_env("OPENAI_API_KEY")
        if api_key is None:
            api_key = _get_env("KNOWLEDGEPILOT_OPENAI_API_KEY")

        # Local Ollama configuration
        raw_ollama_url = _get_env(
            "KNOWLEDGEPILOT_OLLAMA_BASE_URL", "http://localhost:11434"
        )
        assert raw_ollama_url is not None
        ollama_url = raw_ollama_url.rstrip("/")

        raw_ollama_model = _get_env("KNOWLEDGEPILOT_OLLAMA_MODEL", "llama3.2:3b")
        assert raw_ollama_model is not None
        ollama_model = raw_ollama_model.strip() or "llama3.2:3b"

        raw_gen_provider = _get_env("KNOWLEDGEPILOT_GENERATOR_PROVIDER", "mock")
        assert raw_gen_provider is not None
        gen_provider = raw_gen_provider.lower()
        if gen_provider not in _VALID_GENERATOR_PROVIDERS:
            raise ValueError(
                f"Invalid KNOWLEDGEPILOT_GENERATOR_PROVIDER={raw_gen_provider!r}. "
                f"Expected one of {sorted(_VALID_GENERATOR_PROVIDERS)}."
            )

        raw_storage_path = _get_env("KNOWLEDGEPILOT_STORAGE_PATH")
        storage_path = raw_storage_path.strip() if raw_storage_path else None

        return cls(
            env=env_value,  # type: ignore[arg-type]
            log_level=level_value,  # type: ignore[arg-type]
            host=host_value,
            port=port_value,
            openai_api_key=api_key,
            ollama_base_url=ollama_url,
            ollama_model=ollama_model,
            generator_provider=gen_provider,  # type: ignore[arg-type]
            storage_path=storage_path,
        )

    @property
    def is_production(self) -> bool:
        """Whether the app is running in production."""
        return self.env == "production"

    @property
    def is_test(self) -> bool:
        """Whether the app is running in test mode."""
        return self.env == "test"
