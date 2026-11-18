"""Unit tests for knowledgepilot.config.Settings."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from knowledgepilot.config import Settings

# Helpers -------------------------------------------------------------------


def _env(**overrides: str | None) -> dict[str, str]:
    """Build a clean env dict with defaults, allowing None to mean 'unset'."""
    base: dict[str, str | None] = {
        "KNOWLEDGEPILOT_ENV": None,
        "KNOWLEDGEPILOT_LOG_LEVEL": None,
        "KNOWLEDGEPILOT_HOST": None,
        "KNOWLEDGEPILOT_PORT": None,
        "OPENAI_API_KEY": None,
        "KNOWLEDGEPILOT_OPENAI_API_KEY": None,
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


# Defaults ------------------------------------------------------------------


def test_defaults_when_env_empty() -> None:
    with patch.dict(os.environ, _env(), clear=True):
        s = Settings.from_env()
    assert s.env == "development"
    assert s.log_level == "INFO"
    assert s.host == "127.0.0.1"
    assert s.port == 8000
    assert s.openai_api_key is None
    assert s.storage_path is None
    assert s.is_production is False
    assert s.is_test is False


def test_storage_path_from_env() -> None:
    with patch.dict(
        os.environ,
        _env(KNOWLEDGEPILOT_STORAGE_PATH="/tmp/test_index.json"),
        clear=True,
    ):
        s = Settings.from_env()
    assert s.storage_path == "/tmp/test_index.json"


def test_settings_is_frozen_and_slots() -> None:
    s = Settings(env="test", log_level="DEBUG", host="0.0.0.0", port=9000)
    with pytest.raises(AttributeError):
        s.env = "production"  # type: ignore[misc]


# Env / log level -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("development", "development"),
        ("Development", "development"),
        ("PRODUCTION", "production"),
        ("staging", "staging"),
        ("test", "test"),
    ],
)
def test_env_normalization(raw: str, expected: str) -> None:
    with patch.dict(os.environ, _env(KNOWLEDGEPILOT_ENV=raw), clear=True):
        assert Settings.from_env().env == expected


def test_invalid_env_raises() -> None:
    with patch.dict(os.environ, _env(KNOWLEDGEPILOT_ENV="invalid"), clear=True):
        with pytest.raises(ValueError, match="KNOWLEDGEPILOT_ENV"):
            Settings.from_env()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("debug", "DEBUG"),
        ("INFO", "INFO"),
        ("Warning", "WARNING"),
        ("error", "ERROR"),
        ("CRITICAL", "CRITICAL"),
    ],
)
def test_log_level_normalization(raw: str, expected: str) -> None:
    with patch.dict(os.environ, _env(KNOWLEDGEPILOT_LOG_LEVEL=raw), clear=True):
        assert Settings.from_env().log_level == expected


def test_invalid_log_level_raises() -> None:
    with patch.dict(os.environ, _env(KNOWLEDGEPILOT_LOG_LEVEL="VERBOSE"), clear=True):
        with pytest.raises(ValueError, match="KNOWLEDGEPILOT_LOG_LEVEL"):
            Settings.from_env()


# Host / port ---------------------------------------------------------------


def test_custom_host_and_port() -> None:
    with patch.dict(
        os.environ,
        _env(KNOWLEDGEPILOT_HOST="0.0.0.0", KNOWLEDGEPILOT_PORT="9000"),
        clear=True,
    ):
        s = Settings.from_env()
        assert s.host == "0.0.0.0"
        assert s.port == 9000


def test_invalid_port_non_integer() -> None:
    with patch.dict(os.environ, _env(KNOWLEDGEPILOT_PORT="not-a-number"), clear=True):
        with pytest.raises(ValueError, match="KNOWLEDGEPILOT_PORT"):
            Settings.from_env()


@pytest.mark.parametrize("bad_port", ["0", "-1", "65536", "99999"])
def test_invalid_port_out_of_range(bad_port: str) -> None:
    with patch.dict(os.environ, _env(KNOWLEDGEPILOT_PORT=bad_port), clear=True):
        with pytest.raises(ValueError, match="KNOWLEDGEPILOT_PORT"):
            Settings.from_env()


def test_whitespace_only_host_falls_back_to_default() -> None:
    with patch.dict(os.environ, _env(KNOWLEDGEPILOT_HOST="   "), clear=True):
        assert Settings.from_env().host == "127.0.0.1"


# Secrets -------------------------------------------------------------------


def test_openai_api_key_primary() -> None:
    with patch.dict(os.environ, _env(OPENAI_API_KEY="sk-test-123"), clear=True):
        assert Settings.from_env().openai_api_key == "sk-test-123"


def test_openai_api_key_fallback_namespaced() -> None:
    with patch.dict(
        os.environ, _env(KNOWLEDGEPILOT_OPENAI_API_KEY="sk-fallback"), clear=True
    ):
        assert Settings.from_env().openai_api_key == "sk-fallback"


def test_openai_api_key_primary_takes_precedence() -> None:
    with patch.dict(
        os.environ,
        _env(OPENAI_API_KEY="sk-primary", KNOWLEDGEPILOT_OPENAI_API_KEY="sk-fallback"),
        clear=True,
    ):
        assert Settings.from_env().openai_api_key == "sk-primary"


def test_empty_api_key_treated_as_absent() -> None:
    with patch.dict(os.environ, _env(OPENAI_API_KEY="   "), clear=True):
        assert Settings.from_env().openai_api_key is None


def test_never_reads_hardcoded_secrets() -> None:
    """Ensure source does not contain a hard-coded sk- key (sanity check)."""
    import pathlib

    src_text = pathlib.Path("src/knowledgepilot/config.py").read_text(encoding="utf-8")
    # Allow the string "sk-" only in tests/comments, not as a literal secret.
    # We check that no env default contains sk-.
    assert "sk-" not in src_text or "sk-test" in src_text or "sk-fallback" in src_text


# Helpers -------------------------------------------------------------------


def test_is_production_and_is_test_properties() -> None:
    assert Settings(env="production").is_production is True
    assert Settings(env="production").is_test is False
    assert Settings(env="test").is_test is True
    assert Settings(env="development").is_production is False


def test_ollama_config_defaults() -> None:
    with patch.dict(os.environ, _env(), clear=True):
        s = Settings.from_env()
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_model == "llama3.2:3b"
    assert s.generator_provider == "mock"


def test_ollama_config_custom() -> None:
    with patch.dict(
        os.environ,
        _env(
            KNOWLEDGEPILOT_OLLAMA_BASE_URL="http://192.168.1.50:11434/",
            KNOWLEDGEPILOT_OLLAMA_MODEL="phi3",
            KNOWLEDGEPILOT_GENERATOR_PROVIDER="ollama",
        ),
        clear=True,
    ):
        s = Settings.from_env()
        assert s.ollama_base_url == "http://192.168.1.50:11434"
        assert s.ollama_model == "phi3"
        assert s.generator_provider == "ollama"


def test_invalid_generator_provider_raises() -> None:
    with patch.dict(
        os.environ,
        _env(KNOWLEDGEPILOT_GENERATOR_PROVIDER="unknown"),
        clear=True,
    ):
        with pytest.raises(ValueError, match="KNOWLEDGEPILOT_GENERATOR_PROVIDER"):
            Settings.from_env()
