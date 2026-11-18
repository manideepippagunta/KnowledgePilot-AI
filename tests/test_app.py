"""Unit tests for knowledgepilot.app."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from knowledgepilot.app import configure_logging, create_app, main
from knowledgepilot.config import Settings


def test_configure_logging_sets_level() -> None:
    configure_logging("DEBUG")
    import logging

    assert logging.getLogger().level == logging.DEBUG
    # restore to INFO for other tests
    configure_logging("INFO")


def test_configure_logging_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Invalid log level"):
        configure_logging("VERBOSE")


def test_create_app_with_injected_settings_does_not_touch_env() -> None:
    injected = Settings(env="test", log_level="DEBUG", host="127.0.0.1", port=8000)
    result = create_app(settings=injected)
    assert result is injected


def test_create_app_from_env() -> None:
    with patch.dict(
        os.environ,
        {
            "KNOWLEDGEPILOT_ENV": "test",
            "KNOWLEDGEPILOT_LOG_LEVEL": "WARNING",
            "KNOWLEDGEPILOT_HOST": "127.0.0.1",
            "KNOWLEDGEPILOT_PORT": "8000",
        },
        clear=True,
    ):
        s = create_app()
        assert s.env == "test"
        assert s.log_level == "WARNING"


def test_main_success_prints_banner(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.dict(
        os.environ,
        {
            "KNOWLEDGEPILOT_ENV": "development",
            "KNOWLEDGEPILOT_LOG_LEVEL": "INFO",
            "KNOWLEDGEPILOT_HOST": "127.0.0.1",
            "KNOWLEDGEPILOT_PORT": "8000",
        },
        clear=True,
    ):
        code = main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "KnowledgePilot AI is running" in out
    assert "development" in out


def test_main_configuration_error_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.dict(os.environ, {"KNOWLEDGEPILOT_ENV": "bad_env"}, clear=True):
        code = main([])
    assert code == 2
    err = capsys.readouterr().err
    assert "Configuration error" in err


def test_main_with_injected_settings_via_create_app() -> None:
    # Ensure main path still covers the Settings injection via create_app indirectly.
    # Call create_app directly with bad port already covered; here just sanity.
    s = Settings(env="development", log_level="INFO", host="127.0.0.1", port=8000)
    result = create_app(settings=s)
    assert result.env == "development"


def test_create_app_no_misleading_openai_warning_for_mock_or_ollama(
    capsys: pytest.CaptureFixture[str],
) -> None:
    s_mock = Settings(
        env="development",
        generator_provider="mock",
        openai_api_key=None,
    )
    create_app(settings=s_mock)
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" not in err

    s_ollama = Settings(
        env="development",
        generator_provider="ollama",
        openai_api_key=None,
    )
    create_app(settings=s_ollama)
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" not in err


def test_create_app_warns_when_api_key_provider_lacks_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    s_openai = Settings(
        env="development",
        generator_provider="openai",
        openai_api_key=None,
    )
    create_app(settings=s_openai)
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert "Provider 'openai'" in err


def test_main_serve_mode(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("uvicorn.run") as mock_uvicorn_run:
        code = main(["--serve"])
    assert code == 0
    assert mock_uvicorn_run.call_count == 1
    out = capsys.readouterr().out
    assert "KnowledgePilot AI - Web Application Server" in out
    assert "http://127.0.0.1:8000" in out
