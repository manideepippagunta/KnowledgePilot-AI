"""Answer generation package for KnowledgePilot AI."""

from __future__ import annotations

from knowledgepilot.generation.base import BaseGenerator
from knowledgepilot.generation.errors import GenerationError
from knowledgepilot.generation.factory import create_generator
from knowledgepilot.generation.mock import MockGenerator
from knowledgepilot.generation.ollama import OllamaGenerator
from knowledgepilot.generation.prompts import (
    DEFAULT_SYSTEM_INSTRUCTION,
    format_grounded_qa_prompt,
)

__all__ = [
    "DEFAULT_SYSTEM_INSTRUCTION",
    "BaseGenerator",
    "GenerationError",
    "MockGenerator",
    "OllamaGenerator",
    "create_generator",
    "format_grounded_qa_prompt",
]
