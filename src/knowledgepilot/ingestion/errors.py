"""Exception hierarchy for document ingestion and parsing."""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for all document ingestion and processing errors."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when an unsupported file type or extension is encountered."""


class EmptyDocumentError(IngestionError):
    """Raised when an ingested file contains no extractable text."""


class MalformedDocumentError(IngestionError):
    """Raised when a file cannot be parsed due to corruption or encryption."""
