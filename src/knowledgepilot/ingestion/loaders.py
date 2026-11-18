"""Document loaders for plain text and PDF formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError

from knowledgepilot.ingestion.errors import (
    EmptyDocumentError,
    MalformedDocumentError,
    UnsupportedFileTypeError,
)
from knowledgepilot.ingestion.models import Document, DocumentMetadata


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    def __init__(self, *, allow_empty: bool = False) -> None:
        """Initialize the loader.

        Args:
            allow_empty: If True, empty files return an empty list instead
                of raising EmptyDocumentError.
        """
        self.allow_empty = allow_empty

    @abstractmethod
    def load(self, path: str | Path) -> list[Document]:
        """Load and extract text documents from the given file path.

        Args:
            path: Path to the target document.

        Returns:
            A list of Document instances extracted from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            EmptyDocumentError: If the file contains no text and allow_empty is False.
            MalformedDocumentError: If the file cannot be decoded or parsed.
        """


class TxtLoader(BaseLoader):
    """Loader for plain text (.txt) files."""

    def load(self, path: str | Path) -> list[Document]:
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Document file does not exist: {path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        try:
            raw_text = file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise MalformedDocumentError(
                f"Failed to decode text file with UTF-8: {file_path.name}: {exc}"
            ) from exc
        except OSError as exc:
            raise MalformedDocumentError(
                f"I/O error reading file {file_path.name}: {exc}"
            ) from exc

        content = raw_text.strip()
        if not content:
            if not self.allow_empty:
                raise EmptyDocumentError(
                    f"Text document contains no extractable text: {file_path.name}"
                )
            return []

        doc = Document(
            text=content,
            metadata=DocumentMetadata(
                source=str(file_path),
                file_name=file_path.name,
                file_type="txt",
                page_number=1,
                total_pages=1,
            ),
        )
        return [doc]


class PdfLoader(BaseLoader):
    """Loader for Portable Document Format (.pdf) files."""

    def load(self, path: str | Path) -> list[Document]:
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Document file does not exist: {path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        if file_path.stat().st_size == 0:
            if not self.allow_empty:
                raise EmptyDocumentError(
                    f"PDF document is empty (0 bytes): {file_path.name}"
                )
            return []

        try:
            reader = PdfReader(str(file_path))
        except (PdfReadError, PdfStreamError, ValueError, IndexError) as exc:
            raise MalformedDocumentError(
                f"Malformed or corrupted PDF file {file_path.name}: {exc}"
            ) from exc
        except Exception as exc:
            raise MalformedDocumentError(
                f"Unexpected error opening PDF file {file_path.name}: {exc}"
            ) from exc

        if reader.is_encrypted:
            try:
                decrypt_code = reader.decrypt("")
                if reader.is_encrypted or decrypt_code == 0:
                    raise MalformedDocumentError(
                        f"PDF file is password protected or encrypted: {file_path.name}"
                    )
            except MalformedDocumentError:
                raise
            except Exception as exc:
                raise MalformedDocumentError(
                    f"Failed to decrypt PDF {file_path.name}: {exc}"
                ) from exc

        total_pages = len(reader.pages)
        if total_pages == 0:
            if not self.allow_empty:
                raise EmptyDocumentError(
                    f"PDF document contains no pages: {file_path.name}"
                )
            return []

        documents: list[Document] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_raw = page.extract_text() or ""
            except Exception as exc:
                raise MalformedDocumentError(
                    f"Error extracting text from page {page_idx} in "
                    f"{file_path.name}: {exc}"
                ) from exc

            page_text = page_raw.strip()
            if page_text:
                documents.append(
                    Document(
                        text=page_text,
                        metadata=DocumentMetadata(
                            source=str(file_path),
                            file_name=file_path.name,
                            file_type="pdf",
                            page_number=page_idx,
                            total_pages=total_pages,
                        ),
                    )
                )

        if not documents:
            if not self.allow_empty:
                raise EmptyDocumentError(
                    f"PDF document contains no extractable text: {file_path.name}"
                )
            return []

        return documents


def load_document(path: str | Path, *, allow_empty: bool = False) -> list[Document]:
    """Ingest a document from disk based on its file extension.

    Supports .txt and .pdf files.

    Args:
        path: Path to the document.
        allow_empty: If True, empty files return an empty list instead of
            raising EmptyDocumentError.

    Returns:
        A list of Document objects extracted from the file.

    Raises:
        FileNotFoundError: If the document does not exist.
        UnsupportedFileTypeError: If the extension is not supported.
        EmptyDocumentError: If the file contains no text and allow_empty is False.
        MalformedDocumentError: If the file is corrupted or encrypted.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Document file does not exist: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    ext = file_path.suffix.lower()
    if ext == ".txt":
        return TxtLoader(allow_empty=allow_empty).load(file_path)
    if ext == ".pdf":
        return PdfLoader(allow_empty=allow_empty).load(file_path)

    raise UnsupportedFileTypeError(
        f"Unsupported file format '{ext}' for file {file_path.name}. "
        "Supported formats: .txt, .pdf"
    )
