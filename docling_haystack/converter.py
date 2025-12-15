#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Docling Haystack converter module."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from docling.chunking import BaseChunk, BaseChunker, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter
from haystack import Document, component


class ExportType(str, Enum):
    """Enumeration of available export types."""

    MARKDOWN = "markdown"
    DOC_CHUNKS = "doc_chunks"


class BaseMetaExtractor(ABC):
    """BaseMetaExtractor."""

    @abstractmethod
    def extract_chunk_meta(self, chunk: BaseChunk) -> dict[str, Any]:
        """Extract chunk meta."""
        raise NotImplementedError()

    @abstractmethod
    def extract_dl_doc_meta(self, dl_doc: DoclingDocument) -> dict[str, Any]:
        """Extract Docling document meta."""
        raise NotImplementedError()


class MetaExtractor(BaseMetaExtractor):
    """MetaExtractor."""

    def extract_chunk_meta(self, chunk: BaseChunk) -> dict[str, Any]:
        """Extract chunk meta."""
        return {"dl_meta": chunk.export_json_dict()}

    def extract_dl_doc_meta(self, dl_doc: DoclingDocument) -> dict[str, Any]:
        """Extract Docling document meta."""
        return (
            {"dl_meta": {"origin": dl_doc.origin.model_dump(exclude_none=True)}}
            if dl_doc.origin
            else {}
        )


@component
class DoclingConverter:
    """Docling Haystack converter."""

    def __init__(
        self,
        converter: Optional[DocumentConverter] = None,
        convert_kwargs: Optional[dict[str, Any]] = None,
        export_type: ExportType = ExportType.DOC_CHUNKS,
        md_export_kwargs: Optional[dict[str, Any]] = None,
        chunker: Optional[BaseChunker] = None,
        meta_extractor: Optional[BaseMetaExtractor] = None,
    ):
        """Create a Docling Haystack converter.

        Args:
            converter: The Docling `DocumentConverter` to use; if not set, a system
                default is used.
            convert_kwargs: Any parameters to pass to Docling conversion; if not set, a
                system default is used.
            export_type: The export mode to use: set to `ExportType.MARKDOWN` if you
                want to capture each input document as a separate Haystack document, or
                `ExportType.DOC_CHUNKS` (default), if you want to first have each input
                document chunked and to then capture each individual chunk as a separate
                Haystack document downstream.
            md_export_kwargs: Any parameters to pass to Markdown export (applicable in
                case of `ExportType.MARKDOWN`).
            chunker: The Docling chunker instance to use; if not set, a system default
                is used.
            meta_extractor: The extractor instance to use for populating the output
                document metadata; if not set, a system default is used.
        """
        self._converter = converter or DocumentConverter()
        self._convert_kwargs = convert_kwargs if convert_kwargs is not None else {}
        self._export_type = export_type
        self._md_export_kwargs = (
            md_export_kwargs
            if md_export_kwargs is not None
            else {"image_placeholder": ""}
        )
        if self._export_type == ExportType.DOC_CHUNKS:
            self._chunker = chunker or HybridChunker()
        self._meta_extractor = meta_extractor or MetaExtractor()

    @component.output_types(documents=list[Document])
    def run(
        self,
        paths: Iterable[Union[Path, str]],
    ):
        """Run the DoclingConverter.

        Args:
            paths: The input document locations, either as local paths or URLs.

        Returns:
            list[Document]: The output Haystack Documents.
        """
        documents: list[Document] = []
        for filepath in paths:
            dl_doc = self._converter.convert(
                source=filepath,
                **self._convert_kwargs,
            ).document

            if self._export_type == ExportType.DOC_CHUNKS:
                chunk_iter = self._chunker.chunk(dl_doc=dl_doc)
                hs_docs = [
                    Document(
                        content=self._chunker.contextualize(chunk=chunk),
                        meta=self._meta_extractor.extract_chunk_meta(chunk=chunk),
                    )
                    for chunk in chunk_iter
                ]
                documents.extend(hs_docs)
            elif self._export_type == ExportType.MARKDOWN:
                hs_doc = Document(
                    content=dl_doc.export_to_markdown(**self._md_export_kwargs),
                    meta=self._meta_extractor.extract_dl_doc_meta(dl_doc=dl_doc),
                )
                documents.append(hs_doc)
            else:
                raise RuntimeError(f"Unexpected export type: {self._export_type}")
        return {"documents": documents}
