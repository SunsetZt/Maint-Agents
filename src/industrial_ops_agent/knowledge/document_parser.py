"""Structured PDF parsing with governed native-text and PaddleOCR fallbacks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

from industrial_ops_agent.auth.identity import IdentityContext
from industrial_ops_agent.knowledge.files import (
    KnowledgeDocumentParseError,
    ParsedKnowledgeDocument,
)
from industrial_ops_agent.knowledge.ingestion import ExtractedKnowledgeStructure
from industrial_ops_agent.model_gateway.service import ModelGatewayError
from industrial_ops_agent.multimodal.models import EvidenceSchemaError, VisualFinding
from industrial_ops_agent.multimodal.providers import (
    MultimodalProviderUnavailable,
    OcrProvider,
    VlmProvider,
)

_MAX_PDF_PAGES = 200
_MIN_NATIVE_PAGE_CHARACTERS = 40
_MAX_EMBEDDED_FIGURES = 24
_MIN_EMBEDDED_FIGURE_AREA = 0.005


@dataclass(frozen=True, slots=True)
class EmbeddedPdfFigure:
    figure_id: str
    page_number: int
    bounding_box: dict[str, float]


@dataclass(frozen=True, slots=True)
class StructuredPdfExtraction:
    pages: tuple[tuple[int, str], ...]
    structures: tuple[ExtractedKnowledgeStructure, ...]
    parser_version: str
    figures: tuple[EmbeddedPdfFigure, ...] = ()


class StructuredPdfParser(Protocol):
    def extract(self, content: bytes) -> StructuredPdfExtraction: ...


class DoclingStructuredPdfParser:
    """Use Docling layout/table models while keeping OCR disabled."""

    def __init__(self, artifacts_path: Path | None = None) -> None:
        self._artifacts_path = artifacts_path

    def extract(self, content: bytes) -> StructuredPdfExtraction:
        try:
            from docling.datamodel.base_models import (  # type: ignore[import-not-found]
                DocumentStream,
                InputFormat,
            )
            from docling.datamodel.pipeline_options import (  # type: ignore[import-not-found]
                PdfPipelineOptions,
                TableFormerMode,
            )
            from docling.document_converter import (  # type: ignore[import-not-found]
                DocumentConverter,
                PdfFormatOption,
            )
        except ImportError as exc:
            raise KnowledgeDocumentParseError("knowledge_docling_unavailable") from exc

        try:
            option_values: dict[str, Any] = {
                "do_ocr": False,
                "do_table_structure": True,
            }
            if self._artifacts_path is not None:
                option_values["artifacts_path"] = self._artifacts_path
            pipeline_options = PdfPipelineOptions(**option_values)
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
            result = converter.convert(
                DocumentStream(name="knowledge-source.pdf", stream=BytesIO(content)),
                raises_on_error=True,
                max_num_pages=_MAX_PDF_PAGES,
                max_file_size=len(content),
            )
            document = result.document
            page_numbers = tuple(sorted(int(page_number) for page_number in document.pages))
            if page_numbers != tuple(range(1, len(page_numbers) + 1)) or not page_numbers:
                raise KnowledgeDocumentParseError("knowledge_docling_pages_invalid")
            if len(page_numbers) > _MAX_PDF_PAGES:
                raise KnowledgeDocumentParseError("knowledge_pdf_page_limit_exceeded")
            pages = tuple(
                (
                    page_number,
                    str(
                        document.export_to_markdown(
                            page_no=page_number,
                            image_placeholder="",
                            include_annotations=False,
                        )
                    ).strip(),
                )
                for page_number in page_numbers
            )
            structures = _docling_structures(document, frozenset(page_numbers))
            figures = _docling_figures(document, frozenset(page_numbers))
        except KnowledgeDocumentParseError:
            raise
        except Exception as exc:
            raise KnowledgeDocumentParseError("knowledge_docling_parse_failed") from exc
        try:
            package_version = version("docling")
        except PackageNotFoundError:
            package_version = "unknown"
        return StructuredPdfExtraction(
            pages=pages,
            structures=structures,
            parser_version=f"docling-{package_version}-ocr-off-tableformer-accurate",
            figures=figures,
        )


class GovernedKnowledgeDocumentParser:
    def __init__(
        self,
        ocr: OcrProvider | None = None,
        structured_pdf: StructuredPdfParser | None = None,
        vlm: VlmProvider | None = None,
    ) -> None:
        self._ocr = ocr
        self._structured_pdf = structured_pdf
        self._vlm = vlm

    async def parse(
        self,
        content: bytes,
        *,
        mime_type: str,
        identity: IdentityContext,
        ingestion_id: str,
        request_id: str,
        device_models: tuple[str, ...] = (),
    ) -> ParsedKnowledgeDocument:
        if mime_type in {"text/plain", "text/markdown"}:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise KnowledgeDocumentParseError("knowledge_text_encoding_invalid") from exc
            return ParsedKnowledgeDocument(
                pages=((1, text, "text"),),
                parser_version="governed-utf8-parser-v1",
            )
        if mime_type != "application/pdf":
            raise KnowledgeDocumentParseError("knowledge_file_type_invalid")

        structured: StructuredPdfExtraction | None = None
        if self._structured_pdf is not None:
            try:
                structured = await asyncio.to_thread(self._structured_pdf.extract, content)
            except KnowledgeDocumentParseError as exc:
                if exc.reason != "knowledge_docling_unavailable":
                    raise

        if structured is None:
            extracted, pdfium_version = await asyncio.to_thread(_extract_pdf_pages, content)
            return await self._finish_pdf_pages(
                extracted,
                identity=identity,
                ingestion_id=ingestion_id,
                request_id=request_id,
                parser_versions=[f"pypdfium2-{pdfium_version}"],
            )

        _validate_structured_extraction(structured)
        sparse_pages = {page_number for page_number, text in structured.pages if not text.strip()}
        fallback_by_page: dict[int, tuple[str, bytes]] = {}
        parser_versions = [structured.parser_version]
        if sparse_pages:
            extracted, pdfium_version = await asyncio.to_thread(_extract_pdf_pages, content)
            if tuple(page_number for page_number, _, _ in extracted) != tuple(
                page_number for page_number, _ in structured.pages
            ):
                raise KnowledgeDocumentParseError("knowledge_pdf_page_count_mismatch")
            fallback_by_page = {
                page_number: (native_text, image)
                for page_number, native_text, image in extracted
                if page_number in sparse_pages
            }
            parser_versions.append(f"pypdfium2-{pdfium_version}")

        pages: list[tuple[int, str, str]] = []
        ocr_versions: set[str] = set()
        structured_pages: set[int] = set()
        for page_number, structured_text in structured.pages:
            if structured_text.strip():
                pages.append((page_number, structured_text, "docling"))
                structured_pages.add(page_number)
                continue
            native_text, image = fallback_by_page[page_number]
            if len(native_text.strip()) >= _MIN_NATIVE_PAGE_CHARACTERS:
                pages.append((page_number, native_text, "native"))
                continue
            page_text, ocr_version = await self._recognize_page(
                page_number,
                image,
                identity=identity,
                ingestion_id=ingestion_id,
                request_id=request_id,
            )
            pages.append((page_number, page_text, "paddleocr"))
            ocr_versions.add(ocr_version)
        parser_versions.extend(sorted(ocr_versions))
        figure_structures, figure_marker = await self._analyze_figures(
            content,
            structured.figures,
            identity=identity,
            ingestion_id=ingestion_id,
            request_id=request_id,
            asset_model=device_models[0]
            if len(device_models) == 1
            else "knowledge-document-figure",
        )
        if figure_marker is not None:
            parser_versions.append(figure_marker)
        pages = _append_figure_semantics(pages, figure_structures)
        parser_version = "+".join(parser_versions)
        _validate_parser_version(parser_version)
        return ParsedKnowledgeDocument(
            pages=tuple(pages),
            parser_version=parser_version,
            structures=tuple(
                item for item in structured.structures if item.page_number in structured_pages
            )
            + figure_structures,
        )

    async def _finish_pdf_pages(
        self,
        extracted: list[tuple[int, str, bytes]],
        *,
        identity: IdentityContext,
        ingestion_id: str,
        request_id: str,
        parser_versions: list[str],
    ) -> ParsedKnowledgeDocument:
        pages: list[tuple[int, str, str]] = []
        ocr_versions: set[str] = set()
        for page_number, native_text, image in extracted:
            if len(native_text.strip()) >= _MIN_NATIVE_PAGE_CHARACTERS:
                pages.append((page_number, native_text, "native"))
                continue
            page_text, ocr_version = await self._recognize_page(
                page_number,
                image,
                identity=identity,
                ingestion_id=ingestion_id,
                request_id=request_id,
            )
            pages.append((page_number, page_text, "paddleocr"))
            ocr_versions.add(ocr_version)
        parser_versions.extend(sorted(ocr_versions))
        parser_version = "+".join(parser_versions)
        _validate_parser_version(parser_version)
        return ParsedKnowledgeDocument(tuple(pages), parser_version)

    async def _recognize_page(
        self,
        page_number: int,
        image: bytes,
        *,
        identity: IdentityContext,
        ingestion_id: str,
        request_id: str,
    ) -> tuple[str, str]:
        if self._ocr is None:
            raise KnowledgeDocumentParseError("knowledge_pdf_ocr_required")
        try:
            result = await self._ocr.recognize(
                image,
                mime_type="image/png",
                tenant_id=identity.tenant_id,
                subject_id=identity.subject_id,
                trace_id=request_id,
                source_id=f"{ingestion_id}-page-{page_number}",
            )
        except MultimodalProviderUnavailable as exc:
            raise KnowledgeDocumentParseError("knowledge_pdf_ocr_unavailable") from exc
        page_text = "\n".join(block.text for block in result.blocks if block.text.strip())
        if not page_text.strip():
            raise KnowledgeDocumentParseError("knowledge_pdf_page_empty")
        return page_text, result.processor_version

    async def _analyze_figures(
        self,
        content: bytes,
        figures: tuple[EmbeddedPdfFigure, ...],
        *,
        identity: IdentityContext,
        ingestion_id: str,
        request_id: str,
        asset_model: str,
    ) -> tuple[tuple[ExtractedKnowledgeStructure, ...], str | None]:
        if not figures:
            return (), None
        images = await asyncio.to_thread(_render_pdf_figure_images, content, figures)
        structures: list[ExtractedKnowledgeStructure] = []
        processor_versions: set[str] = set()
        degraded = False
        for figure in figures:
            image = images.get(figure.figure_id)
            if not image:
                raise KnowledgeDocumentParseError("knowledge_pdf_figure_render_failed")
            image_digest = sha256(image).hexdigest()
            status = "UNAVAILABLE"
            failure_reason: str | None = "knowledge_figure_vlm_not_configured"
            processor_version: str | None = None
            model_release_ids: list[str] = []
            findings: tuple[VisualFinding, ...] = ()
            if self._vlm is not None:
                try:
                    result = await self._vlm.inspect(
                        image,
                        mime_type="image/png",
                        asset_model=asset_model,
                        tenant_id=identity.tenant_id,
                        subject_id=identity.subject_id,
                        trace_id=request_id,
                        source_id=f"{ingestion_id}-{figure.figure_id}",
                    )
                    processor_version = result.processor_version
                    processor_versions.add(result.processor_version)
                    findings = result.findings
                    model_release_ids = sorted(
                        {finding.model_release_id for finding in result.findings}
                    )
                    failure_reason = None
                    if result.unsupported_model:
                        status = "UNSUPPORTED"
                    elif result.findings:
                        status = "ANALYZED"
                    else:
                        status = "NO_FINDINGS"
                except (
                    EvidenceSchemaError,
                    ModelGatewayError,
                    MultimodalProviderUnavailable,
                ):
                    degraded = True
                    failure_reason = "knowledge_figure_vlm_unavailable"
            text = _figure_semantic_text(figure, status, findings)
            structures.append(
                ExtractedKnowledgeStructure(
                    kind="figure",
                    page_number=figure.page_number,
                    text=text,
                    bounding_box=figure.bounding_box,
                    metadata={
                        "figure_id": figure.figure_id,
                        "analysis_status": status,
                        "source_image_sha256": image_digest,
                        "processor_version": processor_version,
                        "model_release_ids": model_release_ids,
                        "finding_count": len(findings),
                        "failure_reason": failure_reason,
                    },
                )
            )
        if self._vlm is None:
            return tuple(structures), "figure-vlm-disabled"
        if degraded:
            return tuple(structures), "figure-vlm-degraded"
        version_digest = sha256("\0".join(sorted(processor_versions)).encode()).hexdigest()[:12]
        return tuple(structures), f"figure-vlm-{version_digest}"


def _docling_structures(
    document: Any,
    page_numbers: frozenset[int],
) -> tuple[ExtractedKnowledgeStructure, ...]:
    structures: list[ExtractedKnowledgeStructure] = []
    for item, _level in document.iterate_items():
        label_value = getattr(getattr(item, "label", None), "value", "")
        kind: str | None = None
        text = ""
        if label_value in {"title", "section_header"}:
            kind = "section"
            text = str(getattr(item, "text", "")).strip()
        elif label_value == "table":
            kind = "table"
            text = str(item.export_to_markdown(doc=document)).strip()
        if kind is None or not text:
            continue
        provenance = tuple(getattr(item, "prov", ()))
        if not provenance:
            continue
        source = provenance[0]
        page_number = int(source.page_no)
        if page_number not in page_numbers:
            raise KnowledgeDocumentParseError("knowledge_docling_provenance_invalid")
        structures.append(
            ExtractedKnowledgeStructure(
                kind=kind,
                page_number=page_number,
                text=text,
                bounding_box=_normalized_docling_box(document, page_number, source.bbox),
            )
        )
    return tuple(structures)


def _docling_figures(
    document: Any,
    page_numbers: frozenset[int],
) -> tuple[EmbeddedPdfFigure, ...]:
    figures: list[EmbeddedPdfFigure] = []
    page_ordinals: dict[int, int] = {}
    for item, _level in document.iterate_items():
        label_value = getattr(getattr(item, "label", None), "value", "")
        if label_value not in {"picture", "figure"}:
            continue
        provenance = tuple(getattr(item, "prov", ()))
        if not provenance:
            continue
        source = provenance[0]
        page_number = int(source.page_no)
        if page_number not in page_numbers:
            raise KnowledgeDocumentParseError("knowledge_docling_provenance_invalid")
        bounding_box = _normalized_docling_box(document, page_number, source.bbox)
        if bounding_box["width"] * bounding_box["height"] < _MIN_EMBEDDED_FIGURE_AREA:
            continue
        page_ordinals[page_number] = page_ordinals.get(page_number, 0) + 1
        figures.append(
            EmbeddedPdfFigure(
                figure_id=f"figure-page-{page_number}-{page_ordinals[page_number]}",
                page_number=page_number,
                bounding_box=bounding_box,
            )
        )
    if len(figures) > _MAX_EMBEDDED_FIGURES:
        raise KnowledgeDocumentParseError("knowledge_pdf_figure_limit_exceeded")
    return tuple(figures)


def _append_figure_semantics(
    pages: list[tuple[int, str, str]],
    figures: tuple[ExtractedKnowledgeStructure, ...],
) -> list[tuple[int, str, str]]:
    by_page: dict[int, list[str]] = {}
    for figure in figures:
        by_page.setdefault(figure.page_number, []).append(figure.text)
    return [
        (
            page_number,
            "\n\n".join(
                part
                for part in (text.strip(), *by_page.get(page_number, ()))
                if part
            ),
            method,
        )
        for page_number, text, method in pages
    ]


def _figure_semantic_text(
    figure: EmbeddedPdfFigure,
    status: str,
    findings: tuple[VisualFinding, ...],
) -> str:
    heading = f"内嵌图片候选语义（{figure.figure_id}）"
    if status == "ANALYZED":
        observations = [
            (
                f"- {finding.label[:128]}：{finding.description[:1000]}"
                f"（置信度 {finding.confidence:.2f}，{finding.evidence_level}）"
            )
            for finding in findings[:16]
        ]
        return f"{heading}\n" + "\n".join(observations)
    if status == "NO_FINDINGS":
        return f"{heading}\n受治理 VLM 未产生可报告的工业候选观察。"
    if status == "UNSUPPORTED":
        return f"{heading}\n当前已发布 VLM 不支持该图片，未生成候选语义。"
    return f"{heading}\n图片语义处理暂不可用，该图片未形成可检索的诊断观察。"


def _normalized_docling_box(
    document: Any,
    page_number: int,
    bounding_box: Any,
) -> dict[str, float]:
    page_size = document.pages[page_number].size
    normalized = bounding_box.to_top_left_origin(page_size.height).normalized(page_size)
    raw_left = min(float(normalized.l), float(normalized.r))
    raw_top = min(float(normalized.t), float(normalized.b))
    raw_right = max(float(normalized.l), float(normalized.r))
    raw_bottom = max(float(normalized.t), float(normalized.b))
    values = (raw_left, raw_top, raw_right, raw_bottom)
    if any(not isfinite(value) for value in values):
        raise KnowledgeDocumentParseError("knowledge_docling_provenance_invalid")
    left = min(1.0, max(0.0, raw_left))
    top = min(1.0, max(0.0, raw_top))
    right = min(1.0, max(0.0, raw_right))
    bottom = min(1.0, max(0.0, raw_bottom))
    width = right - left
    height = bottom - top
    if width <= 0.0 or height <= 0.0:
        raise KnowledgeDocumentParseError("knowledge_docling_provenance_invalid")
    return {
        "x": left,
        "y": top,
        "width": width,
        "height": height,
    }


def _validate_structured_extraction(extraction: StructuredPdfExtraction) -> None:
    page_numbers = tuple(page_number for page_number, _ in extraction.pages)
    if (
        not page_numbers
        or len(page_numbers) > _MAX_PDF_PAGES
        or page_numbers != tuple(range(1, len(page_numbers) + 1))
    ):
        raise KnowledgeDocumentParseError("knowledge_docling_pages_invalid")
    figure_ids: set[str] = set()
    page_set = frozenset(page_numbers)
    if len(extraction.figures) > _MAX_EMBEDDED_FIGURES:
        raise KnowledgeDocumentParseError("knowledge_pdf_figure_limit_exceeded")
    for figure in extraction.figures:
        box = figure.bounding_box
        if (
            not figure.figure_id
            or len(figure.figure_id) > 128
            or figure.figure_id in figure_ids
            or figure.page_number not in page_set
            or set(box) != {"x", "y", "width", "height"}
            or any(not isfinite(float(value)) for value in box.values())
            or float(box["x"]) < 0
            or float(box["y"]) < 0
            or float(box["width"]) <= 0
            or float(box["height"]) <= 0
            or float(box["x"]) + float(box["width"]) > 1.001
            or float(box["y"]) + float(box["height"]) > 1.001
        ):
            raise KnowledgeDocumentParseError("knowledge_docling_figure_invalid")
        figure_ids.add(figure.figure_id)
    _validate_parser_version(extraction.parser_version)


def _validate_parser_version(parser_version: str) -> None:
    if not parser_version or len(parser_version) > 128:
        raise KnowledgeDocumentParseError("knowledge_parser_version_invalid")


def _render_pdf_figure_images(
    content: bytes,
    figures: tuple[EmbeddedPdfFigure, ...],
) -> dict[str, bytes]:
    try:
        import pypdfium2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise KnowledgeDocumentParseError("knowledge_pdf_parser_unavailable") from exc
    try:
        document = pypdfium2.PdfDocument(content)
    except Exception as exc:
        raise KnowledgeDocumentParseError("knowledge_pdf_malformed") from exc
    by_page: dict[int, list[EmbeddedPdfFigure]] = {}
    for figure in figures:
        by_page.setdefault(figure.page_number, []).append(figure)
    rendered: dict[str, bytes] = {}
    try:
        for page_number, page_figures in by_page.items():
            page: Any = document[page_number - 1]
            try:
                bitmap = page.render(scale=2.0)
                try:
                    page_image = bitmap.to_pil()
                    try:
                        for figure in page_figures:
                            box = figure.bounding_box
                            left = max(0, round(float(box["x"]) * page_image.width))
                            top = max(0, round(float(box["y"]) * page_image.height))
                            right = min(
                                page_image.width,
                                round(
                                    (float(box["x"]) + float(box["width"]))
                                    * page_image.width
                                ),
                            )
                            bottom = min(
                                page_image.height,
                                round(
                                    (float(box["y"]) + float(box["height"]))
                                    * page_image.height
                                ),
                            )
                            if right - left < 16 or bottom - top < 16:
                                raise KnowledgeDocumentParseError(
                                    "knowledge_pdf_figure_render_failed"
                                )
                            cropped = page_image.crop((left, top, right, bottom))
                            try:
                                output = BytesIO()
                                cropped.save(output, format="PNG")
                                rendered[figure.figure_id] = output.getvalue()
                            finally:
                                cropped.close()
                    finally:
                        page_image.close()
                finally:
                    bitmap.close()
            finally:
                page.close()
    except KnowledgeDocumentParseError:
        raise
    except Exception as exc:
        raise KnowledgeDocumentParseError("knowledge_pdf_figure_render_failed") from exc
    finally:
        document.close()
    return rendered


def _extract_pdf_pages(content: bytes) -> tuple[list[tuple[int, str, bytes]], str]:
    try:
        import pypdfium2
    except ImportError as exc:
        raise KnowledgeDocumentParseError("knowledge_pdf_parser_unavailable") from exc
    try:
        document = pypdfium2.PdfDocument(content)
    except Exception as exc:
        raise KnowledgeDocumentParseError("knowledge_pdf_malformed") from exc
    try:
        page_count = len(document)
        if page_count < 1 or page_count > _MAX_PDF_PAGES:
            raise KnowledgeDocumentParseError("knowledge_pdf_page_limit_exceeded")
        pages: list[tuple[int, str, bytes]] = []
        for index in range(page_count):
            page: Any = document[index]
            try:
                text_page = page.get_textpage()
                try:
                    native_text = str(text_page.get_text_range()).strip()
                finally:
                    text_page.close()
                image = b""
                if len(native_text) < _MIN_NATIVE_PAGE_CHARACTERS:
                    bitmap = page.render(scale=2.0)
                    try:
                        output = BytesIO()
                        bitmap.to_pil().save(output, format="PNG")
                        image = output.getvalue()
                    finally:
                        bitmap.close()
                pages.append((index + 1, native_text, image))
            finally:
                page.close()
    except KnowledgeDocumentParseError:
        raise
    except Exception as exc:
        raise KnowledgeDocumentParseError("knowledge_pdf_parse_failed") from exc
    finally:
        document.close()
    return pages, str(getattr(pypdfium2, "__version__", "unknown"))
