import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from models import DocumentChunk


class DocumentParser(Protocol):
    async def parse(
        self,
        source_path: Path,
        parsed_path: Path,
        document_id: str,
        version: int,
    ) -> list[DocumentChunk]: ...

    async def close(self) -> None: ...


class DoclingParser:
    def __init__(self, embedding_model: str, chunk_tokens: int, model_cache_dir: Path) -> None:
        self.embedding_model = embedding_model
        self.chunk_tokens = chunk_tokens
        self.model_cache_dir = model_cache_dir
        self.pool = ProcessPoolExecutor(max_workers=1)

    async def parse(
        self,
        source_path: Path,
        parsed_path: Path,
        document_id: str,
        version: int,
    ) -> list[DocumentChunk]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.pool,
            _parse_document,
            str(source_path),
            str(parsed_path),
            document_id,
            version,
            self.embedding_model,
            self.chunk_tokens,
            str(self.model_cache_dir),
        )

    async def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def _parse_document(
    source_path: str,
    parsed_path: str,
    document_id: str,
    version: int,
    embedding_model: str,
    chunk_tokens: int,
    model_cache_dir: str,
) -> list[DocumentChunk]:
    cache_dir = Path(model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache_dir / "tiktoken"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))

    import tiktoken
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = True
    pdf_options.do_table_structure = True
    pdf_options.generate_parsed_pages = True
    pdf_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )
    document = converter.convert(source_path).document

    parsed_file = Path(parsed_path)
    parsed_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = parsed_file.with_suffix(".tmp")
    try:
        temporary_file.write_text(
            json.dumps(document.export_to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_file.replace(parsed_file)
    finally:
        temporary_file.unlink(missing_ok=True)

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model(embedding_model),
        max_tokens=chunk_tokens,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
    )
    raw_chunks = list(chunker.chunk(dl_doc=document))
    return build_chunks(raw_chunks, chunker, document_id, version)


def build_chunks(
    raw_chunks: list[object],
    chunker: object,
    document_id: str,
    version: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []

    for index, raw_chunk in enumerate(raw_chunks):
        text = str(getattr(raw_chunk, "text", "")).strip()
        if not text:
            continue

        section = _section_for(raw_chunk)
        chunks.append(
            DocumentChunk(
                chunk_id=str(uuid5(NAMESPACE_URL, f"{document_id}:{version}:chunk:{index}")),
                page=_page_for(raw_chunk),
                section=section,
                text=text,
                parent_section_id=(
                    str(uuid5(NAMESPACE_URL, f"{document_id}:{version}:section:{section}"))
                    if section
                    else None
                ),
                embedding_text=str(chunker.contextualize(chunk=raw_chunk)).strip(),
            )
        )

    return chunks


def _section_for(chunk: object) -> str | None:
    metadata = getattr(chunk, "meta", None)
    headings = getattr(metadata, "headings", None) or []
    clean_headings = [str(heading).strip() for heading in headings if str(heading).strip()]
    return " > ".join(clean_headings) or None


def _page_for(chunk: object) -> int | None:
    metadata = getattr(chunk, "meta", None)
    for item in getattr(metadata, "doc_items", None) or []:
        for provenance in getattr(item, "prov", None) or []:
            page = getattr(provenance, "page_no", None)
            if page is not None:
                return int(page)
    return None
