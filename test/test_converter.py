from docling.chunking import HybridChunker

from docling_haystack.converter import DoclingConverter, ExportType

PATHS = ["test/data/2408.09869v5.pdf"]


def test_convert_doc_chunks(monkeypatch):
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    converter = DoclingConverter(
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    )
    docs = converter.run(sources=PATHS, meta=[{"test_custom_meta": "passed"}])[
        "documents"
    ]
    assert len(docs) >= 50
    assert docs[0].meta["test_custom_meta"] == "passed"
    assert docs[-1].meta["test_custom_meta"] == "passed"
    assert "dl_meta" in docs[0].meta


def test_convert_markdown(monkeypatch):
    converter = DoclingConverter(
        export_type=ExportType.MARKDOWN,
    )
    # BYTE_STREAM_DATA = [open("test/data/2408.09869v5.pdf", "rb").read()] # TODO BytStream format not supported yet
    # docs = converter.run(sources=[ByteStream(BYTE_STREAM_DATA[0])], meta={"test_custom_meta":"passed"})["documents"]# TODO BytStream format not supported yet
    docs = converter.run(sources=PATHS, meta={"test_custom_meta": "passed"})[
        "documents"
    ]
    assert len(docs) == 1
    assert docs[0].meta["test_custom_meta"] == "passed"
    assert "dl_meta" in docs[0].meta
