import json
import pickle
import argparse

import numpy as np
import pytest

from src.index_builder import build_index
from src.preprocessing.chunking import SectionRecursiveConfig


pytestmark = pytest.mark.unit


class FakeChunker:
    keep_tables = False

    def chunk(self, text):
        return [text]


class FakeEmbedder:
    init_count = 0
    encode_calls = []

    def __init__(self, model_path):
        self.model_path = model_path
        FakeEmbedder.init_count += 1

    def encode(self, texts, **kwargs):
        FakeEmbedder.encode_calls.append(list(texts))
        return np.array(
            [[float(i + 1), float(len(text)), 1.0] for i, text in enumerate(texts)],
            dtype=np.float32,
        )


def _write_markdown(path, section_title, content):
    path.write_text(f"## {section_title}\n{content}\n", encoding="utf-8")


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _build(tmp_path, monkeypatch, markdown_files):
    FakeEmbedder.init_count = 0
    FakeEmbedder.encode_calls = []
    monkeypatch.setattr("src.index_builder.SentenceTransformer", FakeEmbedder)

    artifacts_dir = tmp_path / "index"
    build_index(
        markdown_file=markdown_files,
        chunker=FakeChunker(),
        chunk_config=SectionRecursiveConfig(
            recursive_chunk_size=2000,
            recursive_overlap=200,
        ),
        embedding_model_path="fake-model.gguf",
        artifacts_dir=artifacts_dir,
        index_prefix="test_index",
    )
    return artifacts_dir


def test_incremental_indexing_reuses_unchanged_documents(tmp_path, monkeypatch):
    doc1 = tmp_path / "doc1.md"
    doc2 = tmp_path / "doc2.md"
    _write_markdown(doc1, "1 First", "alpha content")
    _write_markdown(doc2, "2 Second", "beta content")

    artifacts_dir = _build(tmp_path, monkeypatch, [doc1, doc2])
    assert FakeEmbedder.init_count == 1
    assert [len(call) for call in FakeEmbedder.encode_calls] == [1, 1]

    artifacts_dir = _build(tmp_path, monkeypatch, [doc1, doc2])
    assert FakeEmbedder.init_count == 0
    assert FakeEmbedder.encode_calls == []

    manifest = json.loads((artifacts_dir / "test_index_manifest.json").read_text())
    assert manifest["stats"]["rebuilt_documents"] == 0
    assert manifest["stats"]["reused_documents"] == 2
    assert manifest["stats"]["chunks_reused"] == 2

    chunks = _load_pickle(artifacts_dir / "test_index_chunks.pkl")
    sources = _load_pickle(artifacts_dir / "test_index_sources.pkl")
    assert chunks == ["alpha content", "beta content"]
    assert sources == [str(doc1), str(doc2)]


def test_incremental_indexing_rebuilds_only_changed_document(tmp_path, monkeypatch):
    doc1 = tmp_path / "doc1.md"
    doc2 = tmp_path / "doc2.md"
    _write_markdown(doc1, "1 First", "alpha content")
    _write_markdown(doc2, "2 Second", "beta content")

    artifacts_dir = _build(tmp_path, monkeypatch, [doc1, doc2])
    _write_markdown(doc2, "2 Second", "beta content changed")

    artifacts_dir = _build(tmp_path, monkeypatch, [doc1, doc2])
    assert FakeEmbedder.init_count == 1
    assert [len(call) for call in FakeEmbedder.encode_calls] == [1]

    manifest = json.loads((artifacts_dir / "test_index_manifest.json").read_text())
    assert manifest["stats"]["rebuilt_documents"] == 1
    assert manifest["stats"]["reused_documents"] == 1
    assert manifest["stats"]["chunks_embedded"] == 1

    chunks = _load_pickle(artifacts_dir / "test_index_chunks.pkl")
    assert chunks == ["alpha content", "beta content changed"]


def test_incremental_indexing_omits_deleted_document(tmp_path, monkeypatch):
    doc1 = tmp_path / "doc1.md"
    doc2 = tmp_path / "doc2.md"
    _write_markdown(doc1, "1 First", "alpha content")
    _write_markdown(doc2, "2 Second", "beta content")

    artifacts_dir = _build(tmp_path, monkeypatch, [doc1, doc2])
    doc2.unlink()

    artifacts_dir = _build(tmp_path, monkeypatch, [doc1])
    assert FakeEmbedder.init_count == 0
    assert FakeEmbedder.encode_calls == []

    manifest = json.loads((artifacts_dir / "test_index_manifest.json").read_text())
    assert manifest["stats"]["deleted_documents"] == 1
    assert manifest["stats"]["reused_documents"] == 1
    assert len(manifest["documents"]) == 1

    chunks = _load_pickle(artifacts_dir / "test_index_chunks.pkl")
    sources = _load_pickle(artifacts_dir / "test_index_sources.pkl")
    assert chunks == ["alpha content"]
    assert sources == [str(doc1)]


def test_run_index_mode_uses_explicit_markdown_file(tmp_path, monkeypatch):
    from src import main as main_module
    from src.config import RAGConfig

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample = data_dir / "sample.md"
    other = data_dir / "wbl.md"
    sample.write_text("## 1.1 Sample\nsample text\n", encoding="utf-8")
    other.write_text("## 1.1 WBL\nwbl text\n", encoding="utf-8")

    captured = {}

    def fake_build_index(**kwargs):
        captured["markdown_file"] = kwargs["markdown_file"]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module, "build_index", fake_build_index)

    args = argparse.Namespace(
        keep_tables=False,
        markdown_file=str(sample),
        index_prefix="sample_only",
        multiproc_indexing=False,
        embed_with_headings=False,
    )

    main_module.run_index_mode(args, RAGConfig())

    assert captured["markdown_file"] == [sample]
