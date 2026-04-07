#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata

Entry point (called by main.py):
    build_index(markdown_file, cfg, keep_tables=True, do_visualize=False)
"""

import os
import hashlib
import pickle
import pathlib
import re
import json
import time
from typing import Any, Iterable, List, Dict, Optional, Union

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer

from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Default keywords to exclude sections
DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']

# ------------------------ Main index builder -----------------------------

def build_index(
    markdown_file: Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]],
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool = False,
    use_headings: bool = False
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(markdown_file, (str, os.PathLike)):
        markdown_files = [pathlib.Path(markdown_file)]
    else:
        markdown_files = [pathlib.Path(path) for path in markdown_file]

    if not markdown_files:
        raise ValueError("At least one markdown file is required to build an index.")

    total_start = time.perf_counter()
    timings = {
        "hash_and_manifest_check": 0.0,
        "document_reuse_load": 0.0,
        "document_rebuild": 0.0,
        "model_load": 0.0,
        "embedding": 0.0,
        "merge_total": 0.0,
        "faiss_rebuild": 0.0,
        "bm25_rebuild": 0.0,
        "artifact_writes": 0.0,
        "manifest_write": 0.0,
    }

    manifest_path = artifacts_dir / f"{index_prefix}_manifest.json"
    old_manifest = _load_manifest(manifest_path)
    old_docs = {
        doc["source_path"]: doc
        for doc in old_manifest.get("documents", [])
        if isinstance(doc, dict) and "source_path" in doc
    }
    config_fingerprint = _config_fingerprint(
        chunk_config=chunk_config,
        embedding_model_path=embedding_model_path,
        use_headings=use_headings,
        keep_tables=getattr(chunker, "keep_tables", None),
    )

    document_artifacts = []
    manifest_docs = []
    stats = {
        "total_documents": len(markdown_files),
        "reused_documents": 0,
        "rebuilt_documents": 0,
        "deleted_documents": max(0, len(set(old_docs) - {str(p) for p in markdown_files})),
        "chunks_reused": 0,
        "chunks_embedded": 0,
    }
    embedder = None

    for path in markdown_files:
        check_start = time.perf_counter()
        source_path = str(path)
        source_hash = _hash_file(path)
        old_doc = old_docs.get(source_path)
        artifact_path = _document_artifact_path(artifacts_dir, index_prefix, path)
        can_reuse = (
            old_doc is not None
            and old_doc.get("source_hash") == source_hash
            and old_doc.get("config_fingerprint") == config_fingerprint
            and pathlib.Path(old_doc.get("artifact_path", "")).exists()
        )
        timings["hash_and_manifest_check"] += time.perf_counter() - check_start

        if can_reuse:
            reuse_start = time.perf_counter()
            doc_artifact = _load_document_artifact(old_doc["artifact_path"])
            timings["document_reuse_load"] += time.perf_counter() - reuse_start
            stats["reused_documents"] += 1
            stats["chunks_reused"] += len(doc_artifact["chunks"])
            manifest_doc = dict(old_doc)
        else:
            rebuild_start = time.perf_counter()
            if embedder is None:
                print(
                    f"Loading embedder {pathlib.Path(embedding_model_path).stem} "
                    "for changed documents..."
                )
                model_load_start = time.perf_counter()
                embedder = SentenceTransformer(embedding_model_path)
                timings["model_load"] += time.perf_counter() - model_load_start
            doc_artifact = _build_document_artifact(
                markdown_file=str(path),
                chunker=chunker,
                chunk_config=chunk_config,
                embedder=embedder,
                embedding_model_path=embedding_model_path,
                use_multiprocessing=use_multiprocessing,
                use_headings=use_headings,
                timings=timings,
            )
            _save_document_artifact(artifact_path, doc_artifact)
            timings["document_rebuild"] += time.perf_counter() - rebuild_start
            stats["rebuilt_documents"] += 1
            stats["chunks_embedded"] += len(doc_artifact["chunks"])
            manifest_doc = {
                "source_path": source_path,
                "source_hash": source_hash,
                "artifact_path": str(artifact_path),
                "chunk_count": len(doc_artifact["chunks"]),
                "config_fingerprint": config_fingerprint,
                "updated_at": int(time.time()),
            }

        document_artifacts.append(doc_artifact)
        manifest_docs.append(manifest_doc)

    _merge_document_artifacts(
        document_artifacts=document_artifacts,
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
        timings=timings,
    )
    manifest = {
        "version": 1,
        "index_prefix": index_prefix,
        "config_fingerprint": config_fingerprint,
        "documents": manifest_docs,
        "stats": stats,
        "timings_seconds": timings,
        "updated_at": int(time.time()),
    }
    manifest_write_start = time.perf_counter()
    _save_manifest(manifest_path, manifest)
    timings["manifest_write"] += time.perf_counter() - manifest_write_start
    timings["total"] = time.perf_counter() - total_start
    manifest["timings_seconds"] = timings
    _save_manifest(manifest_path, manifest)

    print(
        "Incremental indexing summary: "
        f"{stats['rebuilt_documents']} rebuilt, "
        f"{stats['reused_documents']} reused, "
        f"{stats['deleted_documents']} deleted/stale, "
        f"{stats['chunks_embedded']} chunks embedded, "
        f"{stats['chunks_reused']} chunks reused."
    )
    print(
        "Index timing summary: "
        f"total={timings['total']:.3f}s, "
        f"hash/check={timings['hash_and_manifest_check']:.3f}s, "
        f"reuse_load={timings['document_reuse_load']:.3f}s, "
        f"document_rebuild={timings['document_rebuild']:.3f}s, "
        f"model_load={timings['model_load']:.3f}s, "
        f"embedding={timings['embedding']:.3f}s, "
        f"merge={timings['merge_total']:.3f}s, "
        f"faiss={timings['faiss_rebuild']:.3f}s, "
        f"bm25={timings['bm25_rebuild']:.3f}s, "
        f"writes={timings['artifact_writes']:.3f}s."
    )


def _build_document_artifact(
    *,
    markdown_file: str,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedder: SentenceTransformer,
    embedding_model_path: str,
    use_multiprocessing: bool,
    use_headings: bool,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build the reusable chunks, metadata, page map, and embeddings for one document."""
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    # Extract sections from markdown. Exclude some with certain keywords.
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )

    page_to_chunk_ids = {}
    current_page = 1
    total_chunks = 0
    heading_stack = []

    # Step 1: Chunk using DocumentChunker
    for c in sections:
        # Determine current section level
        current_level = c.get('level', 1)

        # Determine current chapter number
        chapter_num = c.get('chapter', 0)

        # Pop sections that are deeper or siblings
        while heading_stack and heading_stack[-1][0] >= current_level:
            heading_stack.pop()
        
        # Push pair of (level, heading)
        if c['heading'] != "Introduction":
            heading_stack.append((current_level, c['heading']))

        # Construct section path
        path_list = [h[1] for h in heading_stack]
        full_section_path = " ".join(path_list)
        full_section_path = f"Chapter {chapter_num} " + full_section_path

        # Use DocumentChunker to recursively split this section
        sub_chunks = chunker.chunk(c['content'])

        # Regex to find page markers like "--- Page 3 ---"
        page_pattern = re.compile(r'--- Page (\d+) ---')

        # Iterate through each chunk produced from this section
        for sub_chunk in sub_chunks:
            # Track all pages this specific chunk touches
            chunk_pages = set()

            # Split the sub_chunk by page markers to see if it
            # spans multiple pages.
            fragments = page_pattern.split(sub_chunk)

            # If there is content before the first page marker,
            # it belongs to the current_page.
            if fragments[0].strip():
                chunk_pages.add(current_page)

            # Process the new pages found within this sub_chunk. 
            # Step by 2 where each pair represents (page number, text after it)
            for i in range(1, len(fragments), 2):
                try:
                    # Get the new page number from the marker
                    new_page = int(fragments[i]) + 1

                    # If there is text after this marker, it belongs to the new_page.
                    if fragments[i+1].strip():
                        chunk_pages.add(new_page)
                    
                    current_page = new_page

                except (IndexError, ValueError):
                    continue

            # Clean sub_chunk by removing page markers
            clean_chunk = re.sub(page_pattern, '', sub_chunk).strip()
            
            # Skip introduction chunks for embedding
            if c["heading"] == "Introduction":
                continue

            chunk_id = len(all_chunks)
            for page in chunk_pages:
                page_to_chunk_ids.setdefault(page, set()).add(chunk_id)
            
            # Prepare metadata
            meta = {
                "filename": markdown_file,
                "mode": chunk_config.to_string(),
                "char_len": len(clean_chunk),
                "word_len": len(clean_chunk.split()),
                "section": c['heading'],
                "section_path": full_section_path,
                "text_preview": clean_chunk[:100],
                "page_numbers": sorted(list(chunk_pages)),
                "chunk_id": chunk_id
            }

            # Prepare chunk with prefix
            if use_headings:
                chunk_prefix = (
                    f"Description: {full_section_path} "
                    f"Content: "
                )
            else:
                chunk_prefix = ""

            all_chunks.append(chunk_prefix+clean_chunk)
            sources.append(markdown_file)
            metadata.append(meta)

        total_chunks += len(sub_chunks)

    # Convert the sets to sorted lists for a clean, predictable output
    page_to_chunk_map = {}
    for page, id_set in page_to_chunk_ids.items():
        page_to_chunk_map[page] = sorted(list(id_set))

    # Step 2: Create embeddings for FAISS index
    print(f"Embedding {len(all_chunks):,} chunks with {pathlib.Path(embedding_model_path).stem} ...")

    embedding_start = time.perf_counter()
    if not all_chunks:
        embeddings = np.empty((0, 0), dtype=np.float32)
    elif use_multiprocessing:
        print("Starting multi-process pool for embeddings...")
        # Start the pool. Adjust number of workers as needed.
        pool = embedder.start_multi_process_pool(num_workers=4)
        try:
            # Compute embeddings in parallel
            embeddings = embedder.encode_multi_process(
                all_chunks, 
                pool, 
                batch_size=32
            )
        finally:
            # Stop the pool to prevent hanging processes
            embedder.stop_multi_process_pool(pool)
    else:
        # Standard single-process embedding
        embeddings = embedder.encode(
            all_chunks, 
            batch_size=8, 
            show_progress_bar=True,
            convert_to_numpy=True 
        )
    if timings is not None:
        timings["embedding"] += time.perf_counter() - embedding_start

    return {
        "chunks": all_chunks,
        "sources": sources,
        "metadata": metadata,
        "embeddings": embeddings.astype("float32"),
        "page_to_chunk_map": page_to_chunk_map,
    }


def _merge_document_artifacts(
    *,
    document_artifacts: List[Dict[str, Any]],
    artifacts_dir: pathlib.Path,
    index_prefix: str,
    timings: Optional[Dict[str, float]] = None,
) -> None:
    merge_start = time.perf_counter()
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []
    page_to_chunk_map: Dict[int, List[int]] = {}
    embeddings = []

    for doc_artifact in document_artifacts:
        offset = len(all_chunks)
        doc_chunks = doc_artifact["chunks"]
        all_chunks.extend(doc_chunks)
        sources.extend(doc_artifact["sources"])
        if len(doc_chunks) > 0:
            embeddings.append(doc_artifact["embeddings"])

        for meta in doc_artifact["metadata"]:
            merged_meta = dict(meta)
            merged_meta["chunk_id"] = offset + int(meta["chunk_id"])
            metadata.append(merged_meta)

        for page, chunk_ids in doc_artifact["page_to_chunk_map"].items():
            merged_ids = [offset + int(chunk_id) for chunk_id in chunk_ids]
            page_to_chunk_map.setdefault(int(page), []).extend(merged_ids)

    if not all_chunks:
        raise ValueError("No chunks were produced from the provided markdown files.")

    embeddings = np.vstack(embeddings).astype("float32")

    if timings is not None:
        timings["merge_total"] += time.perf_counter() - merge_start

    # Step 3: Build FAISS index
    print(f"Building FAISS index for {len(all_chunks):,} chunks...")
    faiss_start = time.perf_counter()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
    if timings is not None:
        timings["faiss_rebuild"] += time.perf_counter() - faiss_start
    print(f"FAISS Index built successfully: {index_prefix}.faiss")

    # Step 4: Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    bm25_start = time.perf_counter()
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    if timings is not None:
        timings["bm25_rebuild"] += time.perf_counter() - bm25_start
    print(f"BM25 Index built successfully: {index_prefix}_bm25.pkl")

    # Step 5: Dump index artifacts
    write_start = time.perf_counter()
    output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    final_map = {
        page: sorted(chunk_ids)
        for page, chunk_ids in sorted(page_to_chunk_map.items())
    }
    with open(output_file, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"Saved page to chunk ID map: {output_file}")
    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    if timings is not None:
        timings["artifact_writes"] += time.perf_counter() - write_start
    print(f"Saved all index artifacts with prefix: {index_prefix}")

# ------------------------ Helper functions ------------------------------

def _hash_file(path: os.PathLike) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _config_fingerprint(
    *,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    use_headings: bool,
    keep_tables: Optional[bool],
) -> str:
    payload = {
        "chunk_config": chunk_config.to_string(),
        "embedding_model_path": embedding_model_path,
        "use_headings": use_headings,
        "keep_tables": keep_tables,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_manifest(path: os.PathLike) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"version": 1, "documents": []}


def _save_manifest(path: os.PathLike, manifest: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _document_artifact_path(
    artifacts_dir: pathlib.Path,
    index_prefix: str,
    markdown_file: os.PathLike,
) -> pathlib.Path:
    source_path = str(pathlib.Path(markdown_file))
    doc_id = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
    return artifacts_dir / f"{index_prefix}_doc_{doc_id}.pkl"


def _load_document_artifact(path: os.PathLike) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_document_artifact(path: os.PathLike, artifact: Dict[str, Any]) -> None:
    with open(path, "wb") as f:
        pickle.dump(artifact, f)

def preprocess_for_bm25(text: str) -> list[str]:
    """
    Simplifies text to keep only letters, numbers, underscores, hyphens,
    apostrophes, plus, and hash — suitable for BM25 tokenization.
    """
    # Convert to lowercase
    text = text.lower()

    # Keep only allowed characters
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)

    # Split by whitespace
    tokens = text.split()

    return tokens
