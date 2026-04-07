# 6423 Project Proposal
> Tai Wei Wu
# Incremental Indexing for TokenSmith

## Problem.
TokenSmith currently rebuilds its entire retrieval index from scratch whenever documents are indexed. This is slow and wasteful, especially for large textbook-based corpora.

## Why it matters.
A local RAG system is only practical if indexing is efficient. Full re-indexing increases latency, wastes compute, and makes it harder to update the corpus incrementally.

## Approach.
I will add incremental indexing at the document level. The system will maintain a manifest with file hashes and artifact metadata, detect which source documents are new or changed, and only re-run parsing, chunking, and embedding for those files. Unchanged artifacts will be reused when rebuilding the global index.

## Validation.
I will verify that incremental indexing produces the same final index contents and query behavior as a full rebuild. I will test add, modify, and delete cases.

## Performance evaluation.
I will compare full vs. incremental indexing on:
* indexing time
* number of documents/chunks reprocessed
* amount of embedding work avoided

## Resources needed.
TokenSmith codebase, provided textbook PDF(s), local GGUF models, Python/Conda environment, FAISS, and BM25.

## Goals.
* 75%: Detect unchanged documents and skip reprocessing them.
* 100%: Store per-document artifacts and rebuild the merged index correctly from changed files only.
* 125%: Extend the design to section-level or chunk-level incremental indexing and evaluate the tradeoffs.