# Incremental Indexing Sample

This small file is intended for manual indexing tests. Copy it into `data/`
when you want a lightweight document that follows the section-heading format
used by TokenSmith's markdown parser.

## 1.1 Storage Layout

Database systems store records on pages so that reads and writes can move data
between disk and memory in predictable units. A page can contain a header, slot
directory, and payload area. The slot directory keeps record locations stable
even when records move inside the page. This makes updates cheaper because the
external record identifier can remain unchanged.

--- Page 1 ---

Heap files store pages in no particular logical order. They are simple to
append to and useful when queries scan many records. Sorted files maintain
records according to a search key. They make range queries faster but make
insertions more expensive because records may need to move.

## 1.2 Buffer Management

The buffer manager is responsible for bringing pages into memory and deciding
which pages to evict when the buffer pool is full. A pin count prevents a page
from being evicted while an operator is actively using it. A dirty bit records
whether the in-memory page must be written back before eviction.

--- Page 2 ---

Replacement policies such as least recently used and clock approximate which
pages are likely to be needed again. The best policy depends on the workload.
A sequential scan can pollute the buffer pool if it evicts pages that will be
needed by other queries.

## 1.3 Index Maintenance

Indexes accelerate lookup by storing search keys with pointers to records or
pages. A dense index stores an entry for every record. A sparse index stores
entries for only some records and depends on ordered data pages. When records
are inserted or deleted, the database must keep indexes consistent with the
base table.

--- Page 3 ---

Incremental indexing follows the same general idea: unchanged data should be
reused, while changed data should be rebuilt and merged back into the global
access structure.
