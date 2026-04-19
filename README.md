# RAGFile v0.1

RAGFile is a binary file format inspired by Hadoop Sequence Files to facilitate fast RAG (Retrieval-Augmented Generation) workflows. It provides an efficient, storage-optimized format for large-scale document retrieval systems.


## Binary File Format

RAGFile follows a structured binary format designed for optimal disk I/O performance:

### Header Structure
- **"RAGFILE" header** (verification of file)
- **RAGFile version** (major,minor,patch)
- **Endianess** (1 - Big Endian, 0 - Little Endian)

### Index Strategy Sections
This section contains any number of {strategy}-({start_byte,end_byte}) pairs:
- Each strategy defines its byte range within the file
- Strategies are separated by a single \x00 byte
- Supported strategies in v0.1.0:
  - keyword-(start_byte,end_byte)
  - vector-(start_byte,end_byte)

### Data Sections

#### Keyword-Content Pairs
- **start-byte**: First byte of first keyword-content pair
- **end-byte**: Last byte of last keyword-content pair  
- **padding**: Null \x00 bytes between pairs (4, 8, or 16)
- Format: {keyword}-{content} (single "-" separator)

#### Embedding-Content Pairs
- **precision**: Numerical precision of embedding vectors
- **start-byte**: First byte of first embedding pair
- **end-byte**: Last byte of last embedding pair
- **padding**: Null \x00 bytes between pairs (4, 8, or 16)
- Format: {embedding}-{content} (single "-" separator)

## Use Cases

### Air-Gapped Environments
RAGFile is ideal for air-gapped or disconnected environments where:
- Network connectivity is limited or unavailable
- Local storage and processing are required
- Data sovereignty and security are critical

### Local Top-K Search
The format enables efficient local top-K similarity searches:
- Fast retrieval without network overhead
- Optimized for disk I/O operations
- Supports multiple indexing strategies

### TinyML Applications
RAGFile is well-suited for TinyML and edge computing scenarios:
- Minimal resource requirements
- Efficient binary format reduces memory footprint
- Fast local processing capabilities
- **No external network calls needed**

## Benchmarks

### Python SQLite Benchmarks

The Python implementation using SQLite provides a different approach to vector similarity searching:

**Key Differences:**
- **Storage Format**: SQLite uses BLOBs for vector storage vs. custom binary format
- **Query Method**: In-memory similarity calculation vs. optimized binary search
- **Dependencies**: Requires SQLite and NumPy vs. pure Go implementation
- **Portability**: Cross-platform with Python vs. Go binary

**Performance Characteristics:**
- **Write Speed**: Generally slower due to SQLite transaction overhead
- **Search Speed**: Slower for large datasets due to in-memory calculations
- **Memory Usage**: Higher due to NumPy array allocations
- **Setup Complexity**: Lower - no compilation required

### Performance Metrics
- **Read Speed**: Optimized for sequential and random access patterns
- **Storage Efficiency**: High data density with minimal overhead
- **Indexing Performance**: Fast strategy-based retrieval
- **Memory Usage**: Low memory footprint during operations
### Comparison with Alternatives
- **vs. Traditional Databases**: Faster local access, no network latency
- **vs. Text Files**: Better compression and indexing capabilities
- **vs. Other Binary Formats**: Specialized for RAG workflows

### Scalability
- **Small Scale**: Efficient for datasets under 1GB
- **Medium Scale**: Handles 1GB-100GB datasets effectively
- **Large Scale**: Optimized for 100GB+ document stores

### Future Enhancements
- Graph and hypergraph searches
- Advanced search space narrowing techniques
- Hardware acceleration support

## Benchmark Comparison: RAGFile vs Python + SQLite

The benchmark compares RAGFile (Go, mmap-backed) against a Python implementation
that stores vectors as BLOBs in an **in-memory SQLite database** — the fastest
possible configuration for SQLite, with zero disk I/O.  Even so, RAGFile is
significantly faster across every operation.

### Why RAGFile Is Faster: mmap vs SQLite RAM

SQLite's in-memory mode allocates a private heap buffer for every row it returns.
Each `SELECT` call copies the BLOB bytes from the internal B-tree page into a new
Python `bytes` object, then NumPy copies those bytes again into a `float32` array
before the dot-product can run.  That is **three copies** of the embedding data
per vector, plus Python object overhead and the GIL.

RAGFile uses **memory-mapped I/O** (`mmap` / `syscall.Mmap`).  The OS maps the
file pages directly into the process address space.  Reading an embedding is a
single pointer arithmetic operation — no copy, no allocation, no system call after
the initial `mmap`.  The CPU's prefetcher can stream the data at full memory
bandwidth because the layout is a flat, contiguous binary block.

| Access model | Copies per vector | Allocation per vector | GIL held |
|---|---|---|---|
| SQLite in-memory (Python) | 3 (B-tree → bytes → ndarray) | Yes (heap) | Yes |
| RAGFile mmap (Go) | 0 (pointer into mapped page) | No | N/A |

### Why Top-K Is Faster: Min-Heap vs Full Sort

Python's SQLite approach fetches **all** rows, computes all similarities, appends
them to a list, then calls `list.sort()` — O(n log n) with a large constant.

RAGFile uses a **fixed-size min-heap of size k** (Go's `container/heap`).  For
each vector it computes the dot product and either discards the result (O(1)) or
replaces the heap minimum (O(log k)).  Total complexity is **O(n log k)**, and
because k ≪ n the heap stays in L1/L2 cache for the entire scan.

| Algorithm | Complexity | Memory | Cache behaviour |
|---|---|---|---|
| Python list + sort | O(n log n) | O(n) — full list | Poor (n grows) |
| RAGFile min-heap | O(n log k) | O(k) — fixed | Excellent (k is tiny) |

### Write Performance: RAGFile vs SQLite (in-memory)

| Dataset | RAGFile write | SQLite write | RAGFile faster by |
|---------|--------------|--------------|-------------------|
| Small  (100 pairs, dim 16)    | 16.4 µs | 0.8 ms  | ~49×  |
| Medium (1 000 pairs, dim 32)  | 14.3 µs | 8.1 ms  | ~567× |
| Large  (10 000 pairs, dim 64) | 1.8 ms  | 72.5 ms | ~40×  |

### Top-K Search Performance: RAGFile vs SQLite (in-memory)

| Dataset | RAGFile top-10 | SQLite top-10 | RAGFile faster by |
|---------|---------------|---------------|-------------------|
| Small  (100 pairs, dim 16)    | 16.0 µs | 1.6 ms   | ~100×   |
| Medium (1 000 pairs, dim 32)  | 15.5 µs | 15.9 ms  | ~1 026× |
| Large  (10 000 pairs, dim 64) | 3.8 ms  | 148.6 ms | ~39×    |

### Concurrent Top-K Search (RAGFile, 10 000 pairs × dim 128, k = 10)

RAGFile ships a `ConcurrentTopK` function that partitions the index across
goroutines and merges partial min-heaps.  Because the mmap region is read-only
and shared, goroutines access it with **zero locking**.

| Workers | Latency | vs 1 worker |
|---------|---------|-------------|
| 1  | 4.14 ms | baseline    |
| 2  | 2.92 ms | 1.4× faster |
| 4  | 1.92 ms | 2.2× faster |
| 8  | 2.02 ms | 2.0× faster |
| 16 | 2.00 ms | 2.1× faster |

Speedup plateaus at 4 workers on this 4-core test machine (Intel N95), which is
the expected result — adding more goroutines than physical cores adds scheduling
overhead without additional parallelism.

### Portability: Cross-Compilation vs Runtime Dependencies

RAGFile is a single statically-linked binary produced by `go build`.  It
cross-compiles to any target with one command:

```
GOOS=linux  GOARCH=arm64  go build ./...   # Raspberry Pi / edge device
GOOS=windows GOARCH=amd64 go build ./...   # Windows air-gapped workstation
```

The Python + SQLite approach requires Python ≥ 3.x, NumPy, and a working SQLite
shared library on every target — none of which are guaranteed in air-gapped or
embedded environments.

### Use Case Fit

| Scenario | RAGFile | SQLite + Python |
|----------|---------|-----------------|
| Air-gapped / offline deployment | ✅ single binary | ❌ runtime deps |
| TinyML / edge device | ✅ cross-compile, minimal RAM | ❌ heavy runtime |
| Local top-K RAG search | ✅ mmap + min-heap | ⚠️ slower, more RAM |
| Rapid prototyping | ⚠️ requires Go toolchain | ✅ easy to iterate |

## Cross-Compilation

RAGFile is a pure-Go project with no CGo dependencies, so it cross-compiles to
any supported GOOS/GOARCH pair with a single command — no toolchain changes, no
Docker, no sysroot.

### Quick reference

```bash
# Linux – x86-64 (servers, desktops)
GOOS=linux   GOARCH=amd64   go build -o ragfile-linux-amd64   ./...

# Linux – ARM64 (Raspberry Pi 4/5, AWS Graviton, Apple M-series VMs)
GOOS=linux   GOARCH=arm64   go build -o ragfile-linux-arm64   ./...

# Linux – ARMv7 (Raspberry Pi 2/3, most embedded Linux boards)
GOOS=linux   GOARCH=arm     GOARM=7 go build -o ragfile-linux-armv7 ./...

# Linux – RISC-V 64 (SiFive, StarFive, VisionFive 2)
GOOS=linux   GOARCH=riscv64 go build -o ragfile-linux-riscv64 ./...

# macOS – Apple Silicon (M1/M2/M3/M4)
GOOS=darwin  GOARCH=arm64   go build -o ragfile-darwin-arm64  ./...

# macOS – Intel
GOOS=darwin  GOARCH=amd64   go build -o ragfile-darwin-amd64  ./...

# Windows – x86-64
GOOS=windows GOARCH=amd64   go build -o ragfile-windows-amd64.exe ./...

# Windows – ARM64 (Surface Pro X, Snapdragon laptops)
GOOS=windows GOARCH=arm64   go build -o ragfile-windows-arm64.exe ./...
```

### Build all targets at once

```bash
#!/usr/bin/env bash
targets=(
  "linux/amd64"
  "linux/arm64"
  "linux/arm"
  "linux/riscv64"
  "darwin/amd64"
  "darwin/arm64"
  "windows/amd64"
  "windows/arm64"
)

for target in "${targets[@]}"; do
  IFS='/' read -r os arch <<< "$target"
  out="ragfile-${os}-${arch}"
  [[ "$os" == "windows" ]] && out="${out}.exe"
  echo "Building $out …"
  GOOS=$os GOARCH=$arch go build -o "$out" ./...
done
echo "Done."
```

### Why cross-compilation matters for RAGFile

| Deployment target | GOOS/GOARCH | Notes |
|---|---|---|
| Air-gapped x86 server | `linux/amd64` | Standard server deployment |
| Raspberry Pi 4/5 | `linux/arm64` | Edge RAG, local LLM inference |
| Raspberry Pi 2/3 | `linux/arm` | Older edge hardware |
| AWS Graviton 3/4 | `linux/arm64` | Cloud cost optimisation |
| Apple M-series Mac | `darwin/arm64` | Developer workstation |
| Windows workstation | `windows/amd64` | Air-gapped enterprise |
| RISC-V SBC | `linux/riscv64` | Open-hardware TinyML |

> **Note on `mmap`**: The `syscall.Mmap` call used by RAGFile is available on
> all Linux and macOS targets listed above.  Windows uses a different syscall
> (`CreateFileMapping` / `MapViewOfFile`); a Windows-compatible mmap shim is
> planned for v0.2.
