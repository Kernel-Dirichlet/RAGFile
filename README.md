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