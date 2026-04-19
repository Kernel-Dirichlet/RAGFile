// Package format contains the binary layout definitions for RAGFile
package format

// SectionType represents the type of a section in the RAGFile format
type SectionType uint8

const (
	// SectionVector indicates a vector block section containing embeddings
	SectionVector SectionType = 0x01
	// SectionKeyword indicates a keyword index section
	SectionKeyword SectionType = 0x02
)

// SectionHeader is the header for each section in the RAGFile
// Binary layout: type (1 byte) + startOffset (8 bytes) + endOffset (8 bytes) = 17 bytes
type SectionHeader struct {
	Type        SectionType
	StartOffset uint64
	EndOffset   uint64
}

type VectorEntry struct { 
	ChunkStart uint64
	ChunkEnd uint64
	EmbeddingStart uint64
	EmbeddingEnd uint64

}

type VectorBlock struct { 
	NumPairs uint32
	IndexOffset uint64
	DataOffset uint64
}

type KeywordEntry struct { 
	KeywordStart uint64
	KeywordEnd uint64
	ChunkStart uint64
	ChunkEnd uint64
}
