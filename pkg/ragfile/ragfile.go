// Package ragfile provides the public API for RAGFile format.
// It enables fast, local vector search for RAG applications.
package ragfile

import (
	"io"

	"github.com/Kernel-Dirichlet/RAGFile/internal/format"
	"github.com/Kernel-Dirichlet/RAGFile/internal/reader"
	"github.com/Kernel-Dirichlet/RAGFile/internal/writer"
)

// SemanticPair represents a chunk of text and its embedding vector.
type SemanticPair = writer.SemanticPair

// SearchResult contains a retrieved chunk and its similarity score.
type SearchResult struct {
	Chunk []byte
	Score float32
	Index int
}

// WriteRAGFile writes semantic pairs to a RAGFile format.
// This is the main entry point for writing vector data.
func WriteRAGFile(w io.Writer, pairs []SemanticPair) error {
	_, err := writer.WriteVectorBlockInternal(w, pairs, 0, 0, 0)
	return err
}

// RAGFile represents an opened RAGFile for reading and searching.
type RAGFile struct {
	mr    *reader.MMapReader
	block *reader.MMapVectorBlock
}

// OpenRAGFile opens a RAGFile from disk for reading and searching.
func OpenRAGFile(path string) (*RAGFile, error) {
	mr, err := reader.OpenMMap(path)
	if err != nil {
		return nil, err
	}

	block, err := mr.ReadVectorBlock(0)
	if err != nil {
		mr.Close()
		return nil, err
	}

	return &RAGFile{mr: mr, block: block}, nil
}

// Close releases resources associated with the RAGFile.
func (rf *RAGFile) Close() error {
	return rf.mr.Close()
}

// Len returns the number of entries in the RAGFile.
func (rf *RAGFile) Len() int {
	return int(rf.block.Meta().NumPairs)
}

// Search performs a TopK search and returns results with chunks and scores.
// Uses safe access (no unsafe pointer operations).
func (rf *RAGFile) Search(query []float32, k int) []SearchResult {
	rawResults := rf.block.TopK(query, k, false)

	results := make([]SearchResult, len(rawResults))
	for i, r := range rawResults {
		results[i] = SearchResult{
			Chunk: rf.block.GetChunk(r.Index),
			Score: r.Score,
			Index: r.Index,
		}
	}
	return results
}

// GetChunk returns the chunk text at a specific index.
func (rf *RAGFile) GetChunk(index int) []byte {
	return rf.block.GetChunk(index)
}

// GetEmbedding returns the embedding vector at a specific index.
// Uses safe access (no unsafe pointer operations).
func (rf *RAGFile) GetEmbedding(index int) []float32 {
	return rf.block.GetEmbeddingSafe(index)
}

// Meta returns the vector block metadata (numPairs, offsets).
func (rf *RAGFile) Meta() format.VectorBlock {
	return rf.block.Meta()
}