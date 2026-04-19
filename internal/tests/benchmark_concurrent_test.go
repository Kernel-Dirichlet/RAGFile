package tests

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/Kernel-Dirichlet/RAGFile/internal/reader"
	"github.com/Kernel-Dirichlet/RAGFile/internal/writer"
)

// workerCounts drives all concurrent benchmarks below.
var workerCounts = []int{1, 2, 4, 8, 16}

// ============================================================================
// CONCURRENT WRITE BENCHMARKS
// Measures throughput of WriteConcurrent at various thread counts.
// Dataset: 10 000 pairs × 128-dim (≈ realistic sentence-embedding workload)
// ============================================================================

func BenchmarkConcurrentWrite(b *testing.B) {
	const (
		numPairs = 10_000
		dim      = 128
	)
	pairs := getOrCreateCorpus(numPairs, dim)

	for _, workers := range workerCounts {
		name := fmt.Sprintf("workers=%d", workers)
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(numPairs) * int64(dim) * 4)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var buf bytes.Buffer
				if err := writer.WriteConcurrent(&buf, pairs, workers); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// ============================================================================
// CONCURRENT SEARCH BENCHMARKS
// Measures top-K search throughput at various thread counts.
// Dataset: 10 000 pairs × 128-dim, k = 10
// ============================================================================

func BenchmarkConcurrentSearch(b *testing.B) {
	const (
		numPairs = 10_000
		dim      = 128
		k        = 10
	)

	path, err := getOrCreateCorpusFile(numPairs, dim)
	if err != nil {
		b.Fatal(err)
	}

	mr, err := reader.OpenMMap(path)
	if err != nil {
		b.Fatal(err)
	}
	defer mr.Close()

	block, err := mr.ReadVectorBlock(0)
	if err != nil {
		b.Fatal(err)
	}

	query := generateRealisticEmbedding(dim)

	for _, workers := range workerCounts {
		name := fmt.Sprintf("workers=%d", workers)
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				results := block.ConcurrentTopK(query, k, workers, false)
				_ = results
			}
		})
	}
}

// ============================================================================
// BASELINE (single-threaded) vs CONCURRENT – side-by-side
// Makes it easy to see the speedup factor at a glance.
// ============================================================================

func BenchmarkConcurrentSearchVsBaseline(b *testing.B) {
	const (
		numPairs = 50_000
		dim      = 256
		k        = 10
	)

	path, err := getOrCreateCorpusFile(numPairs, dim)
	if err != nil {
		b.Fatal(err)
	}

	mr, err := reader.OpenMMap(path)
	if err != nil {
		b.Fatal(err)
	}
	defer mr.Close()

	block, err := mr.ReadVectorBlock(0)
	if err != nil {
		b.Fatal(err)
	}

	query := generateRealisticEmbedding(dim)

	// Single-threaded baseline
	b.Run("baseline_1worker", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			results := block.TopK(query, k, false)
			_ = results
		}
	})

	// Concurrent variants
	for _, workers := range workerCounts {
		name := fmt.Sprintf("concurrent_%dworkers", workers)
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				results := block.ConcurrentTopK(query, k, workers, false)
				_ = results
			}
		})
	}
}

// ============================================================================
// CONCURRENT WRITE THROUGHPUT – side-by-side across sizes
// ============================================================================

func BenchmarkConcurrentWriteScalability(b *testing.B) {
	configs := []struct {
		numPairs int
		dim      int
	}{
		{1_000, 64},
		{10_000, 128},
		{50_000, 256},
	}

	for _, cfg := range configs {
		pairs := getOrCreateCorpus(cfg.numPairs, cfg.dim)

		for _, workers := range workerCounts {
			name := fmt.Sprintf("pairs=%d_dim=%d_workers=%d", cfg.numPairs, cfg.dim, workers)
			b.Run(name, func(b *testing.B) {
				b.SetBytes(int64(cfg.numPairs) * int64(cfg.dim) * 4)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var buf bytes.Buffer
					if err := writer.WriteConcurrent(&buf, pairs, workers); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
