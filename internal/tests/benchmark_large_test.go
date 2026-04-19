package tests

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/Kernel-Dirichlet/RAGFile/internal/reader"
	"github.com/Kernel-Dirichlet/RAGFile/internal/writer"
)

// ============================================================================
// REALISTIC EMBEDDING GENERATION
// Mimics real sentence embedding distributions (L2-normalized, near-zero mean)
// ============================================================================

func generateRealisticEmbedding(dim int) []float32 {
	vec := make([]float32, dim)
	// Normal distribution centered at 0 with small std
	for i := range vec {
		vec[i] = float32(rand.NormFloat64() * 0.02)
	}
	// L2 normalize to unit sphere (like real sentence embeddings)
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm == 0 {
		norm = 1
	}
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}

func generateRealisticPairs(numPairs, dim int) []writer.SemanticPair {
	pairs := make([]writer.SemanticPair, numPairs)
	for i := range pairs {
		chunkLen := rand.Intn(128) + 32 // 32-160 bytes per chunk
		pairs[i] = writer.SemanticPair{
			Chunk:     randBytes(chunkLen),
			Embedding: generateRealisticEmbedding(dim),
		}
	}
	return pairs
}

// ============================================================================
// MEMORY PROFILING UTILITIES
// ============================================================================

func getRSSBytes() uint64 {
	var stat syscall.Rusage
	err := syscall.Getrusage(syscall.RUSAGE_SELF, &stat)
	if err != nil {
		return 0
	}
	// Maxrss is in KB on Linux
	return uint64(stat.Maxrss) * 1024
}

func getMemStats() (heapAlloc, heapInUse, sys uint64) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc, m.HeapInuse, m.Sys
}

// ============================================================================
// HELPER: Write pairs to temp file and return path
// ============================================================================

func writePairsToTempFile(pairs []writer.SemanticPair) (string, error) {
	var buf Buffer
	_, err := writer.WriteVectorBlockInternal(&buf, pairs, 0, 0, 0)
	if err != nil {
		return "", err
	}

	tmp, err := os.CreateTemp("", "ragfile-bench-*.bin")
	if err != nil {
		return "", err
	}
	defer tmp.Close()

	if _, err := tmp.Write(buf.Bytes()); err != nil {
		os.Remove(tmp.Name())
		return "", err
	}

	return tmp.Name(), nil
}

// Buffer is a bytes.Buffer wrapper that implements io.Writer
type Buffer struct {
	data []byte
}

func (b *Buffer) Write(p []byte) (n int, err error) {
	b.data = append(b.data, p...)
	return len(p), nil
}

func (b *Buffer) Bytes() []byte { return b.data }

// ============================================================================
// CORPUS PREPARATION (shared across benchmarks)
// ============================================================================

var (
	corpusCache     = make(map[string][]writer.SemanticPair)
	corpusCacheMu   sync.Mutex
	fileCache       = make(map[string]string) // key -> temp file path
	fileCacheMu     sync.Mutex
)

func corpusKey(numPairs, dim int) string {
	return fmt.Sprintf("%d_%d", numPairs, dim)
}

func getOrCreateCorpus(numPairs, dim int) []writer.SemanticPair {
	key := corpusKey(numPairs, dim)
	corpusCacheMu.Lock()
	defer corpusCacheMu.Unlock()

	if pairs, ok := corpusCache[key]; ok {
		return pairs
	}

	// Use fixed seed for reproducibility
	rng := rand.New(rand.NewSource(int64(numPairs*1000 + dim)))
	pairs := make([]writer.SemanticPair, numPairs)
	for i := range pairs {
		chunkLen := rng.Intn(128) + 32
		chunk := make([]byte, chunkLen)
		rng.Read(chunk)

		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = float32(rng.NormFloat64() * 0.02)
		}
		// L2 normalize
		var norm float32
		for _, v := range emb {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm == 0 {
			norm = 1
		}
		for j := range emb {
			emb[j] /= norm
		}

		pairs[i] = writer.SemanticPair{
			Chunk:     chunk,
			Embedding: emb,
		}
	}

	corpusCache[key] = pairs
	return pairs
}

func getOrCreateCorpusFile(numPairs, dim int) (string, error) {
	key := corpusKey(numPairs, dim)
	fileCacheMu.Lock()
	defer fileCacheMu.Unlock()

	if path, ok := fileCache[key]; ok {
		return path, nil
	}

	pairs := getOrCreateCorpus(numPairs, dim)

	var buf Buffer
	_, err := writer.WriteVectorBlockInternal(&buf, pairs, 0, 0, 0)
	if err != nil {
		return "", err
	}

	tmp, err := os.CreateTemp("", "ragfile-bench-*.bin")
	if err != nil {
		return "", err
	}
	defer tmp.Close()

	if _, err := tmp.Write(buf.data); err != nil {
		os.Remove(tmp.Name())
		return "", err
	}

	fileCache[key] = tmp.Name()
	return tmp.Name(), nil
}

// ============================================================================
// WRITE BENCHMARKS
// ============================================================================

func BenchmarkWrite(b *testing.B) {
	sizes := []int{1_000, 5_000, 10_000, 50_000, 100_000, 250_000}
	dims := []int{64, 128, 256, 512, 1024, 2048, 4096}

	for _, size := range sizes {
		for _, dim := range dims {
			// Skip impractical combinations (>10GB)
			estSize := int64(size) * (128 + int64(dim)*4 + 32) // rough estimate
			if estSize > 10*1024*1024*1024 {
				continue
			}

			name := fmt.Sprintf("size=%d_dim=%d", size, dim)
			b.Run(name, func(b *testing.B) {
				pairs := getOrCreateCorpus(size, dim)
				b.SetBytes(int64(len(pairs)) * int64(dim)*4) // bytes processed

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var buf Buffer
					_, err := writer.WriteVectorBlockInternal(&buf, pairs, 0, 0, 0)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// ============================================================================
// SEARCH BENCHMARKS
// ============================================================================

func BenchmarkSearch(b *testing.B) {
	sizes := []int{10_000, 50_000, 100_000, 250_000}
	dims := []int{64, 128, 256, 512, 1024, 2048, 4096}
	kValues := []int{5, 10, 50, 100}

	for _, size := range sizes {
		for _, dim := range dims {
			// Skip impractical combinations
			estSize := int64(size) * (128 + int64(dim)*4 + 32)
			if estSize > 10*1024*1024*1024 {
				continue
			}

			for _, k := range kValues {
				name := fmt.Sprintf("size=%d_dim=%d_k=%d", size, dim, k)
				b.Run(name, func(b *testing.B) {
					path, err := getOrCreateCorpusFile(size, dim)
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

					// Generate a realistic query vector
					query := generateRealisticEmbedding(dim)

					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						results := block.TopK(query, k, false)
						_ = results
					}
				})
			}
		}
	}
}

// ============================================================================
// SCALABILITY BENCHMARKS (fixed dim, varying size)
// ============================================================================

func BenchmarkScalability(b *testing.B) {
	dims := []int{64, 256, 768, 1536}
	sizes := []int{1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000}
	k := 10

	for _, dim := range dims {
		for _, size := range sizes {
			estSize := int64(size) * (128 + int64(dim)*4 + 32)
			if estSize > 10*1024*1024*1024 {
				continue
			}

			name := fmt.Sprintf("dim=%d_size=%d", dim, size)
			b.Run(name, func(b *testing.B) {
				path, err := getOrCreateCorpusFile(size, dim)
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

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					results := block.TopK(query, k, false)
					_ = results
				}
			})
		}
	}
}

// ============================================================================
// MEMORY BENCHMARKS
// ============================================================================

func BenchmarkMemory(b *testing.B) {
	sizes := []int{10_000, 50_000, 100_000, 250_000}
	dims := []int{64, 256, 768, 1536}

	for _, size := range sizes {
		for _, dim := range dims {
			estSize := int64(size) * (128 + int64(dim)*4 + 32)
			if estSize > 10*1024*1024*1024 {
				continue
			}

			name := fmt.Sprintf("size=%d_dim=%d", size, dim)
			b.Run(name, func(b *testing.B) {
				path, err := getOrCreateCorpusFile(size, dim)
				if err != nil {
					b.Fatal(err)
				}

				// Force GC and measure baseline
				runtime.GC()
				time.Sleep(100 * time.Millisecond)
				rssBefore := getRSSBytes()

				// Open the file
				mr, err := reader.OpenMMap(path)
				if err != nil {
					b.Fatal(err)
				}
				defer mr.Close()

				block, err := mr.ReadVectorBlock(0)
				if err != nil {
					b.Fatal(err)
				}

				rssAfter := getRSSBytes()
				mmapSize := mr.Len()
				heapAlloc, heapInUse, _ := getMemStats()

				b.ReportMetric(float64(rssAfter-rssBefore)/1024/1024, "RSS_MB")
				b.ReportMetric(float64(mmapSize)/1024/1024, "MMAP_MB")
				b.ReportMetric(float64(heapAlloc)/1024/1024, "HeapAlloc_MB")
				b.ReportMetric(float64(heapInUse)/1024/1024, "HeapInUse_MB")

				query := generateRealisticEmbedding(dim)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					results := block.TopK(query, 10, false)
					_ = results
				}
			})
		}
	}
}

// ============================================================================
// ACCESS PATTERN BENCHMARKS
// ============================================================================

func BenchmarkChunkAccess(b *testing.B) {
	size := 100_000
	dim := 256

	path, err := getOrCreateCorpusFile(size, dim)
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % size
		chunk := block.GetChunk(idx)
		_ = chunk
	}
}

func BenchmarkEmbeddingAccess(b *testing.B) {
	size := 100_000
	dim := 256

	path, err := getOrCreateCorpusFile(size, dim)
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % size
		emb := block.GetEmbeddingSafe(idx)
		_ = emb
	}
}