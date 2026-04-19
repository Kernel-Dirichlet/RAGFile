package tests

import (
	"bytes"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/Kernel-Dirichlet/RAGFile/internal/reader"
	"github.com/Kernel-Dirichlet/RAGFile/internal/writer"
)

func benchmarkScenario(b *testing.B, pairs int, embSize int) {

	data := buildPairs(pairs, embSize)

	// -------- WRITE --------
	var buf bytes.Buffer

	start := time.Now()
	_, err := writer.WriteVectorBlockInternal(&buf, data, 0, 0, 0)
	if err != nil {
		b.Fatal(err)
	}
	writeTime := time.Since(start)

	sizeBytes := buf.Len()
	sizeMB := float64(sizeBytes) / (1024 * 1024)

	fmt.Printf("\n--- WRITE ---\n")
	fmt.Printf("Pairs: %d | Emb: %d\n", pairs, embSize)
	fmt.Printf("Size: %d bytes (%.2f MB)\n", sizeBytes, sizeMB)
	fmt.Printf("Write time: %v\n", writeTime)
	fmt.Printf("Write throughput: %.2f MB/s\n", sizeMB/writeTime.Seconds())

	// -------- WRITE FILE --------
	tmp, _ := os.CreateTemp("", "ragbench")
	defer os.Remove(tmp.Name())
	tmp.Write(buf.Bytes())
	tmp.Close()

	// -------- READ --------
	mr, err := reader.OpenMMap(tmp.Name())
	if err != nil {
		b.Fatal(err)
	}
	defer mr.Close()

	start = time.Now()
	block, err := mr.ReadVectorBlock(0)
	if err != nil {
		b.Fatal(err)
	}
	readTime := time.Since(start)

	fmt.Printf("\n--- READ ---\n")
	fmt.Printf("Read time: %v\n", readTime)
	fmt.Printf("Read throughput: %.2f MB/s\n", sizeMB/readTime.Seconds())

	// -------- SEARCH --------
	query := randFloat32Slice(embSize)

	start = time.Now()
	block.TopK(query, 10, false)
	searchTime := time.Since(start)

	fmt.Printf("\n--- SEARCH ---\n")
	fmt.Printf("Search time: %v\n", searchTime)
}

//
// =========================
// BENCHMARKS
// =========================
//

func BenchmarkSmall(b *testing.B) {
	for i := 0; i < b.N; i++ {
		benchmarkScenario(b, 100, 16)
	}
}

func BenchmarkMedium(b *testing.B) {
	for i := 0; i < b.N; i++ {
		benchmarkScenario(b, 1000, 32)
	}
}

func BenchmarkLarge(b *testing.B) {
	for i := 0; i < b.N; i++ {
		benchmarkScenario(b, 10000, 64)
	}
}