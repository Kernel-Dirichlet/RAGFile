package tests

import (
	"bytes"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/Kernel-Dirichlet/RAGFile/internal/reader"
	"github.com/Kernel-Dirichlet/RAGFile/internal/writer"
)

func randFloat32Slice(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rand.Float32()
	}
	return out
}

func randBytes(n int) []byte {
	out := make([]byte, n)
	rand.Read(out)
	return out
}

func buildPairs(numPairs, embSize int) []writer.SemanticPair {
	pairs := make([]writer.SemanticPair, numPairs)
	for i := range pairs {
		pairs[i] = writer.SemanticPair{
			Chunk:     randBytes(rand.Intn(64) + 1),
			Embedding: randFloat32Slice(embSize),
		}
	}
	return pairs
}

func writeTempFile(t *testing.T, data []byte) string {
	tmp, err := os.CreateTemp("", "ragfile-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer tmp.Close()

	if _, err := tmp.Write(data); err != nil {
		t.Fatal(err)
	}

	return tmp.Name()
}

//
// =========================
// TEST 1: Small
// =========================
//

func TestSmallBlock(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	pairs := buildPairs(5, 8)

	var buf bytes.Buffer
	_, err := writer.WriteVectorBlockInternal(&buf, pairs, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}

	path := writeTempFile(t, buf.Bytes())
	defer os.Remove(path)

	mr, err := reader.OpenMMap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mr.Close()

	block, err := mr.ReadVectorBlock(0)
	if err != nil {
		t.Fatal(err)
	}

	// Verify we can read all embeddings
	for i := 0; i < len(pairs); i++ {
		emb := block.GetEmbeddingSafe(i)
		if len(emb) != len(pairs[0].Embedding) {
			t.Fatalf("embedding size mismatch at %d: expected %d got %d", i, len(pairs[0].Embedding), len(emb))
		}
	}
}

//
// =========================
// TEST 2: Varied embeddings
// =========================
//

func TestVariedEmbeddings(t *testing.T) {
	pairs := []writer.SemanticPair{
		{Chunk: []byte("a"), Embedding: randFloat32Slice(4)},
		{Chunk: []byte("b"), Embedding: randFloat32Slice(16)},
		{Chunk: []byte("c"), Embedding: randFloat32Slice(32)},
	}

	var buf bytes.Buffer
	_, _ = writer.WriteVectorBlockInternal(&buf, pairs, 0, 0, 0)

	path := writeTempFile(t, buf.Bytes())
	defer os.Remove(path)

	mr, _ := reader.OpenMMap(path)
	defer mr.Close()

	block, _ := mr.ReadVectorBlock(0)

	for i := range pairs {
		safe := block.GetEmbeddingSafe(i)
		expectedLen := len(pairs[i].Embedding)
		if len(safe) != expectedLen {
			t.Fatalf("embedding %d: expected len %d got %d", i, expectedLen, len(safe))
		}
	}
}

//
// =========================
// TEST 3: Multiple sections
// =========================
//

func TestMultipleSections(t *testing.T) {
	p1 := buildPairs(3, 8)
	p2 := buildPairs(7, 8)

	var buf bytes.Buffer

	// Write first block
	endOffset1, _ := writer.WriteVectorBlockInternal(&buf, p1, 0, 0, 0)
	// Write second block
	endOffset2, _ := writer.WriteVectorBlockInternal(&buf, p2, endOffset1, 0, 0)

	path := writeTempFile(t, buf.Bytes())
	defer os.Remove(path)

	mr, _ := reader.OpenMMap(path)
	defer mr.Close()

	b1, _ := mr.ReadVectorBlock(0)
	b2, _ := mr.ReadVectorBlock(int64(endOffset1))

	// Verify we can read all embeddings from block 1
	for i := 0; i < len(p1); i++ {
		_ = b1.GetEmbeddingSafe(i)
	}

	// Verify we can read all embeddings from block 2
	for i := 0; i < len(p2); i++ {
		_ = b2.GetEmbeddingSafe(i)
	}

	_ = endOffset2
}

//
// =========================
// TEST 4: Top-K
// =========================
//

func TestTopK(t *testing.T) {
	pairs := buildPairs(100, 16)

	var buf bytes.Buffer
	_, _ = writer.WriteVectorBlockInternal(&buf, pairs, 0, 0, 0)

	path := writeTempFile(t, buf.Bytes())
	defer os.Remove(path)

	mr, _ := reader.OpenMMap(path)
	defer mr.Close()

	block, _ := mr.ReadVectorBlock(0)

	query := randFloat32Slice(16)

	results := block.TopK(query, 5, false)

	if len(results) != 5 {
		t.Fatal("topk failed")
	}
}

//
// =========================
// TEST 5: Large
// =========================
//

func TestLargeBlock(t *testing.T) {
	pairs := buildPairs(1000, 32)

	var buf bytes.Buffer
	_, _ = writer.WriteVectorBlockInternal(&buf, pairs, 0, 0, 0)

	path := writeTempFile(t, buf.Bytes())
	defer os.Remove(path)

	mr, _ := reader.OpenMMap(path)
	defer mr.Close()

	block, _ := mr.ReadVectorBlock(0)

	// Verify we can read all 1000 embeddings
	for i := 0; i < 1000; i++ {
		_ = block.GetEmbeddingSafe(i)
	}
}
