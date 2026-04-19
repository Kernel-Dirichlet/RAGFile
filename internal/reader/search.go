package reader

import "container/heap"

// dot product (hot loop)
func dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

type result struct {
	Index int
	Score float32
}

type minHeap []result

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].Score < h[j].Score }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }

func (h *minHeap) Push(x interface{}) {
	*h = append(*h, x.(result))
}

func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// TopK search
func (vb *MMapVectorBlock) TopK(
	query []float32,
	k int,
	unsafeMode bool,
) []result {

	h := &minHeap{}
	heap.Init(h)

	for i := 0; i < len(vb.index); i++ {

		var emb []float32
		if unsafeMode {
			emb = vb.GetEmbeddingUnsafe(i)
		} else {
			emb = vb.GetEmbeddingSafe(i)
		}

		score := dot(query, emb)

		if h.Len() < k {
			heap.Push(h, result{i, score})
		} else if score > (*h)[0].Score {
			(*h)[0] = result{i, score}
			heap.Fix(h, 0)
		}
	}

	out := make([]result, h.Len())
	copy(out, *h)
	return out
}