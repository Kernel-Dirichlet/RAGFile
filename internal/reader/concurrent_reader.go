package reader

import (
	"container/heap"
	"sync"
)

// ConcurrentSearchResult mirrors the unexported result type for external use.
type ConcurrentSearchResult struct {
	Index int
	Score float32
}

// ConcurrentTopK partitions the vector block across numWorkers goroutines,
// runs a min-heap TopK search on each partition, then merges the partial
// results into a final TopK ranking.
//
// This is a NEW function that does NOT modify any existing code.
func (vb *MMapVectorBlock) ConcurrentTopK(
	query []float32,
	k int,
	numWorkers int,
	unsafeMode bool,
) []ConcurrentSearchResult {

	n := len(vb.index)
	if n == 0 {
		return nil
	}
	if numWorkers < 1 {
		numWorkers = 1
	}
	if numWorkers > n {
		numWorkers = n
	}

	type partialResult struct {
		index int
		score float32
	}

	// Channel to collect per-worker top-k results
	partialCh := make(chan []partialResult, numWorkers)

	chunkSize := (n + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		lo := w * chunkSize
		hi := lo + chunkSize
		if hi > n {
			hi = n
		}

		go func(lo, hi int) {
			defer wg.Done()

			h := &minHeap{}
			heap.Init(h)

			for i := lo; i < hi; i++ {
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

			out := make([]partialResult, h.Len())
			for i, r := range *h {
				out[i] = partialResult{index: r.Index, score: r.Score}
			}
			partialCh <- out
		}(lo, hi)
	}

	// Close channel after all workers finish
	go func() {
		wg.Wait()
		close(partialCh)
	}()

	// Merge partial results into a final min-heap of size k
	merged := &minHeap{}
	heap.Init(merged)

	for partial := range partialCh {
		for _, r := range partial {
			if merged.Len() < k {
				heap.Push(merged, result{r.index, r.score})
			} else if r.score > (*merged)[0].Score {
				(*merged)[0] = result{r.index, r.score}
				heap.Fix(merged, 0)
			}
		}
	}

	out := make([]ConcurrentSearchResult, merged.Len())
	for i, r := range *merged {
		out[i] = ConcurrentSearchResult{Index: r.Index, Score: r.Score}
	}
	return out
}
