package writer

import (
	"bytes"
	"io"
	"sync"
)

// ConcurrentWriteResult holds the serialized bytes and any error from
// a single shard written by a goroutine.
type ConcurrentWriteResult struct {
	Data []byte
	Err  error
}

// WriteConcurrent partitions semanticPairs into numWorkers shards,
// serialises each shard in a separate goroutine using the existing
// WriteVectorBlockInternal, then writes all shards sequentially to w.
//
// Each shard becomes an independent VectorBlock in the output stream,
// just as if you had called WriteVectorBlockInternal multiple times.
// This is a NEW function and does NOT modify any existing code.
func WriteConcurrent(
	w io.Writer,
	pairs []SemanticPair,
	numWorkers int,
) error {

	n := len(pairs)
	if n == 0 {
		return nil
	}
	if numWorkers < 1 {
		numWorkers = 1
	}
	if numWorkers > n {
		numWorkers = n
	}

	chunkSize := (n + numWorkers - 1) / numWorkers
	results := make([]ConcurrentWriteResult, numWorkers)

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		lo := w * chunkSize
		hi := lo + chunkSize
		if hi > n {
			hi = n
		}

		go func(idx, lo, hi int) {
			defer wg.Done()

			shard := pairs[lo:hi]
			var buf bytes.Buffer
			_, err := WriteVectorBlockInternal(&buf, shard, 0, 0, 0)
			results[idx] = ConcurrentWriteResult{
				Data: buf.Bytes(),
				Err:  err,
			}
		}(w, lo, hi)
	}

	wg.Wait()

	// Write shards in order; fail-fast on any error.
	for _, r := range results {
		if r.Err != nil {
			return r.Err
		}
		if _, err := w.Write(r.Data); err != nil {
			return err
		}
	}

	return nil
}
