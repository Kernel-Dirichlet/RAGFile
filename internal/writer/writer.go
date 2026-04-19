// Package writer handles serialization of RAGFile format
package writer

import (
	"encoding/binary"
	"io"
	"math"

	"github.com/Kernel-Dirichlet/RAGFile/internal/format"
)

// ================
// CONSTANTS
// ================

const (
	SectionHeaderSize = 17 // type (1 byte) + start offset (8 bytes) + end offset (8 bytes)
	VectorMetaSize    = 20 // numPairs (4 bytes) + indexOffset (8 bytes) + dataOffset (8 bytes)
	VectorEntrySize   = 32 // 4 * uint64 (chunkStart, chunkEnd, embeddingStart, embeddingEnd)
)

// ================
// DATA STRUCTURES
// ================

type SemanticPair struct {
	Chunk     []byte
	Embedding []float32
}

// ===========================
// CORE WRITER IMPLEMENTATION
// ===========================

func WriteVectorBlockInternal(
	w io.Writer,
	semanticPairs []SemanticPair,
	startOffset uint64,
	dataCap, indexCap int,
) (endOffset uint64, err error) {

	/* =====================
	 CAPACITY PLANNING
	
	** Precomputing avoids repeated reallocations during append,
	 which would otherwise cause O(n) copying overhead **
    */

	if dataCap <= 0 {
		total := 0
		for _, pair := range semanticPairs {
			total += len(pair.Chunk) + len(pair.Embedding)*4
		}
		dataCap = total
	}

	if indexCap <= 0 {
		indexCap = len(semanticPairs) * VectorEntrySize
	}

	// Preallocate buffers → avoids growth + copying
	dataBuf := make([]byte, 0, dataCap)
	indexBuf := make([]byte, 0, indexCap)

	// =====================
	// BUILD DATA + INDEX
	// =====================

	for _, pair := range semanticPairs {

		// ---- Chunk ----
		chunkStart := uint64(len(dataBuf))
		dataBuf = append(dataBuf, pair.Chunk...)
		chunkEnd := uint64(len(dataBuf))

		// ---- Embedding ----
		embeddingStart := uint64(len(dataBuf))

		// HOT LOOP:
		// We avoid:
		//   - binary.Write (uses reflection)
		//   - temporary byte slices
		//
		// Instead we manually encode float32 → bytes using bit shifts.
		for _, f := range pair.Embedding {
			bits := math.Float32bits(f)

			// Manual encoding (NO reflection, NO allocation)
			dataBuf = append(dataBuf,
				byte(bits),
				byte(bits>>8),
				byte(bits>>16),
				byte(bits>>24),
			)
		}

		embeddingEnd := uint64(len(dataBuf))

		// ---- Index Entry ----
		// AppendUint64 is efficient:
		// - no reflection
		// - no heap allocation
		indexBuf = binary.LittleEndian.AppendUint64(indexBuf, chunkStart)
		indexBuf = binary.LittleEndian.AppendUint64(indexBuf, chunkEnd)
		indexBuf = binary.LittleEndian.AppendUint64(indexBuf, embeddingStart)
		indexBuf = binary.LittleEndian.AppendUint64(indexBuf, embeddingEnd)
	}

	// =====================
	// METADATA
	// =====================

	meta := format.VectorBlock{
		NumPairs:    uint32(len(semanticPairs)),
		IndexOffset: 0,
		DataOffset:  uint64(len(indexBuf)),
	}

	// =====================
	// SECTION HEADER
	// =====================

	totalSize := uint64(SectionHeaderSize + VectorMetaSize +
		len(indexBuf) + len(dataBuf))

	sectionHeader := format.SectionHeader{
		Type:        format.SectionVector,
		StartOffset: startOffset,
		EndOffset:   startOffset + totalSize,
	}

	// =====================
	// WRITE HEADER (NO REFLECTION)
	// =====================
	// WHY NOT binary.Write(struct)?
	//
	// binary.Write(struct):
	//   - uses reflection to inspect fields at runtime
	//   - incurs type checks + interface overhead
	//   - may introduce padding issues
	//
	// Manual encoding:
	//   - zero reflection
	//   - predictable layout
	//   - faster in tight loops / high throughput systems

	var headerBuf [SectionHeaderSize]byte

	headerBuf[0] = byte(sectionHeader.Type)
	binary.LittleEndian.PutUint64(headerBuf[1:9], sectionHeader.StartOffset)
	binary.LittleEndian.PutUint64(headerBuf[9:17], sectionHeader.EndOffset)

	if _, err := w.Write(headerBuf[:]); err != nil {
		return 0, err
	}

	// =====================
	// WRITE METADATA (NO REFLECTION)
	// =====================

	var metaBuf [VectorMetaSize]byte

	binary.LittleEndian.PutUint32(metaBuf[0:4], meta.NumPairs)
	binary.LittleEndian.PutUint64(metaBuf[4:12], meta.IndexOffset)
	binary.LittleEndian.PutUint64(metaBuf[12:20], meta.DataOffset)

	if _, err := w.Write(metaBuf[:]); err != nil {
		return 0, err
	}

	// =====================
	// WRITE INDEX + DATA
	// =====================

	if _, err := w.Write(indexBuf); err != nil {
		return 0, err
	}

	if _, err := w.Write(dataBuf); err != nil {
		return 0, err
	}

	return sectionHeader.EndOffset, nil
}