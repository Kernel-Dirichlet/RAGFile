package reader

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/Kernel-Dirichlet/RAGFile/internal/format"
)

const (
	SectionHeaderSize = 17
	VectorMetaSize    = 20
	VectorEntrySize   = 32
)

type MMapVectorBlock struct {
	meta  format.VectorBlock
	index []format.VectorEntry
	data  []byte
}

// ReadVectorBlock with strict bounds checking
func (m *MMapReader) ReadVectorBlock(offset int64) (*MMapVectorBlock, error) {

	buf := m.data

	if int(offset)+SectionHeaderSize > len(buf) {
		return nil, fmt.Errorf("section header out of bounds")
	}

	// ---- Section Header ----
	header := format.SectionHeader{
		Type:        format.SectionType(buf[offset]),
		StartOffset: binary.LittleEndian.Uint64(buf[offset+1 : offset+9]),
		EndOffset:   binary.LittleEndian.Uint64(buf[offset+9 : offset+17]),
	}

	if header.Type != format.SectionVector {
		return nil, fmt.Errorf("not vector section")
	}

	if int(header.EndOffset) > len(buf) {
		return nil, fmt.Errorf("section exceeds file bounds")
	}

	// ---- Metadata ----
	metaOffset := offset + SectionHeaderSize

	if int(metaOffset)+VectorMetaSize > len(buf) {
		return nil, fmt.Errorf("metadata out of bounds")
	}

	meta := format.VectorBlock{
		NumPairs:    binary.LittleEndian.Uint32(buf[metaOffset : metaOffset+4]),
		IndexOffset: binary.LittleEndian.Uint64(buf[metaOffset+4 : metaOffset+12]),
		DataOffset:  binary.LittleEndian.Uint64(buf[metaOffset+12 : metaOffset+20]),
	}

	// ---- Index ----
	indexOffset := metaOffset + VectorMetaSize
	indexSize := int(meta.NumPairs) * VectorEntrySize

	if int(indexOffset)+indexSize > len(buf) {
		return nil, fmt.Errorf("index out of bounds")
	}

	index := make([]format.VectorEntry, meta.NumPairs)

		for i := 0; i < int(meta.NumPairs); i++ {
		base := indexOffset + int64(i*VectorEntrySize)

		index[i] = format.VectorEntry{
			ChunkStart:     binary.LittleEndian.Uint64(buf[base : base+8]),
			ChunkEnd:       binary.LittleEndian.Uint64(buf[base+8 : base+16]),
			EmbeddingStart: binary.LittleEndian.Uint64(buf[base+16 : base+24]),
			EmbeddingEnd:   binary.LittleEndian.Uint64(buf[base+24 : base+32]),
		}
	}

	// ---- Data ----
	dataStart := indexOffset + int64(indexSize)
	data := buf[dataStart:header.EndOffset]

	return &MMapVectorBlock{
		meta:  meta,
		index: index,
		data:  data,
	}, nil
}

//
// SAFE ACCESS
//

func (vb *MMapVectorBlock) GetEmbeddingSafe(i int) []float32 {
	entry := vb.index[i]
	raw := vb.data[entry.EmbeddingStart:entry.EmbeddingEnd]

	n := len(raw) / 4
	out := make([]float32, n)

	for j := 0; j < n; j++ {
		bits := binary.LittleEndian.Uint32(raw[j*4 : (j+1)*4])
		out[j] = math.Float32frombits(bits)
	}

	return out
}

func (vb *MMapVectorBlock) GetChunk(i int) []byte {
	entry := vb.index[i]
	return vb.data[entry.ChunkStart:entry.ChunkEnd]
}

//
// UNSAFE ACCESS
//

func (vb *MMapVectorBlock) GetEmbeddingUnsafe(i int) []float32 {
	entry := vb.index[i]
	raw := vb.data[entry.EmbeddingStart:entry.EmbeddingEnd]

	if len(raw) == 0 {
		return nil
	}

	if len(raw)%4 != 0 {
		panic("invalid embedding size")
	}

	// Alignment check (CRITICAL for unsafe)
	ptr := unsafe.Pointer(&raw[0])
	if uintptr(ptr)%4 != 0 {
		panic("unaligned memory access")
	}

	return unsafe.Slice((*float32)(ptr), len(raw)/4)
}

// Meta returns the vector block metadata.
func (vb *MMapVectorBlock) Meta() format.VectorBlock {
	return vb.meta
}
