// Package format contains the binary layout definitions for RAGFile
package format

import (
	"bytes"
	"strconv"
	"time"
)

// RAGFILE Header
type Header struct { 
	Version uint32
	Timestamp int64 // Unix timestamp (portable across architectures)
	NumSections int
	Author string

}

func NewHeader(version int,
	timestamp time.Time,
	numSections int,
	author string) *Header {
		return &Header{
			Version:     uint32(version),
			Timestamp:   time.Now().Unix(),
			NumSections: numSections,
			Author:      author,
		}
	}

// Serialize method 

func (h *Header) Serialize() ([]byte, error) {
	buf := new(bytes.Buffer)
    
	// Magic + version 
	buf.WriteString("RAGFILE-")
	buf.WriteString(strconv.FormatUint(uint64(h.Version), 10))
	buf.WriteByte('\n')

	// timestamp
	buf.WriteString("timestamp:")
	buf.WriteString(strconv.FormatInt(h.Timestamp, 10))
	buf.WriteByte('\n')
	
	// number of sections 
	buf.WriteString("numSections:")
	buf.WriteString(strconv.FormatUint(uint64(h.NumSections), 10))
	buf.WriteByte('\n')
	
	buf.WriteString("author:")
	buf.WriteString(h.Author)
	buf.WriteByte('\n')

	buf.WriteByte(0) // Null terminator for HEADER 
	return buf.Bytes(), nil
}

