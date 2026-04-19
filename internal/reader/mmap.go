package reader

import (
	"fmt"
	"os"
	"syscall"
)

// MMapReader provides zero-copy access to a file
//
// ⚠️ IMPORTANT:
// - Backed by virtual memory (not a copy)
// - Data becomes invalid after Close()
// - Not safe for use after unmapping
type MMapReader struct {
	file *os.File
	data []byte
	size int
}

// OpenMMap maps a file into memory (read-only)
func OpenMMap(path string) (*MMapReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	stat, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}

	size := stat.Size()
	if size == 0 {
		f.Close()
		return nil, fmt.Errorf("cannot mmap empty file")
	}

	data, err := syscall.Mmap(
		int(f.Fd()),
		0,
		int(size),
		syscall.PROT_READ,
		syscall.MAP_SHARED,
	)
	if err != nil {
		f.Close()
		return nil, err
	}

	return &MMapReader{
		file: f,
		data: data,
		size: int(size),
	}, nil
}

// Bytes returns underlying mmap buffer (read-only)
func (m *MMapReader) Bytes() []byte {
	return m.data
}

// Len returns file size
func (m *MMapReader) Len() int {
	return m.size
}

// Close unmaps memory
//
// ⚠️ After this, all slices referencing m.data are INVALID
func (m *MMapReader) Close() error {
	if m.data != nil {
		if err := syscall.Munmap(m.data); err != nil {
			return err
		}
		m.data = nil // prevent accidental use-after-free
	}
	return m.file.Close()
}