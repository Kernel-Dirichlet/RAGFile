import struct
import sys
from typing import List, Dict
import numpy as np
import binascii 
import mmap 

class RAGFileWriter:
    """
    RAGFileWriter (schema v0.1.0)

    Binary layout (in order):

    1) Magic header            : b"RAGFILE" (7 bytes)
    2) Version                 : uint8 major, uint8 minor, uint8 patch
    3) Endianness              : uint8 (0 = little, 1 = big)
    4) Index strategy section   : {section}-({start,end})\x00 repeated
       - Records absolute start/end offsets for each section
       - Allows readers to seek directly to sections without parsing the whole file
    5) Keyword-content section :
         - start-byte (uint64)
         - end-byte   (uint64)
         - padding    (uint8)
         - raw data (each record: {keyword}-{content}, padded)
    6) Embedding-content section :
         - precision  (uint8)
         - start-byte (uint64)
         - end-byte   (uint64)
         - padding    (uint8)
         - raw data (each record: {embedding}-{content}, padded)

    Notes:
    - Sections can be optional beyond header (1-3)
    - Padding ensures alignment for memory-mapped or chunked reads
    - Index strategy section is authoritative for locating sections
    """

    MAGIC = b"RAGFILE"

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.buf = bytearray()
        self.index_entries: List = []

        # Determine endianness
        self.endian_flag = 0 if sys.byteorder == "little" else 1
        self.struct_prefix = "<" if self.endian_flag == 0 else ">"

 
    def _u8(self, v): return struct.pack(self.struct_prefix + "B", v)
    
    def _u64(self, v): return struct.pack(self.struct_prefix + "Q", v)

    @staticmethod
    def _pad(data: bytes, alignment: int) -> bytes:
        pad_len = (alignment - (len(data) % alignment)) % alignment
        return data + (b"\x00" * pad_len)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def write_header(self, major=0, minor=1, patch=0):
        self.buf += self.MAGIC
        self.buf += struct.pack("BBB", major, minor, patch)
        self.buf += struct.pack("B", self.endian_flag)

    # ------------------------------------------------------------------
    # Index strategy section
    # ------------------------------------------------------------------
    def write_index_strategy(self):
        """
        Writes the index strategy section immediately after the header.
        Each entry: {section}-({start},{end})\x00
        This allows readers to seek directly to keyword/vector sections.
        """
        table = bytearray()
        for name, start, end in self.index_entries:
            entry = f"{name}-({start},{end})".encode("utf-8")
            table += entry + b"\x00"

        header_end = len(self.MAGIC) + 3 + 1  # magic + version + endian
        # Extend buffer if needed
        if len(self.buf) < header_end + len(table):
            self.buf += b"\x00" * (header_end + len(table) - len(self.buf))

        self.buf[header_end:header_end + len(table)] = table

    # ------------------------------------------------------------------
    # Keyword-content section
    # ------------------------------------------------------------------
    def write_keyword_section(self, pairs: List[Dict[str, str]], padding: int):
        
        assert padding in (4, 8, 16)

        start = len(self.buf)

        # section metadata
        self.buf += self._u64(0)  # placeholder start
        self.buf += self._u64(0)  # placeholder end
        self.buf += self._u8(padding)

        data_start = len(self.buf)

        # write each keyword-content pair, padded
        for pair in pairs:
            record = f"{pair['keyword']}-{pair['content']}".encode("utf-8")
            self.buf += self._pad(record, padding)

        data_end = len(self.buf)

        # patch start/end bytes
        self.buf[start:start + 8] = self._u64(data_start)
        self.buf[start + 8:start + 16] = self._u64(data_end)

        # register in index entries
        self.index_entries.append(("keyword", data_start, data_end))

    # ------------------------------------------------------------------
    # Embedding-content section
    # ------------------------------------------------------------------
    def write_embedding_section(self, pairs: List[Dict], padding: int, precision: int = 32):
        assert padding in (4, 8, 16)
        assert precision in (16, 32)

        start = len(self.buf)

        # section metadata
        self.buf += self._u8(precision)
        self.buf += self._u64(0)  # placeholder start
        self.buf += self._u64(0)  # placeholder end
        self.buf += self._u8(padding)

        data_start = len(self.buf)

        # write each embedding-content pair, padded
        for pair in pairs:
            emb: np.ndarray = pair["embedding"]
            content: str = pair["content"]
            emb_bytes = emb.astype(np.float16 if precision == 16 else np.float32).tobytes()
            record = emb_bytes + b"-" + content.encode("utf-8")
            self.buf += self._pad(record, padding)

        data_end = len(self.buf)

        # patch start/end bytes
        self.buf[start + 1:start + 9] = self._u64(data_start)
        self.buf[start + 9:start + 17] = self._u64(data_end)

        # register in index entries
        self.index_entries.append(("vector", data_start, data_end))

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def finalize(self):
        # write the index strategy section after all offsets are known
        self.write_index_strategy()

        # write buffer to file
        with open(self.filepath, "wb") as f:
            f.write(self.buf)

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------
    def hexdump(self, length=256):
        print(self.buf[:length].hex())


# -------------------------
# Reader
# -------------------------

class RAGFileReader:
    """
    RAGFileReader (schema v0.1.0)

    Supports:
      - Padding-aware keyword-content parsing
      - Section-aware reading using the absolute offsets in the file
      - Searching keywords without loading unrelated sections

    Schema (strict v0.1.0):

    1) Magic header            : b"RAGFILE" (7 bytes)
    2) Version                 : uint8 major, uint8 minor, uint8 patch
    3) Endianness              : uint8 (0 = little, 1 = big)
    4) Index strategy section  : offsets of other sections (not variable-length null-terminated)
    5) Keyword-content section :
         - start-byte (uint64)
         - end-byte   (uint64)
         - padding    (uint8)
         - raw data (records: {keyword}-{content}, padded to alignment)
    6) Embedding-content section :
         - precision  (uint8)
         - start-byte (uint64)
         - end-byte   (uint64)
         - padding    (uint8)
         - raw data
    """

    MAGIC = b"RAGFILE"

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._index = {}
        self._header_end = 11  # 7 (magic) + 3 (version) + 1 (endianness)

    # -------------------------
    # Parse index section immediately after header
    # -------------------------
    def parse_index(self):
        """
        Reads the index strategy section into self._index
        Format: {strategy}-({start},{end})\x00 repeated
        """
        self._index = {}
        with open(self.filepath, "rb") as f:
            f.seek(self._header_end)
            buf = bytearray()
            while True:
                b = f.read(1)
                if not b or b == b'\x00':
                    break
                buf += b
            # Split entries by null byte
            entries = buf.split(b'\x00')
            for entry in entries:
                if not entry:
                    continue
                # v0.1.0 format: strategy-(start,end)
                try:
                    s = entry.decode("utf-8")
                    name, rest = s.split("-(")
                    start_str, end_str = rest[:-1].split(",")
                    self._index[name] = (int(start_str), int(end_str))
                except Exception:
                    continue

    # -------------------------
    # Padding-aware keyword search using mmap
    # -------------------------
    def search_keyword(self, keyword: str, padding: int = 8) -> List[Dict[str, str]]:
        """
        Fast memory-mapped keyword search.
        - Uses section offsets
        - Reads records in padding-aware strides
        """
        if not self._index:
            self.parse_index()

        if "keyword" not in self._index:
            return []

        start, end = self._index["keyword"]
        results = []

        with open(self.filepath, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos = start
            while pos < end:
                # read one padded block
                block = mm[pos: pos + padding * 256]  # read in reasonable chunks
                if not block:
                    break

                # Split records by null padding
                records = block.split(b'\x00')
                for rec in records:
                    if b"-" not in rec:
                        continue
                    k, v = rec.split(b"-", 1)
                    if k.decode("utf-8") == keyword:
                        results.append({"keyword": k.decode("utf-8"), "content": v.decode("utf-8")})
                        # break here if we only want first match

                pos += len(block)

        return results
        
    # Debug
    # -------------------------
    def hexdump(self, length=256):
        """
        Print a hex dump of the first N bytes.
        """
        with open(self.filepath, "rb") as f:
            raw = f.read(length)
        print(raw.hex())