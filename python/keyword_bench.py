#!/usr/bin/env python3

import argparse
import random
import string
import time
from ragfile_utils import RAGFileWriter, RAGFileReader


def random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def main():
    parser = argparse.ArgumentParser(description="Keyword search benchmark")
    parser.add_argument("--num_keywords", type=int, default=10000)
    parser.add_argument("--str_length", type=int, default=8)
    parser.add_argument("--content_length", type=int, default=64)
    parser.add_argument("--padding", type=int, default=8, choices=[4, 8, 16])
    parser.add_argument("--output", type=str, default="benchmark.ragfile")
    args = parser.parse_args()

    # -------------------------
    # Generate keyword-content pairs
    # -------------------------
    print(f"Generating {args.num_keywords} keyword-content pairs...")
    keyword_content_pairs = []
    for _ in range(args.num_keywords):
        kw = random_string(args.str_length)
        content = random_string(args.content_length)
        keyword_content_pairs.append({"keyword": kw, "content": content})

    # pick a random keyword to search
    search_kw = random.choice(keyword_content_pairs)["keyword"]
    print(f"Searching for keyword: {search_kw}")

    # -------------------------
    # Python in-memory search
    # -------------------------
    start_py = time.time()
    py_found = [p for p in keyword_content_pairs if p["keyword"] == search_kw]
    end_py = time.time()
    print(f"Python search found {len(py_found)} entries in {end_py - start_py:.6f} seconds")

    # -------------------------
    # Write RAGFile
    # -------------------------
    print("Writing RAGFile...")
    writer = RAGFileWriter(args.output)
    writer.write_header()
    writer.write_keyword_section(keyword_content_pairs, padding=args.padding)
    writer.finalize()
    print("RAGFile written.\n")

    # -------------------------
    # RAGFile search
    # -------------------------
    reader = RAGFileReader(args.output)
    start_rag = time.time()
    rag_found = reader.search_keyword(search_kw, padding=args.padding)
    end_rag = time.time()
    print(f"RAGFile search found {len(rag_found)} entries in {end_rag - start_rag:.6f} seconds")

    if py_found and rag_found:
        assert py_found[0]["content"] == rag_found[0]["content"], "Mismatch in search results!"

    speedup = (end_py - start_py) / (end_rag - start_rag) if (end_rag - start_rag) > 0 else float("inf")
    print(f"Speedup: Python / RAGFile = {speedup:.2f}x")


if __name__ == "__main__":
    main()
