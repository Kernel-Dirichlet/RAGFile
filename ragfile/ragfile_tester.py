#!/usr/bin/env python3

import argparse
import numpy as np
import ollama
from ragfile_utils import RAGFileWriter


KEYWORDS = [
    "AI",
    "RAG",
    "Graph",
    "Cybersecurity",
    "Distributed Systems",
]


def generate_content(keyword: str) -> str:
    """
    Generate a short, relevant paragraph for a keyword using gemma3:1b.
    """
    response = ollama.chat(
        model="gemma3:1b",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write a concise technical paragraph explaining the concept of "
                    f"{keyword} for a software engineer."
                ),
            }
        ],
    )

    return response["message"]["content"].strip()


def generate_embedding(text: str, precision: int) -> np.ndarray:
    """
    Generate an embedding using all-minilm and return a NumPy array
    with the requested precision.
    """
    result = ollama.embeddings(
        model="all-minilm",
        prompt=text,
    )

    vector = result["embedding"]

    if precision == 16:
        return np.asarray(vector, dtype=np.float16)
    elif precision == 32:
        return np.asarray(vector, dtype=np.float32)
    else:
        raise ValueError("Unsupported precision")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end RAGFileWriter test using Ollama"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_output.ragfile",
        help="Output .ragfile path",
    )
    parser.add_argument(
        "--padding",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Padding bytes for aligned reads",
    )
    parser.add_argument(
        "--precision",
        type=int,
        choices=[16, 32],
        default=32,
        help="Embedding float precision",
    )

    args = parser.parse_args()

    print("Generating keyword → content pairs using gemma3:1b...\n")

    keyword_content_pairs = []
    for kw in KEYWORDS:
        content = generate_content(kw)
        keyword_content_pairs.append(
            {"keyword": kw, "content": content}
        )
        print(f"[{kw}] {content[:80]}...")

    print("\nGenerating embeddings using all-minilm...\n")

    embedding_content_pairs = []
    for pair in keyword_content_pairs:
        emb = generate_embedding(pair["content"], args.precision)
        embedding_content_pairs.append(
            {
                "embedding": emb,
                "content": pair["content"],
            }
        )

    print("Serializing RAGFile...\n")

    writer = RAGFileWriter(args.output)

    # Write header (magic + version + endian)
    writer.write_header(major=0, minor=1, patch=0)

    # Write sections (RAGFile schema handles offsets internally)
    writer.write_keyword_section(
        keyword_content_pairs,
        padding=args.padding,
    )

    writer.write_embedding_section(
        embedding_content_pairs,
        padding=args.padding,
        precision=args.precision,
    )

    # Finalize + write file
    writer.finalize()

    print(f"RAGFile written to: {args.output}\n")
    print("Hexdump preview:\n")
    writer.hexdump()


if __name__ == "__main__":
    main()
