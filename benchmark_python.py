import sqlite3
import time
import random
import numpy as np
import os
import tempfile

def create_vector_pairs(pairs, emb_size):
    """Create random vector pairs for benchmarking"""
    vectors = []
    for _ in range(pairs):
        vector = [random.uniform(-1, 1) for _ in range(emb_size)]
        vectors.append(vector)
    return vectors

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def benchmark_scenario(pairs, emb_size):
    """Run benchmark scenario for given parameters"""
    data = create_vector_pairs(pairs, emb_size)

    # Create temporary SQLite database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create table for vectors
    cursor.execute('''
        CREATE TABLE vectors (
            id INTEGER PRIMARY KEY,
            vector BLOB
        )
    ''')

    # Create index for faster searching
    cursor.execute('CREATE INDEX idx_vector ON vectors (id)')

    # -------- WRITE --------
    start = time.time()
    for i, vector in enumerate(data):
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        cursor.execute('INSERT INTO vectors (id, vector) VALUES (?, ?)', (i, vector_bytes))
    conn.commit()
    write_time = time.time() - start

    size_bytes = sum(len(row[0]) for row in cursor.execute('SELECT vector FROM vectors'))
    size_mb = size_bytes / (1024 * 1024)

    print(f"\n--- WRITE ---")
    print(f"Pairs: {pairs} | Emb: {emb_size}")
    print(f"Size: {size_bytes} bytes ({size_mb:.2f} MB)")
    print(f"Write time: {write_time:.4f}s")
    print(f"Write throughput: {size_mb/write_time:.2f} MB/s")

    # -------- SEARCH --------
    query = [random.uniform(-1, 1) for _ in range(emb_size)]
    query_bytes = np.array(query, dtype=np.float32).tobytes()

    start = time.time()
    cursor.execute('SELECT id, vector FROM vectors')
    results = cursor.fetchall()

    # Calculate similarities
    similarities = []
    for row in results:
        vector_bytes = row[1]
        vector = np.frombuffer(vector_bytes, dtype=np.float32)
        similarity = cosine_similarity(query, vector)
        similarities.append((row[0], similarity))

    # Get top 10
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_10 = similarities[:10]
    search_time = time.time() - start

    print(f"\n--- SEARCH ---")
    print(f"Search time: {search_time:.4f}s")

    conn.close()
    return write_time, search_time

def benchmark_small():
    """Small benchmark: 100 pairs, 16 embedding size"""
    print("\n=== SMALL BENCHMARK ===")
    return benchmark_scenario(100, 16)

def benchmark_medium():
    """Medium benchmark: 1000 pairs, 32 embedding size"""
    print("\n=== MEDIUM BENCHMARK ===")
    return benchmark_scenario(1000, 32)

def benchmark_large():
    """Large benchmark: 10000 pairs, 64 embedding size"""
    print("\n=== LARGE BENCHMARK ===")
    return benchmark_scenario(10000, 64)

if __name__ == "__main__":
    benchmark_small()
    benchmark_medium()
    benchmark_large()