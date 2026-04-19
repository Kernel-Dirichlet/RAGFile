#!/bin/bash
set -euo pipefail

# RAGFile Comprehensive Benchmark Suite
# ======================================
# Runs benchmarks across various corpus sizes and dimensions.
# Results are saved to benchmark_results.txt and benchmark_results.json

OUT_DIR="./benchmark_output"
mkdir -p "$OUT_DIR"

echo "=== RAGFile Benchmark Suite ==="
echo "Results will be saved to $OUT_DIR/"
echo ""

# Run all tests (including benchmarks) with memory stats
echo "Running benchmarks..."
go test -bench=. -benchmem -benchtime=3s ./internal/tests/ 2>&1 | tee "$OUT_DIR/benchmark_results.txt"

echo ""
echo "=== Benchmark results saved to $OUT_DIR/benchmark_results.txt ==="

# Run a quick subset for fast feedback
echo ""
echo "=== Quick benchmark subset (for fast iteration) ==="
go test -bench="BenchmarkSmall|BenchmarkMedium" -benchmem -benchtime=1s ./internal/tests/ 2>&1 | tee "$OUT_DIR/benchmark_quick.txt"

echo ""
echo "=== Quick results saved to $OUT_DIR/benchmark_quick.txt ==="

# Memory-focused benchmarks
echo ""
echo "=== Memory benchmarks ==="
go test -bench=BenchmarkMemory -benchmem -benchtime=1s ./internal/tests/ 2>&1 | tee "$OUT_DIR/benchmark_memory.txt"

echo ""
echo "=== Memory results saved to $OUT_DIR/benchmark_memory.txt ==="

# Scalability benchmarks
echo ""
echo "=== Scalability benchmarks ==="
go test -bench=BenchmarkScalability -benchmem -benchtime=1s ./internal/tests/ 2>&1 | tee "$OUT_DIR/benchmark_scalability.txt"

echo ""
echo "=== Scalability results saved to $OUT_DIR/benchmark_scalability.txt ==="

echo ""
echo "All benchmarks complete!"
echo "Files generated:"
ls -lh "$OUT_DIR/"