#!/bin/bash
set -euo pipefail

# RAGFile Comprehensive Benchmark Suite
# ======================================
# Runs Go + Python benchmarks and writes ALL results to a single file.

OUT_DIR="./benchmark_output"
RESULTS="$OUT_DIR/all_results.txt"
mkdir -p "$OUT_DIR"

# Wipe previous results
> "$RESULTS"

echo "=== RAGFile Benchmark Suite ===" | tee -a "$RESULTS"
echo "Started: $(date)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# -----------------------------------------------------------------------
# GO – core benchmarks (Small / Medium / Large)
# -----------------------------------------------------------------------
echo "=== Go Benchmarks: Small / Medium / Large ===" | tee -a "$RESULTS"
go test -bench="BenchmarkSmall|BenchmarkMedium|BenchmarkLarge" \
        -benchmem -benchtime=3s ./internal/tests/ 2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# -----------------------------------------------------------------------
# GO – concurrent benchmarks (thread-count sweep)
# -----------------------------------------------------------------------
echo "=== Go Benchmarks: Concurrent Reader/Writer ===" | tee -a "$RESULTS"
go test -bench="BenchmarkConcurrent" \
        -benchmem -benchtime=3s ./internal/tests/ 2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# -----------------------------------------------------------------------
# PYTHON – SQLite in-memory benchmarks
# -----------------------------------------------------------------------
echo "=== Python SQLite Benchmarks ===" | tee -a "$RESULTS"
(cd "$(dirname "$0")" && source venv/bin/activate && python benchmark_python.py) \
    2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

echo "=== All benchmarks complete ===" | tee -a "$RESULTS"
echo "Finished: $(date)" | tee -a "$RESULTS"
echo ""
echo "Results written to $RESULTS"
