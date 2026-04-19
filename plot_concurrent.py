"""
plot_concurrent.py
==================
Runs the concurrent Go benchmarks and plots goroutine-count vs latency for
both mmap-backed WRITES and TOP-K SEARCHES side-by-side.

Usage (from the RAGFile project root, with venv active):
    python plot_concurrent.py
"""

import subprocess
import re
import os
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# 1. Run benchmarks
# ---------------------------------------------------------------------------
print("Running concurrent benchmarks (this takes ~30 s) …")
result = subprocess.run(
    [
        "go", "test",
        "-bench=BenchmarkConcurrentWrite$|BenchmarkConcurrentSearch$",
        "-benchtime=3s",
        "-benchmem",
        "./internal/tests/",
    ],
    capture_output=True,
    text=True,
    cwd=os.path.dirname(os.path.abspath(__file__)),
)

if result.returncode != 0:
    print("go test stderr:\n", result.stderr)
    raise RuntimeError("Benchmark run failed")

# ---------------------------------------------------------------------------
# 2. Parse standard Go benchmark lines
#    Format: BenchmarkXxx/sub-N    <iters>    <ns/op>  …
# ---------------------------------------------------------------------------
write_ns: dict[int, float] = {}
search_ns: dict[int, float] = {}

for line in result.stdout.splitlines():
    m = re.search(
        r"BenchmarkConcurrentWrite/workers=(\d+)-\d+\s+\d+\s+([\d.]+)\s+ns/op",
        line,
    )
    if m:
        write_ns[int(m.group(1))] = float(m.group(2))
        continue

    m = re.search(
        r"BenchmarkConcurrentSearch/workers=(\d+)-\d+\s+\d+\s+([\d.]+)\s+ns/op",
        line,
    )
    if m:
        search_ns[int(m.group(1))] = float(m.group(2))

if not write_ns or not search_ns:
    print("stdout:\n", result.stdout)
    raise RuntimeError("Could not parse benchmark output – check regex patterns")

# Convert ns → ms
def to_ms(d: dict) -> tuple[list, list]:
    workers = sorted(d)
    times   = [d[w] / 1e6 for w in workers]
    return workers, times

w_workers, w_times = to_ms(write_ns)
s_workers, s_times = to_ms(search_ns)

# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "RAGFile – Goroutine Scaling on mmap-backed Data\n"
    "(10 000 pairs × dim 128, Intel N95 – 4 physical cores)",
    fontsize=13,
    fontweight="bold",
)

COLORS = {"line": "#2563EB", "baseline": "#DC2626", "fill": "#BFDBFE"}

for ax, workers, times, label, title in [
    (axes[0], w_workers, w_times, "Concurrent write",
     "Write latency vs goroutine count\n(10 000 pairs × dim 128)"),
    (axes[1], s_workers, s_times, "Concurrent top-K search",
     "Top-K search latency vs goroutine count\n(k = 10)"),
]:
    baseline = times[0]   # workers == 1

    ax.plot(workers, times, color=COLORS["line"], linewidth=2.2,
            marker="o", markersize=7, label=label, zorder=3)
    ax.axhline(baseline, color=COLORS["baseline"], linestyle="--",
               linewidth=1.4, label=f"1-worker baseline ({baseline:.2f} ms)", zorder=2)
    ax.fill_between(workers, times, baseline,
                    color=COLORS["fill"], alpha=0.45, zorder=1)

    # Annotate each point with its latency
    for w, t in zip(workers, times):
        ax.annotate(
            f"{t:.2f} ms",
            xy=(w, t),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("Number of goroutines (workers)", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(workers)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_output")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "concurrent_performance.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved → {out_path}")
