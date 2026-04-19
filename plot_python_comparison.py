"""
plot_python_comparison.py
=========================
Runs the Go (Small/Medium/Large) and Python SQLite benchmarks, then produces
a grouped bar chart comparing write and top-K search latency side-by-side.

Usage (from the RAGFile project root, with venv active):
    python plot_python_comparison.py
"""

import subprocess
import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Run Go benchmarks (Small / Medium / Large)
# ---------------------------------------------------------------------------
print("Running Go benchmarks …")
go_result = subprocess.run(
    [
        "go", "test",
        "-bench=BenchmarkSmall$|BenchmarkMedium$|BenchmarkLarge$",
        "-benchtime=3s",
        "./internal/tests/",
    ],
    capture_output=True,
    text=True,
    cwd=ROOT,
)

# ---------------------------------------------------------------------------
# 2. Run Python SQLite benchmarks
# ---------------------------------------------------------------------------
print("Running Python SQLite benchmarks …")
py_result = subprocess.run(
    ["python", "benchmark_python.py"],
    capture_output=True,
    text=True,
    cwd=ROOT,
)


SIZE_MARKERS = {
    r"Pairs: 100\b":   "Small\n(100 pairs, dim 16)",
    r"Pairs: 1000\b":  "Medium\n(1 000 pairs, dim 32)",
    r"Pairs: 10000\b": "Large\n(10 000 pairs, dim 64)",
}

def parse_time_us(value: str, unit: str) -> float:
    """Return time in microseconds."""
    v = float(value)
    if unit == "µs":
        return v
    if unit == "ms":
        return v * 1_000
    if unit == "s":
        return v * 1_000_000
    return v

go_write:  dict[str, float] = {}
go_search: dict[str, float] = {}

current_label: str | None = None
lines = go_result.stdout.splitlines()

for line in lines:
    for pattern, label in SIZE_MARKERS.items():
        if re.search(pattern, line):
            current_label = label
            break

    if current_label is None:
        continue

    if current_label not in go_write:
        m = re.search(r"Write time:\s*([\d.]+)(µs|ms|s)", line)
        if m:
            go_write[current_label] = parse_time_us(m.group(1), m.group(2))

    if current_label not in go_search:
        m = re.search(r"Search time:\s*([\d.]+)(µs|ms|s)", line)
        if m:
            go_search[current_label] = parse_time_us(m.group(1), m.group(2))

# ---------------------------------------------------------------------------
# 4. Parse Python output
#    Lines like:
#      "Write time: 0.0008s"
#      "Search time: 0.0016s"
# ---------------------------------------------------------------------------
py_write:  dict[str, float] = {}
py_search: dict[str, float] = {}

PY_MARKERS = {
    "SMALL BENCHMARK":  "Small\n(100 pairs, dim 16)",
    "MEDIUM BENCHMARK": "Medium\n(1 000 pairs, dim 32)",
    "LARGE BENCHMARK":  "Large\n(10 000 pairs, dim 64)",
}

current_label = None
for line in py_result.stdout.splitlines():
    for marker, label in PY_MARKERS.items():
        if marker in line:
            current_label = label
            break

    if current_label is None:
        continue

    if current_label not in py_write:
        m = re.search(r"Write time:\s*([\d.]+)s", line)
        if m:
            py_write[current_label] = float(m.group(1)) * 1_000_000  # → µs

    if current_label not in py_search:
        m = re.search(r"Search time:\s*([\d.]+)s", line)
        if m:
            py_search[current_label] = float(m.group(1)) * 1_000_000  # → µs

# ---------------------------------------------------------------------------
# 5. Validate
# ---------------------------------------------------------------------------
labels = [
    "Small\n(100 pairs, dim 16)",
    "Medium\n(1 000 pairs, dim 32)",
    "Large\n(10 000 pairs, dim 64)",
]

missing = [l for l in labels if l not in go_write or l not in go_search
                              or l not in py_write or l not in py_search]
if missing:
    print("Go stdout:\n", go_result.stdout[-2000:])
    print("Python stdout:\n", py_result.stdout)
    raise RuntimeError(f"Missing data for: {missing}")

# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------
go_w  = [go_write[l]  for l in labels]
go_s  = [go_search[l] for l in labels]
py_w  = [py_write[l]  for l in labels]
py_s  = [py_search[l] for l in labels]

x     = np.arange(len(labels))
width = 0.35

GO_COLOR = "#2563EB"   # blue
PY_COLOR = "#F97316"   # orange

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "RAGFile (Go, mmap + min-heap)  vs  Python SQLite (in-memory RAM)",
    fontsize=13,
    fontweight="bold",
)

def add_bars(ax, go_vals, py_vals, ylabel, title):
    b1 = ax.bar(x - width / 2, go_vals,  width, label="RAGFile (Go, mmap)",
                color=GO_COLOR, alpha=0.85)
    b2 = ax.bar(x + width / 2, py_vals,  width, label="SQLite in-memory (Python)",
                color=PY_COLOR, alpha=0.85)

    ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels
    for bar, val in zip(b1, go_vals):
        txt = f"{val:.1f} µs" if val < 1000 else f"{val/1000:.1f} ms"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                txt, ha="center", va="bottom", fontsize=7.5, color=GO_COLOR,
                fontweight="bold")
    for bar, val in zip(b2, py_vals):
        txt = f"{val:.1f} µs" if val < 1000 else f"{val/1000:.1f} ms"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                txt, ha="center", va="bottom", fontsize=7.5, color=PY_COLOR,
                fontweight="bold")

    # Speed-up annotation
    for i, (gv, pv) in enumerate(zip(go_vals, py_vals)):
        ratio = pv / gv
        ax.text(x[i], max(gv, pv) * 3.5, f"{ratio:.0f}×\nfaster",
                ha="center", va="bottom", fontsize=8,
                color="#16A34A", fontweight="bold")

add_bars(ax1, go_w, py_w,
         "Write latency (µs, log scale)",
         "Write Performance\nRAGFile vs SQLite in-memory")

add_bars(ax2, go_s, py_s,
         "Search latency (µs, log scale)",
         "Top-K Search Performance (k = 10)\nRAGFile vs SQLite in-memory")

plt.tight_layout()

out_dir  = os.path.join(ROOT, "benchmark_output")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "python_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved → {out_path}")
