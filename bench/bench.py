"""Honest end-to-end benchmarks against ImageHash on the same PIL images."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import imagehash as upstream  # noqa: E402
import mojo_imagehash as mojo  # noqa: E402


def timeit(function, repeat=9):
    best = math.inf
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, (1536, 2048, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, "RGB")
    cases = [
        ("average_hash (hash_size=32)", lambda: mojo.average_hash(image, 32), lambda: upstream.average_hash(image, 32)),
        ("dhash (hash_size=32)", lambda: mojo.dhash(image, 32), lambda: upstream.dhash(image, 32)),
        ("phash (hash_size=16)", lambda: mojo.phash(image, 16), lambda: upstream.phash(image, 16)),
        ("whash (hash_size=16)", lambda: mojo.whash(image, 16), lambda: upstream.whash(image, 16)),
    ]

    mojo.average_hash(image)
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| algorithm | mojo-imagehash | ImageHash 4.3.2 | speedup |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        mojo_time, mojo_hash = timeit(ours)
        upstream_time, upstream_hash = timeit(theirs)
        if mojo_hash != upstream_hash:
            raise RuntimeError(f"benchmark parity failed for {name}")
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{upstream_time * 1e3:.3f} ms | {upstream_time / mojo_time:.2f}x |"
        )

if __name__ == "__main__":
    main()
