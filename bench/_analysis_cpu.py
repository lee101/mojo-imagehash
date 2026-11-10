"""Measure kernel CPU time from /proc/self/task/*/stat around a ctypes call.

Diagnostic only, run standalone so the flock bench is not disturbed.
"""

from __future__ import annotations

import ctypes
import glob
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_imagehash as mojo  # noqa: E402
from mojo_imagehash._lib import buffer, checked_call, lib, run_dct  # noqa: E402

HZ = os.sysconf("SC_CLK_TCK")


def cpu_ticks():
    total = 0
    for path in glob.glob("/proc/self/task/*/stat"):
        try:
            with open(path) as f:
                fields = f.read().rsplit(")", 1)[1].split()
            total += int(fields[11]) + int(fields[12])  # utime + stime
        except (FileNotFoundError, IndexError):
            pass
    return total


def measure(fn, repeat=200):
    """Return (min wall s, summed CPU s over all threads / repeat)."""
    best = math.inf
    cpu_total = 0
    t0 = cpu_ticks()
    for _ in range(repeat):
        s = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - s)
    cpu_total = cpu_ticks() - t0
    return best, cpu_total / HZ / repeat


def row(label, fn, repeat=200):
    wall, cpu = measure(fn, repeat)
    print(f"{label}|{wall * 1e6:.1f}|{cpu * 1e6:.1f}|{cpu / wall:.2f}")


def main():
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, (1536, 2048, 3), dtype=np.uint8), "RGB")
    call = lib()
    print("label|wall_us|cpu_us|threads_equiv")
    for hs in (8, 32):
        p = mojo._luma(image, (hs, hs))
        bits = np.empty((hs, hs), dtype=np.bool_)
        pa, pl, _ = buffer(p, np.uint8)
        ba, bl, _ = buffer(bits, np.bool_)
        row(f"average_hash hs={hs} kernel", lambda: checked_call("mih_average_hash", pa, pl, p.size, ba, bl), 500)
        row(f"average_hash hs={hs} direct", lambda: call.mih_average_hash(pa, pl, p.size, ba, bl), 500)
    for hs in (8, 32):
        p = mojo._luma(image, (hs + 1, hs))
        bits = np.empty((hs, hs), dtype=np.bool_)
        pa, pl, ps = buffer(p, np.uint8)
        ba, bl, _ = buffer(bits, np.bool_)
        row(f"dhash hs={hs} kernel", lambda: checked_call("mih_difference_hash", pa, pl, ps, hs, ba, bl), 500)
        pv = mojo._luma(image, (hs, hs + 1))
        pa, pl, ps = buffer(pv, np.uint8)
        row(f"dhash_vertical hs={hs} kernel", lambda: checked_call("mih_vertical_difference_hash", pa, pl, ps, hs, ba, bl), 500)
    for hs in (8, 16, 32):
        imsize = hs * 4
        p = mojo._luma(image, (imsize, imsize))
        work_size = 2 * hs * imsize
        storage = np.empty(work_size + hs * hs, dtype=np.float64)
        work = storage[:work_size]
        coeff = storage[work_size:].reshape(hs, hs)
        pa, pl, ps = buffer(p, np.uint8)
        wa, wl, _ = buffer(work, np.float64)
        ca, cl, _ = buffer(coeff, np.float64)
        row(f"phash dct serial hs={hs} ({imsize}^2)", lambda: checked_call("mih_phash_dct", pa, pl, ps, imsize, hs, wa, wl, ca, cl), 100)
        row(f"phash dct run_dct hs={hs}", lambda: run_dct(pa, pl, ps, imsize, hs, wa, wl, ca, cl), 100)
    for hs, scale in ((8, 256), (8, 1024), (16, 1024)):
        p = mojo._luma(image, (scale, scale))
        cw = np.empty((hs, hs), dtype=np.float64)
        pa, pl, ps = buffer(p, np.uint8)
        ca, cl, _ = buffer(cw, np.float64)
        row(f"haar_lowpass {scale}->{hs} kernel", lambda: checked_call("mih_haar_lowpass", pa, pl, ps, scale, hs, ca, cl), 30)


if __name__ == "__main__":
    main()
