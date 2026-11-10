"""Kernel-level timings used to place the parallel size thresholds.

Run through `pixi run bench-parallel`. Prints one row per (kernel, size), giving
the best of `REPEAT` calls. Sizes are chosen so both sides of every candidate
threshold are visible.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojo_imagehash._lib import buffer, checked_call  # noqa: E402

REPEAT = 40


def best(fn, repeat=REPEAT):
    value = None
    fastest = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        fastest = min(fastest, time.perf_counter() - start)
    return fastest, value


def dct_case(label, image_size, hash_size):
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, (image_size, image_size), dtype=np.uint8)
    work_size = 2 * hash_size * image_size
    storage = np.empty(work_size + hash_size * hash_size, dtype=np.float64)
    work = storage[:work_size]
    coeff = storage[work_size:]
    pa, pl, ps = buffer(pixels, np.uint8)
    wa, wl, _ = buffer(work, np.float64)
    ca, cl, _ = buffer(coeff, np.float64)
    flat_coeff = np.empty(hash_size * hash_size, dtype=np.float64)
    fa, fl, _ = buffer(flat_coeff, np.float64)
    t, _ = best(
        lambda: checked_call("mih_phash_dct", pa, pl, ps, image_size, hash_size, wa, wl, ca, cl)
    )
    print(
        f"| dct | {label} | {hash_size} x {image_size} = {hash_size * image_size} | "
        f"{t * 1e3:.4f} ms |"
    )
    return (pa, pl, ps, image_size, hash_size, wa, wl, fa, fl)


def dct_split_case(label, args):
    pa, pl, ps, image_size, hash_size, wa, wl, fa, fl = args
    t, _ = best(
        lambda: (
            checked_call("mih_phash_dct_first", pa, image_size, hash_size, wa, 0, hash_size),
            checked_call("mih_phash_dct_second", wa, image_size, hash_size, fa, 0, hash_size),
        )
    )
    print(f"| dct-split-1 | {label} | {hash_size * image_size} | {t * 1e3:.4f} ms |")


def haar_case(scale, hash_size):
    pixels = np.random.default_rng(3).integers(0, 256, (scale, scale), dtype=np.uint8)
    coeff = np.empty(hash_size * hash_size, dtype=np.float64)
    pa, pl, ps = buffer(pixels, np.uint8)
    ca, cl, _ = buffer(coeff, np.float64)
    t, _ = best(lambda: checked_call("mih_haar_lowpass", pa, pl, ps, scale, hash_size, ca, cl))
    print(f"| haar | {scale}x{scale} | {hash_size} | {t * 1e3:.4f} ms |")


def threshold_case(count):
    pixels = np.random.default_rng(5).integers(0, 256, count, dtype=np.uint8)
    bits = np.empty(count, dtype=np.bool_)
    pa, pl, _ = buffer(pixels, np.uint8)
    ba, bl, _ = buffer(bits, np.bool_)
    threshold = float(pixels.mean())
    t, _ = best(lambda: checked_call("mih_threshold", pa, pl, count, threshold, ba, bl))
    print(f"| threshold | {count} | - | {t * 1e3:.4f} ms |")


def average_case(count):
    pixels = np.random.default_rng(5).integers(0, 256, count, dtype=np.uint8)
    bits = np.empty(count, dtype=np.bool_)
    pa, pl, _ = buffer(pixels, np.uint8)
    ba, bl, _ = buffer(bits, np.bool_)
    t, _ = best(lambda: checked_call("mih_average_hash", pa, pl, count, ba, bl))
    print(f"| average | {count} | - | {t * 1e3:.4f} ms |")


def difference_case(hash_size):
    pixels = np.random.default_rng(5).integers(0, 256, (hash_size, hash_size + 1), dtype=np.uint8)
    bits = np.empty((hash_size, hash_size), dtype=np.bool_)
    pa, pl, ps = buffer(pixels, np.uint8)
    ba, bl, _ = buffer(bits, np.bool_)
    t, _ = best(lambda: checked_call("mih_difference_hash", pa, pl, ps, hash_size, ba, bl))
    print(f"| dhash | {hash_size} | - | {t * 1e3:.4f} ms |")


def main():
    print("| kernel | case | size/param | best |")
    print("|---|---|---|---:|")
    for hash_size, image_size in (
        (8, 32),
        (8, 64),
        (16, 64),
        (32, 128),
        (16, 128),
        (32, 256),
        (64, 512),
        (64, 1024),
        (128, 512),
        (128, 1024),
    ):
        args = dct_case(f"hs={hash_size}", image_size, hash_size)
        dct_split_case(f"hs={hash_size}", args)
    for scale, hash_size in ((64, 8), (256, 16), (512, 16), (1024, 16), (2048, 32)):
        haar_case(scale, hash_size)
    for count in (64, 256, 1024, 4096, 16384, 65536, 262144):
        threshold_case(count)
    for count in (64, 1024, 4096, 16384, 65536, 262144):
        average_case(count)
    for hash_size in (16, 33, 64, 128, 256):
        difference_case(hash_size)


if __name__ == "__main__":
    main()
