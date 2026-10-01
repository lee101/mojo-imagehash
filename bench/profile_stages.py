"""Stage-level breakdown of the end-to-end path (diagnostic, not the table)."""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import imagehash as upstream  # noqa: E402
import mojo_imagehash as mojo  # noqa: E402
from mojo_imagehash._lib import buffer, checked_call, run_dct  # noqa: E402


def best(fn, repeat=15):
    b = math.inf
    r = None
    for _ in range(repeat):
        s = time.perf_counter()
        r = fn()
        b = min(b, time.perf_counter() - s)
    return b, r


def main():
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, (1536, 2048, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, "RGB")
    gray = image.convert("L")
    print("Pillow convert(L)      %.3f ms" % (best(lambda: image.convert("L"))[0] * 1e3))
    for size in ((32, 32), (33, 32), (64, 64), (1024, 1024)):
        t, _ = best(lambda s=size: gray.resize(s, Image.Resampling.LANCZOS))
        print("  resize %-12s %.3f ms" % (str(size), t * 1e3))

    print()
    print("== average_hash ==")
    t, _ = best(lambda: mojo.average_hash(image, 32))
    print("  mojo total          %.3f ms" % (t * 1e3))
    t, _ = best(lambda: upstream.average_hash(image, 32))
    print("  upstream total      %.3f ms" % (t * 1e3))

    print("== dhash ==")
    t, _ = best(lambda: mojo.dhash(image, 32))
    print("  mojo total          %.3f ms" % (t * 1e3))
    t, _ = best(lambda: upstream.dhash(image, 32))
    print("  upstream total      %.3f ms" % (t * 1e3))

    print("== phash ==")
    t, _ = best(lambda: mojo.phash(image, 16))
    print("  mojo total          %.3f ms" % (t * 1e3))
    t, _ = best(lambda: upstream.phash(image, 16))
    print("  upstream total      %.3f ms" % (t * 1e3))

    print("== whash ==")
    t, _ = best(lambda: mojo.whash(image, 16))
    print("  mojo total          %.3f ms" % (t * 1e3))
    t, _ = best(lambda: upstream.whash(image, 16))
    print("  upstream total      %.3f ms" % (t * 1e3))

    # ---- kernel-only timings ----
    print()
    print("== kernel only ==")
    p32 = np.ascontiguousarray(np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS)), dtype=np.uint8)
    bits = np.empty((32, 32), dtype=np.bool_)
    pa, pl, _ = buffer(p32, np.uint8)
    ba, bl, _ = buffer(bits, np.bool_)
    t, _ = best(lambda: checked_call("mih_average_hash", pa, pl, p32.size, ba, bl))
    print("  mih_average_hash    %.4f ms" % (t * 1e3))

    p33 = np.ascontiguousarray(np.asarray(gray.resize((33, 32), Image.Resampling.LANCZOS)), dtype=np.uint8)
    pa, pl, ps = buffer(p33, np.uint8)
    t, _ = best(lambda: checked_call("mih_difference_hash", pa, pl, ps, 32, ba, bl))
    print("  mih_difference_hash %.4f ms" % (t * 1e3))

    hs, imsize = 16, 64
    p64 = np.ascontiguousarray(np.asarray(gray.resize((imsize, imsize), Image.Resampling.LANCZOS)), dtype=np.uint8)
    work_size = 2 * hs * imsize
    storage = np.empty(work_size + hs * hs, dtype=np.float64)
    work = storage[:work_size]
    coeff = storage[work_size:].reshape(hs, hs)
    pa, pl, ps = buffer(p64, np.uint8)
    wa, wl, _ = buffer(work, np.float64)
    ca, cl, _ = buffer(coeff, np.float64)
    t, _ = best(lambda: run_dct(pa, pl, ps, imsize, hs, wa, wl, ca, cl))
    print("  run_dct(16x64)      %.4f ms" % (t * 1e3))
    t, _ = best(lambda: checked_call("mih_phash_dct", pa, pl, ps, imsize, hs, wa, wl, ca, cl))
    print("  mih_phash_dct ser   %.4f ms" % (t * 1e3))

    scale = 1024
    p1k = np.ascontiguousarray(np.asarray(gray.resize((scale, scale), Image.Resampling.LANCZOS)), dtype=np.uint8)
    cw = np.empty((16, 16), dtype=np.float64)
    pa, pl, ps = buffer(p1k, np.uint8)
    ca, cl, _ = buffer(cw, np.float64)
    t, _ = best(lambda: checked_call("mih_haar_lowpass", pa, pl, ps, scale, 16, ca, cl))
    print("  mih_haar_lowpass    %.4f ms" % (t * 1e3))
    t, _ = best(lambda: np.median(cw))
    print("  np.median(coeff)    %.4f ms" % (t * 1e3))


if __name__ == "__main__":
    main()