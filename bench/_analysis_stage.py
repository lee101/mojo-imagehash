"""Stage split for bench/analysis.md: Pillow setup, Python/ctypes glue, Mojo kernel.

Diagnostic only; does not change production code. Run with
`pixi run python bench/_analysis_stage.py`.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import imagehash as upstream  # noqa: E402
import mojo_imagehash as mojo  # noqa: E402
from mojo_imagehash._lib import buffer, checked_call, lib, run_dct  # noqa: E402


def best(fn, repeat=25):
    b = math.inf
    r = None
    for _ in range(repeat):
        s = time.perf_counter()
        r = fn()
        b = min(b, time.perf_counter() - s)
    return b, r


def med(fn, repeat=25):
    xs = []
    for _ in range(repeat):
        s = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - s)
    xs.sort()
    return xs[len(xs) // 2]


def fmt(s):
    return f"{s * 1e3:.3f}"


def main():
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, (1536, 2048, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, "RGB")
    gray = image.convert("L")

    print("machine:", open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(" :"))
    print()

    cases = [
        ("average_hash hash_size=32", lambda: mojo.average_hash(image, 32), lambda: upstream.average_hash(image, 32)),
        ("average_hash hash_size=8", lambda: mojo.average_hash(image, 8), lambda: upstream.average_hash(image, 8)),
        ("dhash hash_size=32", lambda: mojo.dhash(image, 32), lambda: upstream.dhash(image, 32)),
        ("dhash hash_size=8", lambda: mojo.dhash(image, 8), lambda: upstream.dhash(image, 8)),
        ("dhash_vertical hash_size=32", lambda: mojo.dhash_vertical(image, 32), lambda: upstream.dhash_vertical(image, 32)),
        ("phash hash_size=16", lambda: mojo.phash(image, 16), lambda: upstream.phash(image, 16)),
        ("phash hash_size=8", lambda: mojo.phash(image, 8), lambda: upstream.phash(image, 8)),
        ("phash hash_size=32", lambda: mojo.phash(image, 32), lambda: upstream.phash(image, 32)),
        ("whash hash_size=16", lambda: mojo.whash(image, 16), lambda: upstream.whash(image, 16)),
        ("whash hash_size=8", lambda: mojo.whash(image, 8), lambda: upstream.whash(image, 8)),
    ]

    print("== end to end (min of 25) ==")
    print("| call | mojo ms | upstream ms | mojo/upstream |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        tm, hm = best(ours)
        tu, hu = best(theirs)
        assert hm == hu, name
        print(f"| {name} | {fmt(tm)} | {fmt(tu)} | {tm / tu:.3f} |")

    print()
    print("== Pillow / numpy stage split (min of 25, ms) ==")
    for label, size in [
        ("convert L", None),
        ("convert L + resize 32x32", (32, 32)),
        ("convert L + resize 33x32", (33, 32)),
        ("convert L + resize 32x33", (32, 33)),
        ("convert L + resize 64x64", (64, 64)),
        ("convert L + resize 128x128", (128, 128)),
        ("convert L + resize 1024x1024", (1024, 1024)),
    ]:
        if size is None:
            t, _ = best(lambda: image.convert("L"))
        else:
            t, _ = best(lambda s=size: mojo._luma(image, s))
        print(f"| {label} | {fmt(t)} |")

    print()
    print("== kernel-only, in-process ctypes (min of 25, ms) ==")
    # average hash kernel, hash_size 8 / 32
    for hs in (8, 32):
        p = mojo._luma(image, (hs, hs))
        bits = np.empty((hs, hs), dtype=np.bool_)
        pa, pl, _ = buffer(p, np.uint8)
        ba, bl, _ = buffer(bits, np.bool_)
        t, _ = best(lambda: checked_call("mih_average_hash", pa, pl, p.size, ba, bl))
        print(f"| mih_average_hash hs={hs} ({p.size} elems) | {fmt(t)} |")
        tnp, _ = best(lambda: p.mean())
        print(f"| numpy p.mean() hs={hs} | {fmt(tnp)} |")
        thr = float(p.mean())
        call = lib().mih_average_hash
        t, _ = best(lambda: call(pa, pl, p.size, ba, bl))
        print(f"| mih_average_hash direct (no checked_call) hs={hs} | {fmt(t)} |")
        t, _ = best(lambda: p.mean())
        print(f"|   (numpy mean again) | {fmt(t)} |")

    # dhash kernel
    for hs in (8, 32):
        p = mojo._luma(image, (hs + 1, hs))
        bits = np.empty((hs, hs), dtype=np.bool_)
        pa, pl, ps = buffer(p, np.uint8)
        ba, bl, _ = buffer(bits, np.bool_)
        t, _ = best(lambda: checked_call("mih_difference_hash", pa, pl, ps, hs, ba, bl))
        print(f"| mih_difference_hash hs={hs} ({p.size} elems) | {fmt(t)} |")
        pv = mojo._luma(image, (hs, hs + 1))
        pa, pl, ps = buffer(pv, np.uint8)
        t, _ = best(lambda: checked_call("mih_vertical_difference_hash", pa, pl, ps, hs, ba, bl))
        print(f"| mih_vertical_difference_hash hs={hs} ({pv.size} elems) | {fmt(t)} |")
        # numpy baseline: horizontal diff
        a = p.astype(np.int16)
        t, _ = best(lambda: a[:, 1:] > a[:, :-1])
        print(f"| numpy (p[:,1:] > p[:,:-1]) hs={hs} | {fmt(t)} |")

    # dct
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
        t_ser, _ = best(lambda: checked_call("mih_phash_dct", pa, pl, ps, imsize, hs, wa, wl, ca, cl))
        t_run, _ = best(lambda: run_dct(pa, pl, ps, imsize, hs, wa, wl, ca, cl))
        print(f"| mih_phash_dct serial hs={hs} ({imsize}x{imsize}) | {fmt(t_ser)} |")
        print(f"| run_dct (as called) hs={hs} | {fmt(t_run)} |")
        t = med(lambda: run_dct(pa, pl, ps, imsize, hs, wa, wl, ca, cl))
        print(f"| run_dct med hs={hs} | {fmt(t)} |")
        # numpy reference dct via scipy for a flops sanity baseline
        from scipy.fft import dctn
        t, _ = best(lambda: dctn(p.astype(np.float64), norm="ortho"))
        print(f"| scipy dctn reference hs={hs} | {fmt(t)} |")
        # allocation cost in the shim
        def alloc():
            st = np.empty(work_size + hs * hs, dtype=np.float64)
            return st[:work_size], st[work_size:].reshape(hs, hs)
        t, _ = best(alloc)
        print(f"| np.empty work+coeff hs={hs} | {fmt(t)} |")
        t, _ = best(lambda: np.median(coeff))
        print(f"| np.median(coeff) hs={hs} | {fmt(t)} |")

    # haar
    for hs, scale in ((8, 1024), (16, 1024), (8, 256), (16, 256)):
        p = mojo._luma(image, (scale, scale))
        cw = np.empty((hs, hs), dtype=np.float64)
        pa, pl, ps = buffer(p, np.uint8)
        ca, cl, _ = buffer(cw, np.float64)
        t, _ = best(lambda: checked_call("mih_haar_lowpass", pa, pl, ps, scale, hs, ca, cl))
        print(f"| mih_haar_lowpass {scale}x{scale} -> {hs}x{hs} | {fmt(t)} |")
        pywt = __import__("pywt")
        vals = p / 255.0
        llmax = int(np.log2(scale))
        level = int(np.log2(hs))
        dwt_level = llmax - level
        def full_pywt():
            c = list(pywt.wavedec2(vals, "haar", level=llmax))
            c[0] *= 0
            v = pywt.waverec2(c, "haar")
            return pywt.wavedec2(v, "haar", level=dwt_level)[0]
        t, _ = best(full_pywt, repeat=9)
        print(f"| pywt full haar path {scale}->{hs} | {fmt(t)} |")
        t, _ = best(lambda: p / 255.0)
        print(f"| numpy p/255.0 {scale} | {fmt(t)} |")

    # glue costs
    print()
    print("== glue cost micro (ns unless noted, min of 2000) ==")
    bits = np.empty((32, 32), dtype=np.bool_)
    p = mojo._luma(image, (32, 32))
    pa, pl, _ = buffer(p, np.uint8)
    ba, bl, _ = buffer(bits, np.bool_)
    call = lib().mih_average_hash
    t, _ = best(lambda: call(pa, pl, p.size, ba, bl), repeat=2000)
    print(f"| direct ctypes call mih_average_hash (32x32) | {t * 1e6:.0f} ns |")
    t, _ = best(lambda: checked_call("mih_average_hash", pa, pl, p.size, ba, bl), repeat=2000)
    print(f"| checked_call wrapper mih_average_hash | {t * 1e6:.0f} ns |")
    t, _ = best(lambda: buffer(p, np.uint8), repeat=2000)
    print(f"| buffer() validation | {t * 1e6:.0f} ns |")
    t, _ = best(lambda: ctypes.cast(pa, ctypes.c_void_p).value, repeat=2000)
    print(f"| ctypes.cast(int) floor | {t * 1e6:.0f} ns |")


if __name__ == "__main__":
    main()
