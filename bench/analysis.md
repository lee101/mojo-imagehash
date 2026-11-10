# Kernel analysis

This is the analysis record the acceleration phases gate on. The numbers below
are derived from the shipped kernels in `src/imagehash.mojo` and from the call
sizes `python/mojo_imagehash/__init__.py` actually passes; the timings are
measured on this machine with `bench/parallel_measure.py`, `bench/bench.py`, and
the CPU/GPU probes described in the GPU section.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; NVIDIA GeForce RTX 5090
(shared with production workloads).

## Arithmetic intensity

Computing intensity requires two inputs: the arithmetic performed per output
element, and the bytes moved. One multiply and one add count as 2 flops. Bytes
are the bytes read from and written to memory for the whole call.

| kernel | call size real callers pass | flops | bytes | flops/byte |
|---|---|---:|---:|---:|
| `average_hash` | hash_size 8 | 64 | 128 | 0.50 |
| `average_hash` | hash_size 32 | 1,024 | 2,048 | 0.50 |
| `dhash` / `dhash_vertical` | hash_size 8 | 64 | 128 | 0.50 |
| `dhash` / `dhash_vertical` | hash_size 32 | 1,024 | 2,048 | 0.50 |
| `phash` DCT | hash_size 8 (32x32, default) | 20,480 | 5,632 | 3.64 |
| `phash` DCT | hash_size 16 (64x64) | 163,840 | 22,528 | 7.27 |
| `phash` DCT | hash_size 32 (128x128) | 1,310,720 | 90,112 | 14.55 |
| `haar_lowpass` | 256x256 -> 8x8 | 65,536 | 65,536 | 1.00 |
| `haar_lowpass` | 1024x1024 -> 16x16 | 1,048,576 | 1,048,576 | 1.00 |

`phash` DCT flops: pass one is `image_size` fused multiply-adds per output for
`hash_size * image_size` outputs; pass two is `image_size` fused multiply-adds
per output for `hash_size^2` outputs. Bytes: one `uint8` plane in, the
`float64` work buffer, and the `hash_size^2` coefficients out.

Haar is 1.0 flops/byte because each `uint8` input pixel is read once and folded
into a block sum; the pyramid is cache-resident and the reduction is a single
add per input.

## Gate

Only kernels above roughly 2 flops per byte are GPU candidates.

- `average_hash`, `dhash`, `dhash_vertical`: **0.50 flops/byte — CPU only,
  memory-bound.** Not candidates; no GPU work is justified.
- `haar_lowpass`: **1.00 flops/byte — CPU only, memory-bound.** Not a
  candidate, even at 1024x1024.
- `phash` DCT: **3.64 to 14.55 flops/byte — a candidate.** It cleared the gate
  and was carried into the GPU phase, where it lost at the sizes callers pass.
  See the GPU section of the README.

## Sizes real callers pass

`phash(image, hash_size=8, highfreq_factor=4)` is the default call and produces
a 32x32 pixel plane: 20,480 flops over 5.6 KB. The README benchmark and the
test suite use hash sizes 4, 8, 16, and the benchmark also covers 32. These are
the sizes the GPU decision must be measured at; larger planes exist only in the
diagnostic sweep, not in caller traffic.

## CPU kernel timings used by the gate

| kernel | image_size | hash_size | best (CPU) |
|---|---:|---:|---:|
| `mih_phash_dct` | 32 | 8 | 9.5 us |
| `mih_phash_dct` | 64 | 16 | 48.4 us |
| `mih_phash_dct` | 128 | 32 | 312 us |
| `mih_phash_dct` | 256 | 64 | 2,566 us |

End to end, `phash` on a 2048x1536 RGB image is 12.078 ms (hash_size 8),
12.527 ms (16), and 13.813 ms (32). Pillow's grayscale conversion and Lanczos
resize dominate every one of them; the DCT kernel is 0.08% to 2.3% of the call.
