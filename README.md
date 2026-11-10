# mojo-imagehash

`mojo-imagehash` is a Mojo port of the compute-heavy core of the Python
[ImageHash](https://pypi.org/project/ImageHash/) library. It computes compact
perceptual fingerprints for similarity search and duplicate detection while
keeping a familiar Python API.

The covered subset is:

- `average_hash(image, hash_size=8, mean=np.mean)` (`ahash`)
- `phash(image, hash_size=8, highfreq_factor=4)`
- `dhash(image, hash_size=8)` and `dhash_vertical(image, hash_size=8)`
- `whash(image, hash_size=8, image_scale=None, mode="haar",
  remove_max_haar_ll=True)`
- `ImageHash`, `hex_to_hash`, and `old_hex_to_hash`

The canonical ImageHash function names and signatures are preserved. Importing
the package as `imagehash` makes covered calls a direct substitution:

```python
import mojo_imagehash as imagehash
```

`phash_simple`, `colorhash`, `crop_resistant_hash`, and the multi-hash classes
are not implemented. The default Haar wavelet calculation runs in Mojo.
`mode="db4"` and exact tie resolution use PyWavelets for compatibility.

## Install

The repository pins a tested Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-imagehash.so`. The Python package can then be
used through Pixi:

```bash
pixi run python
```

For packaging or deployment, set `MOJO_IMAGEHASH_LIB` to the absolute path of
an already-built shared library.

## Usage

```python
from PIL import Image
import mojo_imagehash as imagehash

image = Image.linear_gradient("L").resize((256, 256))

print(imagehash.average_hash(image))  # 00000000ffffffff
print(imagehash.phash(image))         # 8000800080008000
print(imagehash.dhash(image))         # 0000000000000000
print(imagehash.whash(image))         # 00000000ffffffff

other = Image.open("candidate.jpg")
distance = imagehash.phash(image) - imagehash.phash(other)
```

Hashes stringify as hexadecimal, compare for equality, and subtract to produce
their Hamming distance, matching ImageHash.

## Correctness

The test suite compares every covered algorithm directly with ImageHash 4.3.2
on noise, gradients, checkerboards, and constant images. It exercises hash
sizes 4, 8, and 16, alternate pHash frequency factors, explicit and natural
wavelet scales, Haar and db4 modes, custom average functions, serialization,
and validation behavior.

```bash
pixi run build
pixi run test
```

The current suite contains 42 passing parity and behavior tests, including
non-SIMD-aligned pHash sizes and both sides of the parallel launch threshold.

## Benchmarks

These are end-to-end timings, including Pillow grayscale conversion and
Lanczos resizing, on the same 2048 by 1536 RGB image. Lower is better. Both
the before and after columns are real `pixi run bench` output from 2026-11-10:
the unmodified library and the unrolled library were run alternately, three
times each, and each cell is the fastest of those three runs. The machine-wide
benchmark lock was saturated for the whole session (a waiting queue of 13-20
processes from other repositories that never drained within two hours), so
these runs were taken without it, under a load average of 17-21. Single-run
values wander by about 0.3 ms (average hash, dHash, pHash) and about 1.2 ms
(wHash); the kernel table below separates code effects from that noise.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, Linux x86-64.

| algorithm | before | after | ImageHash 4.3.2 |
|---|---:|---:|---:|
| average_hash (hash_size=32) | 11.71 ms | 11.95 ms | 11.74 ms |
| dhash (hash_size=32) | 12.01 ms | 11.84 ms | 11.77 ms |
| phash (hash_size=16) | 12.43 ms | 12.46 ms | 12.37 ms |
| whash (hash_size=16) | 26.39 ms | 25.69 ms | 100.15 ms |

The ImageHash column is the fastest of the same three runs.

Only wHash moves: its Haar kernel dropped from 1.14 ms to 0.32 ms, but the
whole call is ~26 ms because Pillow's grayscale conversion and Lanczos resize
to 1024 by 1024 take ~22 ms, so the end-to-end difference (~0.7 ms) is near the
run-to-run spread and should be read as such. The other three calls keep their
kernels at 0.003 ms, 0.002 ms, and 0.05 ms, which Pillow preprocessing and the
resize dominate completely; their before/after differences here are scheduler
noise, not code changes, and no claim is made from them.

Kernel-only timings on the shipped library, minimum of 60 calls, fixed inputs:

| kernel | before | after | change |
|---|---:|---:|---:|
| `mih_haar_lowpass` 1024x1024 -> 16x16 | 1.138 ms | 0.325 ms | 3.5x |
| `mih_haar_lowpass` 512x512 -> 16x16 | 0.286 ms | 0.087 ms | 3.3x |
| `mih_phash_dct` serial, 64x64 image_size, hash_size 16 | 0.0586 ms | 0.0589 ms | unchanged |
| `mih_phash_dct` serial, 128x128 image_size, hash_size 32 | 0.364 ms | 0.366 ms | unchanged |

Pillow preprocessing and FFI overhead dominate the small average-hash and
dHash kernels. In this run average hash was effectively tied, dHash was tied,
pHash was ~1.02x faster than ImageHash, and Haar wHash was ~3.8x faster.

## GPU phase

The gate is in `bench/analysis.md`: only kernels above roughly 2 flops per
byte are candidates. Average hash and both dHashes are 0.50 flops/byte and Haar
low-pass is 1.00 flops/byte, so those are **CPU only: memory-bound** and were
not taken further. Only `phash` DCT clears the gate at the sizes callers pass
(3.64 flops/byte at the default `hash_size=8`, rising to 14.55 at 32).

**No GPU path ships.** The device was free enough to use
(`nvidia-smi --query-gpu=memory.free` reported 4,557 MiB, above the 4,000 MiB
floor), so this is a measured decision, not a skip. A kernel with the same
arithmetic as `mih_phash_dct` — one thread per output, the same sequential
summation order, one copy-in and one copy-out — was run on the device with
`DeviceContext` from `max.gpu.host` and `block_idx`/`thread_idx` from
`max.gpu`:

| `phash` DCT | CPU serial | GPU, copies included | GPU/CPU |
|---|---:|---:|---:|
| hash_size 8 (32x32, default) | 9.5 us | 25.8 us | 0.37x |
| hash_size 16 (64x64) | 48.4 us | 33.8 us | 1.43x |
| hash_size 32 (128x128) | 312 us | 52.4 us | 6.0x |
| hash_size 64 (256x256) | 2,566 us | 87.7 us | 29x |

Best of 20 runs on the shared RTX 5090, run while the CPU numbers above were
taken. The floor for the simplest possible single launch on this device is
6.3-7.3 us for launch plus synchronisation, and 10.9 us once an 8 KB buffer is
copied in, so at the default 32x32 plane the transfer alone is larger than the
entire CPU computation. The GPU loses at the call most users make.

It does win from `hash_size=16` upward, but that win is invisible end to end:
`phash` on a 2048x1536 image is 12.078 ms (hash_size 8), 12.527 ms (16) and
13.813 ms (32), because Pillow's grayscale conversion and Lanczos resize take
about 12 ms of every call. Replacing a 312 us kernel with a 52 us one at
`hash_size=32` moves the total by 2.3%, inside the +-0.3 ms to +-1.2 ms
run-to-run spread this README already documents. It would win only on a plane
large enough that nobody calls `phash` with it.

Three further findings rule the path out on this toolchain. `cos` on
`DType.float64` is not supported on NVIDIA GPU (`DType.float64 is not
supported for cos on NVIDIA GPU`), so the cosine table has to be built on the
host and uploaded, which adds a second copy-in that must be cached per size to
avoid repeating it per call. The two DCT passes need a barrier between them, so
a faithful version is two launches per call, not one. And the result is not
bit-identical to the CPU kernel: with the host table the largest coefficient
difference was 3.6e-10, which changes no hash bit on the tested image but does
break the bit-for-bit guarantee the rest of the library keeps.

So the GPU code was reverted rather than shipped; the CPU path is unchanged and
remains the only path. No GPU test was added, because there is no GPU path to
test: `pixi run test` runs the same 42 parity and behaviour tests on hosts with
and without a device, and none of them were skipped. Should a future revision
revisit this, `bench/analysis.md` holds the gate and the pool of real call
sizes is hash size 4 to 32.

Run the benchmark only through the locked task:

```bash
pixi run bench
```

## How it works

Pillow performs grayscale conversion and Lanczos resizing, exactly as
ImageHash does. Contiguous row-major `uint8` pixel planes then remain owned by
NumPy for the duration of each synchronous call. Their addresses, element
counts, and row strides cross a small C ABI through `ctypes`. Python validates
the arrays' dtype, contiguity, alignment, and non-null storage; Mojo validates
the complete buffer contract before reconstructing pointers and returns a
checked status. Mojo writes into caller-owned one-byte Boolean or `float64`
output buffers. No allocation crosses the ABI.

One Mojo compilation unit contains mean thresholding, horizontal and vertical
differences, a separable low-frequency DCT-II, and Haar low-pass block
reduction. Haar block sums run over four partial `float64` accumulators, one
per pixel lane, because a single running sum serialises on floating-point add
latency. The sums are formed within one block in a fixed order, so the Haar
coefficient is still exactly the same value (and the same hash) as before.
Average thresholding and both difference directions use
`float64` SIMD loads and reductions, also with scalar tails. The DCT splits into
two range entry points, `mih_phash_dct_first` and `mih_phash_dct_second`, which
the Python shim runs over a `ThreadPoolExecutor`; the second pass reads every
projected row, so the two passes are separated by a barrier. Measured on this
box, eight workers score 0.16x at 8,192 row-elements and 4.3x at 32,768, so
independent frequency rows only split at 16,384 row-elements or more. The split
is bit-for-bit identical to the single-core `mih_phash_dct` entry.
Python uses one allocation for DCT scratch and coefficients, applies the median
threshold, and wraps the row-major Boolean matrix in `ImageHash`. When integer Haar
coefficients tie at the median, PyWavelets resolves the floating-point tie so
the result remains bit-for-bit compatible with ImageHash.

The Haar accumulation is the only summation order that changed in the unrolling
pass, and it cannot change any output: the four partial sums are recombined
inside a single pixel block, and each total is a sum of at most 4,096 integers,
which `float64` represents exactly. Every other kernel is untouched, so the
hashes are bit-identical to the previous revision.

## License

MIT
