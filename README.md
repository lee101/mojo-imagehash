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
Lanczos resizing, on the same 2048 by 1536 RGB image. Lower is better. The
table is real output from `pixi run bench` on 2026-08-24; the machine-wide
benchmark lock was active.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, Linux x86-64.

| algorithm | mojo-imagehash | ImageHash 4.3.2 | speedup |
|---|---:|---:|---:|
| average_hash (hash_size=32) | 11.926 ms | 11.257 ms | 0.94x |
| dhash (hash_size=32) | 11.156 ms | 11.673 ms | 1.05x |
| phash (hash_size=16) | 12.179 ms | 15.466 ms | 1.27x |
| whash (hash_size=16) | 27.629 ms | 109.158 ms | 3.95x |

Pillow preprocessing and FFI overhead dominate the small average-hash and
dHash kernels. In this run average hash was effectively tied, dHash was 1.05x
faster, pHash was 1.27x faster, and Haar wHash remained 3.95x faster.

No GPU path is included. Average hash, dHash, and Haar low-pass reduction have
too little arithmetic intensity, while the practical pHash DCT sizes are
too short to justify adding device transfer and launch overhead.

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
reduction. Average thresholding and both difference directions use
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

## License

MIT
