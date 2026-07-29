"""Perceptual image hashes accelerated by Mojo."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ._lib import buffer, checked_call

__version__ = "0.1.0"


def _binary_array_to_hex(array):
    bit_string = "".join(str(bit) for bit in 1 * array.flatten())
    width = int(np.ceil(len(bit_string) / 4))
    return f"{int(bit_string, 2):0>{width}x}"


class ImageHash:
    """Hash encapsulation compatible with the upstream imagehash package."""

    def __init__(self, binary_array):
        self.hash = binary_array

    def __str__(self):
        return _binary_array_to_hex(self.hash.flatten())

    def __repr__(self):
        return repr(self.hash)

    def __sub__(self, other):
        if other is None:
            raise TypeError("Other hash must not be None.")
        if self.hash.size != other.hash.size:
            raise TypeError(
                "ImageHashes must be of the same shape.",
                self.hash.shape,
                other.hash.shape,
            )
        return np.count_nonzero(self.hash.flatten() != other.hash.flatten())

    def __eq__(self, other):
        if other is None:
            return False
        return np.array_equal(self.hash.flatten(), other.hash.flatten())

    def __ne__(self, other):
        if other is None:
            return False
        return not np.array_equal(self.hash.flatten(), other.hash.flatten())

    def __hash__(self):
        return sum(2 ** (i % 8) for i, value in enumerate(self.hash.flatten()) if value)

    def __len__(self):
        return self.hash.size


def _luma(image, size: tuple[int, int]) -> np.ndarray:
    resized = image.convert("L").resize(size, Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(resized), dtype=np.uint8)


def _empty_hash(hash_size: int) -> np.ndarray:
    return np.empty((hash_size, hash_size), dtype=np.bool_)


def average_hash(image, hash_size=8, mean=np.mean):
    """Compute an average hash with the upstream imagehash signature."""
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")
    pixels = _luma(image, (hash_size, hash_size))
    bits = _empty_hash(hash_size)
    pixels_addr, pixels_len, _ = buffer(pixels, np.uint8)
    bits_addr, bits_len, _ = buffer(bits, np.bool_)
    if mean is np.mean:
        checked_call(
            "mih_average_hash",
            pixels_addr,
            pixels_len,
            pixels.size,
            bits_addr,
            bits_len,
        )
    else:
        threshold = float(mean(pixels))
        checked_call(
            "mih_threshold",
            pixels_addr,
            pixels_len,
            pixels.size,
            threshold,
            bits_addr,
            bits_len,
        )
    return ImageHash(bits)


def phash(image, hash_size=8, highfreq_factor=4):
    """Compute a DCT perceptual hash."""
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")
    if highfreq_factor < 1:
        raise ValueError("High frequency factor must be greater than or equal to 1")
    image_size = hash_size * highfreq_factor
    pixels = _luma(image, (image_size, image_size))
    work_size = 2 * hash_size * image_size
    storage = np.empty(work_size + hash_size * hash_size, dtype=np.float64)
    work = storage[:work_size]
    coeff = storage[work_size:].reshape(hash_size, hash_size)
    pixels_addr, pixels_len, pixels_stride = buffer(pixels, np.uint8)
    work_addr, work_len, _ = buffer(work, np.float64)
    coeff_addr, coeff_len, _ = buffer(coeff, np.float64)
    checked_call(
        "mih_phash_dct",
        pixels_addr,
        pixels_len,
        pixels_stride,
        image_size,
        hash_size,
        work_addr,
        work_len,
        coeff_addr,
        coeff_len,
    )
    return ImageHash(coeff > np.median(coeff))


def dhash(image, hash_size=8):
    """Compute a horizontal difference hash."""
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")
    pixels = _luma(image, (hash_size + 1, hash_size))
    bits = _empty_hash(hash_size)
    pixels_addr, pixels_len, pixels_stride = buffer(pixels, np.uint8)
    bits_addr, bits_len, _ = buffer(bits, np.bool_)
    checked_call(
        "mih_difference_hash",
        pixels_addr,
        pixels_len,
        pixels_stride,
        hash_size,
        bits_addr,
        bits_len,
    )
    return ImageHash(bits)


def dhash_vertical(image, hash_size=8):
    """Compute a vertical difference hash."""
    pixels = _luma(image, (hash_size, hash_size + 1))
    bits = _empty_hash(hash_size)
    pixels_addr, pixels_len, pixels_stride = buffer(pixels, np.uint8)
    bits_addr, bits_len, _ = buffer(bits, np.bool_)
    checked_call(
        "mih_vertical_difference_hash",
        pixels_addr,
        pixels_len,
        pixels_stride,
        hash_size,
        bits_addr,
        bits_len,
    )
    return ImageHash(bits)


def whash(
    image,
    hash_size=8,
    image_scale=None,
    mode="haar",
    remove_max_haar_ll=True,
):
    """Compute a wavelet hash, accelerating the default Haar path in Mojo."""
    if image_scale is not None:
        assert image_scale & (image_scale - 1) == 0, "image_scale is not power of 2"
    else:
        image_natural_scale = 2 ** int(np.log2(min(image.size)))
        image_scale = max(image_natural_scale, hash_size)

    ll_max_level = int(np.log2(image_scale))
    level = int(np.log2(hash_size))
    assert hash_size & (hash_size - 1) == 0, "hash_size is not power of 2"
    assert level <= ll_max_level, "hash_size in a wrong range"

    pixels = _luma(image, (image_scale, image_scale))
    if mode == "haar":
        coeff = np.empty((hash_size, hash_size), dtype=np.float64)
        pixels_addr, pixels_len, pixels_stride = buffer(pixels, np.uint8)
        coeff_addr, coeff_len, _ = buffer(coeff, np.float64)
        checked_call(
            "mih_haar_lowpass",
            pixels_addr,
            pixels_len,
            pixels_stride,
            image_scale,
            hash_size,
            coeff_addr,
            coeff_len,
        )
        median = np.median(coeff)
        if not np.any(coeff == median):
            return ImageHash(coeff > median)

    import pywt

    values = pixels / 255.0
    if remove_max_haar_ll:
        coeffs = list(pywt.wavedec2(values, "haar", level=ll_max_level))
        coeffs[0] *= 0
        values = pywt.waverec2(coeffs, "haar")
    dwt_level = ll_max_level - level
    dwt_low = pywt.wavedec2(values, mode, level=dwt_level)[0]
    return ImageHash(dwt_low > np.median(dwt_low))


def hex_to_hash(hexstr):
    hash_size = int(np.sqrt(len(hexstr) * 4))
    binary_array = f"{int(hexstr, 16):0>{hash_size * hash_size}b}"
    bit_rows = [
        binary_array[i : i + hash_size]
        for i in range(0, len(binary_array), hash_size)
    ]
    hash_array = np.array([[bool(int(digit)) for digit in row] for row in bit_rows])
    return ImageHash(hash_array)


def old_hex_to_hash(hexstr, hash_size=8):
    array = []
    count = hash_size * (hash_size // 4)
    if len(hexstr) != count:
        raise ValueError(f"Expected hex string size of {count}.")
    for i in range(count // 2):
        value = int("0x" + hexstr[i * 2 : i * 2 + 2], 16)
        array.append([value & 2**bit > 0 for bit in range(8)])
    return ImageHash(np.array(array))


ahash = average_hash
difference_hash = dhash
vertical_difference_hash = dhash_vertical
wavelet_hash = whash

__all__ = [
    "ImageHash",
    "ahash",
    "average_hash",
    "dhash",
    "dhash_vertical",
    "difference_hash",
    "hex_to_hash",
    "old_hex_to_hash",
    "phash",
    "vertical_difference_hash",
    "wavelet_hash",
    "whash",
]
