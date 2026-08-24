"""Parity with ImageHash 4.x on identical PIL images."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import imagehash as upstream
import mojo_imagehash as mojo
from mojo_imagehash._lib import buffer, lib


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(20260729)
    noise = rng.integers(0, 256, (91, 137, 3), dtype=np.uint8)
    y, x = np.mgrid[:96, :128]
    gradient = np.stack(
        ((2 * x) % 256, (3 * y) % 256, (x + y) % 256), axis=-1
    ).astype(np.uint8)
    checker = (((x // 8 + y // 8) & 1) * 255).astype(np.uint8)
    return [
        Image.fromarray(noise, "RGB"),
        Image.fromarray(gradient, "RGB"),
        Image.fromarray(checker, "L"),
        Image.new("RGB", (73, 55), (12, 140, 231)),
    ]


@pytest.mark.parametrize("hash_size", [4, 5, 8, 16])
def test_average_hash_parity(images, hash_size):
    for image in images:
        assert mojo.average_hash(image, hash_size) == upstream.average_hash(
            image, hash_size
        )


def test_average_hash_custom_mean(images):
    for image in images:
        assert mojo.average_hash(
            image, hash_size=5, mean=np.median
        ) == upstream.average_hash(image, hash_size=5, mean=np.median)


@pytest.mark.parametrize("hash_size", [4, 5, 8, 16])
def test_dhash_parity(images, hash_size):
    for image in images:
        assert mojo.dhash(image, hash_size) == upstream.dhash(image, hash_size)


@pytest.mark.parametrize("hash_size", [4, 5, 8, 16])
def test_vertical_dhash_parity(images, hash_size):
    for image in images:
        assert mojo.dhash_vertical(image, hash_size) == upstream.dhash_vertical(
            image, hash_size
        )


@pytest.mark.parametrize("hash_size,highfreq_factor", [(4, 1), (8, 2), (8, 4), (16, 4)])
def test_phash_parity(images, hash_size, highfreq_factor):
    for image in images:
        assert mojo.phash(image, hash_size, highfreq_factor) == upstream.phash(
            image, hash_size, highfreq_factor
        )


@pytest.mark.parametrize(
    "hash_size,highfreq_factor",
    [(5, 1), (5, 3), (32, 15), (32, 16)],
)
def test_phash_simd_tail_and_parallel_parity(images, hash_size, highfreq_factor):
    for image in images[:2]:
        assert mojo.phash(image, hash_size, highfreq_factor) == upstream.phash(
            image, hash_size, highfreq_factor
        )


@pytest.mark.parametrize(
    "hash_size,image_scale",
    [(4, 16), (8, 16), (8, 64), (16, 16), (16, 64)],
)
def test_whash_haar_parity(images, hash_size, image_scale):
    for image in images:
        assert mojo.whash(image, hash_size, image_scale) == upstream.whash(
            image, hash_size, image_scale
        )


@pytest.mark.parametrize("remove_low", [False, True])
def test_whash_db4_parity(images, remove_low):
    for image in images[:2]:
        assert mojo.whash(
            image,
            hash_size=8,
            image_scale=64,
            mode="db4",
            remove_max_haar_ll=remove_low,
        ) == upstream.whash(
            image,
            hash_size=8,
            image_scale=64,
            mode="db4",
            remove_max_haar_ll=remove_low,
        )


def test_whash_natural_scale_parity(images):
    for image in images:
        assert mojo.whash(image) == upstream.whash(image)


@pytest.mark.parametrize("function", [mojo.average_hash, mojo.phash, mojo.dhash])
def test_hash_size_validation(function, images):
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        function(images[0], hash_size=1)


def test_whash_validation(images):
    with pytest.raises(AssertionError, match="image_scale is not power of 2"):
        mojo.whash(images[0], image_scale=63)
    with pytest.raises(AssertionError, match="hash_size is not power of 2"):
        mojo.whash(images[0], hash_size=7, image_scale=64)
    with pytest.raises(AssertionError, match="hash_size in a wrong range"):
        mojo.whash(images[0], hash_size=128, image_scale=64)


def test_imagehash_protocol(images):
    ours = mojo.phash(images[0])
    theirs = upstream.phash(images[0])
    assert str(ours) == str(theirs)
    assert repr(ours) == repr(theirs)
    assert len(ours) == len(theirs) == 64
    assert hash(ours) == hash(theirs)
    assert ours - mojo.phash(images[1]) == theirs - upstream.phash(images[1])
    assert not (ours == None)  # noqa: E711
    assert (ours != None) is False  # noqa: E711
    with pytest.raises(TypeError, match="must not be None"):
        _ = ours - None
    with pytest.raises(TypeError, match="same shape"):
        _ = ours - mojo.phash(images[1], hash_size=4)


@pytest.mark.parametrize(
    "text",
    ["0000000000000000", "0123456789abcdef", "ffffffffffffffff"],
)
def test_hex_round_trip(text):
    ours = mojo.hex_to_hash(text)
    theirs = upstream.hex_to_hash(text)
    assert ours == theirs
    assert str(ours) == text


def test_old_hex_parser_parity():
    text = "0123456789abcdef"
    assert mojo.old_hex_to_hash(text) == upstream.old_hex_to_hash(text)
    with pytest.raises(ValueError, match="Expected hex string size"):
        mojo.old_hex_to_hash("abc")


def test_documented_aliases(images):
    image = images[0]
    assert mojo.ahash(image) == mojo.average_hash(image)
    assert mojo.difference_hash(image) == mojo.dhash(image)
    assert mojo.vertical_difference_hash(image) == mojo.dhash_vertical(image)
    assert mojo.wavelet_hash(image) == mojo.whash(image)


def test_ffi_buffer_contract_validation():
    with pytest.raises(TypeError, match="dtype"):
        buffer(np.empty((2, 2), dtype=np.float32), np.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        buffer(np.empty((2, 3), dtype=np.uint8)[:, ::2], np.uint8)
    with pytest.raises(ValueError, match="non-empty"):
        buffer(np.empty((0, 0), dtype=np.uint8), np.uint8)


def test_ffi_kernel_rejects_null_and_wrong_lengths():
    function = lib().mih_average_hash
    assert function(0, 4, 4, 0, 4) != 0
    pixels = np.zeros((2, 2), dtype=np.uint8)
    bits = np.empty((2, 2), dtype=np.bool_)
    assert function(pixels.ctypes.data, 3, 4, bits.ctypes.data, 4) != 0


def test_upstream_edge_case_parity(images):
    image = images[0]
    assert mojo.dhash_vertical(image, 1) == upstream.dhash_vertical(image, 1)
    assert mojo.whash(image, hash_size=1, image_scale=8) == upstream.whash(
        image, hash_size=1, image_scale=8
    )
    with pytest.raises(ValueError):
        mojo.phash(image, highfreq_factor=0)
