"""Numeric kernels for perceptual image hashes."""

from std.math import cos
from std.sys.info import simd_width_of as simdwidthof

comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


@export("mih_average_hash")
def mih_average_hash(
    pixels_addr: Int, pixels_len: Int, count: Int, bits_addr: Int, bits_len: Int
) abi("C") -> Int:
    if pixels_addr == 0 or bits_addr == 0 or count <= 0:
        return 1
    if pixels_len != count or bits_len != count:
        return 1
    var pixels = U8Ptr(unsafe_from_address=pixels_addr)
    var bits = U8Ptr(unsafe_from_address=bits_addr)
    var total = Int64(0)
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    var totals = SIMD[DType.int64, W](0)
    while i + W <= count:
        totals += pixels.unsafe_load[width=W](i).cast[DType.int64]()
        i += W
    total += totals.reduce_add()
    while i < count:
        total += Int64(pixels[i])
        i += 1
    var average = Float64(total) / Float64(count)
    i = 0
    while i + W <= count:
        bits.store(
            i,
            pixels.unsafe_load[width=W](i).cast[DType.float64]().gt(average).cast[
                DType.uint8
            ](),
        )
        i += W
    while i < count:
        bits[i] = UInt8(1) if Float64(pixels[i]) > average else UInt8(0)
        i += 1
    return 0


@export("mih_threshold")
def mih_threshold(
    pixels_addr: Int,
    pixels_len: Int,
    count: Int,
    threshold: Float64,
    bits_addr: Int,
    bits_len: Int,
) abi("C") -> Int:
    if pixels_addr == 0 or bits_addr == 0 or count <= 0:
        return 1
    if pixels_len != count or bits_len != count:
        return 1
    var pixels = U8Ptr(unsafe_from_address=pixels_addr)
    var bits = U8Ptr(unsafe_from_address=bits_addr)
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    while i + W <= count:
        bits.store(
            i,
            pixels.unsafe_load[width=W](i).cast[DType.float64]().gt(threshold).cast[
                DType.uint8
            ](),
        )
        i += W
    while i < count:
        bits[i] = UInt8(1) if Float64(pixels[i]) > threshold else UInt8(0)
        i += 1
    return 0


@export("mih_difference_hash")
def mih_difference_hash(
    pixels_addr: Int,
    pixels_len: Int,
    row_stride: Int,
    hash_size: Int,
    bits_addr: Int,
    bits_len: Int,
) abi("C") -> Int:
    if pixels_addr == 0 or bits_addr == 0 or hash_size < 2:
        return 1
    if hash_size > 1_000_000 or row_stride != hash_size + 1:
        return 1
    if pixels_len != row_stride * hash_size:
        return 1
    if bits_len != hash_size * hash_size:
        return 1
    var pixels = U8Ptr(unsafe_from_address=pixels_addr)
    var bits = U8Ptr(unsafe_from_address=bits_addr)
    comptime W = simdwidthof[DType.float64]()
    for y in range(hash_size):
        var x = 0
        while x + W <= hash_size:
            var i = y * row_stride + x
            bits.store(
                y * hash_size + x,
                pixels.unsafe_load[width=W](i + 1).gt(
                    pixels.unsafe_load[width=W](i)
                ).cast[DType.uint8](),
            )
            x += W
        while x < hash_size:
            var i = y * row_stride + x
            bits[y * hash_size + x] = UInt8(1) if pixels[i + 1] > pixels[
                i
            ] else UInt8(0)
            x += 1
    return 0


@export("mih_vertical_difference_hash")
def mih_vertical_difference_hash(
    pixels_addr: Int,
    pixels_len: Int,
    row_stride: Int,
    hash_size: Int,
    bits_addr: Int,
    bits_len: Int,
) abi("C") -> Int:
    if pixels_addr == 0 or bits_addr == 0 or hash_size < 1:
        return 1
    if hash_size > 1_000_000 or row_stride != hash_size:
        return 1
    if pixels_len != row_stride * (hash_size + 1):
        return 1
    if bits_len != hash_size * hash_size:
        return 1
    var pixels = U8Ptr(unsafe_from_address=pixels_addr)
    var bits = U8Ptr(unsafe_from_address=bits_addr)
    comptime W = simdwidthof[DType.float64]()
    for y in range(hash_size):
        var x = 0
        while x + W <= hash_size:
            var i = y * row_stride + x
            bits.store(
                y * hash_size + x,
                pixels.unsafe_load[width=W](i + row_stride).gt(
                    pixels.unsafe_load[width=W](i)
                ).cast[DType.uint8](),
            )
            x += W
        while x < hash_size:
            var i = y * row_stride + x
            bits[y * hash_size + x] = UInt8(1) if pixels[
                i + row_stride
            ] > pixels[i] else UInt8(
                0
            )
            x += 1
    return 0


@export("mih_phash_dct")
def mih_phash_dct(
    pixels_addr: Int,
    pixels_len: Int,
    row_stride: Int,
    image_size: Int,
    hash_size: Int,
    work_addr: Int,
    work_len: Int,
    coeff_addr: Int,
    coeff_len: Int,
) abi("C") -> Int:
    if pixels_addr == 0 or work_addr == 0 or coeff_addr == 0:
        return 1
    if image_size < 2 or hash_size < 2 or hash_size > image_size:
        return 1
    if image_size > 1_000_000 or row_stride != image_size:
        return 1
    if pixels_len != image_size * image_size:
        return 1
    if work_len != 2 * hash_size * image_size:
        return 1
    if coeff_len != hash_size * hash_size:
        return 1
    var pixels = U8Ptr(unsafe_from_address=pixels_addr)
    var work = F64Ptr(unsafe_from_address=work_addr)
    var coeff = F64Ptr(unsafe_from_address=coeff_addr)
    var temp_offset = hash_size * image_size

    dct_first_rows(
        pixels, work, image_size, hash_size, temp_offset, 0, hash_size
    )
    dct_second_rows(
        work, coeff, image_size, hash_size, temp_offset, 0, hash_size
    )
    var zero_scale = abs(coeff[0]) * 1.0e-13
    if zero_scale > 0.0:
        for i in range(hash_size * hash_size):
            if abs(coeff[i]) < zero_scale:
                coeff[i] = 0.0
    return 0


    """Project pixel rows [k0, k1); each span fills its own cosine rows."""
def dct_first_rows(
    pixels: U8Ptr,
    work: F64Ptr,
    image_size: Int,
    hash_size: Int,
    temp_offset: Int,
    k0: Int,
    k1: Int,
):
    comptime W = simdwidthof[DType.float64]()
    var angle_scale = 3.1415926535897932384626433832795 / (
        2.0 * Float64(image_size)
    )
    for k in range(k0, k1):
        for i in range(image_size):
            work[k * image_size + i] = cos(
                Float64(k * (2 * i + 1)) * angle_scale
            )
        var x = 0
        while x + W <= image_size:
            var acc = SIMD[DType.float64, W](0.0)
            for y in range(image_size):
                acc += (
                    pixels.unsafe_load[width=W](y * image_size + x).cast[
                        DType.float64
                    ]()
                    * work[k * image_size + y]
                )
            work.store(temp_offset + k * image_size + x, 2.0 * acc)
            x += W
        while x < image_size:
            var scalar_acc = 0.0
            for y in range(image_size):
                scalar_acc += (
                    Float64(pixels[y * image_size + x]) * work[k * image_size + y]
                )
            work[temp_offset + k * image_size + x] = 2.0 * scalar_acc
            x += 1


def dct_second_rows(
    work: F64Ptr,
    coeff: F64Ptr,
    image_size: Int,
    hash_size: Int,
    temp_offset: Int,
    k0: Int,
    k1: Int,
):
    comptime W = simdwidthof[DType.float64]()
    for k in range(k0, k1):
        for l in range(hash_size):
            var acc_vec = SIMD[DType.float64, W](0.0)
            var x = 0
            while x + W <= image_size:
                acc_vec += work.unsafe_load[width=W](
                    temp_offset + k * image_size + x
                ) * work.unsafe_load[width=W](l * image_size + x)
                x += W
            var acc = Float64(acc_vec.reduce_add())
            while x < image_size:
                acc += (
                    work[temp_offset + k * image_size + x]
                    * work[l * image_size + x]
                )
                x += 1
            coeff[k * hash_size + l] = 2.0 * acc


@export("mih_phash_dct_first")
def mih_phash_dct_first(
    pixels_addr: Int,
    image_size: Int,
    hash_size: Int,
    work_addr: Int,
    k0: Int,
    k1: Int,
) abi("C") -> Int:
    """Project pixel rows [k0, k1) of the DCT; the shim fans this out."""
    if pixels_addr == 0 or work_addr == 0 or k0 < 0 or k1 > hash_size or k0 >= k1:
        return 1
    dct_first_rows(
        U8Ptr(unsafe_from_address=pixels_addr),
        F64Ptr(unsafe_from_address=work_addr),
        image_size,
        hash_size,
        hash_size * image_size,
        k0,
        k1,
    )
    return 0


@export("mih_phash_dct_second")
def mih_phash_dct_second(
    work_addr: Int,
    image_size: Int,
    hash_size: Int,
    coeff_addr: Int,
    k0: Int,
    k1: Int,
) abi("C") -> Int:
    """Correlate DCT rows [k0, k1); runs after every first pass has landed."""
    if work_addr == 0 or coeff_addr == 0 or k0 < 0 or k1 > hash_size or k0 >= k1:
        return 1
    dct_second_rows(
        F64Ptr(unsafe_from_address=work_addr),
        F64Ptr(unsafe_from_address=coeff_addr),
        image_size,
        hash_size,
        hash_size * image_size,
        k0,
        k1,
    )
    return 0



@export("mih_haar_lowpass")
def mih_haar_lowpass(
    pixels_addr: Int,
    pixels_len: Int,
    row_stride: Int,
    image_scale: Int,
    hash_size: Int,
    coeff_addr: Int,
    coeff_len: Int,
) abi("C") -> Int:
    if pixels_addr == 0 or coeff_addr == 0:
        return 1
    if image_scale < 1 or hash_size < 1 or hash_size > image_scale:
        return 1
    if image_scale > 1_000_000 or image_scale % hash_size != 0:
        return 1
    if row_stride != image_scale or pixels_len != image_scale * image_scale:
        return 1
    if coeff_len != hash_size * hash_size:
        return 1
    var pixels = U8Ptr(unsafe_from_address=pixels_addr)
    var coeff = F64Ptr(unsafe_from_address=coeff_addr)
    var block = image_scale // hash_size
    for by in range(hash_size):
        for bx in range(hash_size):
            var acc = 0.0
            for y in range(by * block, (by + 1) * block):
                for x in range(bx * block, (bx + 1) * block):
                    acc += Float64(pixels[y * image_scale + x])
            coeff[by * hash_size + bx] = acc
    return 0
