"""Load the compiled Mojo image-hash kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "imagehash.mojo")
BUILD_SCRIPT = os.path.join(ROOT, "build", "build.sh")
LIB = os.environ.get("MOJO_IMAGEHASH_LIB") or os.path.join(
    ROOT, "dist", "libmojo-imagehash.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mih_average_hash": ([I, I, I, I, I], I),
    "mih_threshold": ([I, I, I, F, I, I], I),
    "mih_difference_hash": ([I, I, I, I, I, I], I),
    "mih_vertical_difference_hash": ([I, I, I, I, I, I], I),
    "mih_phash_dct": ([I, I, I, I, I, I, I, I, I], I),
    "mih_haar_lowpass": ([I, I, I, I, I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_IMAGEHASH_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_IMAGEHASH_LIB does not exist: {LIB}")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= max(os.path.getmtime(SRC), os.path.getmtime(BUILD_SCRIPT))
    ):
        return LIB
    proc = subprocess.run(
        ["bash", BUILD_SCRIPT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        initialize_cpu = getattr(
            _library, "KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", None
        )
        if initialize_cpu is not None:
            initialize_cpu.argtypes = []
            initialize_cpu.restype = ctypes.c_void_p
            if not initialize_cpu():
                raise RuntimeError("Mojo CPU runtime initialization failed")
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def buffer(array: np.ndarray, dtype: np.dtype) -> tuple[int, int, int]:
    """Return a checked C buffer description while the caller retains `array`."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"FFI buffer dtype must be {np.dtype(dtype)}, got {array.dtype}")
    if array.itemsize != np.dtype(dtype).itemsize:
        raise TypeError("FFI buffer item size does not match its declared dtype")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if not array.flags.aligned:
        raise ValueError("FFI buffers must be naturally aligned")
    address = int(array.ctypes.data)
    if address == 0 or array.size == 0:
        raise ValueError("FFI buffers must be non-empty and non-null")
    row_stride = array.strides[0] // array.itemsize if array.ndim >= 2 else array.size
    return address, int(array.size), int(row_stride)


def checked_call(name: str, *args) -> None:
    status = getattr(lib(), name)(*args)
    if status != 0:
        raise RuntimeError(f"{name} rejected an invalid buffer contract")
