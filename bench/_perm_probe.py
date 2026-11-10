import mmap, sys
n = 1 << 20
buf = bytearray(n)
with open("/proc/self/pagemap", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    import struct
    import ctypes
    p = ctypes.addressof((ctypes.c_char * n).from_buffer(buf))
    print("pagemap size:", len(mm))
    print("cpu time:", open("/proc/self/stat").read().split()[13])
