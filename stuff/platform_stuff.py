"""Platform detection (Jetson), GPU count, CUDA cache clear, and pthread TSS key stats."""
import os
import platform
import ctypes
from ctypes import c_uint, byref

def is_jetson():
    return platform.machine() == "aarch64"

def platform_num_gpus():
    if is_jetson():
        return 1
    else:
        import torch
        return torch.cuda.device_count()

def platform_clear_caches():
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache() # try hard not to run out of GPU memory

def platform_tss_key_stats():
    # 1) ceiling
    max_keys = os.sysconf("SC_THREAD_KEYS_MAX")

    # 2) bind the pthread calls
    libc = ctypes.CDLL(None)
    pthread_key_create = libc.pthread_key_create
    pthread_key_create.argtypes = [ctypes.POINTER(c_uint), ctypes.c_void_p]
    pthread_key_create.restype  = ctypes.c_int

    pthread_key_delete = libc.pthread_key_delete
    pthread_key_delete.argtypes = [c_uint]
    pthread_key_delete.restype  = ctypes.c_int

    # 3) brute‑force allocation
    allocated = []
    for _ in range(max_keys):
        key = c_uint()
        if pthread_key_create(byref(key), None) != 0:
            break
        allocated.append(key.value)

    # 4) report
    used_at_startup = max_keys - len(allocated)
    print(f"TSS key ceiling:    {max_keys}")
    print(f"Keys already in use: {used_at_startup}")
    print(f"Keys we just drank:  {len(allocated)}")  # should be ceiling - used_at_startup

    # 5) clean up our test keys
    for k in allocated:
        pthread_key_delete(k)
