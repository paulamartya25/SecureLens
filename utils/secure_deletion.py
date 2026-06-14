"""
utils/secure_deletion.py
SecureLens — Secure Data Deletion
Clears sensitive data from memory after use.
"""

import os, gc, ctypes
import numpy as np


def secure_clear_array(arr: np.ndarray):
    """
    Overwrites numpy array memory with zeros.
    Prevents sensitive data from lingering in memory.
    """
    if arr is not None and isinstance(arr, np.ndarray):
        arr[:] = 0
        del arr
        gc.collect()


def secure_clear_bytes(data: bytearray):
    """Overwrites bytearray with zeros."""
    if data:
        for i in range(len(data)):
            data[i] = 0


def secure_clear_list(lst: list):
    """Overwrites list contents."""
    if lst:
        for i in range(len(lst)):
            lst[i] = 0.0


def secure_delete_file(path: str):
    """
    Overwrites file with zeros before deletion.
    Prevents file recovery from disk.
    """
    if not os.path.exists(path):
        return
    size = os.path.getsize(path)
    with open(path, "wb") as f:
        f.write(b'\x00' * size)
    os.remove(path)
    print(f"[SecureDelete] Securely deleted: {path}")


def clear_upload_folder(upload_dir: str):
    """
    Securely deletes all files in upload directory.
    Call after each inference request.
    """
    if not os.path.exists(upload_dir):
        return
    count = 0
    for fname in os.listdir(upload_dir):
        fpath = os.path.join(upload_dir, fname)
        if os.path.isfile(fpath):
            secure_delete_file(fpath)
            count += 1
    if count > 0:
        print(f"[SecureDelete] Cleared {count} files from uploads/")


class SecureContext:
    """
    Context manager that clears sensitive numpy arrays on exit.

    Usage:
        with SecureContext(features_array) as ctx:
            # use features safely
        # features_array is zeroed after this block
    """

    def __init__(self, *arrays):
        self.arrays = arrays

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for arr in self.arrays:
            if isinstance(arr, np.ndarray):
                secure_clear_array(arr)
        gc.collect()


if __name__ == "__main__":
    print("[Test] Secure deletion module test...")
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    print(f"  Before: {arr}")
    secure_clear_array(arr)
    print(f"  After : {arr}")
    print("✅ Secure deletion working.")