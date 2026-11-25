#!/usr/bin/env python3
"""Check LDG coordinate mappings on the host, without a device runtime.

Integer labels track positions; they are not BF16 numeric conversions.
The modeled byte widths and layouts are the guide's explicit examples.
"""

import json
import numpy as np


def physical_offset(n, k, logical_k=128):
    """Compute BF16 Dense bytes independently from the 7D permutation."""
    mini = (n // 128) * ((logical_k + 63) // 64) + k // 64
    return mini * 16384 + (n % 128) * 128 + (k % 64) * 2


def pack_bf16_nk(logical, blocksize=128, mode="normal"):
    """Split a logical matrix, permute axes, and retain padded coordinates."""
    if logical.ndim != 2 or blocksize < 128:
        raise ValueError("Expected a matrix and blocksize >= 128")
    if blocksize & (blocksize - 1):
        raise ValueError("The demonstrated blocksize must be a power of two")
    if mode not in {"normal", "vcache"}:
        raise ValueError("Unknown layout mode")
    k1, c1, k0, c0 = (8, 1, 16, 64)
    if mode == "vcache":
        k1, c1, k0, c0 = (1, 8, 64, 16)
    rows, cols = logical.shape
    kg = (blocksize + k1 * k0 - 1) // (k1 * k0)
    kb = (rows + kg * k1 * k0 - 1) // (kg * k1 * k0)
    cg = (cols + c1 * c0 - 1) // (c1 * c0)
    padded = np.zeros((kb * kg * k1 * k0, cg * c1 * c0),
                      dtype=logical.dtype)
    padded[:rows, :cols] = logical
    split = padded.reshape(kb, kg, k1, k0, cg, c1, c0)
    return split.transpose(0, 1, 4, 2, 5, 3, 6).copy()


def unpack_bf16_nk(packed, logical_shape):
    """Undo the permutation before removing padding."""
    split = packed.transpose(0, 1, 3, 5, 2, 4, 6)
    rows = np.prod(split.shape[:4])
    cols = np.prod(split.shape[4:])
    return split.reshape(rows, cols)[
        :logical_shape[0], :logical_shape[1]
    ]


def pack_blocked(logical, block_n, block_k):
    """Pack independent fixed-size blocks, N blocks before K blocks."""
    rows, cols = logical.shape
    chunks = []
    for n0 in range(0, rows, block_n):
        for k0 in range(0, cols, block_k):
            part = logical[n0:n0 + block_n, k0:k0 + block_k]
            block = np.zeros((block_n, block_k), dtype=logical.dtype)
            block[:part.shape[0], :part.shape[1]] = part
            chunks.append(pack_bf16_nk(block).ravel())
    return np.concatenate(chunks)


def pack_uscale_labels(logical):
    """Model BF16 scale lanes and 16 KiB pages using reshape/permutation."""
    rows, groups = logical.shape
    bands = (rows + 127) // 128
    padded = np.zeros((bands * 128, groups), dtype=logical.dtype)
    padded[:rows] = logical
    lanes = padded.reshape(bands, 8, 16, groups)
    lanes = lanes.transpose(1, 0, 3, 2).reshape(8, -1)
    pages = (lanes.shape[1] + 1023) // 1024
    storage = np.zeros((8, pages * 1024), dtype=logical.dtype)
    storage[:, :lanes.shape[1]] = lanes
    return storage.reshape(8, pages, 1024).transpose(1, 0, 2)


def uscale_offset(n, group, groups):
    """Map one BF16 scale into a page, lane, slot, and row."""
    slot = (n // 128) * groups + group
    return ((slot // 64) * 16384 + ((n % 128) // 16) * 2048
            + (slot % 64) * 32 + (n % 16) * 2)


def check_dense(rows, cols):
    labels = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
    packed = pack_bf16_nk(labels)
    np.testing.assert_array_equal(
        unpack_bf16_nk(packed, labels.shape), labels
    )
    n, k = np.indices(labels.shape)
    offsets = physical_offset(n, k, cols)
    np.testing.assert_array_equal(packed.ravel()[offsets // 2], labels)
    if np.unique(offsets).size != labels.size:
        raise AssertionError("Two logical coordinates share a byte offset")
    return {"shape_nk": [rows, cols], "shape_7d": list(packed.shape),
            "coordinates": labels.size, "allocated_bytes": packed.size * 2}


def main():
    dense = [check_dense(*shape) for shape in
             [(128, 128), (256, 128), (256, 256), (130, 70)]]
    labels = np.arange(256 * 256).reshape(256, 256)
    blocked = pack_blocked(labels, 256, 128)
    n, k = np.indices(labels.shape)
    offsets = (k // 128) * 65536 + physical_offset(n, k % 128)
    np.testing.assert_array_equal(blocked[offsets // 2], labels)
    scales = []
    for rows, groups in [(256, 4), (256, 65), (130, 3)]:
        labels = np.arange(rows * groups).reshape(rows, groups)
        packed = pack_uscale_labels(labels)
        n, g = np.indices(labels.shape)
        off = uscale_offset(n, g, groups)
        np.testing.assert_array_equal(packed.ravel()[off // 2], labels)
        scales.append({"shape": [rows, groups],
                       "pages": packed.shape[0],
                       "coordinates": labels.size})
    # Dedicated K/V writes have token-major inputs and different permutations.
    token, dim = 1000, np.arange(128)
    page = np.arange(1024 * 128).reshape(1024, 128)
    k_offsets = physical_offset(token, dim)
    v_offsets = ((token // 64) * 16384 + (dim // 16) * 2048
                 + (token % 64) * 32 + (dim % 16) * 2)
    t_all, d_all = np.indices(page.shape)
    all_k = physical_offset(t_all, d_all)
    all_v = ((t_all // 64) * 16384 + (d_all // 16) * 2048
             + (t_all % 64) * 32 + (d_all % 16) * 2)
    for mode, offsets in [("normal", all_k), ("vcache", all_v)]:
        packed = pack_bf16_nk(page, blocksize=1024, mode=mode)
        np.testing.assert_array_equal(
            packed.ravel()[offsets // 2], page
        )
    # Two K pages are stored in reverse order after a 32 KiB index table.
    logical = np.arange(2048 * 128).reshape(2048, 128)
    index = np.zeros((2, 8, 512), dtype=np.int32)
    index[0, :, 0] = 18
    index[0, :, 1] = 2
    payload = np.concatenate([
        np.full(32768 // 2, -1),
        pack_bf16_nk(logical[1024:], 1024).ravel(),
        pack_bf16_nk(logical[:1024], 1024).ravel(),
    ])
    t_all, d_all = np.indices(logical.shape)
    addresses = (index[0, 0, t_all // 1024] * 16384
                 + physical_offset(t_all % 1024, d_all))
    np.testing.assert_array_equal(payload[addresses // 2], logical)
    for lane in range(8):
        np.testing.assert_array_equal(index[0, lane, :2], [18, 2])
    print(json.dumps({
        "execution": "host_layout_only", "status": "PASS",
        "dense": dense, "blocked_coordinates": 256 * 256,
        "uscale": scales,
        "dedicated_k_coordinates": page.size,
        "dedicated_v_coordinates": page.size,
        "indexed_k_coordinates": logical.size,
        "index_entries_and_copies": 2 * 8,
        "indexed_token1025_dim0_address": int(addresses[1025, 0]),
        "worked_element": {"B_kn": [70, 145], "W_nk": [145, 70],
                           "miniblock": 3, "lane": 1,
                           "byte_offset": physical_offset(145, 70)},
        "uscale_145_2_groups4": uscale_offset(145, 2, 4),
        "k_token1000_segments": k_offsets[::64].tolist(),
        "v_token1000_segments": v_offsets[::16].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
