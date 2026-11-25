#!/usr/bin/env python3
"""Inspect one BF16 LDG weight layout using host-side coordinate labels.

This program models packing order only. It does not allocate device LDG memory
or replace the SDK packer. Integer coordinate labels make every move visible.
"""

import json
import numpy as np


def physical_offset(n, k):
    """Return the byte offset for the documented 128 by 128 normal layout."""
    return (((k // 64) * 8 + n // 16) * 16 + n % 16) * 128 + 2 * (k % 64)


def pack_labels(logical):
    if logical.shape != (128, 128):
        raise ValueError("This worked example requires shape (128, 128)")
    split = logical.reshape(1, 1, 8, 16, 2, 1, 64)
    return split.transpose(0, 1, 4, 2, 5, 3, 6).copy()


def unpack_labels(packed):
    if packed.shape != (1, 1, 2, 8, 1, 16, 64):
        raise ValueError("Unexpected physical shape")
    return packed.transpose(0, 1, 3, 5, 2, 4, 6).reshape(128, 128)


def main():
    labels = np.arange(128 * 128, dtype=np.int32).reshape(128, 128)
    packed = pack_labels(labels)
    np.testing.assert_array_equal(unpack_labels(packed), labels)
    n, k = np.indices(labels.shape)
    offsets = physical_offset(n, k)
    np.testing.assert_array_equal(
        packed.ravel()[offsets // 2], labels
    )
    np.testing.assert_array_equal(
        np.sort(offsets.ravel()), np.arange(0, 32768, 2)
    )
    result = {
        "logical_shape": list(labels.shape),
        "physical_shape": list(packed.shape),
        "bf16_payload_bytes": int(labels.size * 2),
        "checked_coordinates": int(labels.size),
        "roundtrip": "PASS",
        "address_bijection": "PASS",
        "points": [
            {"n": n0, "k": k0, "logical_byte_offset": (n0 * 128 + k0) * 2,
             "physical_byte_offset": physical_offset(n0, k0)}
            for n0, k0 in [(0, 0), (0, 64), (16, 0), (127, 127)]
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
