# SIM_PARAMS: new_token:ptr:in:[1,128]:bf16 cache_page:wgt_ptr:inout:[1,16,128,64]:bf16:raw token_offset:i32
# SIM_GOLDEN: append_kcache_token_bf16
# SIM_RANGE: token_offset=1000:1000
"""Append a token to one resolved KCache payload page.

The launch uses one core. The pointer list contains a payload address, not a
hardware KV index table. This worked case fixes BF16, head dimension 128,
and a physical page size of 1024. Running this file prints host-side offsets;
compilation and functional simulation are provided by validate_examples.py.
"""

import argparse
import json
import numpy as np
import triton
import triton.language as tl


@triton.jit
def append_kcache_token(
    new_token_ptr, cache_page_ptr_list, token_offset,
    PAGE_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    tl.static_assert(PAGE_SIZE == 1024)
    tl.static_assert(HEAD_DIM == 128)
    core_id = tl.program_id(0)
    address = tl.load(cache_page_ptr_list + core_id)
    payload = address.to(tl.pointer_type(tl.bfloat16))
    src = tl.make_block_ptr(
        base=new_token_ptr,
        shape=(1, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(1, HEAD_DIM),
        order=(1, 0),
    )
    token = tl.load(src)
    dst = tl.make_block_ptr(
        base=payload,
        shape=(PAGE_SIZE, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(token_offset, 0),
        block_shape=(1, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(
        dst, token, memory_type="kcache",
        store_mode="lw", blocksize=PAGE_SIZE,
    )


signature = {
    "new_token_ptr": "*bf16",
    "cache_page_ptr_list": "*i64",
    "token_offset": "i32",
}
constants = {"PAGE_SIZE": 1024, "HEAD_DIM": 128}


def reference_byte_offsets(token_offset):
    if not 0 <= token_offset < 1024:
        raise ValueError("token_offset must be in [0, 1024)")
    head = np.arange(128, dtype=np.int64)
    group, lane = divmod(token_offset, 128)
    return (((group * 2 + head // 64) * 128 + lane) * 64 + head % 64) * 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-offset", type=int, default=1000)
    args = parser.parse_args()
    offsets = reference_byte_offsets(args.token_offset)
    assert len(np.unique(offsets)) == 128
    assert int(offsets.min()) >= 0
    assert int(offsets.max()) + 2 <= 262144
    starts = np.r_[True, np.diff(offsets) != 2]
    segments = offsets[starts].tolist()
    print(json.dumps({
        "mode": "kcache", "backend": "host_address_reference",
        "token_offset": args.token_offset,
        "page_bytes": 262144, "updated_bytes": 256,
        "segment_starts": segments,
        "segment_bytes": 256 // len(segments),
    }, indent=2))


if __name__ == "__main__":
    main()
