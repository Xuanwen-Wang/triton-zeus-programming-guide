# SIM_PARAMS: act:ptr:in:[1,1,"M","K"]:fp8 weight:wgt_ptr:in:[1,"N","K"]:int4:uscale_128_aligned_65536_bk_128 output:ptr:out:[1,1,"M","N"]:fp32 M:i32 K:i32 N:i32
# SIM_GOLDEN: matmul_uscale
"""One-core FP8 E4M3FN by INT4 projection with BF16 group-128 UScale.

The worked geometry is M=4, K=128, N=256. Weights use one 64 KiB
aligned slot containing row-paired INT4 payload and its UScale page.
Default execution computes a CPU numerical reference, not a device launch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def quantized_matmul(
    act, weight, output, M, K, N,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr, UGROUP: tl.constexpr,
):
    tl.static_assert(BLOCK_K == 128)
    tl.static_assert(BLOCK_N == 256)
    address = tl.load(weight + tl.program_id(0))
    base = address.to(tl.pointer_type(tl.float8e4nv))
    a_desc = tl.make_block_ptr(
        act, (M, K), (K, 1), (0, 0),
        (BLOCK_M, BLOCK_K), (1, 0),
    )
    w_desc = tl.make_block_ptr(
        base, (N, K), (K, 1), (0, 0),
        (BLOCK_N, BLOCK_K), (1, 0),
    )
    a = tl.load(a_desc)
    w = tl.load(
        w_desc, memory_type="weight", weight_dtype="int4",
        ugroup=UGROUP,
    )
    c = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
    out_desc = tl.make_block_ptr(
        output, (M, N), (N, 1), (0, 0),
        (BLOCK_M, BLOCK_N), (1, 0),
    )
    tl.store(out_desc, c)


signature = {
    "act": "*fp8e4nv", "weight": "*i64", "output": "*fp32",
    "M": "i32", "K": "i32", "N": "i32",
}
constants = {
    "BLOCK_M": 4, "BLOCK_K": 128,
    "BLOCK_N": 256, "UGROUP": 128,
}


def reference(a_fp8, w_int4, scale_bf16):
    """Compute from the actual quantized inputs and per-channel group scale."""
    if a_fp8.shape != (4, 128) or w_int4.shape != (256, 128):
        raise ValueError("This example fixes M=4, K=128, N=256")
    if scale_bf16.shape != (256, 1):
        raise ValueError("One BF16 scale per output channel is required")
    if w_int4.min() < -8 or w_int4.max() > 7:
        raise ValueError("Signed INT4 codes must be in [-8, 7]")
    partial = a_fp8.float() @ w_int4.float().T
    return partial * scale_bf16[:, 0].float()[None, :]


def main():
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(4, 128, generator=generator).to(torch.float8_e4m3fn)
    w = torch.randint(-8, 8, (256, 128), generator=generator)
    scale = (torch.rand(256, 1, generator=generator) + 0.25).to(torch.bfloat16)
    y = reference(a, w, scale)
    assert torch.isfinite(y).all()
    print(f"CPU reference: shape={tuple(y.shape)}, dtype={y.dtype}")
    print("INT4 payload=16384 B; valid UScale=512 B; slot=65536 B")


if __name__ == "__main__":
    main()
