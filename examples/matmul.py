# SIM_PARAMS: act:ptr:in:[1,1,"M","K"]:bf16 weight:wgt_ptr:in:[1,"N","K"]:bf16 output:ptr:out:[1,1,"M","N"]:bf16 M:i32 K:i32 N:i32
# SIM_GOLDEN: matmul
"""Single-core BF16 Matmul with an FP32 accumulator.

The right operand is stored in LDG through a per-core address table.
The default command computes a CPU reference. The guide's Matmul example
shows how to prepare LocalMem and launch the kernel with the matching SDK.
"""

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bf16(
    a_ptr, w_ptr_list, c_ptr, M, K, N,
    CORE_NUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # This teaching implementation deliberately uses one physical core.
    tl.static_assert(CORE_NUM == 1)
    core = tl.program_id(0)
    address = tl.load(w_ptr_list + core)
    wp = address.to(tl.pointer_type(tl.bfloat16))
    ap = tl.make_block_ptr(
        a_ptr, (M, K), (K, 1), (0, 0),
        (BLOCK_M, BLOCK_K), (1, 0),
    )
    bp = tl.make_block_ptr(
        wp, (N, K), (K, 1), (0, 0),
        (BLOCK_N, BLOCK_K), (1, 0),
    )
    cp = tl.make_block_ptr(
        c_ptr, (M, N), (N, 1), (0, 0),
        (BLOCK_M, BLOCK_N), (1, 0),
    )
    for mi in range(tl.cdiv(M, BLOCK_M)):
        for ni in range(tl.cdiv(N, BLOCK_N)):
            m = mi * BLOCK_M
            n = ni * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for ki in range(tl.cdiv(K, BLOCK_K)):
                k = ki * BLOCK_K
                a = tl.load(
                    tl.advance(ap, (m, k)),
                    boundary_check=(0, 1), padding_option="zero",
                )
                b_nk = tl.load(
                    tl.advance(bp, (n, k)), memory_type="weight",
                    boundary_check=(0, 1), padding_option="zero",
                )
                acc += tl.dot(a, tl.trans(b_nk))
            result = acc.to(tl.bfloat16)
            tl.store(
                tl.advance(cp, (m, n)), result, boundary_check=(0, 1),
            )


signature = {
    "a_ptr": "*bf16", "w_ptr_list": "*i64", "c_ptr": "*bf16",
    "M": "i32", "K": "i32", "N": "i32",
}
constants = {"CORE_NUM": 1, "BLOCK_M": 128, "BLOCK_K": 128,
             "BLOCK_N": 128}


def reference(a, weight_nk):
    """Physical W[N,K] represents logical B[K,N]."""
    return (a.float() @ weight_nk.float().T).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=384)
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        parser.error("M, N and K must be positive")
    generator = torch.Generator().manual_seed(20261124)
    a = torch.randn(args.m, args.k, generator=generator)
    w = torch.randn(args.n, args.k, generator=generator)
    output = reference(a.to(torch.bfloat16), w.to(torch.bfloat16))
    print(f"CPU reference shape={tuple(output.shape)}")
    print(f"finite={bool(torch.isfinite(output).all())}")
    print("CPU reference only; the Zeus kernel was not executed")


if __name__ == "__main__":
    main()
