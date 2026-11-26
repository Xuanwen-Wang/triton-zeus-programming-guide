# SIM_PARAMS: act:ptr:in:[1,1,"M","K"]:bf16 weight:wgt_ptr:in:[1,"N","K"]:bf16 output:ptr:out:[1,1,"M","N"]:bf16 M:i32 K:i32 N:i32
# SIM_GOLDEN: matmul
"""Single-core BF16 Matmul with an FP32 accumulator.

The right operand is stored in LDG through a per-core address table.
The default command computes a CPU reference. Use --run-device with the
matching SDK to verify the worked 256 x 256 x 384 shape.
"""

import argparse
import json

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
        base=a_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    bp = tl.make_block_ptr(
        base=wp,
        shape=(N, K),
        strides=(K, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    cp = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
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


def run_zeus(a_host, w_host, device):
    import torch_zeus  # noqa: F401

    m, k = a_host.shape
    n = w_host.shape[0]
    if (m, n, k) != (256, 256, 384):
        raise ValueError("the SDK example fixes M=256, N=256 and K=384")
    with torch.zeus.device(device):
        a = a_host.to(device)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        packed = torch.zeus.local_memory.from_tensor(
            w_host, kind="weight", Tr=1, Tc=1, core_id=0,
        )
        matmul_bf16[(1,)](
            a, packed, out, m, k, n, **constants, zeus_enable_jit=True,
        )
        torch.zeus.synchronize(out.device)
        return out.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=384)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--device", default="zeus:0")
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        parser.error("M, N and K must be positive")
    generator = torch.Generator().manual_seed(20261124)
    a = torch.randn(args.m, args.k, generator=generator).to(torch.bfloat16)
    w = torch.randn(args.n, args.k, generator=generator).to(torch.bfloat16)
    output = reference(a, w)
    if args.run_device:
        actual = run_zeus(a, w, args.device)
        torch.testing.assert_close(actual, output, rtol=1e-2, atol=1e-2)
        print(json.dumps({
            "execution": "zeus_runtime",
            "shape": list(actual.shape),
            "max_abs": (actual.float() - output.float()).abs().max().item(),
        }))
        return
    print(f"CPU reference shape={tuple(output.shape)}")
    print(f"finite={bool(torch.isfinite(output).all())}")
    print("CPU reference only; the Zeus kernel was not executed")


if __name__ == "__main__":
    main()
