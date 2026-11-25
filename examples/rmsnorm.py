"""RMSNorm with FP32 arithmetic and a BF16 output conversion.

The default command computes a CPU reference. Device execution is explicit
and requires the paired Zeus Triton, torch_zeus, SDK and a usable device.
"""

import argparse
import math

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_bf16(
    out_ptr, x_ptr, weight_ptr, N, D, eps,
    CORE_NUM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    # One tile covers the entire row; the wrapper checks D <= BLOCK_HIDDEN.
    core = tl.program_id(0)
    tiles = tl.cdiv(N, BLOCK_TOKENS)
    base = tiles // CORE_NUM
    extra = tiles % CORE_NUM
    count = base + tl.where(core < extra, 1, 0)
    first = core * base + tl.minimum(core, extra)
    xp = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_TOKENS, BLOCK_HIDDEN),
        order=(1, 0),
    )
    yp = tl.make_block_ptr(
        base=out_ptr,
        shape=(N, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_TOKENS, BLOCK_HIDDEN),
        order=(1, 0),
    )
    wp = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_HIDDEN,),
        order=(0,),
    )
    w = tl.load(wp, boundary_check=(0,), padding_option="zero")
    w32 = w.to(tl.float32)
    for tile in range(count):
        row = (first + tile) * BLOCK_TOKENS
        x = tl.load(
            tl.advance(xp, (row, 0)),
            boundary_check=(0, 1), padding_option="zero",
        ).to(tl.float32)
        square_sum = tl.sum(x * x, axis=1)
        inverse_rms = tl.rsqrt(square_sum / D + eps)
        y = (x * inverse_rms[:, None] * w32).to(tl.bfloat16)
        tl.store(tl.advance(yp, (row, 0)), y, boundary_check=(0, 1))


signature = {
    "out_ptr": "*bf16", "x_ptr": "*bf16", "weight_ptr": "*bf16",
    "N": "i32", "D": "i32", "eps": "fp32",
}
constants = {"CORE_NUM": 2, "BLOCK_TOKENS": 4, "BLOCK_HIDDEN": 256}


def reference(x, weight, eps):
    """Use already-quantized BF16 inputs and one final RNE conversion."""
    x32 = x.float()
    mean_square = x32.square().mean(dim=-1, keepdim=True)
    return (x32 * torch.rsqrt(mean_square + eps)
            * weight.float()).to(torch.bfloat16)


def run_device(x, weight, eps):
    """Run only the contiguous BF16 contract shown in the guide."""
    import torch_zeus  # noqa: F401

    if x.ndim != 2 or weight.shape != (x.shape[1],):
        raise ValueError("Expected X[N,D] and weight[D]")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("Expected BF16 inputs")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("This example requires contiguous tensors")
    rows, width = x.shape
    if not 1 <= width <= constants["BLOCK_HIDDEN"]:
        raise ValueError("Require 1 <= D <= 256")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Require 1 <= D <= 256 and eps > 0")
    if rows == 0:
        return torch.empty_like(x, device="cpu")
    xd = x.to("zeus")
    wd = weight.to("zeus")
    yd = torch.empty_like(xd)
    rmsnorm_bf16[(constants["CORE_NUM"],)](
        yd, xd, wd, rows, width, eps, **constants,
        zeus_enable_jit=True, zeus_strict=True,
    )
    torch.zeus.synchronize(yd.device)
    return yd.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=65)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--run-device", action="store_true")
    args = parser.parse_args()
    if args.rows < 1 or not 1 <= args.width <= 256:
        parser.error("Require rows >= 1 and 1 <= width <= 256")
    if not math.isfinite(args.eps) or args.eps <= 0:
        parser.error("Require rows >= 1, 1 <= width <= 256, eps > 0")
    generator = torch.Generator().manual_seed(20261124)
    x = torch.randn(args.rows, args.width, generator=generator)
    x = x.to(torch.bfloat16)
    weight = torch.ones(args.width, dtype=torch.bfloat16)
    expected = reference(x, weight, args.eps)
    print(f"CPU reference shape={tuple(expected.shape)}")
    print(f"finite={bool(torch.isfinite(expected).all())}")
    if args.run_device:
        actual = run_device(x, weight, args.eps)
        error = (actual.float() - expected.float()).abs()
        print(f"Zeus max_abs={error.max().item():.8g}")
        print(f"Zeus rms={error.square().mean().sqrt().item():.8g}")
        torch.testing.assert_close(
            actual, expected, atol=1e-2, rtol=1e-2,
        )
        print("Zeus example tolerance check passed")
    else:
        print("CPU reference only; the Zeus kernel was not executed")


if __name__ == "__main__":
    main()
