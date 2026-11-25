"""Fused residual add and RMSNorm with two visible outputs.

The residual is converted to BF16 before FP32 normalization. This preserves
the conversion boundary of an independent BF16 add followed by RMSNorm.
The default command computes only a CPU reference.
"""

import argparse
import math

import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_rmsnorm_bf16(
    x_ptr, residual_ptr, weight_ptr, N, D, eps,
    CORE_NUM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    # X and residual own distinct storage; both are updated in place.
    core = tl.program_id(0)
    tiles = tl.cdiv(N, BLOCK_TOKENS)
    base = tiles // CORE_NUM
    extra = tiles % CORE_NUM
    count = base + tl.where(core < extra, 1, 0)
    first = core * base + tl.minimum(core, extra)
    xp = tl.make_block_ptr(
        x_ptr, (N, D), (D, 1), (0, 0),
        (BLOCK_TOKENS, BLOCK_HIDDEN), (1, 0),
    )
    rp = tl.make_block_ptr(
        residual_ptr, (N, D), (D, 1), (0, 0),
        (BLOCK_TOKENS, BLOCK_HIDDEN), (1, 0),
    )
    wp = tl.make_block_ptr(
        weight_ptr, (D,), (1,), (0,), (BLOCK_HIDDEN,), (0,),
    )
    w = tl.load(wp, boundary_check=(0,), padding_option="zero")
    w32 = w.to(tl.float32)
    for tile in range(count):
        row = (first + tile) * BLOCK_TOKENS
        xb = tl.advance(xp, (row, 0))
        rb = tl.advance(rp, (row, 0))
        x = tl.load(
            xb, boundary_check=(0, 1), padding_option="zero",
        ).to(tl.float32)
        residual = tl.load(
            rb, boundary_check=(0, 1), padding_option="zero",
        ).to(tl.float32)
        updated = (x + residual).to(tl.bfloat16)
        tl.store(rb, updated, boundary_check=(0, 1))
        # Widen the existing rounded value without another global load.
        value = updated.to(tl.float32)
        square_sum = tl.sum(value * value, axis=1)
        inverse_rms = tl.rsqrt(square_sum / D + eps)
        output = value * inverse_rms[:, None] * w32
        tl.store(xb, output.to(tl.bfloat16), boundary_check=(0, 1))


signature = {
    "x_ptr": "*bf16", "residual_ptr": "*bf16", "weight_ptr": "*bf16",
    "N": "i32", "D": "i32", "eps": "fp32",
}
constants = {"CORE_NUM": 2, "BLOCK_TOKENS": 4, "BLOCK_HIDDEN": 256}


def reference(x, residual, weight, eps):
    """Return normalized output and the independently visible residual."""
    updated = (x.float() + residual.float()).to(torch.bfloat16)
    value = updated.float()
    mean_square = value.square().mean(dim=-1, keepdim=True)
    output = (value * torch.rsqrt(mean_square + eps)
              * weight.float()).to(torch.bfloat16)
    return output, updated


def run_device(x, residual, weight, eps):
    """Own separate device allocations so the in-place writes cannot alias."""
    import torch_zeus  # noqa: F401

    if x.ndim != 2 or x.shape != residual.shape:
        raise ValueError("X and residual must have the same [N,D] shape")
    if weight.shape != (x.shape[1],):
        raise ValueError("Expected weight[D]")
    tensors = (x, residual, weight)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise TypeError("Expected BF16 inputs")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("This example requires contiguous tensors")
    rows, width = x.shape
    if not 1 <= width <= constants["BLOCK_HIDDEN"]:
        raise ValueError("Require 1 <= D <= 256")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Require 1 <= D <= 256 and eps > 0")
    if rows == 0:
        return x.cpu().clone(), residual.cpu().clone()
    # copy=True also preserves the ownership contract for device input tensors.
    xd = x.to("zeus", copy=True)
    rd = residual.to("zeus", copy=True)
    wd = weight.to("zeus")
    fused_add_rmsnorm_bf16[(constants["CORE_NUM"],)](
        xd, rd, wd, rows, width, eps, **constants,
        zeus_enable_jit=True, zeus_strict=True,
    )
    torch.zeus.synchronize()
    return xd.cpu(), rd.cpu()


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
    residual = torch.randn(args.rows, args.width, generator=generator)
    x, residual = x.to(torch.bfloat16), residual.to(torch.bfloat16)
    weight = torch.ones(args.width, dtype=torch.bfloat16)
    expected = reference(x, residual, weight, args.eps)
    print(f"CPU reference output/residual shape={tuple(x.shape)}")
    if args.run_device:
        actual = run_device(x, residual, weight, args.eps)
        for name, result, ref in zip(("output", "residual"), actual, expected):
            error = (result.float() - ref.float()).abs()
            print(f"Zeus {name} max_abs={error.max().item():.8g}")
            tolerance = 0.0 if name == "residual" else 1e-2
            torch.testing.assert_close(
                result, ref, atol=tolerance, rtol=tolerance,
            )
        print("Zeus example tolerance checks passed")
    else:
        print("CPU reference only; the Zeus kernel was not executed")


if __name__ == "__main__":
    main()
