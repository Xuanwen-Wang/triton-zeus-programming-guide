"""BF16 vector addition used throughout the Triton Zeus guide.

The default CLI runs a CPU reference only. Pass --run-device explicitly to
execute through an installed Zeus runtime. The selected runtime determines
whether execution uses a simulator or physical hardware.
"""

# SIM_PARAMS: out:ptr:out:[1,1,1,"numel"]:bf16 in1:ptr:in:[1,1,1,"numel"]:bf16 in2:ptr:in:[1,1,1,"numel"]:bf16 numel:i32 alpha:f32
# SIM_GOLDEN: add

import argparse
import json
import math

import torch
import triton
import triton.language as tl


@triton.jit
def add_bf16(
    out_ptr, in1_ptr, in2_ptr, numel, alpha,
    BLOCK: tl.constexpr, CORE_NUM: tl.constexpr,
):
    # Each core owns disjoint, consecutive tiles, including the tail.
    core_id = tl.program_id(0)
    total_blocks = tl.cdiv(numel, BLOCK)
    quotient = total_blocks // CORE_NUM
    remainder = total_blocks % CORE_NUM
    count = quotient + tl.where(core_id < remainder, 1, 0)
    start = core_id * quotient + tl.minimum(core_id, remainder)

    x_block = tl.make_block_ptr(
        base=in1_ptr,
        shape=(numel,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    z_block = tl.make_block_ptr(
        base=in2_ptr,
        shape=(numel,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    y_block = tl.make_block_ptr(
        base=out_ptr,
        shape=(numel,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )

    for local_tile in range(count):
        offset = (start + local_tile) * BLOCK
        x = tl.load(
            tl.advance(x_block, (offset,)),
            boundary_check=(0,), padding_option="zero",
        ).to(tl.float32)
        z = tl.load(
            tl.advance(z_block, (offset,)),
            boundary_check=(0,), padding_option="zero",
        ).to(tl.float32)
        y = (x + alpha * z).to(tl.bfloat16)
        tl.store(
            tl.advance(y_block, (offset,)), y,
            boundary_check=(0,),
        )


signature = {
    "out_ptr": "*bf16",
    "in1_ptr": "*bf16",
    "in2_ptr": "*bf16",
    "numel": "i32",
    "alpha": "fp32",
}
constants = {"BLOCK": 1024, "CORE_NUM": 2}


def make_inputs(numel):
    """Use exactly representable finite values for the first smoke test."""
    if not 0 <= numel < 2**31:
        raise ValueError("numel must fit a nonnegative i32")
    index = torch.arange(numel, dtype=torch.int64)
    x = ((index % 33 - 16).float() / 8).to(torch.bfloat16)
    z = ((index % 17 - 8).float() / 8).to(torch.bfloat16)
    return x, z


def reference(x, z, alpha=1.0):
    """Compute from BF16 inputs in FP32, then round once to BF16 (RNE)."""
    if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        raise TypeError("reference inputs must be BF16")
    if x.shape != z.shape:
        raise ValueError("reference inputs must have identical shapes")
    if not math.isfinite(alpha):
        raise ValueError("this example requires a finite alpha")
    return (x.float() + float(alpha) * z.float()).to(torch.bfloat16)


def launch(x, z, alpha=1.0, *, block=1024, cores=2):
    """Launch for same-device contiguous 1D BF16 inputs without aliasing."""
    if x.device.type != "zeus" or z.device != x.device:
        raise ValueError("inputs must be on the same Zeus device")
    if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        raise TypeError("this example requires BF16 inputs")
    if x.ndim != 1 or x.shape != z.shape:
        raise ValueError("inputs must be 1D with identical shapes")
    if not x.is_contiguous() or not z.is_contiguous():
        raise ValueError("this example requires contiguous inputs")
    if not 0 <= x.numel() < 2**31:
        raise ValueError("numel must fit a nonnegative i32")
    if cores not in (1, 2):
        raise ValueError("this example supports one or two cores")
    if block <= 0 or block & (block - 1):
        raise ValueError("block must be a positive power of two")
    if x.numel() + block - 1 > 2**31 - 1:
        raise ValueError("numel + block - 1 must fit i32 for cdiv")
    if not math.isfinite(alpha):
        raise ValueError("this example requires a finite alpha")
    out = torch.empty_like(x)
    # Empty tensors have no legal data transfer and need no kernel launch.
    if x.numel() == 0:
        return out
    add_bf16[(cores,)](
        out, x, z, x.numel(), float(alpha),
        BLOCK=block, CORE_NUM=cores,
        zeus_enable_jit=True,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numel", type=int, default=2500)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--device", default="zeus:0")
    args = parser.parse_args()
    if args.run_device and args.alpha != 1.0:
        parser.error("--run-device uses the exact alpha=1.0 smoke test")
    x, z = make_inputs(args.numel)
    expected = reference(x, z, args.alpha)
    report = {
        "execution": "cpu_reference_only",
        "kernel_launched": False,
        "numel": args.numel,
        "alpha": args.alpha,
        "input_dtype": "bfloat16",
        "compute_dtype": "float32",
        "output_dtype": "bfloat16",
        "preview": expected[:8].float().tolist(),
    }
    if args.run_device:
        import torch_zeus  # noqa: F401

        actual = launch(
            x.to(args.device), z.to(args.device), args.alpha
        )
        launched = args.numel > 0
        if launched:
            torch.zeus.synchronize(actual.device)
        actual_cpu = actual.cpu()
        torch.testing.assert_close(
            actual_cpu, expected, rtol=0, atol=0
        )
        error = (actual_cpu.float() - expected.float()).abs()
        report.update(
            execution="zeus_runtime" if launched else "empty_output",
            kernel_launched=launched,
            device=str(actual.device),
            exact_match=True,
            max_abs=float(error.max()) if error.numel() else 0.0,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
