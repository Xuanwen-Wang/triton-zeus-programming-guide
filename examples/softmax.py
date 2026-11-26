"""FP32 row softmax with the Zeus Softmax_ATT operation.

Run without --run-device to evaluate an independent CPU reference.
Device execution requires the matching Triton, torch_zeus and SDK.
Each row must fit inside one static 1024-element tile.
"""

# SIM_PARAMS: output:ptr:out:[1,1,"num_tokens","hidden_size"]:fp32 input:ptr:in:[1,1,"num_tokens","hidden_size"]:fp32 num_tokens:i32 hidden_size:i32
# SIM_GOLDEN: softmax_att

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_fp32(
    output_ptr,
    input_ptr,
    num_tokens,
    hidden_size,
    CORE_NUM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    # A complete row is required by this Softmax_ATT template.
    core = tl.program_id(0)
    total = tl.cdiv(num_tokens, BLOCK_TOKENS)
    quotient = total // CORE_NUM
    remainder = total % CORE_NUM
    count = quotient + tl.where(core < remainder, 1, 0)
    start = core * quotient + tl.minimum(core, remainder)
    source = tl.make_block_ptr(
        base=input_ptr,
        shape=(num_tokens, hidden_size),
        strides=(hidden_size, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_TOKENS, BLOCK_HIDDEN),
        order=(1, 0),
    )
    target = tl.make_block_ptr(
        base=output_ptr,
        shape=(num_tokens, hidden_size),
        strides=(hidden_size, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_TOKENS, BLOCK_HIDDEN),
        order=(1, 0),
    )
    for index in range(count):
        row = (start + index) * BLOCK_TOKENS
        x = tl.load(
            tl.advance(source, (row, 0)),
            boundary_check=(0, 1), padding_option="zero",
        )
        output = tl.zeus.softmax(x, axis=1)
        tl.store(
            tl.advance(target, (row, 0)), output,
            boundary_check=(0, 1),
        )


signature = {
    "output_ptr": "*fp32",
    "input_ptr": "*fp32",
    "num_tokens": "i32",
    "hidden_size": "i32",
}

constants = {
    "CORE_NUM": 2,
    "BLOCK_TOKENS": 1,
    "BLOCK_HIDDEN": 1024,
}


def reference(x):
    """Evaluate stable softmax on the full logical row using PyTorch."""
    return torch.softmax(x.float(), dim=-1)


def run_device(x):
    """Run finite, contiguous FP32 rows with a compatible Zeus runtime."""
    import torch_zeus  # noqa: F401

    if x.ndim != 2 or x.dtype != torch.float32:
        raise ValueError("x must be a two-dimensional FP32 tensor")
    if not x.is_contiguous() or not torch.isfinite(x).all().item():
        raise ValueError("x must be contiguous and finite")
    rows, width = x.shape
    if not 1 <= width <= constants["BLOCK_HIDDEN"]:
        raise ValueError("width must be in [1, 1024]")
    if rows == 0:
        return x.clone()
    source = x.to("zeus")
    output = torch.empty_like(source)
    softmax_fp32[(constants["CORE_NUM"],)](
        output, source, rows, width, **constants,
        zeus_enable_jit=True,
    )
    torch.zeus.synchronize(output.device)
    return output.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-device", action="store_true")
    args = parser.parse_args()
    if args.rows < 0 or not 1 <= args.width <= 1024:
        parser.error("rows must be nonnegative; width must be in [1, 1024]")
    generator = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.rows, args.width, generator=generator)
    expected = reference(x)
    if not args.run_device:
        error = (expected.sum(dim=-1) - 1.0).abs()
        max_error = error.max().item() if error.numel() else 0.0
        print(f"CPU reference: shape={tuple(expected.shape)}")
        print(f"max_row_sum_error={max_error:.8g}")
        return
    actual = run_device(x)
    torch.testing.assert_close(
        actual, expected, atol=1e-5, rtol=1e-3,
    )
    difference = (actual - expected).abs()
    max_error = difference.max().item() if difference.numel() else 0.0
    print(f"Zeus execution passed: max_abs_error={max_error:.8g}")


if __name__ == "__main__":
    main()
