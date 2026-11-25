"""Logical Host reference for the single-request paged GQA example."""

import json
import math

import torch


def physical_slots(page_ids, length, page_size):
    """Resolve the valid logical prefix in request order."""
    logical = torch.arange(length, dtype=torch.int64)
    pages = torch.as_tensor(page_ids, dtype=torch.int64)
    return pages[logical // page_size] * page_size + logical % page_size


def reference(q, k, v, page_ids, length, *, page_size):
    """Return the FP32 partial output and LSE from unpacked BF16 inputs."""
    slots = physical_slots(page_ids, length, page_size)
    if not length:
        return torch.zeros_like(q, dtype=torch.float32), torch.full(
            q.shape[:-1], -math.inf, dtype=torch.float32,
        )
    heads_per_group = q.shape[1] // k.shape[0]
    kv_head = torch.arange(q.shape[1]) // heads_per_group
    keys = k.index_select(1, slots)[kv_head].float()
    values = v.index_select(1, slots)[kv_head].float()
    scores = torch.einsum("bhd,hsd->bhs", q.float(), keys) / math.sqrt(q.shape[-1])
    return torch.einsum("bhs,hsd->bhd", scores.softmax(dim=-1), values), scores.logsumexp(-1)


def check_outputs(partial, lse, expected, expected_lse, *, partial_atol, lse_atol):
    """Compare both nonempty SDK outputs using the caller's error budget."""
    if partial.shape != expected.shape or partial.dtype != torch.bfloat16:
        raise ValueError("expected BF16 partial with shape [B, Hq, D]")
    if lse.shape != expected_lse.shape or lse.dtype != torch.float32:
        raise ValueError("expected FP32 LSE with shape [B, Hq]")
    output_cpu, lse_cpu = partial.cpu().float(), lse.cpu()
    torch.testing.assert_close(output_cpu, expected, atol=partial_atol, rtol=0)
    torch.testing.assert_close(lse_cpu, expected_lse, atol=lse_atol, rtol=0)
    errors = {
        "partial_max_abs": (output_cpu - expected).abs().max().item(),
        "lse_max_abs": (lse_cpu - expected_lse).abs().max().item(),
    }
    print(json.dumps(errors, indent=2))
    return errors


def main():
    generator = torch.Generator().manual_seed(7)
    k = torch.randn((2, 1024, 256), generator=generator, dtype=torch.bfloat16)
    v = torch.randn((2, 1024, 256), generator=generator, dtype=torch.bfloat16)
    q = torch.randn((1, 16, 256), generator=generator, dtype=torch.bfloat16)
    slots = physical_slots((3, 2), 390, 256)
    expected, lse = reference(q, k, v, (3, 2), 390, page_size=256)
    print(json.dumps({
        "execution": "cpu_reference_only",
        "slots": slots[[0, 255, 256, -1]].tolist(),
        "partial_shape": list(expected.shape),
        "lse_shape": list(lse.shape),
        "finite": bool(torch.isfinite(expected).all() and torch.isfinite(lse).all()),
    }, indent=2))


if __name__ == "__main__":
    main()
