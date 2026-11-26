"""Capture and replay the vector-add kernel with different inputs."""

import argparse
import json

import torch

from vector_add import add_bf16, make_inputs, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="zeus:0")
    args = parser.parse_args()

    import torch_zeus  # noqa: F401
    from torch_zeus.zeus import ZEUSGraph, Stream, graph

    device = torch.device(args.device)
    hx, hz = make_inputs(2500)
    x, z = hx.to(device), hz.to(device)
    out = torch.empty_like(x)
    capture_stream = Stream(device=x.device)

    def submit():
        add_bf16[(2,)](
            out, x, z, x.numel(), 1.0, BLOCK=1024, CORE_NUM=2,
            zeus_enable_jit=True,
        )

    with torch.zeus.stream(capture_stream):
        submit()
    capture_stream.synchronize()
    recorded = ZEUSGraph()
    with graph(recorded, stream=capture_stream):
        submit()

    previews = []
    try:
        for updated in (hx * 2, -hx):
            with torch.zeus.stream(capture_stream):
                x.copy_(updated)
                recorded.replay()
            capture_stream.synchronize()
            actual = out.cpu()
            torch.testing.assert_close(actual, reference(updated, hz), rtol=0, atol=0)
            previews.append(actual[:8].float().tolist())
    finally:
        recorded.reset()
    print(json.dumps({"device": str(out.device), "replays": len(previews), "previews": previews}))


if __name__ == "__main__":
    main()
