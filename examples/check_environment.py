"""Inspect the active Python and compiler installation without a launch."""

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import sys


def module_record(name):
    """Report import resolution without importing optional native runtimes."""
    spec = importlib.util.find_spec(name)
    result = {"available": spec is not None}
    if spec is not None:
        result["origin"] = spec.origin
    try:
        result["version"] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result["version"] = "not recorded as a distribution"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-runtime", action="store_true")
    args = parser.parse_args()
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "modules": {
            name: module_record(name)
            for name in ("torch", "triton", "torch_zeus", "zecc_jit")
        },
        "tools": {
            name: shutil.which(name)
            for name in ("zecc", "zeus-compiler", "zeusv3-backend")
        },
    }
    try:
        tl = importlib.import_module("triton.language")
        report["zeus_language_extension"] = hasattr(tl, "zeus")
    except Exception as exc:
        report["triton_import_error"] = str(exc)
    if args.probe_runtime:
        try:
            import torch
            import torch_zeus  # noqa: F401

            report["runtime_probe"] = {
                "registered": hasattr(torch, "zeus"),
                "available": bool(torch.zeus.is_available()),
            }
        except Exception as exc:
            report["runtime_probe"] = {"error": str(exc)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
