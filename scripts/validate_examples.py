#!/usr/bin/env python3
"""Validate tutorial references and optionally compile with an explicit toolchain."""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = {"vector_add": "add_bf16", "rmsnorm": "rmsnorm_bf16",
           "matmul": "matmul_bf16", "fused_add_rmsnorm": "fused_add_rmsnorm_bf16",
           "append_kcache": "append_kcache_token",
           "append_vcache": "append_vcache_token",
           "softmax": "softmax_fp32",
           "quantized_matmul": "quantized_matmul"}


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zeus-source", type=Path)
    parser.add_argument("--triton-source", type=Path)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/validation-v03")
    parser.add_argument("--only", nargs="+", choices=list(KERNELS) + ["ldg_layout"])
    args = parser.parse_args()
    if bool(args.zeus_source) != bool(args.triton_source):
        parser.error("Pass both --zeus-source and --triton-source")
    if args.simulate and not args.zeus_source:
        parser.error("Simulation requires an explicit compiler source and Triton source")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    results = {"python": sys.executable, "started_utc": datetime.now(timezone.utc).isoformat(),
               "sources": {}, "runs": []}
    compiled = set()
    selected = set(args.only) if args.only else set(KERNELS) | {"ldg_layout"}
    if args.zeus_source:
        zs, ts = args.zeus_source.resolve(), args.triton_source.resolve()
        env.update(PYTHONPATH=str(ts/"python"), TRITON_DIR=str(ts),
                   TRITON_SHARED_OPT_PATH=str(zs/"build/bin/triton-shared-opt"),
                   ZEUS_COMPILER=str(zs/"build/bin/zeus-compiler"),
                   ZEUSV3_SIMULATOR_DIR=str(zs/"sim_check/Zeus3FunctionalSimulator"),
                   TRITON_CACHE_DIR=str(out/"cache"))
        results["toolchain"] = {
            "source_checkout_head": git_head(zs), "triton_checkout_head": git_head(ts),
            "note": "Existing binary hashes identify this run; checkout HEAD alone does not certify a build.",
            "binaries": {name: {"realpath": str((zs/"build/bin"/name).resolve()),
                                "sha256": sha(zs/"build/bin"/name)}
                         for name in ("zecc", "zeus-compiler", "triton-shared-opt", "zeusv3-backend")},
            "libtriton_sha256": sha(ts/"python/triton/_C/libtriton.so"),
        }

    def run(label, kind, command, timeout=180):
        print(label, flush=True)
        p = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=timeout)
        log = out / f"{label}.log"
        log.write_text(p.stdout+p.stderr)
        results["runs"].append({"name": label, "kind": kind, "command": command,
                                "exit_code": p.returncode,
                                "log": str(log.relative_to(ROOT))})
        print("exit", p.returncode, flush=True)
        return p.returncode

    for name, kernel in KERNELS.items():
        if name not in selected:
            continue
        source = ROOT/"examples"/f"{name}.py"
        tree = ast.parse(source.read_text())
        results["sources"][str(source.relative_to(ROOT))] = sha(source)
        configs = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("signature", "constants"):
                        configs[target.id] = ast.literal_eval(node.value)
        if args.zeus_source:
            cmd = [str(zs/"build/bin/zecc"), "--compile-only-zbin", "-c", str(source),
                   "-o", str(out/f"{name}.zbin"), "--kernel-name", kernel, "--loc=full"]
            for key, value in configs["signature"].items():
                cmd += ["--signature", f"{key}={value}"]
            for key, value in configs["constants"].items():
                cmd += ["--constant", f"{key}={value}"]
            if run(f"compile-{name}", "compile_only", cmd) == 0:
                compiled.add(name)
    cases = [("vector_add", ["--numel", str(n)]) for n in (0, 1, 2048, 2500)]
    cases += [(name, options) for name in ("rmsnorm", "fused_add_rmsnorm")
              for options in (["--rows", "65", "--width", "256"],
                              ["--rows", "3", "--width", "250"])]
    cases += [("matmul", []), ("matmul", ["--m", "129", "--n", "130", "--k", "257"])]
    cases += [("append_kcache", ["--token-offset", str(n)])
              for n in (0, 127, 128, 1000, 1023)]
    cases += [("append_vcache", ["--token-offset", str(n)])
              for n in (0, 63, 64, 1000, 1023)]
    cases += [("ldg_layout", [])]
    cases += [("quantized_matmul", [])]
    cases += [("softmax", ["--rows", str(r), "--width", str(w)])
              for r, w in ((5, 1024), (5, 1000), (1, 1), (0, 1024))]
    for index, (name, options) in enumerate(cases):
        if name not in selected:
            continue
        results["sources"][f"examples/{name}.py"] = sha(ROOT/"examples"/f"{name}.py")
        kind = "host_layout_reference" if name.startswith("append_") or name == "ldg_layout" else "cpu_reference_only"
        run(f"cpu-{index+1:02}-{name}", kind,
            [sys.executable, str(ROOT/"examples"/f"{name}.py")]+options)
    if args.simulate:
        simulator = zs/"sim_check/Zeus3FunctionalSimulator"
        results["simulator_checkout_head"] = git_head(simulator)
        sim_cases = [("vector_add", ["numel=2048", "alpha=1.0"]),
                              ("vector_add", ["numel=2500", "alpha=1.0"]),
                              ("matmul", ["M=256", "N=256", "K=384"])]
        sim_cases += [("append_kcache", [f"token_offset={n}"]) for n in (127, 128, 1000, 1023)]
        sim_cases += [("append_vcache", [f"token_offset={n}"]) for n in (63, 64, 1000, 1023)]
        sim_cases += [("softmax", [f"num_tokens={r}", f"hidden_size={w}"])
                      for r, w in ((5, 1024), (5, 1000), (1, 1))]
        sim_cases += [("quantized_matmul", ["M=4", "K=128", "N=256"])]
        for name, dynamic in sim_cases:
            if name not in selected:
                continue
            if name not in compiled:
                results["runs"].append({"name": f"sim-{name}-skipped",
                                        "kind": "skipped_after_compile_failure", "exit_code": 1})
                continue
            label = f"sim-{name}-" + "-".join(v.split("=")[1] for v in dynamic)
            run(label, "python_functional_simulator", [sys.executable,
                str(zs/"sim_check/sim_test.py"), str(ROOT/"examples"/f"{name}.py"),
                "--dynamic", *dynamic, "--seed", "42", "--prebuilt-zbin", str(out/f"{name}.zbin"),
                "--keep-dir", str(out/label), "--no-trace"], timeout=240)
    results["artifacts"] = {str(p.relative_to(ROOT)): sha(p)
                            for p in sorted(out.glob("*.zbin"))}
    (out/"results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
    failures = [r["name"] for r in results["runs"] if r["exit_code"]]
    print(json.dumps({"runs": len(results["runs"]), "failures": failures}))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
