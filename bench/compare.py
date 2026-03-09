#!/usr/bin/env python3
"""Compare AOT (HSACO) vs JIT (SPIR-V) performance for IREE kernels.

Usage:
  # Compare a matmul with auto-generated MLIR:
  ./bench/compare.py matmul 2048 2048 2048 --dtype f32

  # Compare with an existing MLIR file (all functions benchmarked):
  ./bench/compare.py file bench/matmul_f32.mlir

  # Just one function from a multi-function file:
  ./bench/compare.py file bench/matmul_f16.mlir --function matmul_2048x2048x2048

  # Elementwise add:
  ./bench/compare.py add 2048 2048 --dtype f32

  # More reps, dump ISA:
  ./bench/compare.py matmul 4096 4096 4096 --dtype f32 --reps 10 --dump-isa

  # Skip benchmarking, just compile and compare ISA:
  ./bench/compare.py matmul 2048 2048 2048 --dtype f32 --isa-only
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BUILD = ROOT / "build"
IREE_COMPILE = BUILD / "tools" / "iree-compile"
IREE_BENCHMARK = BUILD / "tools" / "iree-benchmark-module"
IREE_RUN = BUILD / "tools" / "iree-run-module"
ROCM_SDK = ROOT / "venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin"
OBJDUMP = ROCM_SDK / "llvm-objdump"
_TARGET_CHIP = [os.environ.get("IREE_ROCM_TARGET", "gfx1201")]


def get_target():
    return _TARGET_CHIP[0]


def set_target(chip):
    _TARGET_CHIP[0] = chip


def run(cmd, **kwargs):
    """Run a command, return CompletedProcess."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    return subprocess.run(cmd, **kwargs)


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------


def _zero_literal(dtype):
    """Return the zero constant literal for a given MLIR type."""
    if dtype.startswith("i") or dtype.startswith("ui") or dtype.startswith("si"):
        return f"0 : {dtype}"
    return f"0.0 : {dtype}"


def gen_matmul_mlir(M, N, K, dtype, acc_dtype=None, transpose_b=False):
    if acc_dtype is None:
        acc_dtype = "f32" if dtype in ("f16", "bf16") else dtype
    variant = "transpose_b" if transpose_b else ""
    suffix = "_tb" if transpose_b else ""
    name = f"matmul_{M}x{N}x{K}_{dtype}{suffix}"
    zero = _zero_literal(acc_dtype)
    if transpose_b:
        # matmul_transpose_b: A[M,K] x B[N,K] -> C[M,N]
        rhs_shape = f"{N}x{K}"
        indexing = " indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]"
    else:
        # matmul: A[M,K] x B[K,N] -> C[M,N]
        rhs_shape = f"{K}x{N}"
        indexing = ""
    return name, textwrap.dedent(
        f"""\
        func.func @{name}(%lhs: tensor<{M}x{K}x{dtype}>, %rhs: tensor<{rhs_shape}x{dtype}>) -> tensor<{M}x{N}x{acc_dtype}> {{
          %cst = arith.constant {zero}
          %init = tensor.empty() : tensor<{M}x{N}x{acc_dtype}>
          %fill = linalg.fill ins(%cst : {acc_dtype}) outs(%init : tensor<{M}x{N}x{acc_dtype}>) -> tensor<{M}x{N}x{acc_dtype}>
          %result = linalg.matmul{indexing} ins(%lhs, %rhs : tensor<{M}x{K}x{dtype}>, tensor<{rhs_shape}x{dtype}>) outs(%fill : tensor<{M}x{N}x{acc_dtype}>) -> tensor<{M}x{N}x{acc_dtype}>
          return %result : tensor<{M}x{N}x{acc_dtype}>
        }}
    """
    )


def gen_add_mlir(M, N, dtype):
    name = f"add_{M}x{N}_{dtype}"
    return name, textwrap.dedent(
        f"""\
        func.func @{name}(%lhs: tensor<{M}x{N}x{dtype}>, %rhs: tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}> {{
          %result = arith.addf %lhs, %rhs : tensor<{M}x{N}x{dtype}>
          return %result : tensor<{M}x{N}x{dtype}>
        }}
    """
    )


# ---------------------------------------------------------------------------
# Parse function signatures from MLIR to extract input shapes
# ---------------------------------------------------------------------------


def parse_mlir_functions(mlir_path):
    """Return list of (func_name, [(shape, dtype), ...]) for each public func."""
    text = Path(mlir_path).read_text()
    funcs = []
    for m in re.finditer(r"func\.func\s+@(\w+)\(([^)]*)\)", text):
        name = m.group(1)
        args_str = m.group(2)
        inputs = []
        for am in re.finditer(r"tensor<([^>]+)>", args_str):
            spec = am.group(1)  # e.g. "2048x2048xf32"
            inputs.append(spec)
        funcs.append((name, inputs))
    return funcs


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


def compile_variant(mlir_path, vmfb_path, intermediates_dir, spirv=False):
    cmd = [
        str(IREE_COMPILE),
        f"--iree-hal-target-device=hip",
        f"--iree-rocm-target={get_target()}",
        f"--iree-hal-dump-executable-intermediates-to={intermediates_dir}",
        str(mlir_path),
        "-o",
        str(vmfb_path),
    ]
    if spirv:
        cmd.append("--iree-rocm-use-spirv")
    result = run(cmd)
    if result.returncode != 0:
        return False, result.stderr
    return True, ""


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark(vmfb_path, func_name, input_specs, reps=5):
    """Run iree-benchmark-module, return median time in ms or None."""
    cmd = [
        str(IREE_BENCHMARK),
        f"--module={vmfb_path}",
        "--device=hip",
        f"--function={func_name}",
        f"--benchmark_repetitions={reps}",
    ]
    for spec in input_specs:
        cmd.append(f"--input={spec}")
    result = run(cmd, timeout=300)
    if result.returncode != 0:
        return None, result.stderr
    # Parse median line
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        if "median" in line and "real_time" in line:
            m = re.search(r"(\d+\.?\d*)\s+ms", line)
            if m:
                return float(m.group(1)), ""
    return None, "Could not parse median from benchmark output"


# ---------------------------------------------------------------------------
# ISA extraction and comparison
# ---------------------------------------------------------------------------


def get_jit_isa(vmfb_path, func_name, input_specs):
    """Run the SPIR-V vmfb once with AMD_COMGR_SAVE_TEMPS to get JIT ISA."""
    import glob as globmod

    # Remove all existing comgr temps to avoid picking up stale results.
    for d in globmod.glob("/tmp/comgr-*"):
        shutil.rmtree(d, ignore_errors=True)

    env = os.environ.copy()
    env["AMD_COMGR_SAVE_TEMPS"] = "1"
    cmd = [
        str(IREE_RUN),
        f"--module={vmfb_path}",
        "--device=hip",
        f"--function={func_name}",
    ]
    for spec in input_specs:
        cmd.append(f"--input={spec}")
    result = run(cmd, env=env, timeout=120)
    if result.returncode != 0:
        return None, result.stderr

    # Find comgr temp with a.so in output/ (final linked code object).
    temps = sorted(globmod.glob("/tmp/comgr-*"))
    for d in reversed(temps):
        so_path = os.path.join(d, "output", "a.so")
        if os.path.exists(so_path):
            return so_path, ""
    return None, "No comgr output a.so found"


def disassemble(elf_path):
    """Disassemble an ELF with llvm-objdump, return text."""
    if not OBJDUMP.exists():
        return None
    result = run([str(OBJDUMP), "-d", str(elf_path)])
    if result.returncode == 0:
        return result.stdout
    return None


def find_aot_asm(intermediates_dir):
    """Find the .rocmasm file in intermediates."""
    for f in Path(intermediates_dir).iterdir():
        if f.suffix == ".rocmasm":
            return f.read_text()
    return None


def isa_stats(asm_text):
    """Count key instruction types in ISA text."""
    if not asm_text:
        return {}
    return {
        "buffer_load": len(re.findall(r"buffer_load", asm_text)),
        "buffer_store": len(re.findall(r"buffer_store", asm_text)),
        "global_load": len(re.findall(r"global_load", asm_text)),
        "global_store": len(re.findall(r"global_store", asm_text)),
        "scratch": len(re.findall(r"scratch_", asm_text)),
        "v_fmac": len(re.findall(r"v_(dual_)?fmac", asm_text)),
        "v_fma": len(re.findall(r"v_fma_f32", asm_text)),
        "v_wmma": len(re.findall(r"v_wmma", asm_text)),
        "s_wait": len(re.findall(r"s_wait", asm_text)),
        "v_dual": len(re.findall(r"v_dual_", asm_text)),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_table(headers, rows):
    """Print a simple aligned table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def run_comparison(mlir_path, func_name, input_specs, args):
    """Run one AOT vs JIT comparison for a single function."""
    print(f"\n{'='*60}")
    print(f"  {func_name}")
    print(f"  inputs: {', '.join(input_specs)}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory(prefix="iree_compare_") as tmpdir:
        tmpdir = Path(tmpdir)
        hsaco_vmfb = tmpdir / "hsaco.vmfb"
        spirv_vmfb = tmpdir / "spirv.vmfb"
        hsaco_intermediates = tmpdir / "hsaco_intermediates"
        spirv_intermediates = tmpdir / "spirv_intermediates"
        hsaco_intermediates.mkdir()
        spirv_intermediates.mkdir()

        # --- Compile ---
        print("\nCompiling HSACO (AOT)...", end=" ", flush=True)
        ok, err = compile_variant(mlir_path, hsaco_vmfb, hsaco_intermediates)
        if not ok:
            print("FAILED")
            print(f"  {err[:200]}")
            return
        print("ok")

        print("Compiling SPIR-V (JIT)...", end=" ", flush=True)
        ok, err = compile_variant(
            mlir_path, spirv_vmfb, spirv_intermediates, spirv=True
        )
        if not ok:
            print("FAILED")
            print(f"  {err[:200]}")
            return
        print("ok")

        # --- ISA ---
        aot_asm = find_aot_asm(hsaco_intermediates)
        jit_asm = None

        print("Extracting JIT ISA...", end=" ", flush=True)
        so_path, err = get_jit_isa(spirv_vmfb, func_name, input_specs)
        if so_path:
            jit_asm = disassemble(so_path)
            print("ok")
        else:
            print(f"FAILED ({err[:100]})")

        aot_stats = isa_stats(aot_asm)
        jit_stats = isa_stats(jit_asm)

        if aot_stats or jit_stats:
            print("\nISA instruction counts:")
            all_keys = sorted(set(list(aot_stats.keys()) + list(jit_stats.keys())))
            headers = ["instruction", "AOT", "JIT", "delta"]
            rows = []
            for k in all_keys:
                a = aot_stats.get(k, 0)
                j = jit_stats.get(k, 0)
                if a == 0 and j == 0:
                    continue
                delta = j - a
                sign = "+" if delta > 0 else ""
                rows.append([k, a, j, f"{sign}{delta}"])
            print_table(headers, rows)

        # Save ISA if requested
        if args.dump_isa:
            isa_dir = SCRIPT_DIR / "isa_comparison"
            isa_dir.mkdir(exist_ok=True)
            safe_name = func_name.replace("/", "_")
            if aot_asm:
                (isa_dir / f"{safe_name}_aot.s").write_text(aot_asm)
            if jit_asm:
                (isa_dir / f"{safe_name}_jit.s").write_text(jit_asm)
            print(f"\nISA saved to {isa_dir}/")

        if args.isa_only:
            return

        # --- Benchmark ---
        print(f"\nBenchmarking HSACO ({args.reps} reps)...", end=" ", flush=True)
        hsaco_ms, err = benchmark(hsaco_vmfb, func_name, input_specs, args.reps)
        if hsaco_ms is not None:
            print(f"{hsaco_ms:.3f} ms")
        else:
            print(f"FAILED ({err[:100]})")

        print(f"Benchmarking SPIR-V ({args.reps} reps)...", end=" ", flush=True)
        spirv_ms, err = benchmark(spirv_vmfb, func_name, input_specs, args.reps)
        if spirv_ms is not None:
            print(f"{spirv_ms:.3f} ms")
        else:
            print(f"FAILED ({err[:100]})")

        # --- Summary ---
        if hsaco_ms is not None and spirv_ms is not None:
            ratio = spirv_ms / hsaco_ms if hsaco_ms > 0 else float("inf")
            print(f"\n  HSACO: {hsaco_ms:.3f} ms")
            print(f"  SPIR-V: {spirv_ms:.3f} ms")
            print(f"  Ratio: {ratio:.1f}x slower via JIT")

            # Compute FLOPS for matmul
            if "matmul" in func_name:
                # Extract M, N, K from function name or input specs
                dims = re.findall(r"(\d+)x(\d+)x(\w+)", input_specs[0])
                if len(dims) >= 1:
                    M, K = int(dims[0][0]), int(dims[0][1])
                    dims2 = re.findall(r"(\d+)x(\d+)x(\w+)", input_specs[1])
                    N = int(dims2[0][1]) if dims2 else K
                    flops = 2 * M * N * K
                    hsaco_tflops = flops / (hsaco_ms * 1e-3) / 1e12
                    spirv_tflops = flops / (spirv_ms * 1e-3) / 1e12
                    print(f"  HSACO: {hsaco_tflops:.2f} TFLOPS")
                    print(f"  SPIR-V: {spirv_tflops:.2f} TFLOPS")


def main():
    parser = argparse.ArgumentParser(
        description="Compare AOT (HSACO) vs JIT (SPIR-V) kernel performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            examples:
              %(prog)s matmul 2048 2048 2048 --dtype f32
              %(prog)s matmul 4096 4096 4096 --dtype f16
              %(prog)s add 2048 2048 --dtype f32
              %(prog)s file bench/matmul_f32.mlir
              %(prog)s file bench/matmul_f16.mlir --function matmul_2048x2048x2048
        """
        ),
    )
    sub = parser.add_subparsers(dest="mode")

    # matmul subcommand
    p_mm = sub.add_parser("matmul", help="Generate and compare a matmul")
    p_mm.add_argument("M", type=int)
    p_mm.add_argument("N", type=int)
    p_mm.add_argument("K", type=int)
    p_mm.add_argument("--dtype", default="f32", help="Element type (default: f32)")
    p_mm.add_argument(
        "--acc-dtype",
        default=None,
        help="Accumulator type (default: f32 for f16/bf16, same as dtype otherwise)",
    )
    p_mm.add_argument(
        "--transpose-b",
        action="store_true",
        help="Use linalg.matmul_transpose_b (B is NxK instead of KxN)",
    )

    # add subcommand
    p_add = sub.add_parser("add", help="Generate and compare an elementwise add")
    p_add.add_argument("M", type=int)
    p_add.add_argument("N", type=int)
    p_add.add_argument("--dtype", default="f32")

    # file subcommand
    p_file = sub.add_parser("file", help="Compare using an existing MLIR file")
    p_file.add_argument("mlir", type=str, help="Path to MLIR file")
    p_file.add_argument(
        "--function",
        default=None,
        help="Run only this function (default: all public functions)",
    )

    # Common options
    for p in [p_mm, p_add, p_file]:
        p.add_argument("--reps", type=int, default=5, help="Benchmark repetitions")
        p.add_argument(
            "--dump-isa",
            action="store_true",
            help="Save ISA disassembly to bench/isa_comparison/",
        )
        p.add_argument(
            "--isa-only",
            action="store_true",
            help="Skip benchmarking, only compile and compare ISA",
        )
        p.add_argument(
            "--target",
            default=None,
            help=f"Target chip (default: $IREE_ROCM_TARGET or {get_target()})",
        )

    args = parser.parse_args()
    if not args.mode:
        parser.print_help()
        sys.exit(1)

    if args.target:
        set_target(args.target)

    # Check tools exist
    for tool in [IREE_COMPILE, IREE_BENCHMARK, IREE_RUN]:
        if not tool.exists():
            print(f"Error: {tool} not found. Build IREE first.", file=sys.stderr)
            sys.exit(1)

    if args.mode == "file":
        mlir_path = Path(args.mlir)
        if not mlir_path.exists():
            print(f"Error: {mlir_path} not found", file=sys.stderr)
            sys.exit(1)
        funcs = parse_mlir_functions(mlir_path)
        if not funcs:
            print("Error: no functions found in MLIR file", file=sys.stderr)
            sys.exit(1)
        if args.function:
            funcs = [(n, i) for n, i in funcs if n == args.function]
            if not funcs:
                print(f"Error: function '{args.function}' not found", file=sys.stderr)
                sys.exit(1)
        for func_name, tensor_specs in funcs:
            # Convert tensor specs to iree-benchmark-module input format
            input_specs = [spec for spec in tensor_specs]
            run_comparison(mlir_path, func_name, input_specs, args)

    elif args.mode == "matmul":
        tb = getattr(args, "transpose_b", False)
        name, mlir_text = gen_matmul_mlir(
            args.M,
            args.N,
            args.K,
            args.dtype,
            args.acc_dtype,
            transpose_b=tb,
        )
        acc = args.acc_dtype or ("f32" if args.dtype in ("f16", "bf16") else args.dtype)
        with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
            f.write(mlir_text)
            mlir_path = f.name
        if tb:
            rhs_shape = f"{args.N}x{args.K}"
        else:
            rhs_shape = f"{args.K}x{args.N}"
        input_specs = [
            f"{args.M}x{args.K}x{args.dtype}",
            f"{rhs_shape}x{args.dtype}",
        ]
        try:
            run_comparison(mlir_path, name, input_specs, args)
        finally:
            os.unlink(mlir_path)

    elif args.mode == "add":
        name, mlir_text = gen_add_mlir(args.M, args.N, args.dtype)
        with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
            f.write(mlir_text)
            mlir_path = f.name
        input_specs = [
            f"{args.M}x{args.N}x{args.dtype}",
            f"{args.M}x{args.N}x{args.dtype}",
        ]
        try:
            run_comparison(mlir_path, name, input_specs, args)
        finally:
            os.unlink(mlir_path)


if __name__ == "__main__":
    main()
