#!/usr/bin/env python3
"""Run llc on the AOT/JIT repro IR and report static instruction counts."""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_LLVM_BIN = REPO_ROOT / "build" / "llvm-project" / "bin"
DEFAULT_TEMP_PREFIX = "wmma_spill_llc_"

INSTRUCTION_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)\b", re.MULTILINE)
INSTRUCTION_PREFIXES = (
    "s_",
    "v_",
    "buffer_",
    "global_",
    "flat_",
    "ds_",
    "scratch_",
    "image_",
    "tbuffer_",
)
STAT_KEYS = (
    "instructions",
    "buffer_load",
    "buffer_store",
    "global_load",
    "global_store",
    "scratch",
    "v_fmac",
    "v_fma",
    "v_wmma",
    "s_wait",
    "v_dual",
)



def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True)


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def is_instruction_mnemonic(mnemonic):
    return mnemonic.startswith(INSTRUCTION_PREFIXES)


def isa_stats(asm):
    mnemonics = [
        m.group(1)
        for m in INSTRUCTION_RE.finditer(asm)
        if is_instruction_mnemonic(m.group(1)) and m.group(1) != "s_code_end"
    ]
    return {
        "instructions": len(mnemonics),
        "buffer_load": sum(1 for m in mnemonics if m.startswith("buffer_load")),
        "buffer_store": sum(1 for m in mnemonics if m.startswith("buffer_store")),
        "global_load": sum(1 for m in mnemonics if m.startswith("global_load")),
        "global_store": sum(1 for m in mnemonics if m.startswith("global_store")),
        "scratch": sum(1 for m in mnemonics if m.startswith("scratch_")),
        "v_fmac": sum(1 for m in mnemonics if re.match(r"v_(dual_)?fmac", m)),
        "v_fma": sum(1 for m in mnemonics if m.startswith("v_fma_f32")),
        "v_wmma": sum(1 for m in mnemonics if m.startswith("v_wmma")),
        "s_wait": sum(1 for m in mnemonics if m.startswith("s_wait")),
        "v_dual": sum(1 for m in mnemonics if m.startswith("v_dual_")),
    }


def compile_and_count(name, ll_path, obj_path, llvm_bin):
    llc = llvm_bin / "llc"
    objdump = llvm_bin / "llvm-objdump"
    cmd = [
        str(llc),
        "-mtriple=amdgcn-amd-amdhsa",
        "-mcpu=gfx1201",
        "-filetype=obj",
        str(ll_path),
        "-o",
        str(obj_path),
    ]

    result = run(cmd)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"{name}: llc failed with exit code {result.returncode}")

    result = run([str(objdump), "-d", str(obj_path)])
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"{name}: llvm-objdump failed with exit code {result.returncode}")

    return {
        "name": name,
        **isa_stats(result.stdout),
        "object": obj_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llvm-bin",
        type=Path,
        default=Path(os.environ.get("LLVM_BIN", DEFAULT_LLVM_BIN)),
        help="Directory containing llc and llvm-objdump.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for generated object files. Defaults to a temporary directory.",
    )
    args = parser.parse_args()

    llvm_bin = args.llvm_bin.resolve()
    for tool in ("llc", "llvm-objdump"):
        if not (llvm_bin / tool).exists():
            raise FileNotFoundError(f"missing {llvm_bin / tool}")

    if args.out_dir is None:
        out_dir_ctx = tempfile.TemporaryDirectory(prefix=DEFAULT_TEMP_PREFIX)
        out_dir = Path(out_dir_ctx.name)
    else:
        out_dir_ctx = None
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = []
        for name in ("aot", "jit"):
            results.append(
                compile_and_count(
                    name=name,
                    ll_path=SCRIPT_DIR / f"{name}.ll",
                    obj_path=out_dir / f"{name}.o",
                    llvm_bin=llvm_bin,
                )
            )

        print(f"LLVM_BIN={llvm_bin}")
        aot_stats, jit_stats = results
        headers = ["instruction", "AOT", "JIT", "delta"]
        rows = []
        for key in STAT_KEYS:
            aot_value = aot_stats.get(key, 0)
            jit_value = jit_stats.get(key, 0)
            if aot_value == 0 and jit_value == 0:
                continue
            delta = jit_value - aot_value
            sign = "+" if delta > 0 else ""
            rows.append([key, aot_value, jit_value, f"{sign}{delta}"])
        print_table(headers, rows)
    finally:
        if out_dir_ctx is not None:
            out_dir_ctx.cleanup()


if __name__ == "__main__":
    main()
