#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Template processor for MLIR microkernel code generation.

This script processes .mlir.in template files and generates .mlir output files
with expression evaluation.

Usage:
    python mlir_ukernel_gen.py template.mlir.in -o output.mlir \\
        -D INTRINSICS_M=4 INTRINSICS_N=8 ARCH=gfx942

Template syntax:
    ${VAR}              - Variable substitution
    ${VAR * 2}          - Expression evaluation
    ${VAR1 * VAR2 + 1}  - Complex expressions
"""

import argparse
import re
from typing import Dict, Any

ELEM_TYPE_BITS = {
    "bf16": 16,
    "f16": 16,
    "f32": 32,
    "f4E2M1FN": 4,
    "f8E4M3FN": 8,
    "f8E4M3FNUZ": 8,
}


def fold1(val):
    """
    Fold dimension if value is 1.
    Returns 'valx' if val != 1, else empty string.
    """
    return f"{val}x" if val != 1 else ""


def process_template(text: str, params: Dict[str, Any]) -> str:
    """Process ${...} substitutions in template text."""
    eval_context = params.copy()
    eval_context["fold1"] = fold1

    def replacer(match):
        expr = match.group(1).strip()
        try:
            result = eval(expr, {"__builtins__": {}}, eval_context)
            return str(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '${{{expr}}}': {e}")

    return re.sub(r"\$\{([^}]+)\}", replacer, text)


def parse_define(define_str: str) -> tuple:
    """Parse a -D VAR=VALUE string into (var, value)."""
    if "=" in define_str:
        var, value = define_str.split("=", 1)
        # Try to convert to int if possible
        try:
            value = int(value)
        except ValueError:
            pass
        return var.strip(), value
    else:
        return define_str.strip(), True


def main():
    parser = argparse.ArgumentParser(
        description="Process MLIR microkernel templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python mlir_ukernel_gen.py template.mlir.in -o output.mlir \\
        -D INTRINSICS_M=4 INTRINSICS_N=8 ARCH=gfx942
        """,
    )

    parser.add_argument("template", type=str, help="Path to the .mlir.in template file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "-D",
        "--define",
        type=str,
        nargs="+",
        default=[],
        help="Define parameters: -D VAR1=VALUE1 VAR2=VALUE2 ...",
    )

    args = parser.parse_args()

    # Default parameter values
    DEFAULT_PARAMS = {
        # Common defaults
        "ARCH": "gfx942",
        "ELEM_TYPE": "f16",
        "INTRINSIC": "MFMA_F32_16x16x16_F16",
        # Benefit
        "BENEFIT": 1,
        # Constraint 0: size_min=0, size_max=INT32_MAX, size_div=1
        "SIZE_MIN_0": 0,
        "SIZE_MAX_0": 2147483647,
        "SIZE_DIV_0": 1,
        # Constraint 1: size_min=0, size_max=INT32_MAX, size_div=1
        "SIZE_MIN_1": 0,
        "SIZE_MAX_1": 2147483647,
        "SIZE_DIV_1": 1,
        # Constraint 2: size_min=0, size_max=INT32_MAX, size_div=1
        "SIZE_MIN_2": 0,
        "SIZE_MAX_2": 2147483647,
        "SIZE_DIV_2": 1,
        # MMA layout attributes
        "INTRINSICS_M": 1,
        "INTRINSICS_N": 1,
        "INTRINSICS_K": 1,
        "SUBGROUPS_M": 1,
        "SUBGROUPS_N": 1,
    }

    # Start with defaults, then override with user params
    params = DEFAULT_PARAMS.copy()
    for define in args.define:
        var, value = parse_define(define)
        params[var] = value

    # Derive ELEM_BITS from ELEM_TYPE
    elem_type = params["ELEM_TYPE"]
    assert elem_type in ELEM_TYPE_BITS, f"Invalid element type: {elem_type}"
    params["ELEM_BITS"] = ELEM_TYPE_BITS[elem_type]

    # Read template
    with open(args.template, "r") as f:
        template = f.read()

    # Process template
    output = process_template(template, params)

    # Write output
    with open(args.output, "w") as f:
        f.write(output)
    print(f"Generated: {args.output}")


if __name__ == "__main__":
    main()
