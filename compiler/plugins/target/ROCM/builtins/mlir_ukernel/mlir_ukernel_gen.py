#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Template processor for MLIR microkernel code generation.

This script processes .mlir.in template files and generates .mlir output files
using Python-like syntax.

Usage:
    python mlir_ukernel_gen.py template.mlir.in -o output.mlir \\
        -D INTRINSICS_M=4 INTRINSICS_N=8 ARCH=gfx942

Template syntax:
    ${VAR}              - Variable substitution
    ${VAR * 2}          - Expression evaluation
    $for i in range(4): - Loop unrolling
    $if condition:      - Conditional generation
    $else:              - Alternative branch
"""

import argparse
import io
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


def extract_internal_k(intrinsic_name: str) -> int:
    """
    Extract the K dimension from an intrinsic name and divide by 4.

    This aligns with:
    https://github.com/iree-org/iree/blob/de381bdda5bbcfdd3e0897b2c30b9e781c4d3846/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp#L230-L239

    Examples:
        MFMA_F32_16x16x16_F16 -> 16 // 4 = 4
        MFMA_F32_16x16x32_F8E4M3FNUZ -> 32 // 4 = 8
        WMMA_F32_16x16x16_F16 -> 16 // 4 = 4
    """
    match = re.search(r"_(\d+)x(\d+)x(\d+)_", intrinsic_name)
    if not match:
        raise ValueError(
            f"Could not extract K dimension from intrinsic: {intrinsic_name}"
        )

    m, n, k = int(match.group(1)), int(match.group(2)), int(match.group(3))

    # Assert that M and N are 16 as expected for supported intrinsics.
    assert m == 16, f"Expected M=16 in intrinsic {intrinsic_name}, got M={m}"
    assert n == 16, f"Expected N=16 in intrinsic {intrinsic_name}, got N={n}"

    return k // 4


def process_template(text: str, params: Dict[str, Any]) -> str:
    """Process template with Python-like control flow.

    Lines starting with $ (but not ${) are treated as Python statements.
    Other lines are output directly with ${expr} evaluated and substituted.

    Example (INTRINSICS_M=4):
        $for i in range(INTRINSICS_M):
          %val_${i} = arith.constant ${i} : index

        Generates:
          %val_0 = arith.constant 0 : index
          %val_1 = arith.constant 1 : index
          %val_2 = arith.constant 2 : index
          %val_3 = arith.constant 3 : index
    """

    def extract_leading_whitespace(line):
        """Get leading whitespace from a line."""
        match = re.match(r"^\s*", line)
        return match.group(0) if match else ""

    def escape_line(line):
        """Convert a line with ${expr} to Python print statement parts."""
        output_parts = []
        while "${" in line:
            start_pos = line.index("${")
            end_pos = line.index("}", start_pos + 2)
            if start_pos != 0:
                output_parts.append('"' + line[:start_pos].replace('"', '\\"') + '"')
            output_parts.append("str(" + line[start_pos + 2 : end_pos] + ")")
            line = line[end_pos + 1 :]
        if line:
            output_parts.append('"' + line.replace('"', '\\"') + '"')
        return " + ".join(output_parts) if output_parts else '""'

    # Convert template to Python code.
    input_lines = text.splitlines()
    python_lines = []
    blank_lines = 0
    last_indent = ""
    indent_stack = [("", "")]
    python_block_start = True

    for input_line in input_lines:
        if input_line == "":
            blank_lines += 1
            continue

        input_indent = extract_leading_whitespace(input_line)

        # Adjust indentation stack.
        if python_block_start:
            assert input_indent.startswith(
                last_indent
            ), f"Indentation error: expected indent to start with '{last_indent}', got '{input_indent}'"
            extra_indent = input_indent[len(last_indent) :]
            python_indent = indent_stack[-1][1] + extra_indent
            indent_stack.append((input_indent, python_indent))
        else:
            while not input_indent.startswith(indent_stack[-1][0]):
                indent_stack.pop()

        python_block_start = False
        python_indent = indent_stack[-1][1]
        stripped_line = input_line.strip()

        if stripped_line.startswith("$") and not stripped_line.startswith("${"):
            if stripped_line.endswith(":"):
                python_block_start = True

            while blank_lines > 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1

            python_lines.append(python_indent + stripped_line[1:])
        else:
            while blank_lines > 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1

            line_after_indent = input_line[len(python_indent) :]
            python_lines.append(
                python_indent
                + "print(%s, file=OUT_STREAM)" % escape_line(line_after_indent)
            )

        last_indent = input_indent

    while blank_lines > 0:
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1

    # Set up execution context.
    exec_globals = params.copy()
    exec_globals["FOLD1"] = fold1
    output_stream = io.StringIO()
    exec_globals["OUT_STREAM"] = output_stream

    # Compile and execute the generated Python code.
    python_code = "\n".join(python_lines)
    try:
        python_bytecode = compile(python_code, "<template>", "exec")
        exec(python_bytecode, exec_globals)
    except Exception as e:
        # Show the generated Python code for debugging.
        print("Generated Python code:")
        for i, line in enumerate(python_lines, 1):
            print(f"{i:4d}: {line}")
        raise ValueError(f"Failed to execute template: {e}")

    return output_stream.getvalue()


def parse_define(define_str: str) -> tuple:
    """Parse a -D VAR=VALUE string into (var, value)."""
    if "=" in define_str:
        var, value = define_str.split("=", 1)
        # Try to convert to int if possible.
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

    # Default parameter values.
    DEFAULT_PARAMS = {
        # Common defaults.
        "ARCH": "gfx942",
        "ELEM_TYPE": "f16",
        "INTRINSIC": "MFMA_F32_16x16x16_F16",
        # Benefit.
        "BENEFIT": 1,
        # Constraint 0: size_min=0, size_max=INT32_MAX, size_div=1.
        "SIZE_MIN_0": 0,
        "SIZE_MAX_0": 2147483647,
        "SIZE_DIV_0": 1,
        # Constraint 1: size_min=0, size_max=INT32_MAX, size_div=1.
        "SIZE_MIN_1": 0,
        "SIZE_MAX_1": 2147483647,
        "SIZE_DIV_1": 1,
        # Constraint 2: size_min=0, size_max=INT32_MAX, size_div=1.
        "SIZE_MIN_2": 0,
        "SIZE_MAX_2": 2147483647,
        "SIZE_DIV_2": 1,
        # MMA layout attributes.
        "INTRINSICS_M": 1,
        "INTRINSICS_N": 1,
        "INTRINSICS_K": 1,
        "SUBGROUPS_M": 1,
        "SUBGROUPS_N": 1,
    }

    # Start with defaults, then override with user params.
    params = DEFAULT_PARAMS.copy()
    for define in args.define:
        var, value = parse_define(define)
        params[var] = value

    # Derive ELEM_BITS from ELEM_TYPE.
    elem_type = params["ELEM_TYPE"]
    assert elem_type in ELEM_TYPE_BITS, f"Invalid element type: {elem_type}"
    params["ELEM_BITS"] = ELEM_TYPE_BITS[elem_type]

    # Derive INTERNAL_K from INTRINSIC.
    intrinsic = params["INTRINSIC"]
    params["INTERNAL_K"] = extract_internal_k(intrinsic)

    # Note: Parameter validation now happens via ${ASSERT(...)} in templates.

    # Read template.
    with open(args.template, "r") as f:
        template = f.read()

    # Process template with control flow and expressions.
    output = process_template(template, params)

    # Write output.
    with open(args.output, "w") as f:
        f.write(output)
    print(f"Generated: {args.output}")


if __name__ == "__main__":
    main()
