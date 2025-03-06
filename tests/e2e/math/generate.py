#!/usr/bin/env python3

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate e2e tests of math ops.

Users looking to regenerate tests: do not run this directly, run generate.sh instead.
"""

import argparse
import math


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate e2e tests of math ops")
    parser.add_argument(
        "--type",
        type=str,
        help="MLIR floating-point type, e.g. f32",
        required=True,
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Target backend, e.g. llvm-cpu. Used to set tolerances wrt known bugs.",
        required=True,
    )
    parser.add_argument(
        "--header",
        type=str,
        help="File header. Echoed at the start of output.",
        required=True,
    )
    return parser.parse_args()


def evaluate_unary_float_op(op, x):
    if op == "cbrt":
        return x ** (1.0 / 3)
    if op == "exp2":
        return 2.0**x
    if op == "roundeven":
        return float(round(x))
    if op == "round":
        if 2.0 * x == round(2.0 * x):
            return math.copysign(math.ceil(math.fabs(x)), x)
        return float(round(x))
    if op == "rsqrt":
        return x**-0.5
    return float(eval(f"math.{op}({x})"))


def evaluate_binary_float_op(op, x, y):
    if op == "powf" or op == "fpowi":
        return 0.0 if x == 0 else math.pow(x, y)
    return float(eval(f"math.{op}({x}, {y})"))


def generate_unary_float_op(op, type, tolerance, input_values):
    output_values = [evaluate_unary_float_op(op, x) for x in input_values]
    size = len(input_values)
    print(
        f"""
func.func @test_{op}_{type}() -> () {{
  %input = util.unfoldable_constant dense<{input_values}> : tensor<{size}x{type}>
  %result_empty = tensor.empty() : tensor<{size}x{type}>
  %result = linalg.generic {{indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ], iterator_types = ["parallel"]}}
    ins(%input : tensor<{size}x{type}>) outs(%result_empty : tensor<{size}x{type}>) {{
  ^bb0(%in: {type}, %out: {type}):
    %result = math.{op} %in : {type}
    linalg.yield %result : {type}
  }} -> tensor<{size}x{type}>
  check.expect_almost_eq_const(%result,
    dense<{output_values}> : tensor<{size}x{type}>,
    tolerance {float(tolerance)}) :  tensor<{size}x{type}>
  return
}}"""
    )


def generate_binary_float_op(op, type, tolerance, input_values):
    input_values_x = [x for x, _ in input_values]
    input_values_y = [y for _, y in input_values]
    output_values = [evaluate_binary_float_op(op, x, y) for x, y in input_values]
    size = len(input_values)
    print(
        f"""
func.func @test_{op}_{type}() -> () {{
  %input_x = util.unfoldable_constant dense<{input_values_x}> : tensor<{size}x{type}>
  %input_y = util.unfoldable_constant dense<{input_values_y}> : tensor<{size}x{type}>
  %result_empty = tensor.empty() : tensor<{size}x{type}>
  %result = linalg.generic {{indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ], iterator_types = ["parallel"]}}
    ins(%input_x, %input_y : tensor<{size}x{type}>, tensor<{size}x{type}>) outs(%result_empty : tensor<{size}x{type}>) {{
  ^bb0(%in_x: {type}, %in_y : {type}, %out: {type}):
    %result = math.{op} %in_x, %in_y : {type}
    linalg.yield %result : {type}
  }} -> tensor<{size}x{type}>
  check.expect_almost_eq_const(%result,
    dense<{output_values}> : tensor<{size}x{type}>,
    tolerance {float(tolerance)}) :  tensor<{size}x{type}>
  return
}}"""
    )


def generate_unary_float_input_values(predicate):
    # Some functions like math.round and math.roundeven care
    # specifically about half-integral values. To ensure coverage of
    # half-integral cases, unit_subdivisions should be even.
    # Note that the correctness of the logic here rests on exact representability
    # of small integer values scaled by small powers of two.
    unit_subdivisions = 4
    float_range_radius = 10
    index_range_radius = int(float_range_radius * unit_subdivisions)
    values = []
    for x in range(-index_range_radius, index_range_radius):
        val = float(x) / unit_subdivisions
        if predicate(val):
            values.append(val)
    return values


def generate_binary_float_input_values(predicate):
    # Be conservative due to quadratic growth.
    unit_subdivisions = 4
    float_range_radius = 2
    index_range_radius = int(float_range_radius * unit_subdivisions)
    values = []
    for x in range(-index_range_radius, index_range_radius):
        val_x = float(x) / unit_subdivisions
        for y in range(-index_range_radius, index_range_radius):
            val_y = float(y) / unit_subdivisions
            if predicate(val_x, val_y):
                values.append((val_x, val_y))
    return values


def main(args):
    print(args.header)

    unary_float_ops = [
        ("acos", 1e-4 if args.type == "f32" else 1e-2, lambda x: -1 <= x <= 1),
        ("acosh", 1e-4 if args.type == "f32" else 1e-2, lambda x: x >= 1),
        ("asin", 1e-4 if args.type == "f32" else 1e-3, lambda x: -1 <= x <= 1),
        ("asinh", 1e-4 if args.type == "f32" else 1e-1, lambda x: True),
        ("atan", 1e-4 if args.type == "f32" else 1e-3, lambda x: True),
        ("atanh", 1e-4 if args.type == "f32" else 1e-3, lambda x: -1 < x < 1),
        ("cbrt", 1e-4 if args.type == "f32" else 1e-3, lambda x: x >= 0),
        ("ceil", 0, lambda x: True),
        ("cos", 1e-4 if args.type == "f32" else 1e-3, lambda x: True),
        (
            "cosh",
            1e-4 if args.type == "f32" else 1e-2,
            # Large values would require large tolerance.
            lambda x: -2 <= x <= 2,
        ),
        ("erf", 1e-4 if args.type == "f32" else 1e-3, lambda x: True),
        # TODO(#20164): uncomment erfc when compilation is fixed on CPU.
        # ("erfc", 1e-4, lambda x: True),
        (
            "exp",
            1e-4 if args.type == "f32" else 1e-3,
            # Large values would require large tolerance.
            lambda x: x <= 2,
        ),
        (
            "exp2",
            # TODO(#20163): stop tolerating 1e0 error once the bug is fixed.
            1e-4 if args.type == "f32" else 1e0 if args.backend == "llvm-cpu" else 1e-2,
            # Large values would require large tolerance.
            lambda x: x <= 2,
        ),
        (
            "expm1",
            1e-4 if args.type == "f32" else 1e-3,
            # Large values would require large tolerance.
            lambda x: x <= 2,
        ),
        ("floor", 1e-4 if args.type == "f32" else 1e-3, lambda x: True),
        ("log", 1e-4 if args.type == "f32" else 1e-3, lambda x: x > 0),
        # TODO(#20165): uncomment log10 when compilation is fixed on CPU.
        ("log1p", 1e-4 if args.type == "f32" else 1e-3, lambda x: x > 0),
        ("log2", 1e-4 if args.type == "f32" else 1e-3, lambda x: x > 0),
        ("round", 0, lambda x: True),
        ("roundeven", 0, lambda x: True),
        ("rsqrt", 1e-4 if args.type == "f32" else 1e-3, lambda x: x > 0),
        ("sin", 1e-4 if args.type == "f32" else 1e-3, lambda x: True),
        (
            "sinh",
            1e-4 if args.type == "f32" else 2e-3 if args.backend == "rocm" else 1e-3,
            # Large values would require large tolerance.
            lambda x: -2 <= x <= 2,
        ),
        ("sqrt", 1e-4 if args.type == "f32" else 1e-3, lambda x: x >= 0),
        (
            "tan",
            1e-4 if args.type == "f32" else 1e-2,
            # Large values would require large tolerance.
            lambda x: -2 <= x <= 2,
        ),
        ("tanh", 1e-4 if args.type == "f32" else 1e-3, lambda x: True),
        # TODO(#20165): uncomment trunc when compilation is fixed on CPU.
        # ("trunc", 0, lambda x: True),
    ]
    for op, tolerance, domain_predicate in unary_float_ops:
        input_values = generate_unary_float_input_values(domain_predicate)
        generate_unary_float_op(op, args.type, tolerance, input_values)

    binary_float_ops = [
        ("atan2", 1e-4 if args.type == "f32" else 1e-3, lambda x, y: x != 0 or y != 0),
        (
            "powf",
            # TODO(#20163): stop tolerating 1e0 error once the bug is fixed.
            1e-4 if args.type == "f32" else 1e0 if args.backend == "llvm-cpu" else 1e-2,
            # Avoid large power values, which are hard to constrain in the space of input values
            # because they arise when x is small and y is negative.
            # TODO: determine upstream if math.powf should implement pow(0,0)==1
            # as in the floating point standard. Current lowerings do not. Doing it
            # would incur a small overhead.
            lambda x, y: x > 0 and math.pow(x, y) < 10,
        ),
    ]
    for op, tolerance, domain_predicate in binary_float_ops:
        input_values = generate_binary_float_input_values(domain_predicate)
        generate_binary_float_op(op, args.type, tolerance, input_values)


if __name__ == "__main__":
    main(parse_arguments())
