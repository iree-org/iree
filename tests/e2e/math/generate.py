#!/usr/bin/env python3

# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate e2e tests of math ops."""

import argparse
import dataclasses
import enum
import math
import json
import sys
import typing


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate e2e tests of math ops")
    parser.add_argument(
        "--testcases",
        type=str,
        help="JSON file listing the ops to be tested.",
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


def generate_unary_float_op(op, type, atol, rtol, input_values):
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
    atol {float(atol):.3e}, rtol {float(rtol):.3e}) :  tensor<{size}x{type}>
  return
}}"""
    )


def generate_binary_float_op(op, type, atol, rtol, input_values):
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
    atol {float(atol):.3e}, rtol {float(rtol):.3e}) :  tensor<{size}x{type}>
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
    float_range_radius = 16
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
    float_range_radius = 4
    index_range_radius = int(float_range_radius * unit_subdivisions)
    values = []
    for x in range(-index_range_radius, index_range_radius):
        val_x = float(x) / unit_subdivisions
        for y in range(-index_range_radius, index_range_radius):
            val_y = float(y) / unit_subdivisions
            if predicate(val_x, val_y):
                values.append((val_x, val_y))
    return values


@enum.unique
class MathOpKind(enum.Enum):
    UNARY_FLOAT = 1
    BINARY_FLOAT = 2


@dataclasses.dataclass
class MathOpInfo:
    kind: MathOpKind
    domain: typing.Callable


def get_math_op_info():
    return {
        "acos": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: -1 <= x <= 1,
        ),
        "acosh": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: x >= 1,
        ),
        "asin": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: -1 <= x <= 1,
        ),
        "asinh": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "atan": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "atan2": MathOpInfo(
            kind=MathOpKind.BINARY_FLOAT,
            domain=lambda x, y: x != 0 or y != 0,
        ),
        "atanh": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: -1 < x < 1,
        ),
        "cbrt": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: x >= 0,
        ),
        "ceil": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "cos": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "cosh": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "erf": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "exp": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "exp2": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "expm1": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "floor": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "log": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: x > 0,
        ),
        "log1p": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: x > -1,
        ),
        "log2": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: x > 0,
        ),
        "powf": MathOpInfo(
            kind=MathOpKind.BINARY_FLOAT,
            domain=lambda x, y: x > 0,
        ),
        "round": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "roundeven": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "rsqrt": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: x > 0,
        ),
        "sin": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "sinh": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "sqrt": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: x > 0,
        ),
        "tan": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
        "tanh": MathOpInfo(
            kind=MathOpKind.UNARY_FLOAT,
            domain=lambda x: True,
        ),
    }


def main(args):
    with open(args.testcases, "r") as testcases_file:
        testcases = json.load(testcases_file)

    math_op_info = get_math_op_info()
    ops_not_yet_encountered = {op for op in math_op_info}

    for testcase in testcases:
        op = testcase["op"]
        ops_not_yet_encountered.discard(op)
        info = math_op_info[op]
        kind = info.kind
        type = testcase["type"]
        atol = testcase["atol"]
        rtol = testcase["rtol"]
        if kind == MathOpKind.UNARY_FLOAT:
            # Combine the op's inherent domain (info.domain) with optional
            # testcase-specific restrictions.
            domain = lambda x: (
                info.domain(x)
                and (x >= testcase["xmin"] if "xmin" in testcase else True)
                and (x <= testcase["xmax"] if "xmax" in testcase else True)
            )
            input_values = generate_unary_float_input_values(domain)
            generate_unary_float_op(op, type, atol, rtol, input_values)
        elif kind == MathOpKind.BINARY_FLOAT:
            # Combine the op's inherent domain (info.domain) with optional
            # testcase-specific restrictions.
            domain = lambda x, y: (
                info.domain(x, y)
                and (x >= testcase["xmin"] if "xmin" in testcase else True)
                and (x <= testcase["xmax"] if "xmax" in testcase else True)
                and (y >= testcase["ymin"] if "ymin" in testcase else True)
                and (y <= testcase["ymax"] if "ymax" in testcase else True)
            )
            input_values = generate_binary_float_input_values(domain)
            generate_binary_float_op(op, type, atol, rtol, input_values)
        else:
            raise ValueError(f"Unhandled op kind: {info.kind}")

    if ops_not_yet_encountered:
        print(
            f"Warning: did not find testcases covering {', '.join(ops_not_yet_encountered)} in {args.testcases}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main(parse_arguments())
