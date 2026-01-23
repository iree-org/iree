#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""iree_generated_e2e_matmul_test generator for e2e matmul tests."""

from typing import Optional
import argparse
import dataclasses
import typing

from tests.e2e.matmul.common import *
from tests.e2e.matmul.compilation_info import *
from tests.e2e.matmul.generate_code import *


# Returns the list of TestShape's to use for the collection of shapes
# identified by shapes_id. Also for custom tests, optionally removes tests with
# an existing accumulator when accumulate is false.
def get_test_shapes(shapes_id: ShapesId, accumulate=True):
    # Notes:
    # 1. Be conservative in adding more shapes, as that can increase both the
    #    build and execution latency of tests. The build latency is nearly the
    #    same for all shapes, while execution latency grows cubicly i.e.
    #    linearly with m*k*n.
    # 2. Some shapes are commented out: they used to be tested but have been
    #    disabled to improve the trade-off between test coverage and build
    #    latency.
    if shapes_id == ShapesId.DEFAULT:
        return get_test_shapes(ShapesId.SMALL) + get_test_shapes(ShapesId.LARGE)
    if shapes_id == ShapesId.SMALL:
        return [
            # square matrices. Start by the simplest case of 1x1x1.
            TestShape(m=1, k=1, n=1, accumulate=True),
            TestShape(m=1, k=1, n=1, accumulate=False),
            # Test some small powers of two, that exercise in particular the
            # adjustment of data-tiling tile sizes to narrow cases.
            TestShape(m=2, k=2, n=2, accumulate=True),
            TestShape(m=4, k=4, n=4, accumulate=True),
            TestShape(m=8, k=8, n=8, accumulate=True),
            # test 9x9x9 because as many kernel M0/K0/N0 dims are equal to 8,
            # this will often be the smallest value that exercises something above
            # the kernel's size.
            TestShape(m=9, k=9, n=9, accumulate=True),
            # rectangular matrices.
            # >= 2x differences between M/N/K dims may exercise tiling corner cases
            # not exercised by nearly-square matrices.
            TestShape(m=6, k=13, n=3, accumulate=True),
            TestShape(m=15, k=37, n=7, accumulate=False),
            TestShape(m=81, k=19, n=41, accumulate=True),
            # shapes involving vectors (i.e. most rectangular cases)
            # This is particularly relevant because we have dedicated kernels for
            # the matrix*vector / vector*matrix case.
            TestShape(m=1, k=10, n=10, accumulate=True),  # vector*matrix
            TestShape(m=1, k=10, n=10, accumulate=False),  # vector*matrix
            TestShape(m=10, k=1, n=10, accumulate=True),  # outer-product
            TestShape(m=10, k=10, n=1, accumulate=True),  # matrix*vector
            TestShape(m=10, k=10, n=1, accumulate=False),  # matrix*vector
        ]
    if shapes_id == ShapesId.LARGE:
        return [
            # unaligned cases.
            TestShape(m=457, k=330, n=512, accumulate=False),
            TestShape(m=438, k=331, n=513, accumulate=False),
            TestShape(m=540, k=332, n=516, accumulate=False),
            TestShape(m=1000, k=4, n=512, accumulate=False),
            TestShape(m=4, k=1000, n=512, accumulate=False),
            TestShape(m=512, k=1000, n=4, accumulate=False),
            TestShape(m=513, k=128, n=55, accumulate=False),
            TestShape(m=7, k=160, n=31, accumulate=False),
            TestShape(m=512, k=330, n=33, accumulate=False),
            # shapes involving vectors (i.e. most rectangular cases)
            TestShape(m=1, k=1000, n=1000, accumulate=True),  # large vector*matrix
            TestShape(m=1000, k=1000, n=1, accumulate=True),  # large matrix*vector
            TestShape(m=1000, k=1000, n=1, accumulate=False),  # large matrix*vector
            # Be conservative in adding larger shapes. They can result in
            # high latency tests. If you have to, consider splitting them
            # out in a way that constrains the latency impact, e.g. by
            # running on fewer backends/drivers or with fewer generators
            # (see get_test_generators).
        ]
    if shapes_id == ShapesId.EASY_LARGE_STATIC:
        return [
            TestShape(m=512, k=128, n=512, accumulate=True),
            TestShape(m=512, k=128, n=512, accumulate=False),
        ]
    if shapes_id == ShapesId.CUSTOM_MNK:
        # This is used for custom shapes specified by the --mnk= flag.
        # It is expected that the caller will set the m, n, k values
        # in the TestShape instances.
        if ShapesId.custom_mnk_values is None:
            raise ValueError("Custom MNK values not set. Use --mnk=m,n,k")
        m, n, k = ShapesId.custom_mnk_values
        test_shapes = [TestShape(m=m, k=k, n=n, accumulate=False)]
        if accumulate:
            test_shapes += [
                TestShape(m=m, k=k, n=n, accumulate=True),
            ]
        return test_shapes

    raise ValueError(shapes_id)


# Generates all output files' contents as strings.
def generate(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    mx_scale_type: MatrixElemTypeId,
    mx_block_size: int,
    shapes_id: ShapesId,
    transpose_rhs: bool,
    compilation_info_id: CompilationInfoId,
    accumulate: bool,
):
    functions = {}
    calls = []

    for compilation_info in get_test_compilation_infos(
        compilation_info_id, lhs_rhs_type
    ):
        for shape in get_test_shapes(shapes_id, accumulate):
            for dynamicities in get_dynamicities(shapes_id):
                function = generate_function(
                    lhs_rhs_type=lhs_rhs_type,
                    acc_type=acc_type,
                    mx_scale_type=mx_scale_type,
                    mx_block_size=mx_block_size,
                    shape=shape,
                    transpose_rhs=transpose_rhs,
                    dynamicities=dynamicities,
                    compilation_info=compilation_info,
                )
                # Different testcases may differ only by runtime parameters but
                # share the same code. For example, dynamic-shapes testcases
                # share the same code involing tensor<?x?xf32> even though the runtime
                # value in the trace are different. That's why we append conditionally
                # to calls, but unconditionally to function_definitions.
                if function.name not in functions:
                    functions[function.name] = function
                calls.append(
                    generate_call(
                        function=function,
                        lhs_rhs_type=lhs_rhs_type,
                        acc_type=acc_type,
                        mx_scale_type=mx_scale_type,
                        mx_block_size=mx_block_size,
                        shape=shape,
                        transpose_rhs=transpose_rhs,
                    )
                )

    return (functions, calls)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generator of e2e matmul tests")
    parser.add_argument(
        "--output_matmul_mlir",
        type=str,
        help="Path of output .mlir file containing the generated matmuls",
        required=True,
    )
    parser.add_argument(
        "--output_calls_mlir",
        type=str,
        help="Path of output .mlir file containing the calls",
        required=True,
    )
    parser.add_argument(
        "--lhs_rhs_type",
        type=str,
        choices=[
            "i32",
            "i8",
            "f64",
            "f32",
            "f16",
            "bf16",
            "f8E5M2",
            "f8E4M3FN",
            "f8E5M2FNUZ",
            "f8E4M3FNUZ",
            "f6E3M2FN",
            "f6E2M3FN",
            "f4E2M1FN",
        ],
        help="Numeric type of input LHS and RHS matrices",
        required=True,
    )
    parser.add_argument(
        "--acc_type",
        type=str,
        choices=["i32", "f64", "f32", "f16", "bf16"],
        help="Numeric type of the accumulator and result matrices",
        required=True,
    )
    parser.add_argument(
        "--mx_scale_type",
        type=str,
        choices=[
            "f8E8M0FNU",
        ],
        help="Numeric type of input microscaling scales matrices",
        required=False,
    )
    parser.add_argument(
        "--mx_block_size",
        type=int,
        choices=[
            32,
        ],
        help="Numeric type of input microscaling scales matrices",
        required=False,
    )
    parser.add_argument(
        "--shapes",
        type=str,
        choices=[s.value for s in ShapesId],
        help="Collection of matrix shapes to test",
        default="default",
        required=False,
    )
    parser.add_argument(
        "--transpose_rhs",
        action="store_true",
        help="Whether to transpose RHS",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--compilation_info",
        type=str,
        choices=[i.value for i in CompilationInfoId],
        help="Collection of compilation info setups to test",
        default="",
        required=False,
    )
    parser.add_argument(
        "--requirements",
        type=str,
        help="Target requirements for this module. Comma-separated. As in -iree-llvmcpu-target-cpu-features. If the target device does not meet all of the requirements, the test will be skipped.",
        required=False,
    )
    parser.add_argument(
        "--mnk",
        type=str,
        help="Custom MNK values for CUSTOM_MNK shape. Format: m,n,k (e.g., --mnk=64,128,256)",
        required=False,
    )
    parser.add_argument(
        "--mnk_dynamicities",
        type=str,
        help="Custom dynamicity mask for m,n,k. Format: dynamic|static,dynamic|static,dynamic|static (e.g., --mnk_dynamicities=dynamic,static,static)",
        required=False,
    )
    parser.add_argument(
        "--accumulate",
        action=argparse.BooleanOptionalAction,
        help="Remove/add custom shape tests with existing accumulators, useful to set for extremely large shapes that may cause memory issues",
        required=False,
    )
    return parser.parse_args()


def write_code_file(functions, filename):
    with open(filename, "w") as file:
        for function in functions.values():
            file.write(function.definition + "\n")


def write_calls_file(functions, calls, filename, requirements):
    # Module-level reflection information used to control the test tool.
    reflection = ""
    if requirements:
        reflection = (
            "iree.reflection = {"
            'target_features = "'
            + ",".join([req.lstrip("+") for req in requirements.split(",")])
            + '"'
            "}"
        )
    module_definition = (
        f"builtin.module @calls attributes {{\n" f"  {reflection}\n" f"}} {{\n\n"
    )

    # Declare the custom module that generates arguments.
    module_definition = module_definition + (
        "util.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view\n"
        "util.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)\n"
        "\n"
    )

    # Declare the functions that will be called.
    for function in functions.values():
        module_definition = module_definition + function.import_declaration + "\n"
    module_definition = module_definition + "\n"

    # Emit the test cases for each call.
    for call in calls:
        module_definition = module_definition + call.op + "\n"

    module_definition = module_definition + "\n}\n"

    with open(filename, "w") as file:
        file.write(module_definition)


def main(args):
    # Parse custom MNK values if provided
    shapes_id = ShapesId(args.shapes)
    ShapesId.set_custom_mnk(shapes_id, args.mnk)
    ShapesId.set_dynamicity(shapes_id, args.mnk_dynamicities)

    (functions, calls) = generate(
        lhs_rhs_type=MatrixElemTypeId(args.lhs_rhs_type),
        acc_type=MatrixElemTypeId(args.acc_type),
        mx_scale_type=args.mx_scale_type,
        mx_block_size=args.mx_block_size,
        shapes_id=shapes_id,
        transpose_rhs=args.transpose_rhs,
        compilation_info_id=CompilationInfoId(args.compilation_info),
        accumulate=args.accumulate,
    )

    write_code_file(functions, args.output_matmul_mlir)
    write_calls_file(
        functions,
        calls,
        args.output_calls_mlir,
        args.requirements,
    )


if __name__ == "__main__":
    main(parse_arguments())
