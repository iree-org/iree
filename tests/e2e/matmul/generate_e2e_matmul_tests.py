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


# Returns the list of TestShape's to use for the collection of shapes
# identified by shapes_id.
def get_test_shapes(shapes_id: ShapesId):
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
        return [
            TestShape(m=m, k=k, n=n, accumulate=True),
            TestShape(m=m, k=k, n=n, accumulate=False),
        ]

    raise ValueError(shapes_id)


# Describes the fully resolved shape dimensions of all 3 input matrices,
# LHS, RHS, and Accumulator, in a testcase.
# Each value is a string, which may either represent a positive integer such as "123",
# or a "?" string, meaning a dynamic dimension as in MLIR.
# These string values are used to generate MLIR function names and tensor shapes.
@dataclasses.dataclass
class TestInputMatricesShapes:
    lhs_rows: DimSize
    lhs_cols: DimSize
    rhs_rows: DimSize
    rhs_cols: DimSize
    acc_rows: DimSize
    acc_cols: DimSize


# Helper for generate_function. Generates TestInputMatricesShapes, i.e.
# converts from the runtime shape dimensions in TestShape and given dynamicity to
# the set of shapes to be used in a test function's input tensors.
def generate_shapes(shape: TestShape, transpose_rhs: bool, dynamicity: Dynamicity):
    lhs_rows = shape_dim(shape.m, dynamicity)
    lhs_cols = shape_dim(shape.k, dynamicity)
    acc_rows = shape_dim(shape.m, dynamicity)
    acc_cols = shape_dim(shape.n, dynamicity)
    if transpose_rhs:
        rhs_rows = shape_dim(shape.n, dynamicity)
        rhs_cols = shape_dim(shape.k, dynamicity)
    else:
        rhs_rows = shape_dim(shape.k, dynamicity)
        rhs_cols = shape_dim(shape.n, dynamicity)
    shapes = TestInputMatricesShapes(
        lhs_rows=lhs_rows,
        lhs_cols=lhs_cols,
        rhs_rows=rhs_rows,
        rhs_cols=rhs_cols,
        acc_rows=acc_rows,
        acc_cols=acc_cols,
    )
    return shapes


# Helper for generate_function.
# Generates a name for a test function in the generated MLIR code.
def generate_function_name(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shapes: TestInputMatricesShapes,
    accumulate: bool,
    compilation_info: typing.Optional[CompilationInfo] = None,
):
    input_t = lhs_rhs_type.value
    acc_t = acc_type.value
    lhs_r = int_or_DYN(shapes.lhs_rows)
    lhs_c = int_or_DYN(shapes.lhs_cols)
    rhs_r = int_or_DYN(shapes.rhs_rows)
    rhs_c = int_or_DYN(shapes.rhs_cols)
    acc_r = int_or_DYN(shapes.acc_rows)
    acc_c = int_or_DYN(shapes.acc_cols)

    info = ""
    if compilation_info:
        info = f"_for_{compilation_info.dispatch_lowering_pass_pipeline}"

    matmul_kind = "matmul_accumulate" if accumulate else "matmul"
    return (
        f"{matmul_kind}_{lhs_r}x{lhs_c}x{input_t}_times_"
        + f"{rhs_r}x{rhs_c}x{input_t}_into_{acc_r}x{acc_c}x{acc_t}{info}"
    )


# Represents a generated test function.
@dataclasses.dataclass
class MLIRFunction:
    name: str
    signature: str
    import_declaration: str
    definition: str


# Generates a test function in the generated MLIR code.
# The generated function will take the same arguments as linalg.matmul variants
# and will just call linalg.matmul variants with them, returning its result.
def generate_function(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shape: TestShape,
    transpose_rhs: bool,
    dynamicity: Dynamicity,
    compilation_info: Optional[CompilationInfo] = None,
):
    shapes = generate_shapes(shape, transpose_rhs, dynamicity)
    func_name = generate_function_name(
        lhs_rhs_type, acc_type, shapes, shape.accumulate, compilation_info
    )
    lhs_r = int_or_question_mark(shapes.lhs_rows)
    lhs_c = int_or_question_mark(shapes.lhs_cols)
    rhs_r = int_or_question_mark(shapes.rhs_rows)
    rhs_c = int_or_question_mark(shapes.rhs_cols)
    acc_r = int_or_question_mark(shapes.acc_rows)
    acc_c = int_or_question_mark(shapes.acc_cols)

    lhs_tensor_type = f"tensor<{lhs_r}x{lhs_c}x{lhs_rhs_type.value}>"
    rhs_tensor_type = f"tensor<{rhs_r}x{rhs_c}x{lhs_rhs_type.value}>"
    acc_tensor_type = f"tensor<{acc_r}x{acc_c}x{acc_type.value}>"

    if transpose_rhs:
        op_name = "linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]"
    else:
        op_name = "linalg.matmul"

    # Compilation info is optional; prints empty string by default.
    (
        compilation_info_string,
        compilation_info_attr,
    ) = generate_compilation_info_string_and_attr(compilation_info)
    func_definition = compilation_info_string
    compute = f"  %result = {op_name} {compilation_info_attr}ins(%lhs, %rhs: {lhs_tensor_type}, {rhs_tensor_type}) outs(%acc: {acc_tensor_type}) -> {acc_tensor_type}\n"
    if shape.accumulate:
        signature = f"({lhs_tensor_type}, {rhs_tensor_type}, {acc_tensor_type}) -> {acc_tensor_type}"
        import_declaration = f"util.func private @module.{func_name}(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view"
        func_definition = func_definition + (
            f"util.func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}, %acc: {acc_tensor_type}) -> {acc_tensor_type} {{\n"
            f"{compute}\n"
            f"  util.return %result: {acc_tensor_type}\n"
            f"}}\n"
        )
    else:
        literal_zero_for_acc_type = "0.0" if "f" in acc_type.value else "0"
        if acc_r == "?":
            signature = f"({lhs_tensor_type}, {rhs_tensor_type}) -> {acc_tensor_type}"
            import_declaration = f"util.func private @module.{func_name}(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view"
            func_definition = func_definition + (
                f"util.func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}) -> {acc_tensor_type} {{\n"
                f"  %c0 = arith.constant 0 : index\n"
                f"  %c1 = arith.constant 1 : index\n"
                f"  %acc_dim0 = tensor.dim %lhs, %c0 : {lhs_tensor_type}\n"
                f"  %acc_dim1 = tensor.dim %rhs, %c1 : {rhs_tensor_type}\n"
                f"  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : {acc_tensor_type}\n"
                f"  %c0_acc_type = arith.constant {literal_zero_for_acc_type}: {acc_type.value}\n"
                f"  %acc = linalg.fill ins(%c0_acc_type : {acc_type.value}) outs(%init_acc : {acc_tensor_type}) -> {acc_tensor_type}\n"
                f"{compute}"
                f"  util.return %result: {acc_tensor_type}\n"
                f"}}\n"
            )
        else:
            signature = f"({lhs_tensor_type}, {rhs_tensor_type}) -> {acc_tensor_type}"
            import_declaration = f"util.func private @module.{func_name}(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view"
            func_definition = func_definition + (
                f"util.func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}) -> {acc_tensor_type} {{\n"
                f"  %init_acc = tensor.empty() : {acc_tensor_type}\n"
                f"  %c0_acc_type = arith.constant {literal_zero_for_acc_type}: {acc_type.value}\n"
                f"  %acc = linalg.fill ins(%c0_acc_type : {acc_type.value}) outs(%init_acc : {acc_tensor_type}) -> {acc_tensor_type}\n"
                f"{compute}"
                f"  util.return %result: {acc_tensor_type}\n"
                f"}}\n"
            )
    return MLIRFunction(
        name=func_name,
        signature=signature,
        import_declaration=import_declaration,
        definition=func_definition,
    )


# Represents a call to a generated test function.
@dataclasses.dataclass
class TestCall:
    function: MLIRFunction
    op: str


random_matrix_index = 0


# Generate a matrix function argument of the given size as `%name`.
def generate_random_matrix(
    name: str,
    matrix_shape: list,
    element_type: MatrixElemTypeId,
):
    global random_matrix_index
    random_matrix_index += 1
    return (
        f"  %{name}_dim0 = arith.constant {matrix_shape[0]} : i64\n"
        f"  %{name}_dim1 = arith.constant {matrix_shape[1]} : i64\n"
        f"  %{name}_element_type = hal.element_type<{element_type.value}> : i32\n"
        f"  %{name}_seed = arith.constant {random_matrix_index} : i32\n"
        f"  %{name} = util.call @matmul_test.generate_random_matrix(%device, %{name}_dim0, %{name}_dim1, %{name}_element_type, %{name}_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view\n"
    )


call_id = 0


# Generates the output trace for a testcase i.e. a single test function call,
# as a dictionary to be passed to yaml.dump.
def generate_call(
    function: MLIRFunction,
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shape: TestShape,
    transpose_rhs: bool,
):
    global call_id
    func_name = f"{function.name}_{shape.m}_{shape.k}_{shape.n}"
    if shape.accumulate:
        func_name = f"{func_name}_acc"
    func_name = f"{func_name}_{call_id}"
    call_id = call_id + 1

    description = f"Matmul shape (MxKxN): {shape.m}x{shape.k}x{shape.n}"
    op = (
        f"util.func @{func_name}() attributes {{\n"
        f'  iree.reflection = {{description = "{description}"}}\n'
        "} {\n"
        "  %device_index = arith.constant 0 : index\n"
        "  %device = hal.devices.get %device_index : !hal.device\n"
    )

    lhs_shape = [shape.m, shape.k]
    if transpose_rhs:
        rhs_shape = [shape.n, shape.k]
        transpose_rhs = 1
    else:
        rhs_shape = [shape.k, shape.n]
        transpose_rhs = 0

    op = op + generate_random_matrix("lhs", lhs_shape, lhs_rhs_type)
    op = op + generate_random_matrix("rhs", rhs_shape, lhs_rhs_type)
    if shape.accumulate:
        op = op + generate_random_matrix("acc", [shape.m, shape.n], acc_type)
        # TODO(#16168): there's a bug with in-place input->output aliasing and
        # we work around it here by passing in a unique copy.
        global random_matrix_index
        random_matrix_index -= 1
        op = op + generate_random_matrix("acc_copy", [shape.m, shape.n], acc_type)
        op = op + (
            f"  %result = util.call @module.{function.name}(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view\n"
        )
    else:
        op = op + (
            f"  %acc = util.null : !hal.buffer_view\n"
            f"  %result = util.call @module.{function.name}(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view\n"
        )

    op = op + (
        f"  %m = arith.constant {shape.m} : i64\n"
        f"  %k = arith.constant {shape.k} : i64\n"
        f"  %n = arith.constant {shape.n} : i64\n"
        f"  %transpose_rhs = arith.constant {transpose_rhs} : i32\n"
        f"  util.call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()\n"
    )

    op = op + "  util.return\n"
    op = op + "}\n"

    return TestCall(function=function, op=op)


# Generates all output files' contents as strings.
def generate(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shapes_id: ShapesId,
    transpose_rhs: bool,
    compilation_info_id: CompilationInfoId,
):
    functions = {}
    calls = []

    for compilation_info in get_test_compilation_infos(
        compilation_info_id, lhs_rhs_type
    ):
        for shape in get_test_shapes(shapes_id):
            for dynamicity in get_dynamicities(shapes_id):
                function = generate_function(
                    lhs_rhs_type,
                    acc_type,
                    shape,
                    transpose_rhs,
                    dynamicity,
                    compilation_info,
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
                        function, lhs_rhs_type, acc_type, shape, transpose_rhs
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
    lhs_rhs_type = MatrixElemTypeId(args.lhs_rhs_type)
    acc_type = MatrixElemTypeId(args.acc_type)
    shapes_id = ShapesId(args.shapes)
    compilation_info_id = CompilationInfoId(args.compilation_info)

    # Parse custom MNK values if provided
    ShapesId.set_custom_mnk(shapes_id, args.mnk)

    (functions, calls) = generate(
        lhs_rhs_type, acc_type, shapes_id, args.transpose_rhs, compilation_info_id
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
