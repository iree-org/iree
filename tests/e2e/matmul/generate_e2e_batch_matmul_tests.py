#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generator for e2e batch matmul tests targeting CDNA block intrinsics."""

from typing import Optional
import argparse
import dataclasses
import typing

from tests.e2e.matmul.common import *
from tests.e2e.matmul.compilation_info import *


@dataclasses.dataclass
class BatchTestShape:
    batch: int
    m: int
    k: int
    n: int


# batch=32 is divisible by all block sizes (16, 4, 2).
BATCH_BLOCK_SHAPES = {
    "block_static": [
        BatchTestShape(batch=32, m=128, k=128, n=128),
        BatchTestShape(batch=32, m=256, k=64, n=256),
        BatchTestShape(batch=32, m=96, k=96, n=96),
    ],
}


def get_batch_test_shapes(shapes_id: str) -> list:
    if shapes_id not in BATCH_BLOCK_SHAPES:
        raise ValueError(
            f"Unknown batch shapes id: {shapes_id}. Valid: {list(BATCH_BLOCK_SHAPES)}"
        )
    return BATCH_BLOCK_SHAPES[shapes_id]


batch_call_id = 0
batch_matrix_seed = 0


def _generate_random_matrix_3d(
    name: str, shape: list, element_type: MatrixElemTypeId, increment_seed=True
):
    global batch_matrix_seed
    if increment_seed:
        batch_matrix_seed += 1
    d0, d1, d2 = shape
    return (
        f"  %{name}_dim0 = arith.constant {d0} : i64\n"
        f"  %{name}_dim1 = arith.constant {d1} : i64\n"
        f"  %{name}_dim2 = arith.constant {d2} : i64\n"
        f"  %{name}_element_type = hal.element_type<{element_type.value}> : i32\n"
        f"  %{name}_seed = arith.constant {batch_matrix_seed} : i32\n"
        f"  %{name} = util.call @matmul_test.generate_random_matrix_3d("
        f"%device, %{name}_dim0, %{name}_dim1, %{name}_dim2, "
        f"%{name}_element_type, %{name}_seed) "
        f": (!hal.device, i64, i64, i64, i32, i32) -> !hal.buffer_view\n"
    )


def generate_batch_function(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shape: BatchTestShape,
    compilation_info: Optional[CompilationInfo] = None,
) -> MLIRFunction:
    """Generates a linalg.batch_matmul function with compilation info."""
    b, m, k, n = shape.batch, shape.m, shape.k, shape.n

    pipeline_name = ""
    if compilation_info:
        pipeline_name = compilation_info.dispatch_lowering_pass_pipeline
        if pipeline_name.startswith("#iree_gpu.pipeline<"):
            pipeline_name = pipeline_name[len("#iree_gpu.pipeline<") : -1]

    info_suffix = f"_for_{pipeline_name}" if pipeline_name else ""
    intrinsic = (
        compilation_info.mma_schedule.intrinsic
        if compilation_info and hasattr(compilation_info, "mma_schedule")
        else ""
    )
    intrinsic_suffix = f"_{intrinsic}" if intrinsic else ""
    func_name = (
        f"batch_matmul_{b}x{m}x{k}x{lhs_rhs_type.value}_times_"
        f"{b}x{k}x{n}x{lhs_rhs_type.value}_into_{b}x{m}x{n}x{acc_type.value}"
        f"{info_suffix}{intrinsic_suffix}"
    )

    lhs_type = f"tensor<{b}x{m}x{k}x{lhs_rhs_type.value}>"
    rhs_type = f"tensor<{b}x{k}x{n}x{lhs_rhs_type.value}>"
    acc_tensor_type = f"tensor<{b}x{m}x{n}x{acc_type.value}>"

    (
        compilation_info_string,
        compilation_info_attr,
    ) = generate_compilation_info_string_and_attr(compilation_info)

    func_definition = ""
    if compilation_info_string:
        func_definition += compilation_info_string + "\n"

    func_definition += (
        f"util.func public @{func_name}(\n"
        f"    %lhs: {lhs_type},\n"
        f"    %rhs: {rhs_type},\n"
        f"    %acc: {acc_tensor_type}\n"
        f") -> {acc_tensor_type} {{\n"
        f"  %result = linalg.batch_matmul {compilation_info_attr}"
        f"ins(%lhs, %rhs : {lhs_type}, {rhs_type}) "
        f"outs(%acc : {acc_tensor_type}) -> {acc_tensor_type}\n"
        f"  util.return %result : {acc_tensor_type}\n"
        f"}}\n"
    )

    import_declaration = (
        f"util.func private @module.{func_name}"
        f"(!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view"
    )

    signature = f"{lhs_type}, {rhs_type}, {acc_tensor_type} -> {acc_tensor_type}"

    return MLIRFunction(
        name=func_name,
        signature=signature,
        definition=func_definition,
        import_declaration=import_declaration,
    )


def generate_batch_call(
    function: MLIRFunction,
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shape: BatchTestShape,
) -> TestCall:
    global batch_call_id
    b, m, k, n = shape.batch, shape.m, shape.k, shape.n

    func_name = f"call_{function.name}_{b}x{m}x{k}x{n}_{batch_call_id}"
    batch_call_id += 1

    description = f"Batch matmul shape (BxMxKxN): {b}x{m}x{k}x{n}"
    op = (
        f"util.func @{func_name}() attributes {{\n"
        f'  iree.reflection = {{description = "{description}"}}\n'
        "} {\n"
        "  %device_index = arith.constant 0 : index\n"
        "  %device = hal.devices.get %device_index : !hal.device\n"
    )

    op += _generate_random_matrix_3d("lhs", [b, m, k], lhs_rhs_type)
    op += _generate_random_matrix_3d("rhs", [b, k, n], lhs_rhs_type)
    op += _generate_random_matrix_3d("acc", [b, m, n], acc_type)
    # acc_copy has same values as acc (same seed) but is a different buffer,
    # avoiding in-place aliasing between the outs argument and the result.
    # The check reference uses %acc (the original unmodified buffer).
    op += _generate_random_matrix_3d(
        "acc_copy", [b, m, n], acc_type, increment_seed=False
    )

    op += (
        f"  %result = util.call @module.{function.name}(%lhs, %rhs, %acc_copy)"
        f" : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view\n"
        f"  %batch = arith.constant {b} : i64\n"
        f"  %m = arith.constant {m} : i64\n"
        f"  %k = arith.constant {k} : i64\n"
        f"  %n = arith.constant {n} : i64\n"
        f"  %transpose_rhs = arith.constant 0 : i32\n"
        f"  util.call @matmul_test.check_batch_matmul_results("
        f"%device, %batch, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result)"
        f" : (!hal.device, i64, i64, i64, i64, i32, "
        f"!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()\n"
        f"  util.return\n"
        f"}}\n"
    )

    return TestCall(function=function, op=op)


def generate(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    compilation_info_id: CompilationInfoId,
    shapes_id: str,
):
    functions = {}
    calls = []

    for compilation_info in get_test_compilation_infos(
        compilation_info_id, lhs_rhs_type, acc_type
    ):
        for shape in get_batch_test_shapes(shapes_id):
            function = generate_batch_function(
                lhs_rhs_type=lhs_rhs_type,
                acc_type=acc_type,
                shape=shape,
                compilation_info=compilation_info,
            )
            if function.name not in functions:
                functions[function.name] = function
            calls.append(
                generate_batch_call(
                    function=function,
                    lhs_rhs_type=lhs_rhs_type,
                    acc_type=acc_type,
                    shape=shape,
                )
            )

    return (functions, calls)


def write_code_file(functions, filename):
    with open(filename, "w") as file:
        for function in functions.values():
            file.write(function.definition + "\n")


def write_calls_file(functions, calls, filename):
    module_definition = "builtin.module @calls attributes {\n  \n} {\n\n"
    module_definition += (
        "util.func private @matmul_test.generate_random_matrix_3d("
        "%device: !hal.device, %dim0: i64, %dim1: i64, %dim2: i64, "
        "%element_type: i32, %seed: i32) -> !hal.buffer_view\n"
        "util.func private @matmul_test.check_batch_matmul_results("
        "%device: !hal.device, %batch: i64, %m: i64, %k: i64, %n: i64, "
        "%transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, "
        "%acc: !hal.buffer_view, %actual_result: !hal.buffer_view)\n\n"
    )
    for function in functions.values():
        module_definition += function.import_declaration + "\n"
    module_definition += "\n"
    for call in calls:
        module_definition += call.op + "\n"
    module_definition += "\n}\n"

    with open(filename, "w") as file:
        file.write(module_definition)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generator of e2e batch matmul tests for block intrinsics"
    )
    parser.add_argument("--output_matmul_mlir", type=str, required=True)
    parser.add_argument("--output_calls_mlir", type=str, required=True)
    parser.add_argument(
        "--lhs_rhs_type",
        type=str,
        choices=["i8", "f32", "f16", "bf16"],
        required=True,
    )
    parser.add_argument(
        "--acc_type",
        type=str,
        choices=["i32", "f32"],
        required=True,
    )
    parser.add_argument(
        "--shapes",
        type=str,
        choices=list(BATCH_BLOCK_SHAPES.keys()),
        default="block_static",
        required=False,
    )
    parser.add_argument(
        "--compilation_info",
        type=str,
        choices=[i.value for i in CompilationInfoId],
        default="",
        required=False,
    )
    return parser.parse_args()


def main(args):
    (functions, calls) = generate(
        lhs_rhs_type=MatrixElemTypeId(args.lhs_rhs_type),
        acc_type=MatrixElemTypeId(args.acc_type),
        compilation_info_id=CompilationInfoId(args.compilation_info),
        shapes_id=args.shapes,
    )
    write_code_file(functions, args.output_matmul_mlir)
    write_calls_file(functions, calls, args.output_calls_mlir)


if __name__ == "__main__":
    main(parse_arguments())
