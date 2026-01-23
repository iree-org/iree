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
from tests.e2e.matmul.generate_code_mx import *


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


# Generates a test function in the generated MLIR code.
# The generated function will take the same arguments as linalg.matmul variants
# and will just call linalg.matmul variants with them, returning its result.
def generate_function(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    mx_scale_type: MatrixElemTypeId,
    mx_block_size: int,
    shape: TestShape,
    transpose_rhs: bool,
    dynamicities: tuple[Dynamicity, Dynamicity, Dynamicity],
    compilation_info: Optional[CompilationInfo] = None,
):
    if mx_scale_type:
        return generate_function_mx(
            lhs_rhs_type=lhs_rhs_type,
            acc_type=acc_type,
            mx_scale_type=mx_scale_type,
            mx_block_size=mx_block_size,
            shape=shape,
            transpose_rhs=transpose_rhs,
            dynamicities=dynamicities,
            compilation_info=compilation_info,
        )

    shapes = generate_shapes(shape, transpose_rhs, dynamicities)
    func_name = generate_function_name(
        lhs_rhs_type=lhs_rhs_type,
        acc_type=acc_type,
        shapes=shapes,
        accumulate=shape.accumulate,
        compilation_info=compilation_info,
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

    (
        compilation_info_string,
        compilation_info_attr,
    ) = generate_compilation_info_string_and_attr(compilation_info)

    args = [("%lhs", lhs_tensor_type), ("%rhs", rhs_tensor_type)]
    if shape.accumulate:
        args += [("%acc", acc_tensor_type)]

    func_definition = compilation_info_string + (
        f"util.func @{func_name}("
        + ", ".join([name + ": " + ty for name, ty in args])
        + f") -> {acc_tensor_type} {{\n"
    )

    if not shape.accumulate:
        literal_zero_for_acc_type = "0.0" if "f" in acc_type.value else "0"
        if acc_r == "?" and acc_c == "?":
            func_definition += (
                f"  %c0 = arith.constant 0 : index\n"
                f"  %c1 = arith.constant 1 : index\n"
                f"  %acc_dim0 = tensor.dim %lhs, %c0 : {lhs_tensor_type}\n"
                f"  %acc_dim1 = tensor.dim %rhs, %c1 : {rhs_tensor_type}\n"
                f"  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : {acc_tensor_type}\n"
            )
        elif acc_r == "?":
            func_definition += (
                f"  %c0 = arith.constant 0 : index\n"
                f"  %acc_dim0 = tensor.dim %lhs, %c0 : {lhs_tensor_type}\n"
                f"  %init_acc = tensor.empty(%acc_dim0) : {acc_tensor_type}\n"
            )
        elif acc_c == "?":
            func_definition += (
                f"  %c1 = arith.constant 1 : index\n"
                f"  %acc_dim1 = tensor.dim %rhs, %c1 : {rhs_tensor_type}\n"
                f"  %init_acc = tensor.empty(%acc_dim1) : {acc_tensor_type}\n"
            )
        else:
            func_definition += f"  %init_acc = tensor.empty() : {acc_tensor_type}\n"
        func_definition += (
            f"  %c0_acc_type = arith.constant {literal_zero_for_acc_type}: {acc_type.value}\n"
            f"  %acc = linalg.fill ins(%c0_acc_type : {acc_type.value}) outs(%init_acc : {acc_tensor_type}) -> {acc_tensor_type}\n"
        )

    indexing_maps_attr = ""
    if transpose_rhs:
        indexing_maps_attr = "indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]"
    func_definition += (
        f"  %result = linalg.matmul {indexing_maps_attr} {compilation_info_attr} ins(%lhs, %rhs: {lhs_tensor_type}, {rhs_tensor_type}) outs(%acc: {acc_tensor_type}) -> {acc_tensor_type}\n"
        f"  util.return %result: {acc_tensor_type}\n"
        f"}}\n"
    )

    signature = ", ".join([ty for name, ty in args]) + " -> {acc_tensor_type}"
    import_declaration = (
        f"util.func private @module.{func_name}("
        + ", ".join([name + ": !hal.buffer_view" for name, ty in args])
        + ") -> !hal.buffer_view"
    )
    return MLIRFunction(
        name=func_name,
        signature=signature,
        import_declaration=import_declaration,
        definition=func_definition,
    )


call_id = 0


# Generates the output trace for a testcase i.e. a single test function call,
# as a dictionary to be passed to yaml.dump.
def generate_call(
    function: MLIRFunction,
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    mx_scale_type: MatrixElemTypeId,
    mx_block_size: int,
    shape: TestShape,
    transpose_rhs: bool,
):
    if mx_scale_type:
        return generate_call_mx(
            function=function,
            lhs_rhs_type=lhs_rhs_type,
            acc_type=acc_type,
            mx_scale_type=mx_scale_type,
            mx_block_size=mx_block_size,
            shape=shape,
            transpose_rhs=transpose_rhs,
        )

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
    matmul_args = [
        ("lhs", lhs_shape, lhs_rhs_type),
        ("rhs", rhs_shape, lhs_rhs_type),
    ]
    check_args = matmul_args.copy()

    if shape.accumulate:
        matmul_args += [("acc", [shape.m, shape.n], acc_type)]
    else:
        op += "  %acc = util.null : !hal.buffer_view\n"

    for arg_name, arg_shape, arg_elemtype in matmul_args:
        op = op + generate_random_matrix(arg_name, arg_shape, arg_elemtype)
        # TODO(#16168): there's a bug with in-place input->output aliasing and
        # we work around it here by passing in a unique copy.
        if arg_name == "acc":
            # TODO(#16168): there's a bug with in-place input->output aliasing and
            # we work around it here by passing in a unique copy.
            op = op + generate_random_matrix(
                "acc_copy", arg_shape, arg_elemtype, increment_seed=False
            )
    gen_names_and_types = lambda args_list: (
        ", ".join(["%" + name for name, shape, ty in args_list]),
        ", ".join(["!hal.buffer_view" for a in args_list]),
    )
    matmul_argnames, matmul_argtypes = gen_names_and_types(matmul_args)
    check_argnames, check_argtypes = gen_names_and_types(check_args)
    op += (
        f"  %result = util.call @module.{function.name}({matmul_argnames}) : ({matmul_argtypes}) -> !hal.buffer_view\n"
        f"  %m = arith.constant {shape.m} : i64\n"
        f"  %k = arith.constant {shape.k} : i64\n"
        f"  %n = arith.constant {shape.n} : i64\n"
        f"  %transpose_rhs = arith.constant {transpose_rhs} : i32\n"
        f"  util.call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, {check_argnames}, {'%acc_copy' if shape.accumulate else '%acc'}, %result) : (!hal.device, i64, i64, i64, i32,  {check_argtypes}, !hal.buffer_view, !hal.buffer_view) -> ()\n"
        f"  util.return\n"
        f"}}\n"
    )

    return TestCall(function=function, op=op)
