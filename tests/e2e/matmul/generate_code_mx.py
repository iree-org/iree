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


# Helper for generate_function.
# Generates a name for a test function in the generated MLIR code.
def generate_function_name_mx(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    mx_scale_type: MatrixElemTypeId,
    mx_block_size: int,
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
        f"{matmul_kind}_mx_scale_{mx_scale_type}_{lhs_r}x{lhs_c // mx_block_size}x{mx_block_size}x{input_t}_times_"
        + f"{rhs_r}x{rhs_c // mx_block_size}x{mx_block_size}x{input_t}_into_{acc_r}x{acc_c}x{acc_t}"
    )


# Generates a test function in the generated MLIR code.
def generate_function_mx(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    mx_scale_type: MatrixElemTypeId,
    mx_block_size: int,
    shape: TestShape,
    transpose_rhs: bool,
    dynamicities: tuple[Dynamicity, Dynamicity, Dynamicity],
    compilation_info: Optional[CompilationInfo] = None,
):
    if not transpose_rhs:
        raise ValueError("MX tests require transpose_rhs")
    if dynamicities != (Dynamicity.STATIC, Dynamicity.STATIC, Dynamicity.STATIC):
        raise ValueError("MX tests require static shape")
    if shape.k % mx_block_size:
        raise ValueError(
            f"MX tests require that shape.k ({shape.k}) be a multiple of mx_block_size ({mx_block_size})"
        )

    shapes = generate_shapes(shape, transpose_rhs, dynamicities)
    func_name = generate_function_name_mx(
        lhs_rhs_type=lhs_rhs_type,
        acc_type=acc_type,
        mx_scale_type=mx_scale_type,
        mx_block_size=mx_block_size,
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

    lhs_scales_shape = [shape.m, shape.k // mx_block_size]
    rhs_scales_shape = [shape.n, shape.k // mx_block_size]
    lhs_r = lhs_scales_shape[0]
    lhs_c = lhs_scales_shape[1]
    rhs_r = rhs_scales_shape[0]
    rhs_c = rhs_scales_shape[1]
    mx_block_bytes = mx_block_size * get_size_in_bits(lhs_rhs_type) // 8
    lhs_bytes_tensor_type = f"tensor<{lhs_r}x{lhs_c * mx_block_bytes}xi8>"
    rhs_bytes_tensor_type = f"tensor<{rhs_r}x{rhs_c * mx_block_bytes}xi8>"
    lhs_bytes_expanded_tensor_type = f"tensor<{lhs_r}x{lhs_c}x{mx_block_bytes}xi8>"
    rhs_bytes_expanded_tensor_type = f"tensor<{rhs_r}x{rhs_c}x{mx_block_bytes}xi8>"
    lhs_expanded_tensor_type = (
        f"tensor<{lhs_r}x{lhs_c}x{mx_block_size}x{lhs_rhs_type.value}>"
    )
    rhs_expanded_tensor_type = (
        f"tensor<{rhs_r}x{rhs_c}x{mx_block_size}x{lhs_rhs_type.value}>"
    )
    lhs_scales_tensor_type = f"tensor<{lhs_r}x{lhs_c}x{mx_scale_type}>"
    rhs_scales_tensor_type = f"tensor<{rhs_r}x{rhs_c}x{mx_scale_type}>"
    args = [
        ("%lhs_bytes", lhs_bytes_tensor_type),
        ("%lhs_scales", lhs_scales_tensor_type),
        ("%rhs_bytes", rhs_bytes_tensor_type),
        ("%rhs_scales", rhs_scales_tensor_type),
    ]

    if shape.accumulate:
        args += [("%acc", acc_tensor_type)]

    func_definition = compilation_info_string + (
        f"util.func @{func_name}("
        + ", ".join([name + ": " + ty for name, ty in args])
        + f") -> {acc_tensor_type} {{\n"
    )

    if not shape.accumulate:
        literal_zero_for_acc_type = "0.0" if "f" in acc_type.value else "0"
        if acc_r == "?":
            func_definition += (
                f"  %c0 = arith.constant 0 : index\n"
                f"  %c1 = arith.constant 1 : index\n"
                f"  %acc_dim0 = tensor.dim %lhs, %c0 : {lhs_tensor_type}\n"
                f"  %acc_dim1 = tensor.dim %rhs, %c1 : {rhs_tensor_type}\n"
                f"  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : {acc_tensor_type}\n"
            )
        else:
            func_definition += f"  %init_acc = tensor.empty() : {acc_tensor_type}\n"
        func_definition += (
            f"  %c0_acc_type = arith.constant {literal_zero_for_acc_type}: {acc_type.value}\n"
            f"  %acc = linalg.fill ins(%c0_acc_type : {acc_type.value}) outs(%init_acc : {acc_tensor_type}) -> {acc_tensor_type}\n"
        )

    func_definition += (
        f"  %lhs_bytes_expanded = tensor.expand_shape %lhs_bytes [[0], [1, 2]] output_shape [{lhs_r}, {lhs_c}, {mx_block_bytes}] : {lhs_bytes_tensor_type} into {lhs_bytes_expanded_tensor_type}\n"
        f"  %rhs_bytes_expanded = tensor.expand_shape %rhs_bytes [[0], [1, 2]] output_shape [{rhs_r}, {rhs_c}, {mx_block_bytes}] : {rhs_bytes_tensor_type} into {rhs_bytes_expanded_tensor_type}\n"
        f"  %lhs_expanded = flow.tensor.bitcast %lhs_bytes_expanded : {lhs_bytes_expanded_tensor_type} -> {lhs_expanded_tensor_type}\n"
        f"  %rhs_expanded = flow.tensor.bitcast %rhs_bytes_expanded : {rhs_bytes_expanded_tensor_type} -> {rhs_expanded_tensor_type}\n"
        f"  %result = linalg.generic {{\n"
        f"                indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,\n"
        f"                                 affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>,\n"
        f"                                 affine_map<(d0, d1, d2, d3) -> (d0, d2)>,\n"
        f"                                 affine_map<(d0, d1, d2, d3) -> (d1, d2)>,\n"
        f"                                 affine_map<(d0, d1, d2, d3) -> (d0, d1)>],\n"
        f'                iterator_types = ["parallel", "parallel", "reduction", "reduction"] }}\n'
        f"                ins(%lhs_expanded, %rhs_expanded, %lhs_scales, %rhs_scales : {lhs_expanded_tensor_type}, {rhs_expanded_tensor_type}, {lhs_scales_tensor_type}, {rhs_scales_tensor_type})\n"
        f"                outs(%acc : {acc_tensor_type}) {{\n"
        f"  ^bb0(%l: f4E2M1FN, %r: f4E2M1FN, %ls: f8E8M0FNU, %rs: f8E8M0FNU, %a: f32):\n"
        f"    %lscaled = arith.scaling_extf %l, %ls : f4E2M1FN, f8E8M0FNU to f32\n"
        f"    %rscaled = arith.scaling_extf %r, %rs : f4E2M1FN, f8E8M0FNU to f32\n"
        f"    %m = arith.mulf %lscaled, %rscaled : f32\n"
        f"    %y = arith.addf %a, %m : f32\n"
        f"    linalg.yield %y : f32\n"
        f"  }} -> {acc_tensor_type}\n"
        f"  util.return %result : {acc_tensor_type}\n"
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


call_id_mx = 0


# Generates the output trace for a testcase i.e. a single test function call,
# as a dictionary to be passed to yaml.dump.
def generate_call_mx(
    function: MLIRFunction,
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    mx_scale_type: MatrixElemTypeId,
    mx_block_size: int,
    shape: TestShape,
    transpose_rhs: bool,
):
    if not transpose_rhs:
        raise ValueError("MX tests require transpose_rhs")
    transpose_rhs = 1

    global call_id_mx
    func_name = f"{function.name}_{shape.m}_{shape.k}_{shape.n}"
    if shape.accumulate:
        func_name = f"{func_name}_acc"
    func_name = f"{func_name}_{call_id_mx}"
    call_id_mx = call_id_mx + 1

    description = f"Matmul shape (MxKxN): {shape.m}x{shape.k}x{shape.n}"
    op = (
        f"util.func @{func_name}() attributes {{\n"
        f'  iree.reflection = {{description = "{description}"}}\n'
        "} {\n"
        "  %device_index = arith.constant 0 : index\n"
        "  %device = hal.devices.get %device_index : !hal.device\n"
    )

    lhs_scales_shape = [shape.m, shape.k // mx_block_size]
    rhs_scales_shape = [shape.n, shape.k // mx_block_size]
    lhs_rhs_mx_block_bytes = mx_block_size * get_size_in_bits(lhs_rhs_type) // 8
    lhs_bytes_shape = [
        lhs_scales_shape[0],
        lhs_scales_shape[1] * lhs_rhs_mx_block_bytes,
    ]
    rhs_bytes_shape = [
        rhs_scales_shape[0],
        rhs_scales_shape[1] * lhs_rhs_mx_block_bytes,
    ]
    matmul_args = [
        # UI8 means generate random bytes. Good for MXFP4.
        ("lhs_bytes", lhs_bytes_shape, MatrixElemTypeId.UI8),
        ("lhs_scales", lhs_scales_shape, MatrixElemTypeId(mx_scale_type)),
        ("rhs_bytes", rhs_bytes_shape, MatrixElemTypeId.UI8),
        ("rhs_scales", rhs_scales_shape, MatrixElemTypeId(mx_scale_type)),
    ]

    if shape.accumulate:
        matmul_args += [("acc", [shape.m, shape.n], acc_type)]
    else:
        op += "  %acc = util.null : !hal.buffer_view\n"

    for arg_name, arg_shape, arg_elemtype in matmul_args:
        op = op + generate_random_matrix(arg_name, arg_shape, arg_elemtype)

    gen_names_and_types = lambda args_list: (
        ", ".join(["%" + name for name, shape, ty in args_list]),
        ", ".join(["!hal.buffer_view" for a in args_list]),
    )
    matmul_argnames, matmul_argtypes = gen_names_and_types(matmul_args)
    op = op + (
        f"  %result = util.call @module.{function.name}({matmul_argnames}) : ({matmul_argtypes}) -> !hal.buffer_view\n"
        f"  %m = arith.constant {shape.m} : i64\n"
        f"  %k = arith.constant {shape.k} : i64\n"
        f"  %n = arith.constant {shape.n} : i64\n"
        f"  %transpose_rhs = arith.constant {transpose_rhs} : i32\n"
    )
    lhs_r = lhs_scales_shape[0]
    lhs_c = lhs_scales_shape[1]
    rhs_r = rhs_scales_shape[0]
    rhs_c = rhs_scales_shape[1]
    lhs_rhs_mx_block_bytes = mx_block_size * get_size_in_bits(lhs_rhs_type) // 8
    lhs_bytes_tensor_type = f"tensor<{lhs_r}x{lhs_c * lhs_rhs_mx_block_bytes}xi8>"
    rhs_bytes_tensor_type = f"tensor<{rhs_r}x{rhs_c * lhs_rhs_mx_block_bytes}xi8>"
    lhs_bytes_expanded_tensor_type = (
        f"tensor<{lhs_r}x{lhs_c}x{lhs_rhs_mx_block_bytes}xi8>"
    )
    rhs_bytes_expanded_tensor_type = (
        f"tensor<{rhs_r}x{rhs_c}x{lhs_rhs_mx_block_bytes}xi8>"
    )
    lhs_expanded_tensor_type = (
        f"tensor<{lhs_r}x{lhs_c}x{mx_block_size}x{lhs_rhs_type.value}>"
    )
    rhs_expanded_tensor_type = (
        f"tensor<{rhs_r}x{rhs_c}x{mx_block_size}x{lhs_rhs_type.value}>"
    )
    scaled_lhs_expanded_tensor_type = (
        f"tensor<{lhs_r}x{lhs_c}x{mx_block_size}x{acc_type.value}>"
    )
    scaled_rhs_expanded_tensor_type = (
        f"tensor<{rhs_r}x{rhs_c}x{mx_block_size}x{acc_type.value}>"
    )
    scaled_lhs_tensor_type = f"tensor<{lhs_r}x{lhs_c * mx_block_size}x{acc_type.value}>"
    scaled_rhs_tensor_type = f"tensor<{rhs_r}x{rhs_c * mx_block_size}x{acc_type.value}>"
    lhs_scales_tensor_type = f"tensor<{lhs_r}x{lhs_c}x{mx_scale_type}>"
    rhs_scales_tensor_type = f"tensor<{rhs_r}x{rhs_c}x{mx_scale_type}>"
    op += (
        f"  %mx_block_size = arith.constant {mx_block_size} : i32\n"
        f"  %lhs_bytes_tensor = hal.tensor.import %lhs_bytes : !hal.buffer_view -> {lhs_bytes_tensor_type}\n"
        f"  %rhs_bytes_tensor = hal.tensor.import %rhs_bytes : !hal.buffer_view -> {rhs_bytes_tensor_type}\n"
        f"  %lhs_bytes_expanded = tensor.expand_shape %lhs_bytes_tensor [[0], [1, 2]] output_shape [{lhs_r}, {lhs_c}, {lhs_rhs_mx_block_bytes}] : {lhs_bytes_tensor_type} into {lhs_bytes_expanded_tensor_type}\n"
        f"  %rhs_bytes_expanded = tensor.expand_shape %rhs_bytes_tensor [[0], [1, 2]] output_shape [{rhs_r}, {rhs_c}, {lhs_rhs_mx_block_bytes}] : {rhs_bytes_tensor_type} into {rhs_bytes_expanded_tensor_type}\n"
        f"  %lhs_expanded = flow.tensor.bitcast %lhs_bytes_expanded : {lhs_bytes_expanded_tensor_type} -> {lhs_expanded_tensor_type}\n"
        f"  %rhs_expanded = flow.tensor.bitcast %rhs_bytes_expanded : {rhs_bytes_expanded_tensor_type} -> {rhs_expanded_tensor_type}\n"
        f"  %lhs_scales_tensor = hal.tensor.import %lhs_scales : !hal.buffer_view -> {lhs_scales_tensor_type}\n"
        f"  %rhs_scales_tensor = hal.tensor.import %rhs_scales : !hal.buffer_view -> {rhs_scales_tensor_type}\n"
        f"  %scaled_lhs_expanded_empty = tensor.empty() : {scaled_lhs_expanded_tensor_type}\n"
        f"  %scaled_rhs_expanded_empty = tensor.empty() : {scaled_rhs_expanded_tensor_type}\n"
        f"  %scaled_lhs_expanded_tensor = linalg.generic {{\n"
        f"                indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,\n"
        f"                                 affine_map<(d0, d1, d2) -> (d0, d1)>,\n"
        f"                                 affine_map<(d0, d1, d2) -> (d0, d1, d2)>],\n"
        f'                iterator_types = ["parallel", "parallel", "parallel"] }}\n'
        f"                ins(%lhs_expanded, %lhs_scales_tensor : {lhs_expanded_tensor_type}, {lhs_scales_tensor_type})\n"
        f"                outs(%scaled_lhs_expanded_empty : {scaled_lhs_expanded_tensor_type}) {{\n"
        f"  ^bb0(%a: f4E2M1FN, %s: f8E8M0FNU, %unused: f32):\n"
        f"    %r = arith.scaling_extf %a, %s : f4E2M1FN, f8E8M0FNU to f32\n"
        f"    linalg.yield %r : f32\n"
        f"  }} -> {scaled_lhs_expanded_tensor_type}\n"
        f"  %scaled_rhs_expanded_tensor = linalg.generic {{\n"
        f"                indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,\n"
        f"                                 affine_map<(d0, d1, d2) -> (d0, d1)>,\n"
        f"                                 affine_map<(d0, d1, d2) -> (d0, d1, d2)>],\n"
        f'                iterator_types = ["parallel", "parallel", "parallel"] }}\n'
        f"                ins(%rhs_expanded, %rhs_scales_tensor : {rhs_expanded_tensor_type}, {rhs_scales_tensor_type})\n"
        f"                outs(%scaled_rhs_expanded_empty : {scaled_rhs_expanded_tensor_type}) {{\n"
        f"  ^bb0(%a: f4E2M1FN, %s: f8E8M0FNU, %unused: f32):\n"
        f"    %r = arith.scaling_extf %a, %s : f4E2M1FN, f8E8M0FNU to f32\n"
        f"    linalg.yield %r : f32\n"
        f"  }} -> {scaled_rhs_expanded_tensor_type}\n"
        f"  %scaled_lhs_tensor = tensor.collapse_shape %scaled_lhs_expanded_tensor [[0], [1, 2]] : {scaled_lhs_expanded_tensor_type} into {scaled_lhs_tensor_type}\n"
        f"  %scaled_rhs_tensor = tensor.collapse_shape %scaled_rhs_expanded_tensor [[0], [1, 2]] : {scaled_rhs_expanded_tensor_type} into {scaled_rhs_tensor_type}\n"
        f"  %scaled_lhs = hal.tensor.export %scaled_lhs_tensor : {scaled_lhs_tensor_type} -> !hal.buffer_view\n"
        f"  %scaled_rhs = hal.tensor.export %scaled_rhs_tensor : {scaled_rhs_tensor_type} -> !hal.buffer_view\n"
        f"  util.call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %scaled_lhs, %scaled_rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()\n"
        f"  util.return\n"
        f"}}\n"
    )

    return TestCall(function=function, op=op)
