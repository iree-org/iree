# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir
from iree.compiler.dialects import iree_codegen
from iree.compiler.dialects import affine


def run(fn):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            print("\nTEST:", fn.__name__)
            fn()
    return fn


@run
def root_op():
    module_str = """
        module {
            func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
                %cst = arith.constant 0.000000e+00 : f32
                %0 = tensor.empty() : tensor<4x4xf32>
                %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
                %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
                return %2 : tensor<4x4xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    assert input_module is not None, "Failed to parse input MLIR module"
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    assert len(root_op_list) == 0

    module_str = """
        module {
            func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
                %cst = arith.constant 0.000000e+00 : f32
                %0 = tensor.empty() : tensor<4x4xf32>
                %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
                %2 = linalg.matmul { root_op } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
                return %2 : tensor<4x4xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    assert input_module is not None, "Failed to parse input MLIR module"
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    assert len(root_op_list) == 1
    assert root_op_list[0].name == "linalg.matmul"

    module_str = """
        module {
            func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
                %cst = arith.constant 0.000000e+00 : f32
                %0 = tensor.empty() : tensor<4x4xf32>
                %1 = linalg.fill { root_op } ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
                %2 = linalg.matmul { root_op } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
                return %2 : tensor<4x4xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    assert input_module is not None, "Failed to parse input MLIR module"
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    assert len(root_op_list) == 2
    assert root_op_list[0].name == "linalg.fill"
    assert root_op_list[1].name == "linalg.matmul"


@run
def attention_op_detail():
    dim_exprs = [affine.AffineDimExpr.get(i) for i in range(5)]

    q_map = affine.AffineMap.get(
        5, 0, [dim_exprs[0], dim_exprs[1], dim_exprs[2]]
    )  # (d0, d1, d2).
    k_map = affine.AffineMap.get(
        5, 0, [dim_exprs[0], dim_exprs[3], dim_exprs[2]]
    )  # (d0, d3, d2).
    v_map = affine.AffineMap.get(
        5, 0, [dim_exprs[0], dim_exprs[3], dim_exprs[4]]
    )  # (d0, d3, d4).                                      # ()
    o_map = affine.AffineMap.get(
        5, 0, [dim_exprs[0], dim_exprs[1], dim_exprs[4]]
    )  # (d0, d1, d4).

    result = iree_codegen.get_attention_op_detail(q_map, k_map, v_map, o_map)

    assert result.domain_rank == 5
    assert result.batch_dims == [0]
    assert result.m_dims == [1]
    assert result.k1_dims == [2]
    assert result.k2_dims == [3]
    assert result.n_dims == [4]

    dim_exprs = [affine.AffineDimExpr.get(i) for i in range(4)]

    # Input affine maps that do not follow the expected pattern for an attention operation.
    q_map = affine.AffineMap.get(4, 0, [dim_exprs[0], dim_exprs[1]])  # (d0, d1).
    k_map = affine.AffineMap.get(4, 0, [dim_exprs[0], dim_exprs[2]])  # (d0, d2).
    v_map = affine.AffineMap.get(4, 0, [dim_exprs[0], dim_exprs[3]])  # (d0, d3).
    o_map = affine.AffineMap.get(4, 0, [dim_exprs[0], dim_exprs[1]])  # (d0, d1).

    result = iree_codegen.get_attention_op_detail(q_map, k_map, v_map, o_map)
    assert result.domain_rank == 4
    assert result.batch_dims == [0]
    assert result.m_dims == [1]
    assert result.k1_dims == []
    assert result.k2_dims == [2]
    assert result.n_dims == [3]


@run
def test_isa_attention_op():
    module_str = """
        module {
               func.func @attention_20x4096x64x4096x64(
                    %q : tensor<20x4096x64xf16>,
                    %k : tensor<20x4096x64xf16>,
                    %v : tensor<20x4096x64xf16>,
                    %scale : f16,
                    %output : tensor<20x4096x64xf16>
                ) -> tensor<20x4096x64xf16> {
                    %result = iree_linalg_ext.attention { root_op,
                        indexing_maps = [
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> ()>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
                        ]
                    } ins(%q, %k, %v, %scale : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16)
                        outs(%output : tensor<20x4096x64xf16>) {
                    ^bb0(%score: f32):
                        iree_linalg_ext.yield %score : f32
                    } -> tensor<20x4096x64xf16>
                    return %result : tensor<20x4096x64xf16>
                }
        }
    """
    input_module = ir.Module.parse(module_str)
    assert input_module is not None, "Failed to parse input MLIR module"
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    assert len(root_op_list) == 1
    assert root_op_list[0].name == "iree_linalg_ext.attention"
    assert iree_codegen.isa_attention_op(root_op_list[0])


@run
def test_igemm_conv_details():
    # Test 1: conv_2d_nhwc_hwcf.
    module_str = """
        module {
            func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
                %0 = linalg.conv_2d_nhwc_hwcf { root_op, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
                    ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
                    outs(%arg2 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
                return %0 : tensor<1x14x14x16xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)

    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
    assert details is not None, "IGEMM details should be valid for NHWC_HWCF conv"
    assert details.igemm_loop_bounds == [
        1,
        14,
        14,
        16,
        36,
    ], f"Expected [1,14,14,16,36], got {details.igemm_loop_bounds}"
    assert details.conv_dims_batch == [
        0
    ], f"Expected batch=[0], got {details.conv_dims_batch}"
    assert details.conv_dims_output_image == [
        1,
        2,
    ], f"Expected output_image=[1,2], got {details.conv_dims_output_image}"
    assert details.conv_dims_output_channel == [
        3
    ], f"Expected output_channel=[3], got {details.conv_dims_output_channel}"
    assert details.conv_dims_filter_loop == [
        4,
        5,
    ], f"Expected filter_loop=[4,5], got {details.conv_dims_filter_loop}"
    assert details.conv_dims_input_channel == [
        6
    ], f"Expected input_channel=[6], got {details.conv_dims_input_channel}"
    assert (
        details.is_output_channel_first == False
    ), f"Expected is_output_channel_first=False, got {details.is_output_channel_first}"

    # Test 2: conv_2d_nhwc_fhwc.
    module_str = """
        module {
            func.func @conv_2d_nhwc_fhwc(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<16x3x3x4xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
                %0 = linalg.conv_2d_nhwc_fhwc { root_op, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
                    ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<16x3x3x4xf32>)
                    outs(%arg2 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
                return %0 : tensor<1x14x14x16xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)

    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
    assert details is not None, "IGEMM details should be valid for NHWC_FHWC conv"
    assert details.igemm_loop_bounds == [
        1,
        14,
        14,
        16,
        36,
    ], f"Expected [1,14,14,16,36], got {details.igemm_loop_bounds}"
    assert details.conv_dims_batch == [
        0
    ], f"Expected batch=[0], got {details.conv_dims_batch}"
    assert details.conv_dims_output_image == [
        1,
        2,
    ], f"Expected output_image=[1,2], got {details.conv_dims_output_image}"
    assert details.conv_dims_output_channel == [
        3
    ], f"Expected output_channel=[3], got {details.conv_dims_output_channel}"
    assert isinstance(
        details.is_output_channel_first, bool
    ), "Should have is_output_channel_first flag"

    # Test 3: conv_2d_nchw_fchw.
    module_str = """
        module {
            func.func @conv_2d_nchw_fchw(%arg0: tensor<1x4x16x16xf32>, %arg1: tensor<16x4x3x3xf32>, %arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32> {
                %0 = linalg.conv_2d_nchw_fchw { root_op, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
                    ins(%arg0, %arg1 : tensor<1x4x16x16xf32>, tensor<16x4x3x3xf32>)
                    outs(%arg2 : tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
                return %0 : tensor<1x16x14x14xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)

    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
    assert details is not None, "IGEMM details should be valid for NCHW conv"
    assert details.igemm_loop_bounds == [
        1,
        16,
        14,
        14,
        36,
    ], f"Expected [1,16,14,14,36], got {details.igemm_loop_bounds}"
    assert details.conv_dims_batch == [
        0
    ], f"Expected batch=[0], got {details.conv_dims_batch}"
    assert details.conv_dims_output_image == [
        2,
        3,
    ], f"Expected output_image=[2,3], got {details.conv_dims_output_image}"
    assert details.conv_dims_filter_loop == [
        5,
        6,
    ], f"Expected filter_loop=[5,6], got {details.conv_dims_filter_loop}"

    # Test 4: linalg.generic with convolution pattern (weight backward).
    module_str = """
        module {
            func.func @conv_generic_weight_backward(%arg0: tensor<16x98x64x96xf32>, %arg1: tensor<16x96x64x96xf32>, %arg2: tensor<96x3x96xf32>) -> tensor<96x3x96xf32> {
                %0 = linalg.generic {
                    indexing_maps = [
                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 + d4, d5, d2)>,
                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d0)>,
                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
                    ],
                    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
                } ins(%arg0, %arg1 : tensor<16x98x64x96xf32>, tensor<16x96x64x96xf32>) outs(%arg2 : tensor<96x3x96xf32>) attrs = {root_op} {
                ^bb0(%in: f32, %in_1: f32, %out: f32):
                    %mul = arith.mulf %in, %in_1 : f32
                    %add = arith.addf %out, %mul : f32
                    linalg.yield %add : f32
                } -> tensor<96x3x96xf32>
                return %0 : tensor<96x3x96xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)

    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
    assert (
        details is not None
    ), "IGEMM details should be valid for generic 1D conv weight backward"
    assert details.igemm_loop_bounds == [
        96,
        3,
        96,
        98304,
    ], f"Expected [96,3,96,98304], got {details.igemm_loop_bounds}"
    assert details.conv_dims_output_image == [
        1
    ], f"Expected output_image=[1], got {details.conv_dims_output_image}"
    assert details.conv_dims_filter_loop == [
        4
    ], f"Expected filter_loop=[4], got {details.conv_dims_filter_loop}"

    # Test with a non-conv operation.
    module_str = """
        module {
            func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
                %cst = arith.constant 0.000000e+00 : f32
                %0 = tensor.empty() : tensor<4x4xf32>
                %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
                %2 = linalg.matmul { root_op } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
                return %2 : tensor<4x4xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    matmul_op = root_op_list[0]

    details = iree_codegen.get_igemm_generic_conv_details(matmul_op)
    assert details is None, "IGEMM details should be None for non-conv operation"
