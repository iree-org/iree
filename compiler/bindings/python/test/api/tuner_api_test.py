# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir
from iree.compiler.dialects import iree_codegen
from iree.compiler.dialects import iree_gpu
from iree.compiler.dialects import affine
from iree.compiler.ir import AffineMap, AffineDimExpr


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

    q_map = affine.AffineMap.get(5, 0, [dim_exprs[0], dim_exprs[1], dim_exprs[2]])
    k_map = affine.AffineMap.get(5, 0, [dim_exprs[0], dim_exprs[3], dim_exprs[2]])
    v_map = affine.AffineMap.get(5, 0, [dim_exprs[0], dim_exprs[3], dim_exprs[4]])
    o_map = affine.AffineMap.get(5, 0, [dim_exprs[0], dim_exprs[1], dim_exprs[4]])

    result = iree_codegen.get_attention_op_detail(q_map, k_map, v_map, o_map)

    assert result.domain_rank == 5
    assert result.batch_dims == [0]
    assert result.m_dims == [1]
    assert result.k1_dims == [2]
    assert result.k2_dims == [3]
    assert result.n_dims == [4]

    dim_exprs = [affine.AffineDimExpr.get(i) for i in range(4)]

    # Input affine maps that do not follow the expected pattern for an attention operation.
    q_map = affine.AffineMap.get(4, 0, [dim_exprs[0], dim_exprs[1]])
    k_map = affine.AffineMap.get(4, 0, [dim_exprs[0], dim_exprs[2]])
    v_map = affine.AffineMap.get(4, 0, [dim_exprs[0], dim_exprs[3]])
    o_map = affine.AffineMap.get(4, 0, [dim_exprs[0], dim_exprs[1]])

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
                    %output : tensor<20x4096x64xf16>
                ) -> tensor<20x4096x64xf16> {
                    %result = iree_linalg_ext.attention { root_op,
                        indexing_maps = [
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
                        ]
                    } ins(%q, %k, %v : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>)
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
    assert details.igemm_loop_bounds == [1, 14, 14, 16, 36]

    assert len(details.igemm_contraction_maps) == 3
    maps = [map_attr.value for map_attr in details.igemm_contraction_maps]
    d0, d1, d2, d3, d4 = [AffineDimExpr.get(i) for i in range(5)]
    # For channel-last (NHWC): input (N,H,W,K), filter (K,OC), output (N,H,W,OC).
    assert maps[0] == AffineMap.get(
        5, 0, [d0, d1, d2, d4]
    ), f"Input map mismatch: {maps[0]}"
    assert maps[1] == AffineMap.get(5, 0, [d4, d3]), f"Filter map mismatch: {maps[1]}"
    assert maps[2] == AffineMap.get(
        5, 0, [d0, d1, d2, d3]
    ), f"Output map mismatch: {maps[2]}"
    iter_types = [str(attr) for attr in details.igemm_loop_iterators]
    assert iter_types == [
        '"parallel"',
        '"parallel"',
        '"parallel"',
        '"parallel"',
        '"reduction"',
    ]
    assert details.im2col_output_perm == [0, 1, 2, 3]
    assert details.filter_reassoc_indices == [[0, 1, 2], [3]]
    assert not details.is_output_channel_first
    assert details.conv_to_igemm_dim_map == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}

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
    assert details.igemm_loop_bounds == [1, 14, 14, 16, 36]
    assert len(details.igemm_contraction_maps) == 3
    maps = [map_attr.value for map_attr in details.igemm_contraction_maps]
    # Verify expected affine maps (NHWC_FHWC layout).
    d0, d1, d2, d3, d4 = [AffineDimExpr.get(i) for i in range(5)]
    assert maps[0] == AffineMap.get(
        5, 0, [d0, d1, d2, d4]
    ), f"Input map mismatch: {maps[0]}"
    assert maps[1] == AffineMap.get(5, 0, [d3, d4]), f"Filter map mismatch: {maps[1]}"
    assert maps[2] == AffineMap.get(
        5, 0, [d0, d1, d2, d3]
    ), f"Output map mismatch: {maps[2]}"
    iter_types = [str(attr) for attr in details.igemm_loop_iterators]
    assert iter_types == [
        '"parallel"',
        '"parallel"',
        '"parallel"',
        '"parallel"',
        '"reduction"',
    ]
    assert details.im2col_output_perm == [0, 1, 2, 3]
    assert details.filter_reassoc_indices == [[0], [1, 2, 3]]
    assert not details.is_output_channel_first
    assert details.conv_to_igemm_dim_map == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}

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
    assert details.igemm_loop_bounds == [1, 16, 14, 14, 36]
    assert len(details.igemm_contraction_maps) == 3
    maps = [map_attr.value for map_attr in details.igemm_contraction_maps]
    # Verify expected affine maps for NCHW with loop dims [N, OC, H, W, K].
    # Note: operands are swapped - filter first, then input.
    d0, d1, d2, d3, d4 = [AffineDimExpr.get(i) for i in range(5)]
    assert maps[0] == AffineMap.get(5, 0, [d1, d4]), f"Filter map mismatch: {maps[0]}"
    assert maps[1] == AffineMap.get(
        5, 0, [d0, d2, d3, d4]
    ), f"Input map mismatch: {maps[1]}"
    assert maps[2] == AffineMap.get(
        5, 0, [d0, d1, d2, d3]
    ), f"Output map mismatch: {maps[2]}"
    iter_types = [str(attr) for attr in details.igemm_loop_iterators]
    assert iter_types == [
        '"parallel"',
        '"parallel"',
        '"parallel"',
        '"parallel"',
        '"reduction"',
    ]
    assert details.im2col_output_perm == [0, 1, 2, 3]
    assert details.filter_reassoc_indices == [[0], [1, 2, 3]]
    assert details.is_output_channel_first
    assert details.conv_to_igemm_dim_map == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}

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
    assert details.igemm_loop_bounds == [96, 3, 96, 98304]
    assert len(details.igemm_contraction_maps) == 3
    maps = [map_attr.value for map_attr in details.igemm_contraction_maps]
    d0, d1, d2, d3 = [AffineDimExpr.get(i) for i in range(4)]
    assert maps[0] == AffineMap.get(4, 0, [d3, d0]), f"Map 0 mismatch: {maps[0]}"
    assert maps[1] == AffineMap.get(4, 0, [d1, d3, d2]), f"Map 1 mismatch: {maps[1]}"
    assert maps[2] == AffineMap.get(4, 0, [d0, d1, d2]), f"Map 2 mismatch: {maps[2]}"
    iter_types = [str(attr) for attr in details.igemm_loop_iterators]
    assert iter_types == ['"parallel"', '"parallel"', '"parallel"', '"reduction"']
    assert details.im2col_output_perm == [1, 2, 0]
    assert details.filter_reassoc_indices == [[0, 1, 2], [3]]
    assert details.is_output_channel_first
    assert details.conv_to_igemm_dim_map == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}

    # Test with a non-conv operation.
    module_str = """
        module {
            func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
                %0 = linalg.matmul { root_op } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg2 : tensor<4x4xf32>) -> tensor<4x4xf32>
                return %0 : tensor<4x4xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    matmul_op = root_op_list[0]

    details = iree_codegen.get_igemm_generic_conv_details(matmul_op)
    assert details is None, "IGEMM details should be None for non-conv operation"


@run
def test_isa_scaled_contraction_op():
    # Test 1: Regular matmul is not a scaled contraction.
    module_str = """
        module {
            func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
                %0 = linalg.matmul { root_op } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg2 : tensor<4x4xf32>) -> tensor<4x4xf32>
                return %0 : tensor<4x4xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    assert input_module is not None, "Failed to parse input MLIR module"
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    assert len(root_op_list) == 1
    matmul_op = root_op_list[0]

    assert not iree_codegen.isa_scaled_contraction_op(
        matmul_op
    ), "Regular matmul should not be a scaled contraction"

    # Test 2: Fill op is not a scaled contraction.
    module_str = """
        module {
            func.func @fill(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
                %cst = arith.constant 0.000000e+00 : f32
                %0 = linalg.fill { root_op } ins(%cst : f32) outs(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32>
                return %0 : tensor<4x4xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    assert len(root_op_list) == 1
    fill_op = root_op_list[0]

    assert not iree_codegen.isa_scaled_contraction_op(
        fill_op
    ), "Fill op should not be a scaled contraction"

    # Test 3: Scaled matmul as linalg.generic should be detected.
    # Pattern: linalg.generic with 5 indexing maps (lhs, rhs, lhs_scale, rhs_scale, output),
    # and 4 iterator types (2 parallel for M,N; 2 reduction for Ko,Kb).
    # Uses f4E2M1FN for operands and f8E8M0FNU for scales (matching real scaled matmul pattern).
    module_str = """
        module {
            func.func @scaled_matmul(%lhs: tensor<16x4x32xf4E2M1FN>, %rhs: tensor<16x4x32xf4E2M1FN>,
                                     %lhs_scales: tensor<16x4xf8E8M0FNU>, %rhs_scales: tensor<16x4xf8E8M0FNU>,
                                     %out: tensor<16x16xf32>) -> tensor<16x16xf32> {
                %result = linalg.generic {
                    indexing_maps = [
                        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
                        affine_map<(d0, d1, d2, d3) -> (d1, d2)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1)>
                    ],
                    iterator_types = ["parallel", "parallel", "reduction", "reduction"],
                    root_op
                } ins(%lhs, %rhs, %lhs_scales, %rhs_scales : tensor<16x4x32xf4E2M1FN>, tensor<16x4x32xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<16x4xf8E8M0FNU>)
                  outs(%out : tensor<16x16xf32>) {
                ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %acc: f32):
                    %a_scaled = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
                    %b_scaled = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
                    %prod = arith.mulf %a_scaled, %b_scaled : f32
                    %sum = arith.addf %acc, %prod : f32
                    linalg.yield %sum : f32
                } -> tensor<16x16xf32>
                return %result : tensor<16x16xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    assert len(root_op_list) == 1, "Should have one root op"

    scaled_generic_op = root_op_list[0]
    is_scaled = iree_codegen.isa_scaled_contraction_op(scaled_generic_op)
    assert (
        is_scaled
    ), "linalg.generic with scaled matmul pattern should be detected as scaled contraction"

    dims = iree_codegen.infer_scaled_contraction_dimensions(scaled_generic_op)
    assert dims is not None, "Should be able to infer dimensions for scaled contraction"

    assert dims.m == [0], f"Got {dims.m}"
    assert dims.n == [1], f"Got {dims.n}"
    assert dims.k == [2], f"Got {dims.k}"
    assert dims.kB == [3], f"Got {dims.kB}"
    assert dims.batch == [], f"Got {dims.batch}"


@run
def test_infer_scaled_contraction_dimensions():
    # Test 1: Verify dimension inference on a scaled matmul operation.
    module_str = """
        module {
            func.func @scaled_matmul(%lhs: tensor<16x4x32xf4E2M1FN>, %rhs: tensor<16x4x32xf4E2M1FN>,
                                     %lhs_scales: tensor<16x4xf8E8M0FNU>, %rhs_scales: tensor<16x4xf8E8M0FNU>,
                                     %out: tensor<16x16xf32>) -> tensor<16x16xf32> {
                %result = linalg.generic {
                    indexing_maps = [
                        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
                        affine_map<(d0, d1, d2, d3) -> (d1, d2)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1)>
                    ],
                    iterator_types = ["parallel", "parallel", "reduction", "reduction"],
                    root_op
                } ins(%lhs, %rhs, %lhs_scales, %rhs_scales : tensor<16x4x32xf4E2M1FN>, tensor<16x4x32xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<16x4xf8E8M0FNU>)
                  outs(%out : tensor<16x16xf32>) {
                ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %acc: f32):
                    %a_scaled = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
                    %b_scaled = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
                    %prod = arith.mulf %a_scaled, %b_scaled : f32
                    %sum = arith.addf %acc, %prod : f32
                    linalg.yield %sum : f32
                } -> tensor<16x16xf32>
                return %result : tensor<16x16xf32>
            }
        }
    """
    input_module = ir.Module.parse(module_str)
    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    assert len(root_op_list) == 1, "Should have exactly one root op"
    scaled_op = root_op_list[0]

    assert iree_codegen.isa_scaled_contraction_op(
        scaled_op
    ), "Operation should be recognized as scaled contraction"

    dims = iree_codegen.infer_scaled_contraction_dimensions(scaled_op)
    assert dims is not None, "Should successfully infer dimensions"
    assert dims.m == [0], f"Got {dims.m}"
    assert dims.n == [1], f"Got {dims.n}"
    assert dims.k == [2], f"Got {dims.k}"
    assert dims.kB == [3], f"Got {dims.kB}"
    assert dims.batch == [], f"Got {dims.batch}"

    # Test 2: Non-scaled contraction should return None.
    module_str_regular = """
        module {
            func.func @regular_matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
                %0 = linalg.matmul { root_op } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg2 : tensor<4x4xf32>) -> tensor<4x4xf32>
                return %0 : tensor<4x4xf32>
            }
        }
    """
    input_module_regular = ir.Module.parse(module_str_regular)
    regular_ops = iree_codegen.get_tuner_root_ops(input_module_regular)
    assert len(regular_ops) == 1
    regular_matmul = regular_ops[0]

    # Regular matmul should not have scaled contraction dimensions.
    # Check if all dimensions are empty (indicating it's not a scaled contraction).
    dims_regular = iree_codegen.infer_scaled_contraction_dimensions(regular_matmul)
    if dims_regular is not None:
        all_empty = (
            len(dims_regular.m) == 0
            and len(dims_regular.n) == 0
            and len(dims_regular.k) == 0
            and len(dims_regular.kB) == 0
            and len(dims_regular.batch) == 0
        )
        assert (
            all_empty or dims_regular is None
        ), "Regular matmul should not have valid scaled contraction dimensions"

    # Test 3: Batched scaled matmul.
    module_str_batched = """
        module {
            func.func @batched_scaled_matmul(%lhs: tensor<8x16x4x32xf4E2M1FN>, %rhs: tensor<8x16x4x32xf4E2M1FN>,
                                             %lhs_scales: tensor<8x16x4xf8E8M0FNU>, %rhs_scales: tensor<8x16x4xf8E8M0FNU>,
                                             %out: tensor<8x16x16xf32>) -> tensor<8x16x16xf32> {
                %result = linalg.generic {
                    indexing_maps = [
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
                    ],
                    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"],
                    root_op
                } ins(%lhs, %rhs, %lhs_scales, %rhs_scales : tensor<8x16x4x32xf4E2M1FN>, tensor<8x16x4x32xf4E2M1FN>, tensor<8x16x4xf8E8M0FNU>, tensor<8x16x4xf8E8M0FNU>)
                  outs(%out : tensor<8x16x16xf32>) {
                ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %acc: f32):
                    %a_scaled = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
                    %b_scaled = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
                    %prod = arith.mulf %a_scaled, %b_scaled : f32
                    %sum = arith.addf %acc, %prod : f32
                    linalg.yield %sum : f32
                } -> tensor<8x16x16xf32>
                return %result : tensor<8x16x16xf32>
            }
        }
    """
    input_module_batched = ir.Module.parse(module_str_batched)
    batched_ops = iree_codegen.get_tuner_root_ops(input_module_batched)
    assert len(batched_ops) == 1, "Batched op should be found"
    batched_op = batched_ops[0]
    assert iree_codegen.isa_scaled_contraction_op(
        batched_op
    ), "Batched scaled matmul should be recognized"

    dims_batched = iree_codegen.infer_scaled_contraction_dimensions(batched_op)
    assert (
        dims_batched is not None
    ), "Batch dimension must be present in batched scaled matmul"
    assert dims_batched.batch == [0], f"Got {dims_batched.batch}"
    assert dims_batched.m == [1], f"Got {dims_batched.m}"
    assert dims_batched.n == [2], f"Got {dims_batched.n}"
    assert dims_batched.k == [3], f"Got {dims_batched.k}"
    assert dims_batched.kB == [4], f"Got {dims_batched.kB}"


@run
def test_is_xor_shuffle_valid():
    """Test XOR shuffle validation (pure function, no MLIR attributes)."""
    # Valid: row and access divide tile; row >= access; tile >= row.
    assert iree_gpu.is_xor_shuffle_valid(256, 32, 512)
    assert iree_gpu.is_xor_shuffle_valid(512, 64, 512)
    assert iree_gpu.is_xor_shuffle_valid(32, 8, 512)
    # Invalid: row exceeds tile.
    assert not iree_gpu.is_xor_shuffle_valid(512, 32, 256)
    # Invalid: access exceeds row.
    assert not iree_gpu.is_xor_shuffle_valid(256, 512, 512)
    # Invalid: row does not evenly divide tile.
    assert not iree_gpu.is_xor_shuffle_valid(300, 32, 512)
    # Invalid: access does not evenly divide row.
    assert not iree_gpu.is_xor_shuffle_valid(256, 33, 512)


@run
def test_get_xor_shuffle_bounds():
    """Test XOR shuffle bounds for an MMA intrinsic (for use by SharkTuner)."""
    # Use an MMA intrinsic that supports getXorShuffleBounds (InnerTileDescAttrInterface).
    mma_attr = iree_gpu.MMAAttr.get(iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16)
    bounds = iree_gpu.get_xor_shuffle_bounds(mma_attr, operand_index=0)
    assert bounds is not None, "get_xor_shuffle_bounds should succeed for MMAAttr"
    min_access_elems, total_tile_elems = bounds
    assert min_access_elems == 4
    assert total_tile_elems == 256
    bounds_rhs = iree_gpu.get_xor_shuffle_bounds(mma_attr, operand_index=1)
    assert bounds_rhs is not None
