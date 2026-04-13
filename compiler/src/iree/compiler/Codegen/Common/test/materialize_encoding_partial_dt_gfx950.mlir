// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding{test-gpu-encoding-resolver=gpu_data_tiling}))" \
// RUN:   --iree-gpu-test-target=gfx950 \
// RUN:   --iree-gpu-partial-dt-scaled-mma \
// RUN:   --split-input-file %s | FileCheck %s

// Tests for PartialDataTiledScaledMMAAttr materialization:
//   - Data operands (LHS, RHS) produce linalg.pack + expand_shape (no transpose)
//   - Scale operands produce full pack + expand_shape + transpose
//   - Scaled matmul lowers to inner_tiled with partial_data_tiled_scaled_mma_layout kind

//-----------------------------------------------------------------------------
// LHS data operand (operand_index=0): pack + expand_shape, no transpose
//-----------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs = #iree_encoding.encoding<
  operand_index = 0 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [256, 512, 128, 32]>

func.func @partial_dt_set_encoding_LHS_data(%arg0: tensor<256x128x32xf4E2M1FN>) -> tensor<256x128x32xf4E2M1FN, #encoding_lhs> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<256x128x32xf4E2M1FN> -> tensor<256x128x32xf4E2M1FN, #encoding_lhs>
  return %0 : tensor<256x128x32xf4E2M1FN, #encoding_lhs>
}

// CHECK-LABEL: func.func @partial_dt_set_encoding_LHS_data(
// CHECK:         %[[PACK:.*]] = linalg.pack
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-NOT:     linalg.transpose
// CHECK:         return %[[EXPANDED]]

// -----

//-----------------------------------------------------------------------------
// RHS data operand (operand_index=1): pack + expand_shape, no transpose
//-----------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_rhs = #iree_encoding.encoding<
  operand_index = 1 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [256, 512, 128, 32]>

func.func @partial_dt_set_encoding_RHS_data(%arg0: tensor<512x128x32xf4E2M1FN>) -> tensor<512x128x32xf4E2M1FN, #encoding_rhs> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<512x128x32xf4E2M1FN> -> tensor<512x128x32xf4E2M1FN, #encoding_rhs>
  return %0 : tensor<512x128x32xf4E2M1FN, #encoding_rhs>
}

// CHECK-LABEL: func.func @partial_dt_set_encoding_RHS_data(
// CHECK:         %[[PACK:.*]] = linalg.pack
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-NOT:     linalg.transpose
// CHECK:         return %[[EXPANDED]]

// -----

//-----------------------------------------------------------------------------
// LHS scale operand (operand_index=2): full pack + expand + transpose
//-----------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs_scales = #iree_encoding.encoding<
  operand_index = 2 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [256, 512, 128, 32]>

func.func @partial_dt_set_encoding_LHS_scales(%arg0: tensor<256x128xf8E8M0FNU>) -> tensor<256x128xf8E8M0FNU, #encoding_lhs_scales> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<256x128xf8E8M0FNU> -> tensor<256x128xf8E8M0FNU, #encoding_lhs_scales>
  return %0 : tensor<256x128xf8E8M0FNU, #encoding_lhs_scales>
}

// CHECK-LABEL: func.func @partial_dt_set_encoding_LHS_scales(
// CHECK:         linalg.pack
// CHECK:         tensor.expand_shape
// CHECK:         linalg.transpose
// CHECK:         return

// -----

//-----------------------------------------------------------------------------
// RHS scale operand (operand_index=3): full pack + expand + transpose
//-----------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_rhs_scales = #iree_encoding.encoding<
  operand_index = 3 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [256, 512, 128, 32]>

func.func @partial_dt_set_encoding_RHS_scales(%arg0: tensor<512x128xf8E8M0FNU>) -> tensor<512x128xf8E8M0FNU, #encoding_rhs_scales> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<512x128xf8E8M0FNU> -> tensor<512x128xf8E8M0FNU, #encoding_rhs_scales>
  return %0 : tensor<512x128xf8E8M0FNU, #encoding_rhs_scales>
}

// CHECK-LABEL: func.func @partial_dt_set_encoding_RHS_scales(
// CHECK:         linalg.pack
// CHECK:         tensor.expand_shape
// CHECK:         linalg.transpose
// CHECK:         return

// -----

//-----------------------------------------------------------------------------
// Scaled matmul inner_tiled formation with PartialDataTiledScaledMMAAttr kind
//-----------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [256, 512, 128, 32]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [256, 512, 128, 32]>
#encoding_lhs_scales = #iree_encoding.encoding<operand_index = 2 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [256, 512, 128, 32]>
#encoding_rhs_scales = #iree_encoding.encoding<operand_index = 3 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [256, 512, 128, 32]>
#encoding_result = #iree_encoding.encoding<operand_index = 4 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [256, 512, 128, 32]>

func.func @partial_dt_scaled_matmul_inner_tiled(
    %arg0: tensor<256x128x32xf4E2M1FN, #encoding_lhs>,
    %arg1: tensor<512x128x32xf4E2M1FN, #encoding_rhs>,
    %arg2: tensor<256x128xf8E8M0FNU, #encoding_lhs_scales>,
    %arg3: tensor<512x128xf8E8M0FNU, #encoding_rhs_scales>,
    %arg4: tensor<256x512xf32, #encoding_result>
) -> tensor<256x512xf32, #encoding_result> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
           : tensor<256x128x32xf4E2M1FN, #encoding_lhs>, tensor<512x128x32xf4E2M1FN, #encoding_rhs>,
             tensor<256x128xf8E8M0FNU, #encoding_lhs_scales>, tensor<512x128xf8E8M0FNU, #encoding_rhs_scales>)
      outs(%arg4 : tensor<256x512xf32, #encoding_result>) {
  ^bb0(%in: f4E2M1FN, %in_0: f4E2M1FN, %in_1: f8E8M0FNU, %in_2: f8E8M0FNU, %out: f32):
    %11 = arith.scaling_extf %in, %in_1 : f4E2M1FN, f8E8M0FNU to f32
    %12 = arith.scaling_extf %in_0, %in_2 : f4E2M1FN, f8E8M0FNU to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<256x512xf32, #encoding_result>
  return %0 : tensor<256x512xf32, #encoding_result>
}

// CHECK:     func.func @partial_dt_scaled_matmul_inner_tiled(
// CHECK:       iree_codegen.inner_tiled
// CHECK-SAME:    kind = #iree_gpu.partial_data_tiled_scaled_mma_layout<
