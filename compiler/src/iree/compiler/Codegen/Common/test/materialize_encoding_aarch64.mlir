// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s --check-prefixes=CHECK,NO-SVE
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s --check-prefixes=CHECK,WITH-SVE

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], iteration_sizes = [?, ?, ?]>
func.func @matmul_LHS(%arg0: tensor<8x16xbf16>, %m: index, %n: index, %k: index) -> tensor<8x16xbf16, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+sve", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<8x16xbf16> -> tensor<8x16xbf16, #encoding>
  return %0 : tensor<8x16xbf16, #encoding>
}

/// NOTE: No scalable tiles for LHS, hence no difference between NO-SVE and WITH-SVE

// CHECK-LABEL: func.func @matmul_LHS
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x16xbf16>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [8, 1]
// CHECK:         return %[[PACK]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], iteration_sizes = [?, ?, ?]>
func.func @matmul_RHS(%arg0: tensor<8x16xbf16>, %m: index, %n: index, %k: index) -> tensor<8x16xbf16, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+sve", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<8x16xbf16> -> tensor<8x16xbf16, #encoding>
  return %0 : tensor<8x16xbf16, #encoding>
}
/// NOTE: For RHS, the inner tile corresponding to the "N" dimension is
/// scalable, hence NO-SVE and WITH-SVE differ!

// WITH-SVE: #[[$MAP:.+]] = affine_map<()[s0] -> (16 ceildiv s0)>

// CHECK-LABEL: func.func @matmul_RHS
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<8x16xbf16>
// WITH-SVE-DAG:  %[[PAD:.+]] = arith.constant 0.000000e+00 : bf16

/// SVE: the number of outer tiles corresponding to the inner scalable tile
// WITH-SVE-DAG:  %[[C8:.*]] = arith.constant 8 : index
// WITH-SVE-DAG:  %[[VSCALE:.*]] = vector.vscale
// WITH-SVE:      %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
// WITH-SVE:      %[[OUTER_DIM:.*]] = affine.apply #[[$MAP]]()[%[[C8_VSCALE]]]

/// Init the output tensor
// NO-SVE-DAG:     %[[INIT:.+]] = tensor.empty() : tensor<2x8x8x1xbf16>
// WITH-SVE-DAG:   %[[INIT:.*]] = tensor.empty(%[[OUTER_DIM]], %[[C8_VSCALE]]) : tensor<?x8x?x1xbf16>

/// The newly materialised Pack Op (SVE includes padding)
// CHECK:         %[[PACK:.+]] = linalg.pack %[[SRC]]
// WITH-SVE-SAME:    padding_value(%[[PAD]] : bf16)
// NO-SVE-NOT:        padding_value

// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]

// NO-SVE-SAME:      inner_tiles = [8, 1]
// NO-SVE-SAME:      into %[[INIT]] : tensor<8x16xbf16> -> tensor<2x8x8x1xbf16>

// WITH-SVE-SAME:    inner_tiles = [%[[C8_VSCALE]], 1]
// WITH-SVE-SAME:    into %[[INIT]] : tensor<8x16xbf16> -> tensor<?x8x?x1xbf16>

// CHECK:         return %[[PACK]]

// -----

#map = affine_map<(b, m, n, k) -> (b, m, k)>
#map1 = affine_map<(b, m, n, k) -> (b, k, n)>
#map2 = affine_map<(b, m, n, k) -> (b, m, n)>
#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 32, 320, ?]>
func.func @batch_matmul_RHS(%arg0: tensor<128x32x320xf32>, %k: index) -> tensor<128x32x320xf32, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+sve", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%k} : tensor<128x32x320xf32> -> tensor<128x32x320xf32, #encoding>
  return %0 : tensor<128x32x320xf32, #encoding>
}

/// NOTE: For RHS, the inner tile corresponding to the "N" dimension is
/// scalable, hence NO-SVE and WITH-SVE differ!

// WITH-SVE: #[[$MAP:.+]] = affine_map<()[s0] -> (320 ceildiv s0)>

// CHECK-LABEL: func.func @batch_matmul_RHS
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<128x32x320xf32>
// WITH-SVE-DAG:  %[[PAD:.+]] = arith.constant 0.000000e+00 : f32

/// SVE: the number of outer tiles corresponding to the inner scalable tile
// WITH-SVE-DAG:  %[[C8:.*]] = arith.constant 8 : index
// WITH-SVE-DAG:  %[[VSCALE:.*]] = vector.vscale
// WITH-SVE:      %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
// WITH-SVE:      %[[OUTER_DIM:.*]] = affine.apply #[[$MAP]]()[%[[C8_VSCALE]]]

/// Init the output tensor
// NO-SVE:          %[[INIT:.+]] = tensor.empty() : tensor<128x40x32x8x1xf32>
// WITH-SVE:        %[[INIT:.+]] = tensor.empty(%[[OUTER_DIM]], %[[C8_VSCALE]]) : tensor<128x?x32x?x1xf32>

/// The newly materialised Pack Op (SVE includes padding!)
// CHECK:           %[[PACK:.+]] = linalg.pack %[[SRC]]
// WITH-SVE-SAME:     padding_value(%[[PAD]] : f32)
// NO-SVE-NOT:        padding_value

// CHECK-SAME:        outer_dims_perm = [0, 2, 1]
// CHECK-SAME:        inner_dims_pos = [2, 1]

// NO-SVE-SAME:       inner_tiles = [8, 1] into %[[INIT]] : tensor<128x32x320xf32> -> tensor<128x40x32x8x1xf32>

// WITH-SVE-SAME:     inner_tiles = [%[[C8_VSCALE]], 1] into %[[INIT]] : tensor<128x32x320xf32> -> tensor<128x?x32x?x1xf32>

// CHECK:         return %[[PACK]]

// -----

#map = affine_map<(b, m, n, k) -> (b, m, k)>
#map1 = affine_map<(b, m, n, k) -> (b, k, n)>
#map2 = affine_map<(b, m, n, k) -> (b, m, n)>
#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 80, 320, ?]>
func.func @batch_matmul_RETURN_unset(%arg0: tensor<128x80x320xf32, #encoding>, %k: index) -> tensor<128x80x320xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+sve", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %arg0 encoding_dims{%k} : tensor<128x80x320xf32, #encoding> -> tensor<128x80x320xf32>
  return %0 : tensor<128x80x320xf32>
}

/// NOTE: For RETURN, the inner tile corresponding to the "N" dimension is
/// scalable, hence NO-SVE and WITH-SVE differ!

// CHECK-LABEL: func.func @batch_matmul_RETURN_unset
// NO-SVE-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<128x10x40x8x8xf32>
// WITH-SVE-SAME:  %[[INPUT:[a-zA-Z0-9]+]]: tensor<128x10x?x8x?xf32>

// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty()

/// SVE: the number of outer tiles corresponding to the inner scalable tile
// WITH-SVE-DAG:  %[[C8:.+]] = arith.constant 8 : index
// WITH-SVE:      %[[VSCALE:.+]] = vector.vscale
// WITH-SVE:      %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]] : index

/// The newly materialised UnPack Op
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[INPUT]]
// NO-SVE-SAME:       outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %[[EMPTY]]
// WITH-SVE-SAME:     outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, %[[C8_VSCALE]]] into %[[EMPTY]]

// CHECK:         return %[[UNPACK]]

// -----

#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [16, 1, 16]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [16, 1, 16]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [16, 1, 16]>
func.func @matvec_shaped_matmul_lowering_f32f32f32_aarch64(
    %arg0: tensor<16x16xf32, #encoding_lhs>,
    %arg1: tensor<16x1xf32, #encoding_rhs>,
    %arg2: tensor<16x1xf32, #encoding_result>
) -> tensor<16x1xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<16x16xf32, #encoding_lhs>, tensor<16x1xf32, #encoding_rhs>)
                    outs(%arg2 : tensor<16x1xf32, #encoding_result>) -> tensor<16x1xf32, #encoding_result>
  return %0 : tensor<16x1xf32, #encoding_result>
}

// CHECK-LABEL: func @matvec_shaped_matmul_lowering_f32f32f32_aarch64
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<2x16x8x1xf32>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<1x16x1x1xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x2x1x8xf32>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[RHS]], %[[LHS]]
// CHECK-SAME:       outs(%[[OUT]]
// CHECK:         return %[[MMT4D]]

// -----

// We test with IREE ops to ensure sizes are calculated correctly and be used in
// the ops. They do not belong to materialize_encoding_for_iree_ops.mlir because
// it requires --iree-llvmcpu-enable-scalable-vectorization=true experimental
// flag to be enabled.

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features="+sve", target_triple="aarch64-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf32, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  return
}
//     CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  WITH-SVE-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//   CHECK-LABEL: func @matmul_lowering_f32f32f32_aarch64()
//  WITH-SVE-DAG:   %[[C8:.+]] = arith.constant 8 : index
//     CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//     CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//     CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//     CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//         CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//    CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
///
/// Compute "tiled" N:
///  *  N / 8 for fixed-width tiling,
///  *  N / ( 8 * vscale) for scalable tiling
///
//        NO-SVE:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//      WITH-SVE:   %[[VSCALE:.+]] = vector.vscale
//      WITH-SVE:   %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]]
//      WITH-SVE:   %[[TILED_N:.+]] = affine.apply #[[$MAP1]]()[%[[N]], %[[C8_VSCALE]]]
//         CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_N]], %[[K]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x1xf32>>{%[[TILED_N]], %[[K]], %[[C8_VSCALE]]}
//         CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x?xf32>>{%[[TILED_M]], %[[TILED_N]], %[[C8_VSCALE]]}
//         CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//    CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//         CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], %[[C8_VSCALE]], 1], strides = [1, 1, 1, 1]
//         CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, %[[C8_VSCALE]]], strides = [1, 1, 1, 1]
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//    CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//    CHECK-SAME:       outs(%[[OUTS]] :
//         CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, %[[C8_VSCALE]]], strides = [1, 1, 1, 1]

// -----

#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iteration_sizes = [16, 16]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iteration_sizes= [16, 16]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iteration_sizes = [16, 16]>
func.func @matvec_lowering_f32f32f32_aarch64(
    %lhs: tensor<16x16xf32, #encoding_lhs>,
    %rhs: tensor<16xf32, #encoding_rhs>,
    %init: tensor<16xf32, #encoding_result>
) -> tensor<16xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %result = linalg.matvec
    ins(%lhs, %rhs : tensor<16x16xf32, #encoding_lhs>, tensor<16xf32, #encoding_rhs>)
    outs(%init : tensor<16xf32, #encoding_result>)
    -> tensor<16xf32, #encoding_result>
  return %result : tensor<16xf32, #encoding_result>
}
// CHECK-LABEL: func @matvec_lowering_f32f32f32_aarch64(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<2x16x8x1xf32>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<16x1xf32>
//  CHECK-SAME:   %[[ACC:[a-zA-Z0-9]+]]: tensor<2x8xf32>
//       CHECK:   %[[EXPANDED_RHS:.+]] = tensor.expand_shape %[[RHS]]
//  CHECK-SAME:     [0, 1], [2, 3]
//  CHECK-SAME:     output_shape [1, 16, 1, 1]
//  CHECK-SAME:     : tensor<16x1xf32> into tensor<1x16x1x1xf32>
//       CHECK:   %[[EXPANDED_ACC:.+]] = tensor.expand_shape %[[ACC]]
//  CHECK-SAME:     [0, 1], [2, 3]
//  CHECK-SAME:     output_shape [1, 2, 1, 8]
//  CHECK-SAME:     : tensor<2x8xf32> into tensor<1x2x1x8xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:      ins(%[[EXPANDED_RHS]], %[[LHS]]
//  CHECK-SAME:     outs(%[[EXPANDED_ACC]]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]]
//  CHECK-SAME:     [0, 1], [2, 3]
//  CHECK-SAME:     : tensor<1x2x1x8xf32> into tensor<2x8xf32>
//       CHECK:   return %[[COLLAPSED]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 1, 16]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 1, 16]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 1, 16]>
func.func @matvec_lowering_f32f32f32_aarch64(
    %lhs: tensor<16x16xf32, #encoding_lhs>,
    %rhs: tensor<16x1xf32, #encoding_rhs>,
    %result: tensor<16x1xf32, #encoding_result>
) -> tensor<16x1xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<16x16xf32, #encoding_lhs>, tensor<16x1xf32, #encoding_rhs>)
    outs(%result : tensor<16x1xf32, #encoding_result>)
    -> tensor<16x1xf32, #encoding_result>
  return %matmul : tensor<16x1xf32, #encoding_result>
}
// CHECK-LABEL: func @matvec_lowering_f32f32f32_aarch64(
//  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<2x16x8x1xf32>
//  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<1x16x1x1xf32>
//  CHECK-SAME:     %[[OUTS:[a-zA-Z0-9]+]]: tensor<1x2x1x8xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[RHS]], %[[LHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f16f16f16_aarch64(
    %lhs: tensor<?x?xf16, #encoding_lhs>,
    %rhs: tensor<?x?xf16, #encoding_rhs>,
    %result: tensor<?x?xf16, #encoding_result>
) -> tensor<?x?xf16, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features="+sve", target_triple="aarch64-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xf16, #encoding_lhs>, tensor<?x?xf16, #encoding_rhs>)
    outs(%result : tensor<?x?xf16, #encoding_result>)
    -> tensor<?x?xf16, #encoding_result>
  return %matmul : tensor<?x?xf16, #encoding_result>
}
//   CHECK-LABEL: func @matmul_lowering_f16f16f16_aarch64(
//   NO-SVE-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x1xf16>
//   NO-SVE-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x1xf16>
//   NO-SVE-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xf16>
// WITH-SVE-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x1xf16>
// WITH-SVE-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x1xf16>
// WITH-SVE-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x?xf16>
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//    CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//    CHECK-SAME:       outs(%[[OUTS]] :
//         CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f16f16_aarch64(
    %arg0: index,
    %arg1: index,
    %lhs_f32: tensor<?x?xf32, #encoding_lhs>,
    %rhs: tensor<?x?xf16, #encoding_rhs>,
    %dest: tensor<?x?xf16, #encoding_result>
) -> tensor<?x?xf16, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features="+sve", target_triple="aarch64-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %empty = tensor.empty(%arg0, %arg1) : tensor<?x?xf16, #encoding_lhs>
  %lhs_f16 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%lhs_f32 : tensor<?x?xf32, #encoding_lhs>)
    outs(%empty : tensor<?x?xf16, #encoding_lhs>) {
  ^bb0(%in: f32, %out: f16):
    %trunc = arith.truncf %in : f32 to f16
    linalg.yield %trunc : f16
  } -> tensor<?x?xf16, #encoding_lhs>
  %result = linalg.matmul
    ins(%lhs_f16, %rhs : tensor<?x?xf16, #encoding_lhs>, tensor<?x?xf16, #encoding_rhs>)
    outs(%dest : tensor<?x?xf16, #encoding_result>)
    -> tensor<?x?xf16, #encoding_result>
  return %result : tensor<?x?xf16, #encoding_result>
}
//         CHECK: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-LABEL: func @matmul_lowering_f32f16f16_aarch64(
//         CHECK:   %[[ARG0:[a-zA-Z0-9]+]]: index
//         CHECK:   %[[ARG1:[a-zA-Z0-9]+]]: index
//   NO-SVE-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x1xf32>
//   NO-SVE-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x1xf16>
//   NO-SVE-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xf16>
// WITH-SVE-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x1xf32>
// WITH-SVE-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x1xf16>
// WITH-SVE-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x?xf16>
//         CHECK:   %[[M_CEILDIV_8:.+]] = affine.apply #[[$MAP0]]()[%[[ARG0]]]
//         CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[M_CEILDIV_8]], %[[ARG1]]) : tensor<?x?x8x1xf16>
//         CHECK:   %[[LHS_F16:.+]] = linalg.generic
//    CHECK-SAME:       ins(%[[LHS]] : tensor<?x?x8x1xf32>)
//    CHECK-SAME:       outs(%[[EMPTY]] : tensor<?x?x8x1xf16>)
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//    CHECK-SAME:       ins(%[[LHS_F16]], %[[RHS]] :
//    CHECK-SAME:       outs(%[[OUTS]] :
//         CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_aarch64(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi8, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>, tensor<?x?xi8, #encoding_rhs>)
    outs(%result : tensor<?x?xi32, #encoding_result>)
    -> tensor<?x?xi32, #encoding_result>
  return %matmul : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_aarch64(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?xi8>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?xi8>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_aarch64_dotprod(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi8, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+sve", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>, tensor<?x?xi8, #encoding_rhs>)
    outs(%result : tensor<?x?xi32, #encoding_result>)
    -> tensor<?x?xi32, #encoding_result>
  return %matmul : tensor<?x?xi32, #encoding_result>
}
//   CHECK-LABEL: func @matmul_lowering_i8i8i32_aarch64_dotprod(
//   NO-SVE-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xi8>
//   NO-SVE-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xi8>
//   NO-SVE-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi32>
// WITH-SVE-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xi8>
// WITH-SVE-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x4xi8>
// WITH-SVE-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x?xi32>
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//    CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//    CHECK-SAME:       outs(%[[OUTS]] :
//         CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_aarch64_i8mm(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi8, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+i8mm,+sve", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>, tensor<?x?xi8, #encoding_rhs>)
    outs(%result : tensor<?x?xi32, #encoding_result>)
    -> tensor<?x?xi32, #encoding_result>
  return %matmul : tensor<?x?xi32, #encoding_result>
}
//   CHECK-LABEL: func @matmul_lowering_i8i8i32_aarch64_i8mm(
//   NO-SVE-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi8>
//   NO-SVE-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi8>
//   NO-SVE-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi32>
// WITH-SVE-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi8>
// WITH-SVE-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x8xi8>
// WITH-SVE-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x?xi32>
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//    CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//    CHECK-SAME:       outs(%[[OUTS]] :
//         CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i4i32_aarch64(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi4, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>, tensor<?x?xi4, #encoding_rhs>)
    outs(%result : tensor<?x?xi32, #encoding_result>)
    -> tensor<?x?xi32, #encoding_result>
  return %matmul : tensor<?x?xi32, #encoding_result>
}
//   CHECK-LABEL: func @matmul_lowering_i8i4i32_aarch64(
//   CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x4x2xi8>
//   CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x2xi4>
//   CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x4x16xi32>
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//    CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//    CHECK-SAME:       outs(%[[OUTS]] :
//         CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i4i32_aarch64_dotprod(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi4, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>, tensor<?x?xi4, #encoding_rhs>)
    outs(%result : tensor<?x?xi32, #encoding_result>)
    -> tensor<?x?xi32, #encoding_result>
  return %matmul : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i4i32_aarch64_dotprod(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi8>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi4>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i4i32_aarch64_i8mm(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi4, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+i8mm", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>, tensor<?x?xi4, #encoding_rhs>)
    outs(%result : tensor<?x?xi32, #encoding_result>)
    -> tensor<?x?xi32, #encoding_result>
  return %matmul : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i4i32_aarch64_i8mm(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x4x16xi8>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x16xi4>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x4x8xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]
