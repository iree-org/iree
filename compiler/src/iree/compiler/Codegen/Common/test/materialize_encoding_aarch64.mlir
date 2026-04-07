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

// -----

#map_in  = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh + fh, ow + fw, ic)>
#map_f   = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (fh, fw, ic, oc)>
#map_out = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh, ow, oc)>

#encoding_conv_input = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map_in, #map_f, #map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_input_pack(%arg0: tensor<1x16x16x4xf32>)
    -> tensor<1x16x16x4xf32, #encoding_conv_input>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<1x16x16x4xf32> -> tensor<1x16x16x4xf32, #encoding_conv_input>
  return %0 : tensor<1x16x16x4xf32, #encoding_conv_input>
}
// CHECK-LABEL: func.func @conv_input_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x16x16x4xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x16x16x4xf32> -> tensor<1x1x16x16x16xf32>
// CHECK:         return %[[PACK]]

// -----

#map_in  = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh + fh, ow + fw, ic)>
#map_f   = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (fh, fw, ic, oc)>
#map_out = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh, ow, oc)>

#encoding_conv_filter = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map_in, #map_f, #map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_filter_pack(%arg0: tensor<3x3x4x8xf32>)
    -> tensor<3x3x4x8xf32, #encoding_conv_filter>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<3x3x4x8xf32> -> tensor<3x3x4x8xf32, #encoding_conv_filter>
  return %0 : tensor<3x3x4x8xf32, #encoding_conv_filter>
}
// CHECK-LABEL: func.func @conv_filter_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<3x3x4x8xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [3, 2, 0, 1]
// CHECK-SAME:      inner_dims_pos = [3, 2]
// CHECK-SAME:      inner_tiles = [16, 16]
// CHECK-SAME:      : tensor<3x3x4x8xf32> -> tensor<1x1x3x3x16x16xf32>
// CHECK:         return %[[PACK]]

// -----

#map_in  = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh + fh, ow + fw, ic)>
#map_f   = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (fh, fw, ic, oc)>
#map_out = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh, ow, oc)>

#encoding_conv_output = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map_in, #map_f, #map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_output_unset(%arg0: tensor<1x14x14x8xf32, #encoding_conv_output>)
    -> tensor<1x14x14x8xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %arg0
       : tensor<1x14x14x8xf32, #encoding_conv_output> -> tensor<1x14x14x8xf32>
  return %0 : tensor<1x14x14x8xf32>
}
// CHECK-LABEL: func.func @conv_output_unset
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x14x14x16xf32>
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x1x14x14x16xf32> -> tensor<1x14x14x8xf32>
// CHECK:         return %[[UNPACK]]

// -----

// Full conv materialization: direct-access 9D tiled generic with tensor.extract.
//
// Map invariant (all filter formats must produce the same canonical generics):
//   9D filter:  (d0..d8) -> (d1, d4, d5, d6, d7, d8)
//   9D output:  (d0..d8) -> (d0, d1, d2, d3, d7)
//
// CHECK: #[[$M_FLT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d7, d8)>
// CHECK: #[[$M_OUT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>

#map_in  = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh + fh, ow + fw, ic)>
#map_f   = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (fh, fw, ic, oc)>
#map_out = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh, ow, oc)>

#encoding_in = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map_in, #map_f, #map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>
#encoding_f = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map_in, #map_f, #map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>
#encoding_out = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map_in, #map_f, #map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv2d_nhwc_hwcf_materialize(
    %input  : tensor<1x16x16x4xf32, #encoding_in>,
    %filter : tensor<3x3x4x8xf32, #encoding_f>,
    %output : tensor<1x14x14x8xf32, #encoding_out>)
    -> tensor<1x14x14x8xf32, #encoding_out>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = linalg.conv_2d_nhwc_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %filter
           : tensor<1x16x16x4xf32, #encoding_in>,
             tensor<3x3x4x8xf32, #encoding_f>)
         outs(%output : tensor<1x14x14x8xf32, #encoding_out>)
         -> tensor<1x14x14x8xf32, #encoding_out>
  return %0 : tensor<1x14x14x8xf32, #encoding_out>
}
// CHECK-LABEL: func.func @conv2d_nhwc_hwcf_materialize
// CHECK-SAME:    %[[INPUT:.+]]: tensor<1x1x16x16x16xf32>
// CHECK-SAME:    %[[FILTER:.+]]: tensor<1x1x3x3x16x16xf32>
// CHECK-SAME:    %[[OUTPUT:.+]]: tensor<1x1x14x14x16xf32>
//
// Direct-access 9D tiled computation (input accessed via tensor.extract):
// CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$M_FLT]], #[[$M_OUT]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel",
// CHECK-SAME:                        "reduction", "reduction", "reduction",
// CHECK-SAME:                        "parallel", "reduction"]
// CHECK-SAME:      ins(%[[FILTER]]
// CHECK-SAME:      outs(%[[OUTPUT]]
// CHECK:           tensor.extract %[[INPUT]]
// CHECK:           %[[MUL:.+]] = arith.mulf
// CHECK:           %[[ADD:.+]] = arith.addf %[[MUL]]
// CHECK:           linalg.yield %[[ADD]]
// CHECK:         return %[[RESULT]]

// -----

// NCHW input: pack should canonicalize to [N, H, W, IC/c0, c0].

#nchw_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#nchw_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#nchw_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#encoding_nchw_input = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_map_in, #nchw_map_f, #nchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>

func.func @conv_nchw_input_pack(%arg0: tensor<1x4x16x16xf32>)
    -> tensor<1x4x16x16xf32, #encoding_nchw_input>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<1x4x16x16xf32> -> tensor<1x4x16x16xf32, #encoding_nchw_input>
  return %0 : tensor<1x4x16x16xf32, #encoding_nchw_input>
}
// CHECK-LABEL: func.func @conv_nchw_input_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x4x16x16xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x4x16x16xf32> -> tensor<1x1x16x16x16xf32>
// CHECK:         return %[[PACK]]

// -----

// FCHW filter: pack should canonicalize to [OC/k0, FH, FW, IC/c0, k0, c0].

#nchw_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#nchw_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#nchw_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#encoding_fchw_filter = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_map_in, #nchw_map_f, #nchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>

func.func @conv_fchw_filter_pack(%arg0: tensor<8x4x3x3xf32>)
    -> tensor<8x4x3x3xf32, #encoding_fchw_filter>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<8x4x3x3xf32> -> tensor<8x4x3x3xf32, #encoding_fchw_filter>
  return %0 : tensor<8x4x3x3xf32, #encoding_fchw_filter>
}
// CHECK-LABEL: func.func @conv_fchw_filter_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x4x3x3xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [16, 16]
// CHECK-SAME:      : tensor<8x4x3x3xf32> -> tensor<1x1x3x3x16x16xf32>
// CHECK:         return %[[PACK]]

// -----

// NCHW output: unpack should reverse the canonical [N, OH, OW, OC/k0, k0].

#nchw_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#nchw_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#nchw_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#encoding_nchw_output = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_map_in, #nchw_map_f, #nchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>

func.func @conv_nchw_output_unset(%arg0: tensor<1x8x14x14xf32, #encoding_nchw_output>)
    -> tensor<1x8x14x14xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %arg0
       : tensor<1x8x14x14xf32, #encoding_nchw_output> -> tensor<1x8x14x14xf32>
  return %0 : tensor<1x8x14x14xf32>
}
// CHECK-LABEL: func.func @conv_nchw_output_unset
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x14x14x16xf32>
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x1x14x14x16xf32> -> tensor<1x8x14x14xf32>
// CHECK:         return %[[UNPACK]]

// -----

// Full conv_2d_nchw_fchw materialization: direct-access 9D tiled generic with tensor.extract.
// Packing canonicalizes NCHW/FCHW to the same internal layout as NHWC/HWCF, so the
// 9D computation generic must have identical indexing maps.
//
// CHECK: #[[$M_FLT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d7, d8)>
// CHECK: #[[$M_OUT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>

#nchw_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#nchw_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#nchw_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#encoding_nchw_in = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_map_in, #nchw_map_f, #nchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>
#encoding_nchw_f = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_map_in, #nchw_map_f, #nchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>
#encoding_nchw_out = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_map_in, #nchw_map_f, #nchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>

func.func @conv2d_nchw_fchw_materialize(
    %input  : tensor<1x4x16x16xf32, #encoding_nchw_in>,
    %filter : tensor<8x4x3x3xf32, #encoding_nchw_f>,
    %output : tensor<1x8x14x14xf32, #encoding_nchw_out>)
    -> tensor<1x8x14x14xf32, #encoding_nchw_out>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = linalg.conv_2d_nchw_fchw
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %filter
           : tensor<1x4x16x16xf32, #encoding_nchw_in>,
             tensor<8x4x3x3xf32, #encoding_nchw_f>)
         outs(%output : tensor<1x8x14x14xf32, #encoding_nchw_out>)
         -> tensor<1x8x14x14xf32, #encoding_nchw_out>
  return %0 : tensor<1x8x14x14xf32, #encoding_nchw_out>
}
// CHECK-LABEL: func.func @conv2d_nchw_fchw_materialize
// CHECK-SAME:    %[[INPUT:.+]]: tensor<1x1x16x16x16xf32>
// CHECK-SAME:    %[[FILTER:.+]]: tensor<1x1x3x3x16x16xf32>
// CHECK-SAME:    %[[OUTPUT:.+]]: tensor<1x1x14x14x16xf32>
//
// Direct-access 9D tiled computation (input accessed via tensor.extract):
// CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$M_FLT]], #[[$M_OUT]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel",
// CHECK-SAME:                        "reduction", "reduction", "reduction",
// CHECK-SAME:                        "parallel", "reduction"]
// CHECK-SAME:      ins(%[[FILTER]]
// CHECK-SAME:      outs(%[[OUTPUT]]
// CHECK:           tensor.extract %[[INPUT]]
// CHECK:           %[[MUL:.+]] = arith.mulf
// CHECK:           %[[ADD:.+]] = arith.addf %[[MUL]]
// CHECK:           linalg.yield %[[ADD]]
// CHECK:         return %[[RESULT]]

// -----

// FHWC filter: pack should produce canonical [OC/k0, FH, FW, IC/c0, k0, c0].

#nhwc_fhwc_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#nhwc_fhwc_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#nhwc_fhwc_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#encoding_fhwc_filter = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_fhwc_map_in, #nhwc_fhwc_map_f, #nhwc_fhwc_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_fhwc_filter_pack(%arg0: tensor<8x3x3x4xf32>)
    -> tensor<8x3x3x4xf32, #encoding_fhwc_filter>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<8x3x3x4xf32> -> tensor<8x3x3x4xf32, #encoding_fhwc_filter>
  return %0 : tensor<8x3x3x4xf32, #encoding_fhwc_filter>
}
// CHECK-LABEL: func.func @conv_fhwc_filter_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x3x3x4xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [0, 3]
// CHECK-SAME:      inner_tiles = [16, 16]
// CHECK-SAME:      : tensor<8x3x3x4xf32> -> tensor<1x1x3x3x16x16xf32>
// CHECK:         return %[[PACK]]

// -----

// FHWC input: input packing is independent of filter layout; produces [N, H, W, IC/c0, c0].

#nhwc_fhwc_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#nhwc_fhwc_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#nhwc_fhwc_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#encoding_fhwc_input = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_fhwc_map_in, #nhwc_fhwc_map_f, #nhwc_fhwc_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_fhwc_input_pack(%arg0: tensor<1x16x16x4xf32>)
    -> tensor<1x16x16x4xf32, #encoding_fhwc_input>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0
       : tensor<1x16x16x4xf32> -> tensor<1x16x16x4xf32, #encoding_fhwc_input>
  return %0 : tensor<1x16x16x4xf32, #encoding_fhwc_input>
}
// CHECK-LABEL: func.func @conv_fhwc_input_pack
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x16x16x4xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x16x16x4xf32> -> tensor<1x1x16x16x16xf32>
// CHECK:         return %[[PACK]]

// -----

// FHWC output: output unpacking is independent of filter layout; produces [N, OH, OW, OC] from [N, OH, OW, OC/k0, k0].

#nhwc_fhwc_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#nhwc_fhwc_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#nhwc_fhwc_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#encoding_fhwc_output = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_fhwc_map_in, #nhwc_fhwc_map_f, #nhwc_fhwc_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_fhwc_output_unset(%arg0: tensor<1x14x14x8xf32, #encoding_fhwc_output>)
    -> tensor<1x14x14x8xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %arg0
       : tensor<1x14x14x8xf32, #encoding_fhwc_output> -> tensor<1x14x14x8xf32>
  return %0 : tensor<1x14x14x8xf32>
}
// CHECK-LABEL: func.func @conv_fhwc_output_unset
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x14x14x16xf32>
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x1x14x14x16xf32> -> tensor<1x14x14x8xf32>
// CHECK:         return %[[UNPACK]]

// -----

// Full conv_2d_nhwc_fhwc materialization: direct-access 9D tiled generic with tensor.extract.
// The canonical 9D computation generic must have identical indexing maps to the NHWC/HWCF case.
//
// CHECK: #[[$M_FLT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d7, d8)>
// CHECK: #[[$M_OUT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>

#nhwc_fhwc_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#nhwc_fhwc_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#nhwc_fhwc_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#encoding_nhwc_fhwc_in = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_fhwc_map_in, #nhwc_fhwc_map_f, #nhwc_fhwc_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>
#encoding_nhwc_fhwc_f = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_fhwc_map_in, #nhwc_fhwc_map_f, #nhwc_fhwc_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>
#encoding_nhwc_fhwc_out = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_fhwc_map_in, #nhwc_fhwc_map_f, #nhwc_fhwc_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv2d_nhwc_fhwc_materialize(
    %input  : tensor<1x16x16x4xf32, #encoding_nhwc_fhwc_in>,
    %filter : tensor<8x3x3x4xf32, #encoding_nhwc_fhwc_f>,
    %output : tensor<1x14x14x8xf32, #encoding_nhwc_fhwc_out>)
    -> tensor<1x14x14x8xf32, #encoding_nhwc_fhwc_out>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = linalg.conv_2d_nhwc_fhwc
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %filter
           : tensor<1x16x16x4xf32, #encoding_nhwc_fhwc_in>,
             tensor<8x3x3x4xf32, #encoding_nhwc_fhwc_f>)
         outs(%output : tensor<1x14x14x8xf32, #encoding_nhwc_fhwc_out>)
         -> tensor<1x14x14x8xf32, #encoding_nhwc_fhwc_out>
  return %0 : tensor<1x14x14x8xf32, #encoding_nhwc_fhwc_out>
}
// CHECK-LABEL: func.func @conv2d_nhwc_fhwc_materialize
// CHECK-SAME:    %[[INPUT:.+]]: tensor<1x1x16x16x16xf32>
// CHECK-SAME:    %[[FILTER:.+]]: tensor<1x1x3x3x16x16xf32>
// CHECK-SAME:    %[[OUTPUT:.+]]: tensor<1x1x14x14x16xf32>
//
// Direct-access 9D tiled computation (input accessed via tensor.extract):
// CHECK:         %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$M_FLT]], #[[$M_OUT]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel",
// CHECK-SAME:                        "reduction", "reduction", "reduction",
// CHECK-SAME:                        "parallel", "reduction"]
// CHECK-SAME:      ins(%[[FILTER]]
// CHECK-SAME:      outs(%[[OUTPUT]]
// CHECK:           tensor.extract %[[INPUT]]
// CHECK:           %[[MUL:.+]] = arith.mulf
// CHECK:           %[[ADD:.+]] = arith.addf %[[MUL]]
// CHECK:           linalg.yield %[[ADD]]
// CHECK:         return %[[RESULT]]

// -----

// Full-pack canonicalization for conv_2d_nhwc_hwcf: verifies outer_dims_perm,
// inner_dims_pos, and inner_tiles for all three operands together.
// Input and output are NHWC (identity perm, tile last dim).
// Filter is HWCF (OC promoted to front, both IC and OC tiled).

#nhwc_hwcf_map_in  = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh + fh, ow + fw, ic)>
#nhwc_hwcf_map_f   = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (fh, fw, ic, oc)>
#nhwc_hwcf_map_out = affine_map<(n, oh, ow, oc, fh, fw, ic) -> (n, oh, ow, oc)>

#enc_nhwc_hwcf_in = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_hwcf_map_in, #nhwc_hwcf_map_f, #nhwc_hwcf_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>
#enc_nhwc_hwcf_f = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_hwcf_map_in, #nhwc_hwcf_map_f, #nhwc_hwcf_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>
#enc_nhwc_hwcf_out = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_hwcf_map_in, #nhwc_hwcf_map_f, #nhwc_hwcf_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_nhwc_hwcf_pack_canonicalization(
    %input  : tensor<1x16x16x4xf32>,
    %filter : tensor<3x3x4x8xf32>,
    %output : tensor<1x14x14x8xf32>)
    -> tensor<1x14x14x8xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %enc_in  = iree_encoding.set_encoding %input
      : tensor<1x16x16x4xf32> -> tensor<1x16x16x4xf32, #enc_nhwc_hwcf_in>
  %enc_f   = iree_encoding.set_encoding %filter
      : tensor<3x3x4x8xf32> -> tensor<3x3x4x8xf32, #enc_nhwc_hwcf_f>
  %enc_out = iree_encoding.set_encoding %output
      : tensor<1x14x14x8xf32> -> tensor<1x14x14x8xf32, #enc_nhwc_hwcf_out>
  %0 = linalg.conv_2d_nhwc_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%enc_in, %enc_f
           : tensor<1x16x16x4xf32, #enc_nhwc_hwcf_in>,
             tensor<3x3x4x8xf32, #enc_nhwc_hwcf_f>)
         outs(%enc_out : tensor<1x14x14x8xf32, #enc_nhwc_hwcf_out>)
         -> tensor<1x14x14x8xf32, #enc_nhwc_hwcf_out>
  %1 = iree_encoding.unset_encoding %0
      : tensor<1x14x14x8xf32, #enc_nhwc_hwcf_out> -> tensor<1x14x14x8xf32>
  return %1 : tensor<1x14x14x8xf32>
}
// CHECK-LABEL: func.func @conv_nhwc_hwcf_pack_canonicalization
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x16x16x4xf32>
// CHECK-SAME:    %[[F:[a-zA-Z0-9]+]]: tensor<3x3x4x8xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x14x14x8xf32>
//
// Input pack: NHWC → [N, IC/c0, H, W, c0]  (NCHWc layout)
// CHECK:         %[[PACK_IN:.+]] = linalg.pack %[[IN]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x16x16x4xf32> -> tensor<1x1x16x16x16xf32>
//
// Filter pack: HWCF → [OC/k0, IC/c0, FH, FW, c0, k0]  (XNNPACK convention)
// CHECK:         %[[PACK_F:.+]] = linalg.pack %[[F]]
// CHECK-SAME:      outer_dims_perm = [3, 2, 0, 1]
// CHECK-SAME:      inner_dims_pos = [3, 2]
// CHECK-SAME:      inner_tiles = [16, 16]
// CHECK-SAME:      : tensor<3x3x4x8xf32> -> tensor<1x1x3x3x16x16xf32>
//
// Output pack: NHWC → [N, OC/k0, OH, OW, k0]  (NCHWc layout)
// CHECK:         %[[PACK_OUT:.+]] = linalg.pack %[[OUT]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x14x14x8xf32> -> tensor<1x1x14x14x16xf32>
//
// Output unpack: [N, OC/k0, OH, OW, k0] → NHWC
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %{{[^ ]+}}
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x1x14x14x16xf32> -> tensor<1x14x14x8xf32>

// -----

// Full-pack canonicalization for conv_2d_nchw_fchw: packing must canonicalize
// NCHW input/output and FCHW filter to the same internal layout as NHWC/HWCF.
// Input/output: outer_dims_perm keeps C in-place (already NCHWc-compatible).
// Filter: outer_dims_perm keeps dims in-place (OC, IC already in XNNPACK order).

#nchw_fchw_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#nchw_fchw_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#nchw_fchw_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#enc_nchw_fchw_in = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_fchw_map_in, #nchw_fchw_map_f, #nchw_fchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>
#enc_nchw_fchw_f = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_fchw_map_in, #nchw_fchw_map_f, #nchw_fchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>
#enc_nchw_fchw_out = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nchw_fchw_map_in, #nchw_fchw_map_f, #nchw_fchw_map_out],
  iteration_sizes = [1, 8, 14, 14, 4, 3, 3]>

func.func @conv_nchw_fchw_pack_canonicalization(
    %input  : tensor<1x4x16x16xf32>,
    %filter : tensor<8x4x3x3xf32>,
    %output : tensor<1x8x14x14xf32>)
    -> tensor<1x8x14x14xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %enc_in  = iree_encoding.set_encoding %input
      : tensor<1x4x16x16xf32> -> tensor<1x4x16x16xf32, #enc_nchw_fchw_in>
  %enc_f   = iree_encoding.set_encoding %filter
      : tensor<8x4x3x3xf32> -> tensor<8x4x3x3xf32, #enc_nchw_fchw_f>
  %enc_out = iree_encoding.set_encoding %output
      : tensor<1x8x14x14xf32> -> tensor<1x8x14x14xf32, #enc_nchw_fchw_out>
  %0 = linalg.conv_2d_nchw_fchw
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%enc_in, %enc_f
           : tensor<1x4x16x16xf32, #enc_nchw_fchw_in>,
             tensor<8x4x3x3xf32, #enc_nchw_fchw_f>)
         outs(%enc_out : tensor<1x8x14x14xf32, #enc_nchw_fchw_out>)
         -> tensor<1x8x14x14xf32, #enc_nchw_fchw_out>
  %1 = iree_encoding.unset_encoding %0
      : tensor<1x8x14x14xf32, #enc_nchw_fchw_out> -> tensor<1x8x14x14xf32>
  return %1 : tensor<1x8x14x14xf32>
}
// CHECK-LABEL: func.func @conv_nchw_fchw_pack_canonicalization
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x4x16x16xf32>
// CHECK-SAME:    %[[F:[a-zA-Z0-9]+]]: tensor<8x4x3x3xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x8x14x14xf32>
//
// Input pack: NCHW → [N, IC/c0, H, W, c0]  (identity outer perm, tile dim 1)
// CHECK:         %[[PACK_IN:.+]] = linalg.pack %[[IN]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x4x16x16xf32> -> tensor<1x1x16x16x16xf32>
//
// Filter pack: FCHW → [OC/k0, IC/c0, FH, FW, c0, k0]  (identity outer perm, tile dims 0 and 1)
// CHECK:         %[[PACK_F:.+]] = linalg.pack %[[F]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [16, 16]
// CHECK-SAME:      : tensor<8x4x3x3xf32> -> tensor<1x1x3x3x16x16xf32>
//
// Output pack: NCHW → [N, OC/k0, OH, OW, k0]  (identity outer perm, tile dim 1)
// CHECK:         %[[PACK_OUT:.+]] = linalg.pack %[[OUT]]
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x8x14x14xf32> -> tensor<1x1x14x14x16xf32>
//
// Output unpack: [N, OC/k0, OH, OW, k0] → NCHW
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %{{[^ ]+}}
// CHECK-SAME:      outer_dims_perm = [0, 1, 2, 3]
// CHECK-SAME:      inner_dims_pos = [1]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x1x14x14x16xf32> -> tensor<1x8x14x14xf32>

// -----

// Full-pack canonicalization for conv_2d_nhwc_fhwc: input and output use NHWC
// (identical packing to HWCF variant). Filter uses FHWC: identity outer perm,
// inner_dims_pos = [0, 3] (OC at dim 0, IC at dim 3).

#nhwc_fhwc2_map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#nhwc_fhwc2_map_f   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#nhwc_fhwc2_map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#enc_nhwc_fhwc_in = #iree_encoding.encoding<operand_index = 0, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_fhwc2_map_in, #nhwc_fhwc2_map_f, #nhwc_fhwc2_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>
#enc_nhwc_fhwc_f = #iree_encoding.encoding<operand_index = 1, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_fhwc2_map_in, #nhwc_fhwc2_map_f, #nhwc_fhwc2_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>
#enc_nhwc_fhwc_out = #iree_encoding.encoding<operand_index = 2, op_type = conv,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#nhwc_fhwc2_map_in, #nhwc_fhwc2_map_f, #nhwc_fhwc2_map_out],
  iteration_sizes = [1, 14, 14, 8, 3, 3, 4]>

func.func @conv_nhwc_fhwc_pack_canonicalization(
    %input  : tensor<1x16x16x4xf32>,
    %filter : tensor<8x3x3x4xf32>,
    %output : tensor<1x14x14x8xf32>)
    -> tensor<1x14x14x8xf32>
    attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz",
      {target_triple="aarch64-xyz-xyz", cpu_features="+neon",
       iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %enc_in  = iree_encoding.set_encoding %input
      : tensor<1x16x16x4xf32> -> tensor<1x16x16x4xf32, #enc_nhwc_fhwc_in>
  %enc_f   = iree_encoding.set_encoding %filter
      : tensor<8x3x3x4xf32> -> tensor<8x3x3x4xf32, #enc_nhwc_fhwc_f>
  %enc_out = iree_encoding.set_encoding %output
      : tensor<1x14x14x8xf32> -> tensor<1x14x14x8xf32, #enc_nhwc_fhwc_out>
  %0 = linalg.conv_2d_nhwc_fhwc
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%enc_in, %enc_f
           : tensor<1x16x16x4xf32, #enc_nhwc_fhwc_in>,
             tensor<8x3x3x4xf32, #enc_nhwc_fhwc_f>)
         outs(%enc_out : tensor<1x14x14x8xf32, #enc_nhwc_fhwc_out>)
         -> tensor<1x14x14x8xf32, #enc_nhwc_fhwc_out>
  %1 = iree_encoding.unset_encoding %0
      : tensor<1x14x14x8xf32, #enc_nhwc_fhwc_out> -> tensor<1x14x14x8xf32>
  return %1 : tensor<1x14x14x8xf32>
}
// CHECK-LABEL: func.func @conv_nhwc_fhwc_pack_canonicalization
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x16x16x4xf32>
// CHECK-SAME:    %[[F:[a-zA-Z0-9]+]]: tensor<8x3x3x4xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: tensor<1x14x14x8xf32>
//
// Input pack: NHWC → [N, IC/c0, H, W, c0]  (NCHWc layout, same as HWCF variant)
// CHECK:         %[[PACK_IN:.+]] = linalg.pack %[[IN]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x16x16x4xf32> -> tensor<1x1x16x16x16xf32>
//
// Filter pack: FHWC → [OC/k0, IC/c0, FH, FW, c0, k0]  (XNNPACK convention, tile dims 0 and 3)
// CHECK:         %[[PACK_F:.+]] = linalg.pack %[[F]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [0, 3]
// CHECK-SAME:      inner_tiles = [16, 16]
// CHECK-SAME:      : tensor<8x3x3x4xf32> -> tensor<1x1x3x3x16x16xf32>
//
// Output pack: NHWC → [N, OC/k0, OH, OW, k0]  (NCHWc layout, same as HWCF variant)
// CHECK:         %[[PACK_OUT:.+]] = linalg.pack %[[OUT]]
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x14x14x8xf32> -> tensor<1x1x14x14x16xf32>
//
// Output unpack: [N, OC/k0, OH, OW, k0] → NHWC
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %{{[^ ]+}}
// CHECK-SAME:      outer_dims_perm = [0, 3, 1, 2]
// CHECK-SAME:      inner_dims_pos = [3]
// CHECK-SAME:      inner_tiles = [16]
// CHECK-SAME:      : tensor<1x1x14x14x16xf32> -> tensor<1x14x14x8xf32>
