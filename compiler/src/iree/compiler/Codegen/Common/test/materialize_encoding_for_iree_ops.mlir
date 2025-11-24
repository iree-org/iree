// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s

//----------------------------------------------------------------------------//
// Test suite using generic encoding resolvers, that are defined in Encoding
// dialect.
//----------------------------------------------------------------------------//

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
func.func @matmul_lowering_f32f32f32_identity_resolver_full_slices() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "whatever", {iree.encoding.resolver = #iree_encoding.identity_resolver<>}>
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
// CHECK-LABEL: func @matmul_lowering_f32f32f32_identity_resolver_full_slices()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(2)
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[M]], %[[K]]}
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[K]], %[[N]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>{%[[M]], %[[N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[K]]], strides = [1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[K]], %[[N]]], strides = [1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[N]]], strides = [1, 1]
//       CHECK:   %[[RES:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[RES]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[N]]], strides = [1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 12, bindings = [
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
func.func @matmul_lowering_f32f32f32_identity_resolver_partial_slices() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "whatever", {iree.encoding.resolver = #iree_encoding.identity_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %sizeM = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
  %sizeN = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : index
  %sizeK = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : index
  %offsetM = hal.interface.constant.load layout(#pipeline_layout) ordinal(6) : index
  %offsetN = hal.interface.constant.load layout(#pipeline_layout) ordinal(7) : index
  %offsetK = hal.interface.constant.load layout(#pipeline_layout) ordinal(8) : index
  %strideM = hal.interface.constant.load layout(#pipeline_layout) ordinal(9) : index
  %strideN = hal.interface.constant.load layout(#pipeline_layout) ordinal(10) : index
  %strideK = hal.interface.constant.load layout(#pipeline_layout) ordinal(11) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%offsetM, %offsetK], sizes = [%sizeM, %sizeK], strides = [%strideM, %strideK]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [%offsetK, %offsetN], sizes = [%sizeK, %sizeN], strides = [%strideK, %strideN]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf32, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [%offsetM, %offsetN], sizes = [%sizeM, %sizeN], strides = [%strideM, %strideN]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [%offsetM, %offsetN], sizes = [%sizeM, %sizeN], strides = [%strideM, %strideN]
      : tensor<?x?xf32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  return

}
// CHECK-LABEL: func @matmul_lowering_f32f32f32_identity_resolver_partial_slices()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(2)
//   CHECK-DAG:   %[[SIZE_M:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(3)
//   CHECK-DAG:   %[[SIZE_N:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(4)
//   CHECK-DAG:   %[[SIZE_K:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(5)
//   CHECK-DAG:   %[[OFFSET_M:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(6)
//   CHECK-DAG:   %[[OFFSET_N:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(7)
//   CHECK-DAG:   %[[OFFSET_K:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(8)
//   CHECK-DAG:   %[[STRIDE_M:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(9)
//   CHECK-DAG:   %[[STRIDE_N:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(10)
//   CHECK-DAG:   %[[STRIDE_K:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(11)
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[M]], %[[K]]}
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[K]], %[[N]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>{%[[M]], %[[N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:     offsets = [%[[OFFSET_M]], %[[OFFSET_K]]], sizes = [%[[SIZE_M]], %[[SIZE_K]]], strides = [%[[STRIDE_M]], %[[STRIDE_K]]]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:     offsets = [%[[OFFSET_K]], %[[OFFSET_N]]], sizes = [%[[SIZE_K]], %[[SIZE_N]]], strides = [%[[STRIDE_K]], %[[STRIDE_N]]]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:     offsets = [%[[OFFSET_M]], %[[OFFSET_N]]], sizes = [%[[SIZE_M]], %[[SIZE_N]]], strides = [%[[STRIDE_M]], %[[STRIDE_N]]]
//       CHECK:   %[[RES:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[RES]], %[[OUTS_BINDING]]
//  CHECK-SAME:     offsets = [%[[OFFSET_M]], %[[OFFSET_N]]], sizes = [%[[SIZE_M]], %[[SIZE_N]]], strides = [%[[STRIDE_M]], %[[STRIDE_N]]]

// -----

//----------------------------------------------------------------------------//
// Test suite using CPU encoding resolvers.
//----------------------------------------------------------------------------//

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
func.func @matmul_lowering_f32f32f32_x86_64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
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
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
// CHECK-LABEL: func @matmul_lowering_f32f32f32_x86_64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP1]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x4x1xf32>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x4xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 4, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 4], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 4], strides = [1, 1, 1, 1]

// -----

// VMVX path with ukernel enabled generates iree_codegen.query_tile_sizes for
// dynamic inner tiles. The test exercises that the information is properly used
// by the bindings.

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_vmvx_ukernel() attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all", iree.encoding.resolver = #iree_cpu.vmvx_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP_CEILDIV:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: #[[LHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iteration_sizes = [?, ?, ?]>
//      CHECK: #[[RHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1 : i64, op_type =  matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iteration_sizes = [?, ?, ?]>
//      CHECK: #[[OUT_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 2 : i64, op_type =  matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iteration_sizes = [?, ?, ?]>
//      CHECK: func @matmul_lowering_i8i8i32_vmvx_ukernel()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//      CHECK:   %[[LHS_TILE_SIZES:.+]]:2 = iree_codegen.query_tile_sizes tensor<?x?xi8, #[[LHS_ENCODING]]> -> index, index
//  CHECK-DAG:   %[[LHS_OUTER_SIZE0:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[M]], %[[LHS_TILE_SIZES]]#0]
//  CHECK-DAG:   %[[LHS_OUTER_SIZE1:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[K]], %[[LHS_TILE_SIZES]]#1]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%[[LHS_OUTER_SIZE0]], %[[LHS_OUTER_SIZE1]], %[[LHS_TILE_SIZES]]#0, %[[LHS_TILE_SIZES]]#1}
//      CHECK:   %[[RHS_TILE_SIZES:.+]]:2 = iree_codegen.query_tile_sizes tensor<?x?xi8, #[[RHS_ENCODING]]> -> index, index
//  CHECK-DAG:   %[[RHS_OUTER_SIZE0:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[N]], %[[RHS_TILE_SIZES]]#0]
//  CHECK-DAG:   %[[RHS_OUTER_SIZE1:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[K]], %[[RHS_TILE_SIZES]]#1]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%[[RHS_OUTER_SIZE0]], %[[RHS_OUTER_SIZE1]], %[[RHS_TILE_SIZES]]#0, %[[RHS_TILE_SIZES]]#1}
//      CHECK:   %[[RESULT_TILE_SIZES:.+]]:2 = iree_codegen.query_tile_sizes tensor<?x?xi32, #[[OUT_ENCODING]]> -> index, index
//  CHECK-DAG:   %[[RESULT_OUTER_SIZE0:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[M]], %[[RESULT_TILE_SIZES]]#0]
//  CHECK-DAG:   %[[RESULT_OUTER_SIZE1:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[N]], %[[RESULT_TILE_SIZES]]#1]
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
// CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%[[RESULT_OUTER_SIZE0]], %[[RESULT_OUTER_SIZE1]], %[[RESULT_TILE_SIZES]]#0, %[[RESULT_TILE_SIZES]]#1}
//      CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[LHS_OUTER_SIZE0]], %[[LHS_OUTER_SIZE1]], %[[LHS_TILE_SIZES]]#0, %[[LHS_TILE_SIZES]]#1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[RHS_OUTER_SIZE0]], %[[RHS_OUTER_SIZE1]], %[[RHS_TILE_SIZES]]#0, %[[RHS_TILE_SIZES]]#1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[RESULT_OUTER_SIZE0]], %[[RESULT_OUTER_SIZE1]], %[[RESULT_TILE_SIZES]]#0, %[[RESULT_TILE_SIZES]]#1], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[RESULT_OUTER_SIZE0]], %[[RESULT_OUTER_SIZE1]], %[[RESULT_TILE_SIZES]]#0, %[[RESULT_TILE_SIZES]]#1], strides = [1, 1, 1, 1]

// -----

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
func.func @matmul_lowering_f32f32f32_vmvx_generic() attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_cpu.vmvx_encoding_resolver<>}>
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
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @matmul_lowering_f32f32f32_vmvx_generic()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x4xf32>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x4xf32>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
// CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

//----------------------------------------------------------------------------//
// Test suite using GPU encoding resolvers.
//----------------------------------------------------------------------------//

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {
    abi = "hip",
    iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>,
    iree_codegen.target_info = #iree_gpu.target<arch = "gfx942",
                                       features = "",
                                       wgp = <compute = fp32,
                                              storage =  b32,
                                              subgroup =  none,
                                              mma = [<MFMA_F32_16x16x4_F32>],
                                              subgroup_size_choices = [64],
                                              max_workgroup_sizes = [1024, 1024, 1024],
                                              max_thread_count_per_workgroup = 1024,
                                              max_workgroup_memory_bytes = 65536,
                                              max_workgroup_counts = [2147483647, 2147483647, 2147483647],
                                              max_load_instruction_bits = 128,
                                              simds_per_wgp = 4,
                                              vgpr_space_bits = 16384>>
  }>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_gfx942() attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
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
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 128)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//   CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//   CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @matmul_lowering_f32f32f32_gfx942()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2x4x4x16x4xf32>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x2x4x4x16x4xf32>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x2x2x4x4x4x16x4xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 2, 4, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 2, 4, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0, 0, 0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 2, 2, 4, 4, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
//       CHECK:   %[[GEMM:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]])
//  CHECK-SAME:       outs(%[[OUTS]])
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[GEMM]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0, 0, 0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 2, 2, 4, 4, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]

// -----

// This tests that the padding resolver can handle partial loads/stores. The
// offsets, sizes and strides are arbitrarily chosen in the test.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {
    abi = "hip",
    iree.encoding.resolver = #iree_gpu.gpu_padding_resolver<>,
    iree_codegen.target_info = #iree_gpu.target<arch = "gfx942",
                                       features = "",
                                       wgp = <compute = fp32,
                                              storage =  b32,
                                              subgroup =  none,
                                              mma = [<MFMA_F32_16x16x4_F32>],
                                              subgroup_size_choices = [64],
                                              max_workgroup_sizes = [1024, 1024, 1024],
                                              max_thread_count_per_workgroup = 1024,
                                              max_workgroup_memory_bytes = 65536,
                                              max_workgroup_counts = [2147483647, 2147483647, 2147483647],
                                              max_load_instruction_bits = 128,
                                              simds_per_wgp = 4,
                                              vgpr_space_bits = 16384>>
  }>

#binding_ro = #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
#binding = #hal.pipeline.binding<storage_buffer, Indirect>
#encoding_mmt = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f16, f16, f16]>
#pad_encoding = #iree_encoding.layout<[#iree_encoding.padding<[0, 64]>]>
func.func @set_pad_encoding_and_partial_load_store()  attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %3 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) binding(0) alignment(64) offset(%1) flags("ReadOnly|Indirect")
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>>
  %4 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect)
    : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16, #pad_encoding>>
  %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1024, 1024], strides = [2, 2]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>> -> tensor<1024x1024xf16>
  %6 = iree_encoding.set_encoding %5 : tensor<1024x1024xf16> -> tensor<1024x1024xf16, #encoding_mmt>
  iree_tensor_ext.dispatch.tensor.store %6, %4, offsets = [0, 0], sizes = [1024, 1024], strides = [2, 2]
    : tensor<1024x1024xf16, #encoding_mmt> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16, #pad_encoding>>
  return
}
// CHECK-LABEL: @set_pad_encoding_and_partial_load_store
// CHECK:         %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>>
// CHECK:         %[[B:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2112xf16>>
// CHECK:         %[[LD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[A]], offsets = [0, 0], sizes = [1024, 1024], strides = [2, 2]
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>> -> tensor<1024x1024xf16>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[LD]], %[[B]], offsets = [0, 0], sizes = [1024, 1024], strides = [2, 2]
// CHECK-SAME:                  tensor<1024x1024xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2112xf16>>

// -----

//----------------------------------------------------------------------------//
// Test suite for encodings with resolved layouts.
// All the implementations use interfaces, so we only check with CPU encoding
// resolvers.
//----------------------------------------------------------------------------//

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
#encoding = #iree_encoding.layout<[#iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [1, 1], outerDimsPerm = [0, 1]}}>]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding1 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, 256, ?]>
func.func @set_encoding_LHS_with_layout() attributes {
  hal.executable.target = #executable_target
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x256xf32, #encoding>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xf32>> -> tensor<1x256xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<1x256xf32> -> tensor<1x256xf32, #encoding1>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : tensor<1x256xf32, #encoding1> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x256xf32, #encoding>>
  return
}
// CHECK-LABEL: func.func @set_encoding_LHS_with_layout
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x256x1x1xf32>>
//       CHECK:   %[[INPUT:.+]] = iree_tensor_ext.dispatch.tensor.load %[[INPUT_BINDING]]
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[INPUT]]
//  CHECK-SAME:     outer_dims_perm = [0, 1]
//  CHECK-SAME:     inner_dims_pos = [0, 1]
//  CHECK-SAME:     inner_tiles = [1, 1]
//  CHECK-SAME:     tensor<1x256xf32> -> tensor<1x256x1x1xf32>
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[PACK]], %[[RESULT_BINDING]]

// -----

#encoding = #iree_encoding.layout<[#iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [1, 0], innerTileSizes = [16, 1], outerDimsPerm = [1, 0]}}>]>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, 256, ?]>
func.func @set_encoding_RHS_with_layout() attributes {
  hal.executable.target = #executable_target
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x10xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x10xf32, #encoding>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 10], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x10xf32>> -> tensor<256x10xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<256x10xf32> -> tensor<256x10xf32, #encoding1>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [256, 10], strides = [1, 1] : tensor<256x10xf32, #encoding1> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x10xf32, #encoding>>
  return
}
// CHECK-LABEL: func.func @set_encoding_RHS_with_layout
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x10xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x256x16x1xf32>>
//   CHECK-DAG:   %[[PAD_VALUE:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[INPUT:.+]] = iree_tensor_ext.dispatch.tensor.load %[[INPUT_BINDING]]
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[INPUT]]
//  CHECK-SAME:     padding_value(%[[PAD_VALUE]] : f32)
//  CHECK-SAME:     outer_dims_perm = [1, 0]
//  CHECK-SAME:     inner_dims_pos = [1, 0]
//  CHECK-SAME:     inner_tiles = [16, 1]
//  CHECK-SAME:     tensor<256x10xf32> -> tensor<1x256x16x1xf32>
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[PACK]], %[[RESULT_BINDING]]

// -----

#encoding = #iree_encoding.layout<[#iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [1, 16], outerDimsPerm = [0, 1]}}>]>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", target_triple = "x86_64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#encoding1 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, 10, ?]>
func.func @unset_encoding_RES_with_layout() attributes {
  hal.executable.target = #executable_target
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x10xf32, #encoding>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x10xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 10], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x10xf32, #encoding>> -> tensor<1x10xf32, #encoding1>
  %3 = iree_encoding.unset_encoding %2 : tensor<1x10xf32, #encoding1> -> tensor<1x10xf32>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1, 10], strides = [1, 1] : tensor<1x10xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x10xf32>>
  return
}
// CHECK-LABEL: func.func @unset_encoding_RES_with_layout
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x1x16xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x10xf32>>
//       CHECK:   %[[INPUT:.+]] = iree_tensor_ext.dispatch.tensor.load %[[INPUT_BINDING]]
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[INPUT]]
//  CHECK-SAME:     outer_dims_perm = [0, 1]
//  CHECK-SAME:     inner_dims_pos = [0, 1]
//  CHECK-SAME:     inner_tiles = [1, 16]
//  CHECK-SAME:     tensor<1x1x1x16xf32> -> tensor<1x10xf32>
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[UNPACK]], %[[RESULT_BINDING]]

// -----

// FoldIntoPackUnpack patterns sometimes inhibit fusion, we can pass a controlFn that blocks the application of folding patterns that would
// block fusion in subsequent passes. This test is actually target-independent.
#encoding = #iree_encoding.layout<[#iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [3], innerTileSizes = [32], outerDimsPerm = [0, 2, 1, 3]}}>]>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", target_triple = "x86_64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1, d3)>
func.func @set_encoding_transpose_multi_result() attributes {
  hal.executable.target = #executable_target
} {
    %c0 = arith.constant  0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<56x57x1x64xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x56x57x64xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x56x57x64xf32, #encoding>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [56, 57, 1, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<56x57x1x64xf32>> -> tensor<56x57x1x64xf32>
    %4 = tensor.empty() : tensor<1x56x57x64xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<56x57x1x64xf32>) outs(%4 : tensor<1x56x57x64xf32>) {
    ^bb0(%in: f32 , %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x56x57x64xf32>
    %6 = iree_encoding.set_encoding %5 : tensor<1x56x57x64xf32> -> tensor<1x56x57x64xf32, #encoding>
    iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0, 0, 0], sizes = [1, 56, 57, 64], strides = [1, 1, 1, 1] : tensor<1x56x57x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x56x57x64xf32>>
    iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [1, 56, 57, 64], strides = [1, 1, 1, 1] : tensor<1x56x57x64xf32, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x56x57x64xf32, #encoding>>
    return
}
// CHECK-LABEL: func.func @set_encoding_transpose_multi_result
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<56x57x1x64xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x56x57x64xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING1:.+]] = hal.interface.binding.subspan {{.*}} binding(2) {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x57x56x2x32xf32>>
//       CHECK:   %[[INPUT:.+]] = iree_tensor_ext.dispatch.tensor.load %[[INPUT_BINDING]]
//       CHECK:   %[[TRANSPOSE:.+]] = linalg.generic {{.*}} ins(%[[INPUT]] : tensor<56x57x1x64xf32>)
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[TRANSPOSE]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[TRANSPOSE]], %[[RESULT_BINDING]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[PACK]], %[[RESULT_BINDING1]]

// -----

//----------------------------------------------------------------------------//
// Test suite for layout transfer with mismatched encodings.
//----------------------------------------------------------------------------//

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @load_from_plain_src_produce_encoded_tensor(%arg0: index, %arg1: index) -> tensor<?x?xf32, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg0, %arg1}
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%arg0, %arg1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg0, %arg1} -> tensor<?x?xf32, #encoding>
  return %1 : tensor<?x?xf32, #encoding>
}
// CHECK-LABEL: func.func @load_from_plain_src_produce_encoded_tensor
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[ARG0]], %[[ARG1]]}
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING]]
//  CHECK-SAME:       -> tensor<?x?xf32>
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[LOAD]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 1]
//       CHECK:   return %[[PACK]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @load_from_encoded_src_produce_plain_tensor(%arg0: index, %arg1: index) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%arg0, %arg1}
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%arg0, %arg1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%arg0, %arg1} -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func.func @load_from_encoded_src_produce_plain_tensor
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[TILED:.+]] = affine.apply #[[$MAP]]()[%[[ARG0]]]
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%[[TILED]], %[[ARG1]]}
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING]]
//  CHECK-SAME:       -> tensor<?x?x16x1xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[LOAD]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 1]
//       CHECK:   return %[[UNPACK]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, ?, ?]>
#encoding1 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @load_with_layout_transfer(%arg0: index, %arg1: index) -> tensor<?x?xf32, #encoding1> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%arg0, %arg1}
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%arg0, %arg1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%arg0, %arg1} -> tensor<?x?xf32, #encoding1>
  return %1 : tensor<?x?xf32, #encoding1>
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func.func @load_with_layout_transfer
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x1x1xf32>>{%[[ARG0]], %[[ARG1]]}
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING]]
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[LOAD]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [1, 1]
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP]]()[%[[ARG0]]]
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[UNPACK]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 1]
//       CHECK:   return %[[PACK]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, ?, ?]>
#encoding1 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @load_with_layout_transfer_partial_dynamic(%arg0: index) -> tensor<1024x?xf32, #encoding1> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x?xf32, #encoding>>{%arg0}
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, %arg0], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x?xf32, #encoding>>{%arg0} -> tensor<1024x?xf32, #encoding1>
  return %1 : tensor<1024x?xf32, #encoding1>
}
// CHECK-LABEL: func.func @load_with_layout_transfer_partial_dynamic
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x?x1x1xf32>>{%[[ARG0]]}
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING]]
//  CHECK-SAME:       -> tensor<1024x?x1x1xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[LOAD]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [1, 1]
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[UNPACK]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 1]
//       CHECK:   return %[[PACK]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_result_DxDxD = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @store_plain_tensor_to_encoded_dest(%src: tensor<?x?xf32>, %M: index, %N: index) attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding_result_DxDxD>>{%M, %N}
  iree_tensor_ext.dispatch.tensor.store %src, %0, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding_result_DxDxD>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func.func @store_plain_tensor_to_encoded_dest
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[M:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[N:[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_N:.+]] = affine.apply #[[$MAP]]()[%[[N]]]
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x16x16xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[SRC]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[PACK]], %[[BINDING]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_result_DxDxD = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @store_encoded_tensor_to_plain_dest(%src: tensor<?x?xf32, #encoding_result_DxDxD>, %M: index, %N: index) attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%M, %N}
  iree_tensor_ext.dispatch.tensor.store %src, %0, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result_DxDxD> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%M, %N}
  return
}
// CHECK-LABEL: func.func @store_encoded_tensor_to_plain_dest
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[M:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[N:[a-zA-Z0-9]+]]
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[M]], %[[N]]}
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[SRC]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[UNPACK]], %[[BINDING]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_result_1xDxD = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, ?, ?]>
#encoding_result_DxDxD = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @store_with_layout_transfer(%src: tensor<?x?xf32, #encoding_result_DxDxD>, %M: index, %N: index) attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding_result_1xDxD>>{%M, %N}
  iree_tensor_ext.dispatch.tensor.store %src, %0, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result_DxDxD> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding_result_1xDxD>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func.func @store_with_layout_transfer
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[M:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[N:[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[TILED_N:.+]] = affine.apply #[[$MAP]]()[%[[N]]]
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x1x16xf32>>{%[[M]], %[[TILED_N]]}
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[SRC]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16]
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[UNPACK]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [1, 16]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[PACK]], %[[BINDING]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_result_1xDxD = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, ?, ?]>
#encoding_result_DxDxD = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1024, ?, ?]>
func.func @store_with_layout_transfer_partial_dynamic(%src: tensor<1024x?xf32, #encoding_result_DxDxD>, %N: index) attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x?xf32, #encoding_result_1xDxD>>{%N}
  iree_tensor_ext.dispatch.tensor.store %src, %0, offsets = [0, 0], sizes = [1024, %N], strides = [1, 1]
      : tensor<1024x?xf32, #encoding_result_DxDxD> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x?xf32, #encoding_result_1xDxD>>{%N}
  return
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func.func @store_with_layout_transfer_partial_dynamic
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[N:[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[TILED_N:.+]] = affine.apply #[[$MAP]]()[%[[N]]]
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan {{.*}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x?x1x16xf32>>{%[[TILED_N]]}
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[SRC]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16]
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[UNPACK]]
//  CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [1, 16]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[PACK]], %[[BINDING]]

// -----

func.func @do_not_crash_on_non_ranked_tensor_type() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(0) alignment(32) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:f32>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>) binding(1) alignment(32) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:f32>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:f32> -> tensor<f32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !iree_tensor_ext.dispatch.tensor<writeonly:f32>
  return
}
// CHECK-LABEL: func.func @do_not_crash_on_non_ranked_tensor_type

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d0, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @do_not_crash_on_non_encoded_tensors(%offset0: index, %offset1: index, %size: index) attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
} {
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%offset0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x32x4x32xf16>>{%size}
  %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%offset1) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x32x?x32xf16>>{%size}
  %10 = tensor.empty(%size) : tensor<4x2x32x?x32xf16>
  %11 = iree_tensor_ext.dispatch.tensor.load %8, offsets = [0, 0, 0, 0, 0], sizes = [1, %size, 32, 4, 32], strides = [1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x32x4x32xf16>>{%size} -> tensor<?x32x4x32xf16>
  %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<?x32x4x32xf16>) outs(%10 : tensor<4x2x32x?x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x2x32x?x32xf16>
  iree_tensor_ext.dispatch.tensor.store %12, %9, offsets = [0, 0, 0, 0, 0], sizes = [4, 2, 32, %size, 32], strides = [1, 1, 1, 1, 1] : tensor<4x2x32x?x32xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x32x?x32xf16>>{%size}
  return
}
// CHECK-LABEL: func.func @do_not_crash_on_non_encoded_tensors
