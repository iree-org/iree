// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_riscv(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv32-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %acc, %c0 : tensor<?x?xf32>
  %N = tensor.dim %acc, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %lhs : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %rhs : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_rhs>
  %2 = iree_encoding.set_encoding %acc : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_result>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%2 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  %4 = iree_encoding.unset_encoding %3 : tensor<?x?xf32, #encoding_result> -> tensor<?x?xf32>{%M, %N}
  return %4 : tensor<?x?xf32>
}
// RISC-V targets does not implement data-tiling yet.
// CHECK-LABEL: func @matmul_lowering_f32f32f32_riscv
//       CHECK:   %[[RES:.+]] = linalg.matmul
//       CHECK:   return %[[RES]]

// -----

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
func.func @matmul_lowering_i8i8i32_riscv32_ukernel() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv32-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
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
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
// CHECK-LABEL: func @matmul_lowering_i8i8i32_riscv32_ukernel()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
