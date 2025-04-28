// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [16, 1, 16]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [16, 1, 16]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [16, 1, 16]>
func.func @matvec_shaped_matmul_lowering_f32f32f32_aarch64(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<16x16xf32>
  %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<16x1xf32>
  %2 = hal.tensor.import %arg2 "input2" : !hal.buffer_view -> tensor<16x1xf32>
  %3 = iree_encoding.set_encoding %0 : tensor<16x16xf32> -> tensor<16x16xf32, #encoding_lhs>
  %4 = iree_encoding.set_encoding %1 : tensor<16x1xf32> -> tensor<16x1xf32, #encoding_rhs>
  %5 = iree_encoding.set_encoding %2 : tensor<16x1xf32> -> tensor<16x1xf32, #encoding_result>
  %6 = linalg.matmul ins(%3, %4 : tensor<16x16xf32, #encoding_lhs>, tensor<16x1xf32, #encoding_rhs>) outs(%5 : tensor<16x1xf32, #encoding_result>) -> tensor<16x1xf32, #encoding_result>
  %7 = iree_encoding.unset_encoding %6 : tensor<16x1xf32, #encoding_result> -> tensor<16x1xf32>
  %8 = hal.tensor.export %7 "output0" : tensor<16x1xf32> -> !hal.buffer_view
  func.return %8 : !hal.buffer_view
}
// CHECK-LABEL: func @matvec_shaped_matmul_lowering_f32f32f32_aarch64(
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins({{.*}} : tensor<1x16x1x1xf32>, tensor<2x16x8x1xf32>)
//  CHECK-SAME:        outs({{.*}} : tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>

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
func.func @matmul_lowering_f32f32f32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
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
// CHECK-LABEL: func @matmul_lowering_f32f32f32_aarch64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iteration_sizes = [16, 16]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iteration_sizes= [16, 16]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iteration_sizes = [16, 16]>
func.func @matvec_lowering_f32f32f32_aarch64(%arg0: tensor<16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>) -> tensor<16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
} {
  %c0 = arith.constant 0 : index
  %3 = iree_encoding.set_encoding %arg0 : tensor<16x16xf32> -> tensor<16x16xf32, #encoding_lhs>
  %4 = iree_encoding.set_encoding %arg1 : tensor<16xf32> -> tensor<16xf32, #encoding_rhs>
  %5 = iree_encoding.set_encoding %arg2 : tensor<16xf32> -> tensor<16xf32, #encoding_result>
  %6 = linalg.matvec ins(%3, %4 : tensor<16x16xf32, #encoding_lhs>, tensor<16xf32, #encoding_rhs>) outs(%5 : tensor<16xf32, #encoding_result>) -> tensor<16xf32, #encoding_result>
  %7 = iree_encoding.unset_encoding %6 : tensor<16xf32, #encoding_result> -> tensor<16xf32>
  func.return %7 : tensor<16xf32>
}
// CHECK-LABEL: func @matvec_lowering_f32f32f32_aarch64(
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins({{.*}} : tensor<1x16x1x1xf32>, tensor<2x16x8x1xf32>)
//  CHECK-SAME:        outs({{.*}} : tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 1, 16]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 1, 16]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 1, 16]>
func.func @matvec_lowering_f32f32f32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16xf32, #encoding_lhs>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x1xf32, #encoding_rhs>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x1xf32, #encoding_result>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16xf32, #encoding_lhs>>
      -> tensor<16x16xf32, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x1xf32, #encoding_rhs>>
      -> tensor<16x1xf32, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x1xf32, #encoding_result>>
      -> tensor<16x1xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<16x16xf32, #encoding_lhs>,
                   tensor<16x1xf32, #encoding_rhs>)
      outs(%5 : tensor<16x1xf32, #encoding_result>)
      -> tensor<16x1xf32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : tensor<16x1xf32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x1xf32, #encoding_result>>
  return
}
// CHECK-LABEL: func @matvec_lowering_f32f32f32_aarch64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x16x8x1xf32>>
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x1x1xf32>>
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x2x1x8xf32>>
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [2, 16, 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 16, 1, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 2, 1, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[RHS]], %[[LHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 2, 1, 8], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f16f16f16_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf16, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf16, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
      -> tensor<?x?xf16, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf16, #encoding_result>)
      -> tensor<?x?xf16, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-LABEL: func @matmul_lowering_f16f16f16_aarch64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xf16>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f16f16_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  %lhs_f32 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %rhs = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf16, #encoding_rhs>
  %dest = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
      -> tensor<?x?xf16, #encoding_result>

  %empty = tensor.empty(%M, %K) : tensor<?x?xf16, #encoding_lhs>
  %lhs_f16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
     ins(%lhs_f32 : tensor<?x?xf32, #encoding_lhs>)
     outs(%empty : tensor<?x?xf16, #encoding_lhs>) {
  ^bb0(%in: f32, %out: f16):
    %17 = arith.truncf %in : f32 to f16
    linalg.yield %17 : f16
  } -> tensor<?x?xf16, #encoding_lhs>
  %6 = linalg.matmul
      ins(%lhs_f16, %rhs : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%dest : tensor<?x?xf16, #encoding_result>)
      -> tensor<?x?xf16, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP_CEILDIV_8:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[$MAP_IDENTITY_4D:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @matmul_lowering_f32f16f16_aarch64()
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0) : index
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1) : index
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2) : index
//   CHECK-DAG:   %[[M_CEILDIV_8:.+]] = affine.apply #[[$MAP_CEILDIV_8]]()[%[[M]]]
//   CHECK-DAG:   %[[N_CEILDIV_8:.+]] = affine.apply #[[$MAP_CEILDIV_8]]()[%[[N]]]
//   CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[M_CEILDIV_8]], %[[K]]}
//   CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) {{.*}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[N_CEILDIV_8]], %[[K]]}
//   CHECK-DAG:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2) {{.*}} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xf16>>{%[[M_CEILDIV_8]], %[[N_CEILDIV_8]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M_CEILDIV_8]], %[[K]], 8, 1], {{.*}} -> tensor<?x?x8x1xf32>
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[N_CEILDIV_8]], %[[K]], 8, 1], {{.*}} -> tensor<?x?x8x1xf16>
//       CHECK:   %[[OUT:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUT_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M_CEILDIV_8]], %[[N_CEILDIV_8]], 8, 8], {{.*}} -> tensor<?x?x8x8xf16>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[M_CEILDIV_8]], %[[K]]) : tensor<?x?x8x1xf16>
//       CHECK:   %[[LHS_F16:.+]] = linalg.generic {indexing_maps = [#[[$MAP_IDENTITY_4D]], #[[$MAP_IDENTITY_4D]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS]] : tensor<?x?x8x1xf32>) outs(%[[EMPTY]] : tensor<?x?x8x1xf16>) {
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS_F16]], %[[RHS]] : tensor<?x?x8x1xf16>, tensor<?x?x8x1xf16>) outs(%[[OUT]] : tensor<?x?x8x8xf16>)
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUT_BINDING]],

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
func.func @matmul_lowering_i8i8i32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
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
// CHECK-LABEL: func @matmul_lowering_i8i8i32_aarch64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8>>{%[[M]], %[[K]]}
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8>>{%[[K]], %[[N]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32>>{%[[M]], %[[N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[K]]], strides = [1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[K]], %[[N]]], strides = [1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[N]]], strides = [1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[N]]], strides = [1, 1]

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
func.func @matmul_lowering_i8i8i32_aarch64_dotprod() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
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
// CHECK-LABEL: func @matmul_lowering_i8i8i32_aarch64_dotprod()
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
func.func @matmul_lowering_i8i8i32_aarch64_i8mm() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+i8mm", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
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
// CHECK-LABEL: func @matmul_lowering_i8i8i32_aarch64_i8mm()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP0]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i4i32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi4, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi4, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func @matmul_lowering_i8i4i32_aarch64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x4x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP2]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x2xi4>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x4x16xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 4, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 16], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i4i32_aarch64_dotprod() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi4, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi4, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-LABEL: func @matmul_lowering_i8i4i32_aarch64_dotprod()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP0]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x8xi4>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i4i32_aarch64_i8mm() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+i8mm", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi4, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi4, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-LABEL: func @matmul_lowering_i8i4i32_aarch64_i8mm()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x4x16xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP2]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x16xi4>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x4x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 4, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_aarch64_sve(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+sve", target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>}>
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

// Targets that has "sve" features does not implement data-tiling yet.
// CHECK-LABEL: func @matmul_lowering_f32f32f32_aarch64_sve
//       CHECK:   %[[RES:.+]] = linalg.matmul
//       CHECK:   return %[[RES]]
