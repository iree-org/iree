// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, 1000, ?]>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
func.func @set_encoding_with_padding_semantics_bf16_x86_64_avx512f(%arg0: tensor<1x1000xbf16>)
    -> tensor<1x1000xbf16, #encoding> attributes { hal.executable.target = #executable_target } {
  %0 = iree_encoding.set_encoding %arg0 : tensor<1x1000xbf16> -> tensor<1x1000xbf16, #encoding>
  return %0 : tensor<1x1000xbf16, #encoding>
}
// This tests that
//   1. The padding value is created for linalg.pack ops.
//   2. The inner tile sizes are less than or equal to values in iteration_sizes.
//      We could choose 128 when it is a narrow matrix.
// CHECK-LABEL: func.func @set_encoding_with_padding_semantics_bf16_x86_64_avx512f
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[INIT:.+]] = tensor.empty() : tensor<1x1000x1x1xbf16>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[SRC]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [1, 1]
// CHECK-SAME:      into %[[INIT]] : tensor<1x1000xbf16> -> tensor<1x1000x1x1xbf16>
// CHECK:         return %[[PACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [7, 7, 7]>
func.func @set_encoding_7x7x7_matmul_LHS(%14: tensor<7x7xf32>) -> tensor<7x7xf32, #encoding> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %17 = iree_encoding.set_encoding %14 : tensor<7x7xf32> -> tensor<7x7xf32, #encoding>
  return %17 : tensor<7x7xf32, #encoding>
}
// CHECK-LABEL: func @set_encoding_7x7x7_matmul_LHS(
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<7x7xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.0
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x7x8x1xf32>
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[INPUT]] padding_value(%[[CST]] : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %[[EMPTY]] : tensor<7x7xf32> -> tensor<1x7x8x1xf32>
//       CHECK:   return %[[PACK]] : tensor<1x7x8x1xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 80, 32, ?]>
func.func @set_encoding_128x80x32_batch_matmul_LHS(%14: tensor<128x80x32xf32>) -> tensor<128x80x32xf32, #encoding> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %17 = iree_encoding.set_encoding %14 : tensor<128x80x32xf32> -> tensor<128x80x32xf32, #encoding>
  return %17 : tensor<128x80x32xf32, #encoding>
}
// CHECK-LABEL:    func @set_encoding_128x80x32_batch_matmul_LHS(
//  CHECK-SAME:      %[[INPUT:[a-zA-Z0-9]+]]: tensor<128x80x32xf32>
//   CHECK-DAG:      %[[EMPTY:.+]] = tensor.empty() : tensor<128x10x32x8x1xf32>
//       CHECK:      %[[PACK:.+]] = linalg.pack %[[INPUT]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 1] into %[[EMPTY]] : tensor<128x80x32xf32> -> tensor<128x10x32x8x1xf32>
//       CHECK:      return %[[PACK]] : tensor<128x10x32x8x1xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 32, 320, ?]>
func.func @set_encoding_128x32x320_batch_matmul_RHS(%16: tensor<128x32x320xf32>) -> tensor<128x32x320xf32, #encoding> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %19 = iree_encoding.set_encoding %16 : tensor<128x32x320xf32> -> tensor<128x32x320xf32, #encoding>
  return %19 : tensor<128x32x320xf32, #encoding>
}
// CHECK-LABEL:    func @set_encoding_128x32x320_batch_matmul_RHS(
//  CHECK-SAME:      %[[INPUT:[a-zA-Z0-9]+]]: tensor<128x32x320xf32>
//   CHECK-DAG:      %[[EMPTY:.+]] = tensor.empty() : tensor<128x40x32x8x1xf32>
//       CHECK:      %[[PACK:.+]] = linalg.pack %[[INPUT]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [8, 1] into %[[EMPTY]] : tensor<128x32x320xf32> -> tensor<128x40x32x8x1xf32>
//       CHECK:      return %[[PACK]] : tensor<128x40x32x8x1xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 80, 320, ?]>
func.func @unset_encoding_128x80x320_batch_matmul_RESULT(%arg0: tensor<128x80x320xf32, #encoding>) -> tensor<128x80x320xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<128x80x320xf32, #encoding> -> tensor<128x80x320xf32>
  return %0 : tensor<128x80x320xf32>
}
// CHECK-LABEL: func @unset_encoding_128x80x320_batch_matmul_RESULT(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<128x10x40x8x8xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x80x320xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[ARG0]]
//  CHECK-SAME:       outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %[[EMPTY]]
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @pack_gemm_fill_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #encoding_result>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #encoding_lhs>, tensor<?x?xf32, #encoding_rhs>)
      outs(%3 : tensor<?x?xf32, #encoding_result>) -> tensor<?x?xf32, #encoding_result>
  %5 = iree_encoding.unset_encoding %4 : tensor<?x?xf32, #encoding_result> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-LABEL: func @pack_gemm_fill_dynamic(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[OUT_D0:.+]] = affine.apply #[[$MAP0]]()[%[[D0]]]
//   CHECK-DAG:   %[[OUT_D1:.+]] = affine.apply #[[$MAP0]]()[%[[D1]]]
//   CHECK-DAG:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}%[[ARG0]]
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack
//  CHECK-SAME:     %[[ARG1]]
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[OUT_D0]], %[[OUT_D1]]) : tensor<?x?x8x8xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[EMPTY]] :
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[MMT4D]]
//       CHECK:   return %[[UNPACK]]

// -----

// It tests with bindings and checks that the reshape ops are folded into bindings.

#executable_target_xyz = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [15, 32]>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [15, 32]>
#encoding2 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [15, 32]>
func.func @matvec_f32_x86_64_generic() attributes {hal.executable.target = #executable_target_xyz} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c2048 = arith.constant 2048 : index
  %c2176 = arith.constant 2176 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<15x32xf32, #encoding>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c2048) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32xf32, #encoding1>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c2176) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<15xf32, #encoding2>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [15, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<15x32xf32, #encoding>> -> tensor<15x32xf32, #encoding>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32xf32, #encoding1>> -> tensor<32xf32, #encoding1>
  %5 = tensor.empty() : tensor<15xf32, #encoding2>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<15xf32, #encoding2>) -> tensor<15xf32, #encoding2>
  %7 = linalg.matvec ins(%3, %4 : tensor<15x32xf32, #encoding>, tensor<32xf32, #encoding1>) outs(%6 : tensor<15xf32, #encoding2>) -> tensor<15xf32, #encoding2>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0], sizes = [15], strides = [1] : tensor<15xf32, #encoding2> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<15xf32, #encoding2>>
  return
}
// CHECK-LABEL: func.func @matvec_f32_x86_64_generic()
// CHECK-NOT:     tensor.expand_shape
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x4x1x4xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins({{.+}}) outs(%[[INIT]]
// CHECK:         linalg.mmt4d
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x32x1x1xf32>, tensor<4x32x4x1xf32>
// CHECK-SAME:      outs(%[[FILL]]
// CHECK-NOT:     tensor.collapse_shape

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_x86_64(
  %3: tensor<?x?xf32, #encoding_lhs>,
  %4: tensor<?x?xf32, #encoding_rhs>,
  %5: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %6 : tensor<?x?xf32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_f32f32f32_x86_64(
//  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x1xf32>
//  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x4x1xf32>
//  CHECK-SAME:     %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_x86_64_avx2(
  %3: tensor<?x?xf32, #encoding_lhs>,
  %4: tensor<?x?xf32, #encoding_rhs>,
  %5: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %6 : tensor<?x?xf32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_f32f32f32_x86_64_avx2(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x1xf32>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x1xf32>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_x86_64_avx512f(
  %3: tensor<?x?xf32, #encoding_lhs>,
  %4: tensor<?x?xf32, #encoding_rhs>,
  %5: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %6 : tensor<?x?xf32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_f32f32f32_x86_64_avx512f(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf32>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf32>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f16f16f32_x86_64_avx512f(
  %3: tensor<?x?xf16, #encoding_lhs>,
  %4: tensor<?x?xf16, #encoding_rhs>,
  %5: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %6 : tensor<?x?xf32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_f16f16f32_x86_64_avx512f(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf16>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf16>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f16f16f16_x86_64_avx512f(
  %3: tensor<?x?xf16, #encoding_lhs>,
  %4: tensor<?x?xf16, #encoding_rhs>,
  %5: tensor<?x?xf16, #encoding_result>
) -> tensor<?x?xf16, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf16, #encoding_result>)
      -> tensor<?x?xf16, #encoding_result>
  return %6 : tensor<?x?xf16, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_f16f16f16_x86_64_avx512f(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf16>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf16>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf16>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_bf16bf16f32_x86_64_avx512f(
  %3: tensor<?x?xbf16, #encoding_lhs>,
  %4: tensor<?x?xbf16, #encoding_rhs>,
  %5: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #encoding_lhs>,
                   tensor<?x?xbf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %6 : tensor<?x?xf32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_bf16bf16f32_x86_64_avx512f(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xbf16>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xbf16>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_bf16bf16bf16_x86_64_avx512f(
  %3: tensor<?x?xbf16, #encoding_lhs>,
  %4: tensor<?x?xbf16, #encoding_rhs>,
  %5: tensor<?x?xbf16, #encoding_result>
) -> tensor<?x?xbf16, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #encoding_lhs>,
                   tensor<?x?xbf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xbf16, #encoding_result>)
      -> tensor<?x?xbf16, #encoding_result>
  return %6 : tensor<?x?xbf16, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_bf16bf16bf16_x86_64_avx512f(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xbf16>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xbf16>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x16x16xbf16>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_bf16bf16f32_x86_64_avx512bf16(
  %3: tensor<?x?xbf16, #encoding_lhs>,
  %4: tensor<?x?xbf16, #encoding_rhs>,
  %5: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f,+avx512bf16", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #encoding_lhs>,
                   tensor<?x?xbf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %6 : tensor<?x?xf32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_bf16bf16f32_x86_64_avx512bf16(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x16x2xbf16>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x2xbf16>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_bf16bf16bf16_x86_64_avx512bf16(
  %3: tensor<?x?xbf16, #encoding_lhs>,
  %4: tensor<?x?xbf16, #encoding_rhs>,
  %5: tensor<?x?xbf16, #encoding_result>
) -> tensor<?x?xbf16, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f,+avx512bf16", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #encoding_lhs>,
                   tensor<?x?xbf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xbf16, #encoding_result>)
      -> tensor<?x?xbf16, #encoding_result>
  return %6 : tensor<?x?xbf16, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_bf16bf16bf16_x86_64_avx512bf16(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x16x2xbf16>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x2xbf16>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x16x16xbf16>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f16f16_x86_64_avx512f(
  %M: index,
  %K: index,
  %lhs_f32: tensor<?x?xf32, #encoding_lhs>,
  %rhs: tensor<?x?xf16, #encoding_rhs>,
  %dest: tensor<?x?xf16, #encoding_result>
) -> tensor<?x?xf16, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f,+avx512bf16", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
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
  return %6 : tensor<?x?xf16, #encoding_result>
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @matmul_lowering_f32f16f16_x86_64_avx512f(
//  CHECK-SAME:   %[[M:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[K:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf32>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf16>
//  CHECK-SAME:   %[[OUT:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf16>
//       CHECK:   %[[M_CEILDIV_16:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[M_CEILDIV_16]], %[[K]]) : tensor<?x?x16x1xf16>
//       CHECK:   %[[LHS_F16:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS]] : tensor<?x?x16x1xf32>) outs(%[[EMPTY]] : tensor<?x?x16x1xf16>) {
//   CHECK-DAG:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS_F16]], %[[RHS]] : tensor<?x?x16x1xf16>, tensor<?x?x16x1xf16>) outs(%[[OUT]] : tensor<?x?x16x16xf16>)
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_x86_64_avx2(
  %3: tensor<?x?xi8, #encoding_lhs>,
  %4: tensor<?x?xi8, #encoding_rhs>,
  %5: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx2", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %6 : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_x86_64_avx2(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x2xi8>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x2xi8>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
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
func.func @matmul_lowering_i8i8i32_x86_64_avx512bw(
    %3: tensor<?x?xi8, #encoding_lhs>,
    %4: tensor<?x?xi8, #encoding_rhs>,
    %5: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512bw", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %6 : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_x86_64_avx512bw(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x16x2xi8>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x2xi8>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x16x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
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
func.func @matmul_lowering_i8i8i32_x86_64_avx512vnni(
    %3: tensor<?x?xi8, #encoding_lhs>,
    %4: tensor<?x?xi8, #encoding_rhs>,
    %5: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %6 : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_x86_64_avx512vnni(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x16x2xi8>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x2xi8>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x16x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 1, 11008, 128]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 1, 11008, 128]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 1, 11008, 128]>
func.func @extend_batch_vecmat_explicit_unit_dim(%arg0: tensor<32x1x128xi8>, %arg1: tensor<32x128x11008xi8>) -> tensor<32x1x11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0_i32 = arith.constant 0 : i32
  %4 = iree_encoding.set_encoding %arg0 : tensor<32x1x128xi8> -> tensor<32x1x128xi8, #encoding_lhs>
  %5 = tensor.empty() : tensor<32x1x128xi32, #encoding_lhs>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<32x1x128xi8, #encoding_lhs>) outs(%5 : tensor<32x1x128xi32, #encoding_lhs>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<32x1x128xi32, #encoding_lhs>
  %7 = iree_encoding.set_encoding %arg1 : tensor<32x128x11008xi8> -> tensor<32x128x11008xi8, #encoding_rhs>
  %8 = tensor.empty() : tensor<32x128x11008xi32, #encoding_rhs>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<32x128x11008xi8, #encoding_rhs>) outs(%8 : tensor<32x128x11008xi32, #encoding_rhs>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<32x128x11008xi32, #encoding_rhs>
  %10 = tensor.empty() : tensor<32x1x11008xi32, #encoding_result>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<32x1x11008xi32, #encoding_result>) -> tensor<32x1x11008xi32, #encoding_result>
  %12 = linalg.batch_matmul ins(%6, %9 : tensor<32x1x128xi32, #encoding_lhs>, tensor<32x128x11008xi32, #encoding_rhs>) outs(%11 : tensor<32x1x11008xi32, #encoding_result>) -> tensor<32x1x11008xi32, #encoding_result>
  %13 = iree_encoding.unset_encoding %12 : tensor<32x1x11008xi32, #encoding_result> -> tensor<32x1x11008xi32>
  return %13 : tensor<32x1x11008xi32>
}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @extend_batch_vecmat_explicit_unit_dim(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<32x1x128xi8>, %[[RHS:.+]]: tensor<32x128x11008xi8>) -> tensor<32x1x11008xi32>
//       CHECK:   %[[C0_I32:.+]] = arith.constant 0 : i32
//       CHECK:   %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<32x1x64x1x2xi8>
//       CHECK:   %[[LHS_PACK:.+]] = linalg.pack %[[LHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [1, 2] into %[[INIT_LHS_PACK]] : tensor<32x1x128xi8> -> tensor<32x1x64x1x2xi8>
//       CHECK:   %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<32x1x64x1x2xi32>
//       CHECK:   %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<32x1x64x1x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<32x1x64x1x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[LHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<32x688x64x16x2xi8>
//       CHECK:   %[[RHS_PACK:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %[[INIT_RHS_PACK]] : tensor<32x128x11008xi8> -> tensor<32x688x64x16x2xi8>
//       CHECK:   %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<32x688x64x16x2xi32>
//       CHECK:   %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<32x688x64x16x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<32x688x64x16x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[RHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_FILL:.+]] = tensor.empty() : tensor<32x1x688x1x16xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[INIT_FILL]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[LHS_EXT]], %[[RHS_EXT]] : tensor<32x1x64x1x2xi32>, tensor<32x688x64x16x2xi32>) outs(%[[FILL]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<32x1x11008xi32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[MMT4D]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [1, 16] into %[[INIT_UNPACK]] : tensor<32x1x688x1x16xi32> -> tensor<32x1x11008xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i16i16i32_x86_64_avx2(
    %3: tensor<?x?xi16, #encoding_lhs>,
    %4: tensor<?x?xi16, #encoding_rhs>,
    %5: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx2", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi16, #encoding_lhs>,
                   tensor<?x?xi16, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %6 : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i16i16i32_x86_64_avx2(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x2xi16>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x2xi16>
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i16ui4i32_x86_64_avx512vnni(
    %N: index,
    %K: index,
    %lhs: tensor<?x?xi16, #encoding_lhs>,
    %rhs_i4: tensor<?x?xi4, #encoding_rhs>,
    %outs: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %empty = tensor.empty(%K, %N) : tensor<?x?xi32, #encoding_rhs>
  %rhs_i32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
     ins(%rhs_i4 : tensor<?x?xi4, #encoding_rhs>) outs(%empty : tensor<?x?xi32, #encoding_rhs>) {
  ^bb0(%in: i4, %out: i32):
    %17 = arith.extui %in : i4 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?xi32, #encoding_rhs>
  %result = linalg.matmul
      ins(%lhs, %rhs_i32 : tensor<?x?xi16, #encoding_lhs>,
                   tensor<?x?xi32, #encoding_rhs>)
      outs(%outs : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %result : tensor<?x?xi32, #encoding_result>
}


//   CHECK-DAG: #[[$MAP_CEILDIV_8:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[$MAP_CEILDIV_32:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//   CHECK-DAG: #[[$MAP_IDENTITY_4D:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @matmul_lowering_i16ui4i32_x86_64_avx512vnni(
//  CHECK-SAME:   %[[N:[a-zA-Z0-9]+]]
//  CHECK-SAME:   %[[K:[a-zA-Z0-9]+]]
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x1x8xi16>,
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x32x8xi4>,
//  CHECK-SAME:   %[[OUT:[a-zA-Z0-9]+]]: tensor<?x?x1x32xi32>
//   CHECK-DAG:   %[[K_CEILDIV_8:.+]] = affine.apply #[[$MAP_CEILDIV_8]]()[%[[K]]]
//   CHECK-DAG:   %[[N_CEILDIV_32:.+]] = affine.apply #[[$MAP_CEILDIV_32]]()[%[[N]]]
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[N_CEILDIV_32]], %[[K_CEILDIV_8]]) : tensor<?x?x32x8xi32>
//   CHECK-DAG:   %[[RHS_I32:.+]] = linalg.generic {indexing_maps = [#[[$MAP_IDENTITY_4D]], #[[$MAP_IDENTITY_4D]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS]] : tensor<?x?x32x8xi4>) outs(%[[EMPTY]] : tensor<?x?x32x8xi32>) {
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS]], %[[RHS_I32]] : tensor<?x?x1x8xi16>, tensor<?x?x32x8xi32>) outs(%[[OUT]] : tensor<?x?x1x32xi32>) -> tensor<?x?x1x32xi32>
//       CHECK:   return %[[MMT4D]]

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [11008, 128]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [11008, 128]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [11008, 128]>
func.func @vecmat(%arg0: tensor<128xi8>, %arg1: tensor<128x11008xi8>) -> tensor<11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0_i32 = arith.constant 0 : i32
  %4 = iree_encoding.set_encoding %arg0 : tensor<128xi8> -> tensor<128xi8, #encoding_lhs>
  %5 = tensor.empty() : tensor<128xi32, #encoding_lhs>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4 : tensor<128xi8, #encoding_lhs>) outs(%5 : tensor<128xi32, #encoding_lhs>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<128xi32, #encoding_lhs>
  %7 = iree_encoding.set_encoding %arg1 : tensor<128x11008xi8> -> tensor<128x11008xi8, #encoding_rhs>
  %8 = tensor.empty() : tensor<128x11008xi32, #encoding_rhs>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<128x11008xi8, #encoding_rhs>) outs(%8 : tensor<128x11008xi32, #encoding_rhs>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<128x11008xi32, #encoding_rhs>
  %10 = tensor.empty() : tensor<11008xi32, #encoding_result>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<11008xi32, #encoding_result>) -> tensor<11008xi32, #encoding_result>
  %12 = linalg.vecmat ins(%6, %9 : tensor<128xi32, #encoding_lhs>, tensor<128x11008xi32, #encoding_rhs>) outs(%11 : tensor<11008xi32, #encoding_result>) -> tensor<11008xi32, #encoding_result>
  %13 = iree_encoding.unset_encoding %12 : tensor<11008xi32, #encoding_result> -> tensor<11008xi32>
  return %13 : tensor<11008xi32>
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @vecmat(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<128xi8>, %[[RHS:.+]]: tensor<128x11008xi8>) -> tensor<11008xi32>
//   CHECK-DAG:   %[[C0_I32:.+]] = arith.constant 0 : i32
//       CHECK:   %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<64x2xi8>
//       CHECK:   %[[LHS_PACK:.+]] = linalg.pack %[[LHS]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [2] into %[[INIT_LHS_PACK]] : tensor<128xi8> -> tensor<64x2xi8>
//       CHECK:   %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<64x2xi32>
//       CHECK:   %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<64x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<64x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[LHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<688x64x16x2xi8>
//       CHECK:   %[[RHS_PACK:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %[[INIT_RHS_PACK]] : tensor<128x11008xi8> -> tensor<688x64x16x2xi8>
//       CHECK:   %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<688x64x16x2xi32>
//       CHECK:   %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<688x64x16x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<688x64x16x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[RHS_EXT_OP]] : i32
//   CHECK-DAG:   %[[INIT_FILL:.+]] = tensor.empty() : tensor<1x688x1x16xi32>
//   CHECK-DAG:   %[[EXPAND_LHS:.+]] = tensor.expand_shape %[[LHS_EXT]] {{\[}}[0, 1], [2, 3]] output_shape [1, 64, 1, 2] : tensor<64x2xi32> into tensor<1x64x1x2xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[INIT_FILL]] : tensor<1x688x1x16xi32>) -> tensor<1x688x1x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXPAND_LHS]], %[[RHS_EXT]] : tensor<1x64x1x2xi32>, tensor<688x64x16x2xi32>) outs(%[[FILL]] : tensor<1x688x1x16xi32>) -> tensor<1x688x1x16xi32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0, 1], [2, 3]] : tensor<1x688x1x16xi32> into tensor<688x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<11008xi32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[COLLAPSED]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<688x16xi32> -> tensor<11008xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [11008, 128]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [11008, 128]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [11008, 128]>
func.func @matvec(%arg0: tensor<11008x128xi8>, %arg1: tensor<128xi8>) -> tensor<11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0_i32 = arith.constant 0 : i32
  %4 = iree_encoding.set_encoding %arg0 : tensor<11008x128xi8> -> tensor<11008x128xi8, #encoding_lhs>
  %5 = tensor.empty() : tensor<11008x128xi32, #encoding_lhs>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<11008x128xi8, #encoding_lhs>) outs(%5 : tensor<11008x128xi32, #encoding_lhs>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<11008x128xi32, #encoding_lhs>
  %7 = iree_encoding.set_encoding %arg1 : tensor<128xi8> -> tensor<128xi8, #encoding_rhs>
  %8 = tensor.empty() : tensor<128xi32, #encoding_rhs>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7 : tensor<128xi8, #encoding_rhs>) outs(%8 : tensor<128xi32, #encoding_rhs>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<128xi32, #encoding_rhs>
  %10 = tensor.empty() : tensor<11008xi32, #encoding_result>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<11008xi32, #encoding_result>) -> tensor<11008xi32, #encoding_result>
  %12 = linalg.matvec ins(%6, %9 : tensor<11008x128xi32, #encoding_lhs>, tensor<128xi32, #encoding_rhs>) outs(%11 : tensor<11008xi32, #encoding_result>) -> tensor<11008xi32, #encoding_result>
  %13 = iree_encoding.unset_encoding %12 : tensor<11008xi32, #encoding_result> -> tensor<11008xi32>
  return %13 : tensor<11008xi32>
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @matvec(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<11008x128xi8>, %[[RHS:.+]]: tensor<128xi8>) -> tensor<11008xi32>
//       CHECK:   %[[C0_I32:.+]] = arith.constant 0 : i32
//       CHECK:   %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<688x64x16x2xi8>
//       CHECK:   %[[LHS_PACK:.+]] = linalg.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 2] into %[[INIT_LHS_PACK]] : tensor<11008x128xi8> -> tensor<688x64x16x2xi8>
//       CHECK:   %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<688x64x16x2xi32>
//       CHECK:   %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<688x64x16x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<688x64x16x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[LHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<64x2xi8>
//       CHECK:   %[[RHS_PACK:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [2] into %[[INIT_RHS_PACK]] : tensor<128xi8> -> tensor<64x2xi8>
//       CHECK:   %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<64x2xi32>
//       CHECK:   %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<64x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<64x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[RHS_EXT_OP]] : i32
//   CHECK-DAG:   %[[INIT_FILL:.+]] = tensor.empty() : tensor<1x688x1x16xi32>
//   CHECK-DAG:   %[[EXPAND_RHS:.+]] = tensor.expand_shape %[[RHS_EXT]] {{\[}}[0, 1], [2, 3]] output_shape [1, 64, 1, 2] : tensor<64x2xi32> into tensor<1x64x1x2xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[INIT_FILL]] : tensor<1x688x1x16xi32>) -> tensor<1x688x1x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXPAND_RHS]], %[[LHS_EXT]]  : tensor<1x64x1x2xi32>, tensor<688x64x16x2xi32>) outs(%[[FILL]] : tensor<1x688x1x16xi32>) -> tensor<1x688x1x16xi32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0, 1], [2, 3]] : tensor<1x688x1x16xi32> into tensor<688x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<11008xi32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[COLLAPSED]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<688x16xi32> -> tensor<11008xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [15, 128]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [15, 128]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [15, 128]>
func.func @matvec_with_narrow_M(%arg0: tensor<15x128xi8>, %arg1: tensor<128xi8>) -> tensor<15xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0_i32 = arith.constant 0 : i32
  %4 = iree_encoding.set_encoding %arg0 : tensor<15x128xi8> -> tensor<15x128xi8, #encoding_lhs>
  %5 = tensor.empty() : tensor<15x128xi32, #encoding_lhs>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<15x128xi8, #encoding_lhs>) outs(%5 : tensor<15x128xi32, #encoding_lhs>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<15x128xi32, #encoding_lhs>
  %7 = iree_encoding.set_encoding %arg1 : tensor<128xi8> -> tensor<128xi8, #encoding_rhs>
  %8 = tensor.empty() : tensor<128xi32, #encoding_rhs>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7 : tensor<128xi8, #encoding_rhs>) outs(%8 : tensor<128xi32, #encoding_rhs>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<128xi32, #encoding_rhs>
  %10 = tensor.empty() : tensor<15xi32, #encoding_result>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<15xi32, #encoding_result>) -> tensor<15xi32, #encoding_result>
  %12 = linalg.matvec ins(%6, %9 : tensor<15x128xi32, #encoding_lhs>, tensor<128xi32, #encoding_rhs>) outs(%11 : tensor<15xi32, #encoding_result>) -> tensor<15xi32, #encoding_result>
  %13 = iree_encoding.unset_encoding %12 : tensor<15xi32, #encoding_result> -> tensor<15xi32>
  return %13 : tensor<15xi32>
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @matvec_with_narrow_M(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<15x128xi8>, %[[RHS:.+]]: tensor<128xi8>) -> tensor<15xi32>
//   CHECK-DAG:   %[[C0_I8:.+]] = arith.constant 0 : i8
//   CHECK-DAG:   %[[C0_I32:.+]] = arith.constant 0 : i32
//       CHECK:   %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<1x64x16x2xi8>
//       CHECK:   %[[LHS_PACK:.+]] = linalg.pack %[[LHS]] padding_value(%[[C0_I8]] : i8) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 2] into %[[INIT_LHS_PACK]] : tensor<15x128xi8> -> tensor<1x64x16x2xi8>
//       CHECK:   %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<1x64x16x2xi32>
//       CHECK:   %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<1x64x16x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<1x64x16x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[LHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<64x2xi8>
//       CHECK:   %[[RHS_PACK:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [2] into %[[INIT_RHS_PACK]] : tensor<128xi8> -> tensor<64x2xi8>
//       CHECK:   %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<64x2xi32>
//       CHECK:   %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<64x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<64x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[RHS_EXT_OP]] : i32
//   CHECK-DAG:   %[[INIT_FILL:.+]] = tensor.empty() : tensor<1x1x1x16xi32>
//   CHECK-DAG:   %[[EXPAND_RHS:.+]] = tensor.expand_shape %[[RHS_EXT]] {{\[}}[0, 1], [2, 3]] output_shape [1, 64, 1, 2] : tensor<64x2xi32> into tensor<1x64x1x2xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[INIT_FILL]] : tensor<1x1x1x16xi32>) -> tensor<1x1x1x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXPAND_RHS]], %[[LHS_EXT]]  : tensor<1x64x1x2xi32>, tensor<1x64x16x2xi32>) outs(%[[FILL]] : tensor<1x1x1x16xi32>) -> tensor<1x1x1x16xi32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0, 1], [2, 3]] : tensor<1x1x1x16xi32> into tensor<1x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<15xi32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[COLLAPSED]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<1x16xi32> -> tensor<15xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, 11008, 128]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, 11008, 128]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, 11008, 128]>
func.func @batch_vecmat(%arg0: tensor<32x128xi8>, %arg1: tensor<32x128x11008xi8>) -> tensor<32x11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0_i32 = arith.constant 0 : i32
  %4 = iree_encoding.set_encoding %arg0 : tensor<32x128xi8> -> tensor<32x128xi8, #encoding_lhs>
  %5 = tensor.empty() : tensor<32x128xi32, #encoding_lhs>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<32x128xi8, #encoding_lhs>) outs(%5 : tensor<32x128xi32, #encoding_lhs>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<32x128xi32, #encoding_lhs>
  %7 = iree_encoding.set_encoding %arg1 : tensor<32x128x11008xi8> -> tensor<32x128x11008xi8, #encoding_rhs>
  %8 = tensor.empty() : tensor<32x128x11008xi32, #encoding_rhs>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<32x128x11008xi8, #encoding_rhs>) outs(%8 : tensor<32x128x11008xi32, #encoding_rhs>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<32x128x11008xi32, #encoding_rhs>
  %10 = tensor.empty() : tensor<32x11008xi32, #encoding_result>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<32x11008xi32, #encoding_result>) -> tensor<32x11008xi32, #encoding_result>
  %12 = linalg.batch_vecmat ins(%6, %9 : tensor<32x128xi32, #encoding_lhs>, tensor<32x128x11008xi32, #encoding_rhs>) outs(%11 : tensor<32x11008xi32, #encoding_result>) -> tensor<32x11008xi32, #encoding_result>
  %13 = iree_encoding.unset_encoding %12 : tensor<32x11008xi32, #encoding_result> -> tensor<32x11008xi32>
  return %13 : tensor<32x11008xi32>
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @batch_vecmat(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<32x128xi8>, %[[RHS:.+]]: tensor<32x128x11008xi8>) -> tensor<32x11008xi32>
//       CHECK:   %[[C0_I32:.+]] = arith.constant 0 : i32
//       CHECK:   %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<32x64x2xi8>
//       CHECK:   %[[LHS_PACK:.+]] = linalg.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [2] into %[[INIT_LHS_PACK]] : tensor<32x128xi8> -> tensor<32x64x2xi8>
//       CHECK:   %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<32x64x2xi32>
//       CHECK:   %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<32x64x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<32x64x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[LHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<32x688x64x16x2xi8>
//       CHECK:   %[[RHS_PACK:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %[[INIT_RHS_PACK]] : tensor<32x128x11008xi8> -> tensor<32x688x64x16x2xi8>
//       CHECK:   %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<32x688x64x16x2xi32>
//       CHECK:   %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<32x688x64x16x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<32x688x64x16x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[RHS_EXT_OP]] : i32
//   CHECK-DAG:   %[[INIT_FILL:.+]] = tensor.empty() : tensor<32x1x688x1x16xi32>
//   CHECK-DAG:   %[[EXPAND_LHS:.+]] = tensor.expand_shape %[[LHS_EXT]] {{\[}}[0], [1, 2], [3, 4]] output_shape [32, 1, 64, 1, 2] : tensor<32x64x2xi32> into tensor<32x1x64x1x2xi32>
//   CHECK-DAG:   %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[INIT_FILL]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[EXPAND_LHS]], %[[RHS_EXT]] : tensor<32x1x64x1x2xi32>, tensor<32x688x64x16x2xi32>) outs(%[[FILL]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0], [1, 2], [3, 4]] : tensor<32x1x688x1x16xi32> into tensor<32x688x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<32x11008xi32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[COLLAPSED]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<32x688x16xi32> -> tensor<32x11008xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [11008, 1, 128]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [11008, 1, 128]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [11008, 1, 128]>
func.func @batch_matvec(
    %0: tensor<32x11008x128xi8>,
    %1: tensor<32x128xi8>,
    %2: tensor<32x11008xi32>
) -> tensor<32x11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %3 = iree_encoding.set_encoding %0 : tensor<32x11008x128xi8> -> tensor<32x11008x128xi8, #encoding_lhs>
  %4 = iree_encoding.set_encoding %1 : tensor<32x128xi8> -> tensor<32x128xi8, #encoding_rhs>
  %5 = iree_encoding.set_encoding %2 : tensor<32x11008xi32> -> tensor<32x11008xi32, #encoding_result>
  %6 = linalg.batch_matvec ins(%3, %4 : tensor<32x11008x128xi8, #encoding_lhs>, tensor<32x128xi8, #encoding_rhs>) outs(%5 : tensor<32x11008xi32, #encoding_result>) -> tensor<32x11008xi32, #encoding_result>
  %7 = iree_encoding.unset_encoding %6 : tensor<32x11008xi32, #encoding_result> -> tensor<32x11008xi32>
  return %7 : tensor<32x11008xi32>
}
// CHECK-LABEL: func @batch_matvec(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<32x11008x128xi8>,
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<32x128xi8>,
//  CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<32x11008xi32>
//       CHECK:   %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<32x688x64x16x2xi8>
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack %[[LHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [16, 2] into %[[INIT_LHS_PACK]] : tensor<32x11008x128xi8> -> tensor<32x688x64x16x2xi8>
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<32x64x2xi8>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [2] into %[[INIT_RHS_PACK]] : tensor<32x128xi8> -> tensor<32x64x2xi8>
//       CHECK:   %[[INIT_OUTS_PACK:.+]] = tensor.empty() : tensor<32x688x16xi32>
//       CHECK:   %[[PACK_OUTS:.+]] = linalg.pack %[[OUTS]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [16] into %[[INIT_OUTS_PACK]] : tensor<32x11008xi32> -> tensor<32x688x16xi32>
//       CHECK:   %[[EXPAND_RHS:.+]] = tensor.expand_shape %[[PACK_RHS]]
//  CHECK-SAME:      {{\[}}[0], [1, 2], [3, 4]] output_shape [32, 1, 64, 1, 2]
//  CHECK-SAME:      : tensor<32x64x2xi8> into tensor<32x1x64x1x2xi8>
//       CHECK:   %[[EXPAND_OUTS:.+]] = tensor.expand_shape %[[PACK_OUTS]]
//  CHECK-SAME:      {{\[}}[0], [1, 2], [3, 4]] output_shape [32, 688, 1, 16, 1]
//  CHECK-SAME:      : tensor<32x688x16xi32> into tensor<32x688x1x16x1xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.batch_mmt4d
//  CHECK-SAME:       ins(%[[PACK_LHS]], %[[EXPAND_RHS]] :
//  CHECK-SAME:       outs(%[[EXPAND_OUTS]] :
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]]
//  CHECK-SAME:      {{\[}}[0], [1, 2], [3, 4]] : tensor<32x688x1x16x1xi32> into tensor<32x688x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<32x11008xi32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[COLLAPSED]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<32x688x16xi32> -> tensor<32x11008xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 512, 256]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 512, 256]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 512, 256]>
func.func @matmul_transpose_a_f32f32f32(%arg0: tensor<256x128xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = iree_encoding.set_encoding %arg0 : tensor<256x128xf32> -> tensor<256x128xf32, #encoding_lhs>
  %10 = iree_encoding.set_encoding %arg1 : tensor<256x512xf32> -> tensor<256x512xf32, #encoding_rhs>
  %14 = iree_encoding.set_encoding %arg2 : tensor<128x512xf32> -> tensor<128x512xf32, #encoding_result>
  %15 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d2, d0)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%6, %10 : tensor<256x128xf32, #encoding_lhs>, tensor<256x512xf32, #encoding_rhs>)
    outs(%14 : tensor<128x512xf32, #encoding_result>) -> tensor<128x512xf32, #encoding_result>
  %16 = iree_encoding.unset_encoding %15 : tensor<128x512xf32, #encoding_result> -> tensor<128x512xf32>
  return %16 : tensor<128x512xf32>
}

// CHECK-LABEL: func.func @matmul_transpose_a_f32f32f32(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<256x128xf32>, %[[RHS:.+]]: tensor<256x512xf32>, %[[RESULT:.+]]: tensor<128x512xf32>) -> tensor<128x512xf32>
//       CHECK:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<16x256x8x1xf32>
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack %[[LHS]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<256x128xf32> -> tensor<16x256x8x1xf32>
//       CHECK:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<128x256x4x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<256x512xf32> -> tensor<128x256x4x1xf32>
//       CHECK:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<16x128x8x4xf32>
//       CHECK:   %[[PACK_RES:.+]] = linalg.pack %[[RESULT]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<128x512xf32> -> tensor<16x128x8x4xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
//  CHECK-SAME:       outs(%[[PACK_RES]] :
//   CHECK-DAG:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<128x512xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[MMT4D]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<16x128x8x4xf32> -> tensor<128x512xf32>
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 512, 256]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 512, 256]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 512, 256]>
func.func @matmul_transpose_b_f32f32f32(%arg0: tensor<128x256xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %6 = iree_encoding.set_encoding %arg0 : tensor<128x256xf32> -> tensor<128x256xf32, #encoding_lhs>
  %10 = iree_encoding.set_encoding %arg1 : tensor<512x256xf32> -> tensor<512x256xf32, #encoding_rhs>
  %14 = iree_encoding.set_encoding %arg2 : tensor<128x512xf32> -> tensor<128x512xf32, #encoding_result>
  %15 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%6, %10 : tensor<128x256xf32, #encoding_lhs>, tensor<512x256xf32, #encoding_rhs>)
    outs(%14 : tensor<128x512xf32, #encoding_result>) -> tensor<128x512xf32, #encoding_result>
  %16 = iree_encoding.unset_encoding %15 : tensor<128x512xf32, #encoding_result> -> tensor<128x512xf32>
  return %16 : tensor<128x512xf32>
}

// CHECK-LABEL: func.func @matmul_transpose_b_f32f32f32(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<128x256xf32>, %[[RHS:.+]]: tensor<512x256xf32>, %[[RESULT:.+]]: tensor<128x512xf32>) -> tensor<128x512xf32>
//       CHECK:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<16x256x8x1xf32>
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<128x256xf32> -> tensor<16x256x8x1xf32>
//       CHECK:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<128x256x4x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<512x256xf32> -> tensor<128x256x4x1xf32>
//       CHECK:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<16x128x8x4xf32>
//       CHECK:   %[[PACK_RES:.+]] = linalg.pack %[[RESULT]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<128x512xf32> -> tensor<16x128x8x4xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
//  CHECK-SAME:       outs(%[[PACK_RES]] :
//       CHECK:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<128x512xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[MMT4D]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<16x128x8x4xf32> -> tensor<128x512xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [2, 128, 512, 256]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [2, 128, 512, 256]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [2, 128, 512, 256]>
func.func @batch_matmul_transpose_a_f32f32f32(%arg0: tensor<2x256x128xf32>, %arg1: tensor<2x256x512xf32>, %arg2: tensor<2x128x512xf32>) -> tensor<2x128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %7 = iree_encoding.set_encoding %arg0 : tensor<2x256x128xf32> -> tensor<2x256x128xf32, #encoding_lhs>
  %12 = iree_encoding.set_encoding %arg1 : tensor<2x256x512xf32> -> tensor<2x256x512xf32, #encoding_rhs>
  %17 = iree_encoding.set_encoding %arg2 : tensor<2x128x512xf32> -> tensor<2x128x512xf32, #encoding_result>
  %18 = linalg.batch_matmul
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ]
    ins(%7, %12 : tensor<2x256x128xf32, #encoding_lhs>, tensor<2x256x512xf32, #encoding_rhs>)
    outs(%17 : tensor<2x128x512xf32, #encoding_result>) -> tensor<2x128x512xf32, #encoding_result>
  %19 = iree_encoding.unset_encoding %18 : tensor<2x128x512xf32, #encoding_result> -> tensor<2x128x512xf32>
  return %19 : tensor<2x128x512xf32>
}

// CHECK-LABEL: func.func @batch_matmul_transpose_a_f32f32f32(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<2x256x128xf32>, %[[RHS:.+]]: tensor<2x256x512xf32>, %[[RESULT:.+]]: tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
//       CHECK:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<2x16x256x8x1xf32>
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack %[[LHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<2x256x128xf32> -> tensor<2x16x256x8x1xf32>
//       CHECK:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<2x128x256x4x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<2x256x512xf32> -> tensor<2x128x256x4x1xf32>
//       CHECK:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<2x16x128x8x4xf32>
//       CHECK:   %[[PACK_RES:.+]] = linalg.pack %[[RESULT]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<2x128x512xf32> -> tensor<2x16x128x8x4xf32>
//       CHECK:   %[[BATCH_MMT4D:.+]] = linalg.batch_mmt4d
//       CHECK:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
//       CHECK:       outs(%[[PACK_RES]] :
//       CHECK:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<2x128x512xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[BATCH_MMT4D]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<2x16x128x8x4xf32> -> tensor<2x128x512xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [2, 128, 512, 256]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [2, 128, 512, 256]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [2, 128, 512, 256]>
func.func @batch_matmul_transpose_b_f32f32f32(%arg0: tensor<2x128x256xf32>, %arg1: tensor<2x512x256xf32>, %arg2: tensor<2x128x512xf32>) -> tensor<2x128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %7 = iree_encoding.set_encoding %arg0 : tensor<2x128x256xf32> -> tensor<2x128x256xf32, #encoding_lhs>
  %12 = iree_encoding.set_encoding %arg1 : tensor<2x512x256xf32> -> tensor<2x512x256xf32, #encoding_rhs>
  %17 = iree_encoding.set_encoding %arg2 : tensor<2x128x512xf32> -> tensor<2x128x512xf32, #encoding_result>
  %18 = linalg.batch_matmul
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ]
    ins(%7, %12 : tensor<2x128x256xf32, #encoding_lhs>, tensor<2x512x256xf32, #encoding_rhs>)
    outs(%17 : tensor<2x128x512xf32, #encoding_result>) -> tensor<2x128x512xf32, #encoding_result>
  %19 = iree_encoding.unset_encoding %18 : tensor<2x128x512xf32, #encoding_result> -> tensor<2x128x512xf32>
  return %19 : tensor<2x128x512xf32>
}

// CHECK-LABEL: func.func @batch_matmul_transpose_b_f32f32f32(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<2x128x256xf32>, %[[RHS:.+]]: tensor<2x512x256xf32>, %[[RESULT:.+]]: tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
//       CHECK:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<2x16x256x8x1xf32>
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack %[[LHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<2x128x256xf32> -> tensor<2x16x256x8x1xf32>
//       CHECK:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<2x128x256x4x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<2x512x256xf32> -> tensor<2x128x256x4x1xf32>
//       CHECK:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<2x16x128x8x4xf32>
//       CHECK:   %[[PACK_RES:.+]] = linalg.pack %[[RESULT]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<2x128x512xf32> -> tensor<2x16x128x8x4xf32>
//       CHECK:   %[[BATCH_MMT4D:.+]] = linalg.batch_mmt4d
//  CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
//  CHECK-SAME:       outs(%[[PACK_RES]] :
//   CHECK-DAG:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<2x128x512xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[BATCH_MMT4D]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<2x16x128x8x4xf32> -> tensor<2x128x512xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, 4096, 128]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, 4096, 128]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1, 4096, 128]>
func.func @generic_batch_vecmat_transposed_i16u4i32(%arg0: tensor<32x128xi16>, %arg1: tensor<4096x32x128xi4>, %arg2: tensor<4096x32xi32>) -> tensor<4096x32xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %3 = iree_encoding.set_encoding %arg0 : tensor<32x128xi16> -> tensor<32x128xi16, #encoding_lhs>
  %8 = iree_encoding.set_encoding %arg1 : tensor<4096x32x128xi4> -> tensor<4096x32x128xi4, #encoding_rhs>
  %12 = iree_encoding.set_encoding %arg2 : tensor<4096x32xi32> -> tensor<4096x32xi32, #encoding_result>
  %13 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %8 : tensor<32x128xi16, #encoding_lhs>, tensor<4096x32x128xi4, #encoding_rhs>) outs(%12 : tensor<4096x32xi32, #encoding_result>) {
  ^bb0(%in: i16, %in_2: i4, %out: i32):
    %15 = arith.extsi %in : i16 to i32
    %16 = arith.extui %in_2 : i4 to i32
    %17 = arith.muli %15, %16 : i32
    %18 = arith.addi %17, %out : i32
    linalg.yield %18 : i32
  } -> tensor<4096x32xi32, #encoding_result>
  %14 = iree_encoding.unset_encoding %13 : tensor<4096x32xi32, #encoding_result> -> tensor<4096x32xi32>
  return %14 : tensor<4096x32xi32>
}

//       CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @generic_batch_vecmat_transposed_i16u4i32(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<32x128xi16>, %[[RHS:.+]]: tensor<4096x32x128xi4>, %[[RESULT:.+]]: tensor<4096x32xi32>) -> tensor<4096x32xi32>
//   CHECK-DAG:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<32x16x8xi16>
//   CHECK-DAG:   %[[PACK_LHS:.+]] = linalg.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [8] into %[[PACK_LHS_DEST]] : tensor<32x128xi16> -> tensor<32x16x8xi16>
//   CHECK-DAG:   %[[EXPAND_LHS:.+]] = tensor.expand_shape %[[PACK_LHS]] {{.*}} output_shape [32, 1, 16, 1, 8] : tensor<32x16x8xi16> into tensor<32x1x16x1x8xi16>
//   CHECK-DAG:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<32x128x16x32x8xi4>
//   CHECK-DAG:   %[[PACK_RHS:.+]] = linalg.pack %[[RHS]] outer_dims_perm = [1, 0, 2] inner_dims_pos = [0, 2] inner_tiles = [32, 8] into %[[PACK_RHS_DEST]] : tensor<4096x32x128xi4> -> tensor<32x128x16x32x8xi4>
//   CHECK-DAG:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<32x128x32xi32>
//   CHECK-DAG:   %[[PACK_RES:.+]] = linalg.pack %[[RESULT]] outer_dims_perm = [1, 0] inner_dims_pos = [0] inner_tiles = [32] into %[[PACK_RES_DEST]] : tensor<4096x32xi32> -> tensor<32x128x32xi32>
//   CHECK-DAG:   %[[EXTEND_DEST:.+]] = tensor.empty() : tensor<32x128x16x32x8xi32>
//       CHECK:   %[[EXTEND:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP]]]
//  CHECK-SAME:       ins(%[[PACK_RHS]] : tensor<32x128x16x32x8xi4>)
//  CHECK-SAME:       outs(%[[EXTEND_DEST]] : tensor<32x128x16x32x8xi32>)
//       CHECK:   %[[EXPAND_RES:.+]] = tensor.expand_shape %[[PACK_RES]] {{.*}} output_shape [32, 1, 128, 1, 32] : tensor<32x128x32xi32> into tensor<32x1x128x1x32xi32>
//       CHECK:   %[[BATCH_MMT4D:.+]] = linalg.batch_mmt4d
//  CHECK-SAME:       ins(%[[EXPAND_LHS]], %[[EXTEND]] :
//  CHECK-SAME:       outs(%[[EXPAND_RES]] :
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[BATCH_MMT4D]] {{.*}} : tensor<32x1x128x1x32xi32> into tensor<32x128x32xi32>
//   CHECK-DAG:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<4096x32xi32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[COLLAPSE]] outer_dims_perm = [1, 0] inner_dims_pos = [0] inner_tiles = [32] into %[[UNPACK_DEST]] : tensor<32x128x32xi32> -> tensor<4096x32xi32>

// -----

#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iteration_sizes = [2, 128, 64, ?]>
#encoding_bcast = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iteration_sizes = [2, 128, 64, ?]>
func.func @dequantization(
    %7: tensor<2x128x64xi8, #encoding>,
    %8: tensor<2x64xf32, #encoding_bcast>,
    %9: tensor<2x64xf32, #encoding_bcast>
) -> tensor<2x128x64xf32, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %8, %9 : tensor<2x128x64xi8, #encoding>, tensor<2x64xf32, #encoding_bcast>, tensor<2x64xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %21 = arith.extui %in : i8 to i32
    %22 = arith.uitofp %21 : i32 to f32
    %23 = arith.subf %22, %in_1 : f32
    %24 = arith.mulf %23, %in_0 : f32
    linalg.yield %24 : f32
  } -> tensor<2x128x64xf32, #encoding>
  return %14 : tensor<2x128x64xf32, #encoding>
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
// CHECK-LABEL: func.func @dequantization(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<2x4x128x16x1xi8>
//  CHECK-SAME:   %[[LHS_SCALES:[a-zA-Z0-9]+]]: tensor<2x4x16xf32>
//  CHECK-SAME:   %[[LHS_ZPS:[a-zA-Z0-9]+]]: tensor<2x4x16xf32>
//  CHECK-SAME: ) -> tensor<2x4x128x16x1xf32>
//   CHECK-DAG:   %[[EMPTY_LHS:.+]] = tensor.empty() : tensor<2x4x128x16x1xf32>
//   CHECK-DAG:   %[[LHS_DEQUANT:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP1]], #[[$MAP]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[LHS]], %[[LHS_SCALES]], %[[LHS_ZPS]] : tensor<2x4x128x16x1xi8>, tensor<2x4x16xf32>, tensor<2x4x16xf32>)
//  CHECK-SAME:       outs(%[[EMPTY_LHS]] : tensor<2x4x128x16x1xf32>)
//       CHECK:     arith.extui
//       CHECK:     arith.uitofp
//       CHECK:     arith.subf
//       CHECK:     arith.mulf
//       CHECK:   return %[[LHS_DEQUANT]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iteration_sizes = [2, 128, 64, ?]>
#encoding_bcast = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,  affine_map<(d0, d1, d2) -> (d1, d2)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iteration_sizes = [2, 128, 64, ?]>
func.func @broadcast_batch(
    %8: tensor<128x64xf32, #encoding_bcast>
) -> tensor<2x128x64xf32, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<128x64xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x128x64xf32, #encoding>
  return %14 : tensor<2x128x64xf32, #encoding>
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @broadcast_batch(
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<4x128x16x1xf32>
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x4x128x16x1xf32>
//   CHECK-DAG:   %[[BROADCAST:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<4x128x16x1xf32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<2x4x128x16x1xf32>)
//       CHECK:   return %[[BROADCAST]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iteration_sizes = [2, 128, 64, ?]>
#encoding_bcast = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iteration_sizes = [2, 128, 64, ?]>
func.func @broadcast_M(
    %8: tensor<2x128xf32, #encoding_bcast>
) -> tensor<2x128x64xf32, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<2x128xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x128x64xf32, #encoding>
  return %14 : tensor<2x128x64xf32, #encoding>
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @broadcast_M(
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<2x128x1xf32>
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x4x128x16x1xf32>
//   CHECK-DAG:   %[[BROADCAST:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<2x128x1xf32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<2x4x128x16x1xf32>)
//       CHECK:  return %[[BROADCAST]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iteration_sizes = [2, 128, 64, ?]>
#encoding_bcast = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2) -> (d0, d2)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iteration_sizes = [2, 128, 64, ?]>
func.func @broadcast_N(
    %8: tensor<2x64xf32, #encoding_bcast>
) -> tensor<2x128x64xf32, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<2x64xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x128x64xf32, #encoding>
  return %14 : tensor<2x128x64xf32, #encoding>
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @broadcast_N(
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<2x64x1xf32>
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x8x64x16x1xf32>
//   CHECK-DAG:   %[[BROADCAST:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<2x64x1xf32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<2x8x64x16x1xf32>)
//      CHECK:   return %[[BROADCAST]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iteration_sizes = [2, 128, 64, ?]>
#encoding_bcast = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iteration_sizes = [2, 128, 64, ?]>
func.func @broadcast_K(
    %8: tensor<2x64xf32, #encoding_bcast>
) -> tensor<2x128x64xf32, #encoding> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<2x64xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x128x64xf32, #encoding>
  return %14 : tensor<2x128x64xf32, #encoding>
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @broadcast_K(
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<2x4x16xf32>
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x4x128x16x1xf32>
//   CHECK-DAG:   %[[BROADCAST:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<2x4x16xf32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<2x4x128x16x1xf32>)
//       CHECK:   return %[[BROADCAST]]

// -----

#executable_target_xyz = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> ()>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [2, 4, 3]>
#encoding1 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [2, 4, 3]>
func.func @generic_with_0d_tensor(
    %3: tensor<2x4xf32, #encoding>,
    %5: tensor<f32, #encoding1>
) -> tensor<2x4xf32, #encoding> attributes {hal.executable.target = #executable_target_xyz} {
  %cst = arith.constant 0.000000e+00 : f32
  %6 = tensor.empty() : tensor<2x4xf32, #encoding>
  %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<2x4xf32, #encoding>) -> tensor<2x4xf32, #encoding>
  %8 = linalg.generic {indexing_maps = [#map4, #map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%3, %5 : tensor<2x4xf32, #encoding>, tensor<f32, #encoding1>) outs(%7 : tensor<2x4xf32, #encoding>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %9 = arith.addf %in, %in_0 : f32
    linalg.yield %9 : f32
  } -> tensor<2x4xf32, #encoding>
  return %8 : tensor<2x4xf32, #encoding>
}
// CHECK-LABEL: func.func @generic_with_0d_tensor(
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<1x1x2x4xf32>
//  CHECK-SAME:   %[[INPUT_0D:[a-zA-Z0-9]+]]: tensor<f32>
//       CHECK:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x2x4xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[CST]] : f32)
//  CHECK-SAME:       outs(%[[EMPTY]]
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[INPUT]], %[[INPUT_0D]]
//  CHECK-SAME:     outs(%[[FILL]]
//       CHECK:   return %[[GENERIC]]

// -----

// Scaled contraction (MX matmul) is not yet supported on CPU, so we drop the
// encoding and clone the op as-is.

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#encoding_a = #iree_encoding.encoding<operand_index = 0 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map0, #map1, #map2, #map3, #map4], iteration_sizes = [256, 512, 128, 32]>
#encoding_b = #iree_encoding.encoding<operand_index = 1 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map0, #map1, #map2, #map3, #map4], iteration_sizes = [256, 512, 128, 32]>
#encoding_a_scales = #iree_encoding.encoding<operand_index = 2 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map0, #map1, #map2, #map3, #map4], iteration_sizes = [256, 512, 128, 32]>
#encoding_b_scales = #iree_encoding.encoding<operand_index = 3 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map0, #map1, #map2, #map3, #map4], iteration_sizes = [256, 512, 128, 32]>
#encoding_c = #iree_encoding.encoding<operand_index = 4 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map0, #map1, #map2, #map3, #map4], iteration_sizes = [256, 512, 128, 32]>
func.func @scaled_matmul_f4E2M1FN_f8E8M0FNU_f32(
    %a: tensor<256x128x32xf4E2M1FN, #encoding_a>,
    %b: tensor<512x128x32xf4E2M1FN, #encoding_b>,
    %a_scales: tensor<256x128xf8E8M0FNU, #encoding_a_scales>,
    %b_scales: tensor<512x128xf8E8M0FNU, #encoding_b_scales>,
    %c: tensor<256x512xf32, #encoding_c>) -> tensor<256x512xf32, #encoding_c> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = linalg.generic {
      indexing_maps = [#map0, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%a, %b, %a_scales, %b_scales : tensor<256x128x32xf4E2M1FN, #encoding_a>, tensor<512x128x32xf4E2M1FN, #encoding_b>, tensor<256x128xf8E8M0FNU, #encoding_a_scales>, tensor<512x128xf8E8M0FNU, #encoding_b_scales>)
      outs(%c : tensor<256x512xf32, #encoding_c>) {
  ^bb0(%in_a: f4E2M1FN, %in_b: f4E2M1FN, %in_a_scale: f8E8M0FNU, %in_b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %in_a, %in_a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %in_b, %in_b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<256x512xf32, #encoding_c>
  return %0 : tensor<256x512xf32, #encoding_c>
}
// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK-LABEL: func.func @scaled_matmul_f4E2M1FN_f8E8M0FNU_f32(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: tensor<256x128x32xf4E2M1FN>,
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: tensor<512x128x32xf4E2M1FN>,
//  CHECK-SAME:   %[[A_SCALES:[a-zA-Z0-9]+]]: tensor<256x128xf8E8M0FNU>,
//  CHECK-SAME:   %[[B_SCALES:[a-zA-Z0-9]+]]: tensor<512x128xf8E8M0FNU>,
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: tensor<256x512xf32>)
//  CHECK-SAME:   -> tensor<256x512xf32>
//       CHECK:   %[[RESULT:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(%[[A]], %[[B]], %[[A_SCALES]], %[[B_SCALES]] : tensor<256x128x32xf4E2M1FN>, tensor<512x128x32xf4E2M1FN>, tensor<256x128xf8E8M0FNU>, tensor<512x128xf8E8M0FNU>)
//  CHECK-SAME:       outs(%[[C]] : tensor<256x512xf32>)
