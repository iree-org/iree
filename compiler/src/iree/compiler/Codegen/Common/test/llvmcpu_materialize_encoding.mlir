// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding),canonicalize,cse)" --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], round_dims_to = array<i64: 1, 16, 16>>
func.func @set_encoding_with_padding_semantics_bf16_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
}{
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1000xbf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x1000xbf16, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1000], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1000xbf16>> -> tensor<1x1000xbf16>
  %3 = iree_encoding.set_encoding %2 : tensor<1x1000xbf16> -> tensor<1x1000xbf16, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1, 1000], strides = [1, 1] : tensor<1x1000xbf16, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<1x1000xbf16,  #encoding>>
  return
}
// This tests that
//   1. The padding value is created for tensor.pack ops.
//   2. The inner tile sizes are less than or equal to values in round_dims_to.
//      We could choose 128 when it is a narrow matrix.
// CHECK-LABEL: func.func @set_encoding_with_padding_semantics_bf16_x86_64_avx512f
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[IN_BINDING:.+]] = hal.interface.binding.subspan {{.+}} : !flow.dispatch.tensor<readonly:tensor<1x1000xbf16>>
// CHECK-DAG:     %[[OUT_BINDING:.+]] = hal.interface.binding.subspan {{.+}} : !flow.dispatch.tensor<writeonly:tensor<1x1000x1x1xbf16>>
// CHECK:         %[[SRC:.+]] = flow.dispatch.tensor.load %[[IN_BINDING]]
// CHECK-DAG:     %[[INIT:.+]] = tensor.empty() : tensor<1x1000x1x1xbf16>
// CHECK:         %[[PACK:.+]] = tensor.pack %[[SRC]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [1, 1]
// CHECK-SAME:      into %[[INIT]] : tensor<1x1000xbf16> -> tensor<1x1000x1x1xbf16>
// CHECK:         flow.dispatch.tensor.store %[[PACK]], %[[OUT_BINDING]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @set_encoding_7x7x7_matmul_LHS() attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma"}>
} {
  %c0 = arith.constant 0 : index
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<7x7xf32>>
  %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<7x7xf32, #encoding>>
  %14 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [7, 7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7xf32>> -> tensor<7x7xf32>
  %17 = iree_encoding.set_encoding %14 : tensor<7x7xf32> -> tensor<7x7xf32, #encoding>
  flow.dispatch.tensor.store %17, %11, offsets = [0, 0], sizes = [7, 7], strides = [1, 1] : tensor<7x7xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<7x7xf32, #encoding>>
  return
}
// CHECK-LABEL:  func @set_encoding_7x7x7_matmul_LHS(
//   CHECK-DAG:    %[[CST:.+]] = arith.constant 0.0
//       CHECK:    %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} !flow.dispatch.tensor<readonly:tensor<7x7xf32>>
//       CHECK:    %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} !flow.dispatch.tensor<writeonly:tensor<1x7x8x1xf32>>
//       CHECK:    %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]], offsets = [0, 0], sizes = [7, 7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7xf32>> -> tensor<7x7xf32>
//       CHECK:    %[[EMPTY:.+]] = tensor.empty() : tensor<1x7x8x1xf32>
//       CHECK:    %[[PACK:.+]] = tensor.pack %[[INPUT]] padding_value(%[[CST]] : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<7x7xf32> -> tensor<1x7x8x1xf32>
//       CHECK:    flow.dispatch.tensor.store %[[PACK]], %[[OUTPUT_BINDING]], offsets = [0, 0, 0, 0], sizes = [1, 7, 8, 1], strides = [1, 1, 1, 1] : tensor<1x7x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x7x8x1xf32>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @set_encoding_128x80x32_batch_matmul_LHS() attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma"}>
} {
  %c0 = arith.constant 0 : index
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x80x32xf32>>
  %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x80x32xf32, #encoding>>
  %14 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0], sizes = [128, 80, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x80x32xf32>> -> tensor<128x80x32xf32>
  %17 = iree_encoding.set_encoding %14 : tensor<128x80x32xf32> -> tensor<128x80x32xf32, #encoding>
  flow.dispatch.tensor.store %17, %11, offsets = [0, 0, 0], sizes = [128, 80, 32], strides = [1, 1, 1]
    : tensor<128x80x32xf32, #encoding>
    -> !flow.dispatch.tensor<writeonly:tensor<128x80x32xf32, #encoding>>
  return
}
// CHECK-LABEL:    func @set_encoding_128x80x32_batch_matmul_LHS(
//       CHECK:      %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) {{.*}} !flow.dispatch.tensor<readonly:tensor<128x80x32xf32>>
//       CHECK:      %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) {{.*}} !flow.dispatch.tensor<writeonly:tensor<128x10x32x8x1xf32>>
//       CHECK:      %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]], offsets = [0, 0, 0], sizes = [128, 80, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x80x32xf32>> -> tensor<128x80x32xf32>
//       CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<128x10x32x8x1xf32>
//       CHECK:      %[[PACK:.+]] = tensor.pack %[[INPUT]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 1] into %[[EMPTY]] : tensor<128x80x32xf32> -> tensor<128x10x32x8x1xf32>
//       CHECK:      flow.dispatch.tensor.store %[[PACK]], %[[OUTPUT_BINDING]], offsets = [0, 0, 0, 0, 0], sizes = [128, 10, 32, 8, 1], strides = [1, 1, 1, 1, 1] : tensor<128x10x32x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x10x32x8x1xf32>>

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @set_encoding_128x32x320_batch_matmul_RHS() attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %5 = arith.index_castui %0 {stream.alignment = 64 : index} : i32 to index
  %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x32x320xf32>>
  %13 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<128x32x320xf32, #encoding>>
  %16 = flow.dispatch.tensor.load %10, offsets = [0, 0, 0], sizes = [128, 32, 320], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x32x320xf32>> -> tensor<128x32x320xf32>
  %19 = iree_encoding.set_encoding %16 : tensor<128x32x320xf32> -> tensor<128x32x320xf32, #encoding>
  flow.dispatch.tensor.store %19, %13, offsets = [0, 0, 0], sizes = [128, 32, 320], strides = [1, 1, 1]
    : tensor<128x32x320xf32, #encoding>
    -> !flow.dispatch.tensor<writeonly:tensor<128x32x320xf32, #encoding>>
  return
}
// CHECK-LABEL:    func @set_encoding_128x32x320_batch_matmul_RHS(
//       CHECK:      %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) {{.*}} !flow.dispatch.tensor<readonly:tensor<128x32x320xf32>>
//       CHECK:      %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) {{.*}} !flow.dispatch.tensor<writeonly:tensor<128x40x32x8x1xf32>>
//       CHECK:      %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]], offsets = [0, 0, 0], sizes = [128, 32, 320], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x32x320xf32>> -> tensor<128x32x320xf32>
//       CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<128x40x32x8x1xf32>
//       CHECK:      %[[PACK:.+]] = tensor.pack %[[INPUT]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [8, 1] into %[[EMPTY]] : tensor<128x32x320xf32> -> tensor<128x40x32x8x1xf32>
//       CHECK:      flow.dispatch.tensor.store %[[PACK]], %[[OUTPUT_BINDING]], offsets = [0, 0, 0, 0, 0], sizes = [128, 40, 32, 8, 1], strides = [1, 1, 1, 1, 1] : tensor<128x40x32x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x40x32x8x1xf32>>

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @unset_encoding_128x80x320_batch_matmul_RESULT() attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %3 = arith.index_castui %0 : i32 to index
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x80x320xf32>>
  %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x80x320xf32, #encoding>>
  %10 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0], sizes = [128, 80, 320], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<128x80x320xf32, #encoding>>
      -> tensor<128x80x320xf32, #encoding>
  %11 = iree_encoding.unset_encoding %10 : tensor<128x80x320xf32, #encoding> -> tensor<128x80x320xf32>
  flow.dispatch.tensor.store %11, %6, offsets = [0, 0, 0], sizes = [128, 80, 320], strides = [1, 1, 1] : tensor<128x80x320xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x80x320xf32>>
  return
}
// CHECK-LABEL: func @unset_encoding_128x80x320_batch_matmul_RESULT()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(0)
//       CHECK:   %[[CAST:.+]] = arith.index_castui %[[D0]] : i32 to index
//       CHECK:   %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(64) offset(%[[C0]])
//  CHECK-SAME:       : !flow.dispatch.tensor<writeonly:tensor<128x80x320xf32>>
//       CHECK:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(64) offset(%[[CAST]])
//  CHECK-SAME:       : !flow.dispatch.tensor<readonly:tensor<128x10x40x8x8xf32>>
//       CHECK:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0, 0], sizes = [128, 10, 40, 8, 8], strides = [1, 1, 1, 1, 1]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty()
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[INPUT]]
//  CHECK-SAME:       outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %[[EMPTY]]
//   CHECK-DAG:   flow.dispatch.tensor.store %[[UNPACK]], %[[OUTPUT_BINDING]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @pack_gemm_fill_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma"}>
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
//   CHECK-DAG:   %[[PACK_LHS:.+]] = tensor.pack {{.*}}%[[ARG0]]
//       CHECK:   %[[PACK_RHS:.+]] = tensor.pack
//  CHECK-SAME:     %[[ARG1]]
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[OUT_D0]], %[[OUT_D1]]) : tensor<?x?x8x8xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[EMPTY]] :
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]]
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], round_dims_to = array<i64: 16, 1, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], round_dims_to = array<i64: 16, 1, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], round_dims_to = array<i64: 16, 1, 16>>
func.func @matvec_shaped_matmul_lowering_f32f32f32_aarch64(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz"}>
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f32f32f32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf32, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
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
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], round_dims_to = array<i64: 16, 1, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], round_dims_to = array<i64: 16, 1, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], round_dims_to = array<i64: 16, 1, 16>>
func.func @matvec_lowering_f32f32f32_aarch64(%arg0: tensor<16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>) -> tensor<16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz"}>
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 1, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 1, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 1, 16>>
func.func @matvec_lowering_f32f32f32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<16x16xf32, #encoding_lhs>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<16x1xf32, #encoding_rhs>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<16x1xf32, #encoding_result>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<16x16xf32, #encoding_lhs>>
      -> tensor<16x16xf32, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<16x1xf32, #encoding_rhs>>
      -> tensor<16x1xf32, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<16x1xf32, #encoding_result>>
      -> tensor<16x1xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<16x16xf32, #encoding_lhs>,
                   tensor<16x1xf32, #encoding_rhs>)
      outs(%5 : tensor<16x1xf32, #encoding_result>)
      -> tensor<16x1xf32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : tensor<16x1xf32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<16x1xf32, #encoding_result>>
  return
}
// CHECK-LABEL: func @matvec_lowering_f32f32f32_aarch64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<2x16x8x1xf32>>
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<1x16x1x1xf32>>
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<1x2x1x8xf32>>
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [2, 16, 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 16, 1, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 2, 1, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[RHS]], %[[LHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f16f16f16_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf16, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf16, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
      -> tensor<?x?xf16, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf16, #encoding_result>)
      -> tensor<?x?xf16, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
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
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf16>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f32f32f32_x86_64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf32, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
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
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP1]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x4x1xf32>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x4xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 4, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 4], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 4], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f32f32f32_x86_64_avx2() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf32, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-LABEL: func @matmul_lowering_f32f32f32_x86_64_avx2()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f32f32f32_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf32, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func @matmul_lowering_f32f32f32_x86_64_avx512f()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f16f16f32_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf16, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf16, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func @matmul_lowering_f16f16f32_x86_64_avx512f()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf16>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf16>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f16f16f16_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf16, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf16, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
      -> tensor<?x?xf16, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf16, #encoding_result>)
      -> tensor<?x?xf16, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func @matmul_lowering_f16f16f16_x86_64_avx512f()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf16>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf16>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf16>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_bf16bf16f32_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xbf16, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xbf16, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #encoding_lhs>,
                   tensor<?x?xbf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func @matmul_lowering_bf16bf16f32_x86_64_avx512f()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xbf16>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xbf16>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_bf16bf16bf16_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xbf16, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xbf16, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #encoding_result>>{%M, %N}
      -> tensor<?x?xbf16, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #encoding_lhs>,
                   tensor<?x?xbf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xbf16, #encoding_result>)
      -> tensor<?x?xbf16, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xbf16, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func @matmul_lowering_bf16bf16bf16_x86_64_avx512f()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xbf16>>{%[[TILED_M]], %[[K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xbf16>>{%[[TILED_N]], %[[K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xbf16>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_bf16bf16f32_x86_64_avx512bf16() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f,+avx512bf16"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xbf16, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xbf16, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #encoding_lhs>,
                   tensor<?x?xbf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK-LABEL: func @matmul_lowering_bf16bf16f32_x86_64_avx512bf16()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xbf16>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xbf16>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_bf16bf16bf16_x86_64_avx512bf16() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f,+avx512bf16"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xbf16, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xbf16, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #encoding_result>>{%M, %N}
      -> tensor<?x?xbf16, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #encoding_lhs>,
                   tensor<?x?xbf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xbf16, #encoding_result>)
      -> tensor<?x?xbf16, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xbf16, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK-LABEL: func @matmul_lowering_bf16bf16bf16_x86_64_avx512bf16()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xbf16>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xbf16>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xbf16>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f32f16f16_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  %lhs_f32 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %rhs = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf16, #encoding_rhs>
  %dest = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
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
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
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
//   CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[M_CEILDIV_8]], %[[K]]}
//   CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[N_CEILDIV_8]], %[[K]]}
//   CHECK-DAG:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2) {{.*}} : !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf16>>{%[[M_CEILDIV_8]], %[[N_CEILDIV_8]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M_CEILDIV_8]], %[[K]], 8, 1], {{.*}} -> tensor<?x?x8x1xf32>
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[N_CEILDIV_8]], %[[K]], 8, 1], {{.*}} -> tensor<?x?x8x1xf16>
//       CHECK:   %[[OUT:.+]] = flow.dispatch.tensor.load %[[OUT_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M_CEILDIV_8]], %[[N_CEILDIV_8]], 8, 8], {{.*}} -> tensor<?x?x8x8xf16>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[M_CEILDIV_8]], %[[K]]) : tensor<?x?x8x1xf16>
//       CHECK:   %[[LHS_F16:.+]] = linalg.generic {indexing_maps = [#[[$MAP_IDENTITY_4D]], #[[$MAP_IDENTITY_4D]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS]] : tensor<?x?x8x1xf32>) outs(%[[EMPTY]] : tensor<?x?x8x1xf16>) {
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS_F16]], %[[RHS]] : tensor<?x?x8x1xf16>, tensor<?x?x8x1xf16>) outs(%[[OUT]] : tensor<?x?x8x8xf16>)
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUT_BINDING]],

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f32f16f16_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f,+avx512bf16"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  %lhs_f32 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %rhs = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf16, #encoding_rhs>
  %dest = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
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
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  return
}

//   CHECK-DAG: #[[$MAP_CEILDIV_16:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//   CHECK-DAG: #[[$MAP_IDENTITY_4D:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:     func.func @matmul_lowering_f32f16f16_x86_64_avx512f()
//   CHECK-DAG: %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0) : index
//   CHECK-DAG: %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1) : index
//   CHECK-DAG: %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2) : index
//   CHECK-DAG: %[[M_CEILDIV_16:.+]] = affine.apply #[[$MAP_CEILDIV_16]]()[%[[M]]]
//   CHECK-DAG: %[[N_CEILDIV_16:.+]] = affine.apply #[[$MAP_CEILDIV_16]]()[%[[N]]]
//   CHECK-DAG: %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%[[M_CEILDIV_16]], %[[K]]}
//   CHECK-DAG: %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf16>>{%[[N_CEILDIV_16]], %[[K]]}
//   CHECK-DAG: %[[OUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2) {{.*}} : !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf16>>{%[[M_CEILDIV_16]], %[[N_CEILDIV_16]]}
//       CHECK: %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M_CEILDIV_16]], %[[K]], 16, 1], {{.*}} -> tensor<?x?x16x1xf32>
//       CHECK: %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[N_CEILDIV_16]], %[[K]], 16, 1], {{.*}} -> tensor<?x?x16x1xf16>
//       CHECK: %[[OUT:.+]] = flow.dispatch.tensor.load %[[OUT_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M_CEILDIV_16]], %[[N_CEILDIV_16]], 16, 16], {{.*}} -> tensor<?x?x16x16xf16>
//       CHECK: %[[EMPTY:.+]] = tensor.empty(%[[M_CEILDIV_16]], %[[K]]) : tensor<?x?x16x1xf16>
//       CHECK: %[[LHS_F16:.+]] = linalg.generic {indexing_maps = [#[[$MAP_IDENTITY_4D]], #[[$MAP_IDENTITY_4D]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS]] : tensor<?x?x16x1xf32>) outs(%[[EMPTY]] : tensor<?x?x16x1xf16>) {
//       CHECK-DAG: %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS_F16]], %[[RHS]] : tensor<?x?x16x1xf16>, tensor<?x?x16x1xf16>) outs(%[[OUT]] : tensor<?x?x16x16xf16>)
//       CHECK: flow.dispatch.tensor.store %[[MMT4D]], %[[OUT_BINDING]],

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i8i8i32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_aarch64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?xi8>>{%[[M]], %[[K]]}
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?xi8>>{%[[K]], %[[N]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%[[M]], %[[N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[K]]], strides = [1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[K]], %[[N]]], strides = [1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[N]]], strides = [1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i8i8i32_aarch64_dotprod() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
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
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i8i8i32_aarch64_i8mm() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+i8mm", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
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
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i8i4i32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi4, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi4, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
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
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x4x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP2]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi4>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x4x16xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 4, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i8i4i32_aarch64_dotprod() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi4, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi4, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
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
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x8xi4>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i8i4i32_aarch64_i8mm() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+i8mm", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi4, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi4, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
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
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x4x16xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP2]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x16xi4>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x4x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 4, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f32f32f32_aarch64_sve(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+sve", target_triple="aarch64-xyz-xyz"}>
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

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_f32f32f32_riscv(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv32-xyz-xyz"}>
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i8i8i32_riscv32_ukernel() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv32-xyz-xyz", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
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
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i8i8i32_x86_64_avx2() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx2"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK-LABEL: func @matmul_lowering_i8i8i32_x86_64_avx2()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x2xi8>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i8i8i32_x86_64_avx512bw() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512bw"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK-LABEL: func @matmul_lowering_i8i8i32_x86_64_avx512bw()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi8>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i8i8i32_x86_64_avx512vnni() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK-LABEL: func @matmul_lowering_i8i8i32_x86_64_avx512vnni()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi8>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 16, 16>>
func.func @extend_batch_vecmat_explicit_unit_dim(%arg0: tensor<32x1x128xi8>, %arg1: tensor<32x128x11008xi8>) -> tensor<32x1x11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
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
//       CHECK:   %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [1, 2] into %[[INIT_LHS_PACK]] : tensor<32x1x128xi8> -> tensor<32x1x64x1x2xi8>
//       CHECK:   %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<32x1x64x1x2xi32>
//       CHECK:   %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<32x1x64x1x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<32x1x64x1x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[LHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<32x688x64x16x2xi8>
//       CHECK:   %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %[[INIT_RHS_PACK]] : tensor<32x128x11008xi8> -> tensor<32x688x64x16x2xi8>
//       CHECK:   %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<32x688x64x16x2xi32>
//       CHECK:   %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<32x688x64x16x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<32x688x64x16x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[RHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_FILL:.+]] = tensor.empty() : tensor<32x1x688x1x16xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[INIT_FILL]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[LHS_EXT]], %[[RHS_EXT]] : tensor<32x1x64x1x2xi32>, tensor<32x688x64x16x2xi32>) outs(%[[FILL]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<32x1x11008xi32>
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [1, 16] into %[[INIT_UNPACK]] : tensor<32x1x688x1x16xi32> -> tensor<32x1x11008xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_lowering_i16i16i32_x86_64_avx2() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx2"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi16, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi16, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi16, #encoding_lhs>,
                   tensor<?x?xi16, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK-LABEL: func @matmul_lowering_i16i16i32_x86_64_avx2()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x2xi16>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x2xi16>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
func.func @matmul_lowering_i16ui4i32_x86_64_avx512vnni() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %lhs_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #encoding_lhs>>{%M, %K}
  %rhs_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
  %out_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi16, #encoding_lhs>
  %rhs_i4 = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi4, #encoding_rhs>
  %empty = tensor.empty(%K, %N) : tensor<?x?xi32, #encoding_rhs>
  %rhs_i32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
     ins(%rhs_i4 : tensor<?x?xi4, #encoding_rhs>) outs(%empty : tensor<?x?xi32, #encoding_rhs>) {
  ^bb0(%in: i4, %out: i32):
    %17 = arith.extui %in : i4 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?xi32, #encoding_rhs>
  %out = flow.dispatch.tensor.load %out_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %result = linalg.matmul
      ins(%lhs, %rhs_i32 : tensor<?x?xi16, #encoding_lhs>,
                   tensor<?x?xi32, #encoding_rhs>)
      outs(%out : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %result, %out_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}

//   CHECK-DAG: #[[$MAP_CEILDIV_8:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[$MAP_CEILDIV_32:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//   CHECK-DAG: #[[$MAP_IDENTITY_4D:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @matmul_lowering_i16ui4i32_x86_64_avx512vnni()
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0) : index
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1) : index
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2) : index
//   CHECK-DAG:   %[[K_CEILDIV_8:.+]] = affine.apply #[[$MAP_CEILDIV_8]]()[%[[K]]]
//   CHECK-DAG:   %[[N_CEILDIV_32:.+]] = affine.apply #[[$MAP_CEILDIV_32]]()[%[[N]]]
//   CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x1x8xi16>>{%[[M]], %[[K_CEILDIV_8]]}
//   CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x32x8xi4>>{%[[N_CEILDIV_32]], %[[K_CEILDIV_8]]}
//   CHECK-DAG:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2) {{.*}} : !flow.dispatch.tensor<readwrite:tensor<?x?x1x32xi32>>{%[[M]], %[[N_CEILDIV_32]]}
//   CHECK-DAG:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M]], %[[K_CEILDIV_8]], 1, 8], {{.*}} -> tensor<?x?x1x8xi16>
//   CHECK-DAG:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[N_CEILDIV_32]], %[[K_CEILDIV_8]], 32, 8], {{.*}} -> tensor<?x?x32x8xi4>
//   CHECK-DAG:   %[[OUT:.+]] = flow.dispatch.tensor.load %[[OUT_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M]], %[[N_CEILDIV_32]], 1, 32], {{.*}} -> tensor<?x?x1x32xi32>
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[N_CEILDIV_32]], %[[K_CEILDIV_8]]) : tensor<?x?x32x8xi32>
//   CHECK-DAG:   %[[RHS_I32:.+]] = linalg.generic {indexing_maps = [#[[$MAP_IDENTITY_4D]], #[[$MAP_IDENTITY_4D]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS]] : tensor<?x?x32x8xi4>) outs(%[[EMPTY]] : tensor<?x?x32x8xi32>) {
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS]], %[[RHS_I32]] : tensor<?x?x1x8xi16>, tensor<?x?x32x8xi32>) outs(%[[OUT]] : tensor<?x?x1x32xi32>) -> tensor<?x?x1x32xi32>
//       CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUT_BINDING]],

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 16, 16>>
func.func @vecmat(%arg0: tensor<128xi8>, %arg1: tensor<128x11008xi8>) -> tensor<11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
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
//       CHECK:   %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [2] into %[[INIT_LHS_PACK]] : tensor<128xi8> -> tensor<64x2xi8>
//       CHECK:   %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<64x2xi32>
//       CHECK:   %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<64x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<64x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[LHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<688x64x16x2xi8>
//       CHECK:   %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %[[INIT_RHS_PACK]] : tensor<128x11008xi8> -> tensor<688x64x16x2xi8>
//       CHECK:   %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<688x64x16x2xi32>
//       CHECK:   %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<688x64x16x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<688x64x16x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[RHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_FILL:.+]] = tensor.empty() : tensor<688x16xi32>
//       CHECK:   %[[EXPAND_LHS:.+]] = tensor.expand_shape %[[LHS_EXT]] {{\[}}[0, 1], [2, 3]] output_shape [1, 64, 1, 2] : tensor<64x2xi32> into tensor<1x64x1x2xi32>
//       CHECK:   %[[EXPAND_INIT:.+]] = tensor.expand_shape %[[INIT_FILL:.+]] {{\[}}[0, 1], [2, 3]] output_shape [1, 688, 1, 16] : tensor<688x16xi32> into tensor<1x688x1x16xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EXPAND_INIT]] : tensor<1x688x1x16xi32>) -> tensor<1x688x1x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXPAND_LHS]], %[[RHS_EXT]] : tensor<1x64x1x2xi32>, tensor<688x64x16x2xi32>) outs(%[[FILL]] : tensor<1x688x1x16xi32>) -> tensor<1x688x1x16xi32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0, 1], [2, 3]] : tensor<1x688x1x16xi32> into tensor<688x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<11008xi32>
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[COLLAPSED]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<688x16xi32> -> tensor<11008xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 1, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 1, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 1, 16>>
func.func @matvec(%arg0: tensor<11008x128xi8>, %arg1: tensor<128xi8>) -> tensor<11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
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
//       CHECK:   %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 2] into %[[INIT_LHS_PACK]] : tensor<11008x128xi8> -> tensor<688x64x16x2xi8>
//       CHECK:   %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<688x64x16x2xi32>
//       CHECK:   %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<688x64x16x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<688x64x16x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[LHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<64x2xi8>
//       CHECK:   %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [2] into %[[INIT_RHS_PACK]] : tensor<128xi8> -> tensor<64x2xi8>
//       CHECK:   %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<64x2xi32>
//       CHECK:   %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<64x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<64x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[RHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_FILL:.+]] = tensor.empty() : tensor<688x16xi32>
//       CHECK:   %[[EXPAND_RHS:.+]] = tensor.expand_shape %[[RHS_EXT]] {{\[}}[0, 1], [2, 3]] output_shape [1, 64, 1, 2] : tensor<64x2xi32> into tensor<1x64x1x2xi32>
//       CHECK:   %[[EXPAND_INIT:.+]] = tensor.expand_shape %[[INIT_FILL:.+]] {{\[}}[0, 1], [2, 3]] output_shape [1, 688, 1, 16] : tensor<688x16xi32> into tensor<1x688x1x16xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EXPAND_INIT]] : tensor<1x688x1x16xi32>) -> tensor<1x688x1x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXPAND_RHS]], %[[LHS_EXT]]  : tensor<1x64x1x2xi32>, tensor<688x64x16x2xi32>) outs(%[[FILL]] : tensor<1x688x1x16xi32>) -> tensor<1x688x1x16xi32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0, 1], [2, 3]] : tensor<1x688x1x16xi32> into tensor<688x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<11008xi32>
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[COLLAPSED]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<688x16xi32> -> tensor<11008xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 1, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 1, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 1, 16>>
func.func @matvec_with_narrow_M(%arg0: tensor<15x128xi8>, %arg1: tensor<128xi8>) -> tensor<15xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
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
//       CHECK:   %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] padding_value(%[[C0_I8]] : i8) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 2] into %[[INIT_LHS_PACK]] : tensor<15x128xi8> -> tensor<1x64x16x2xi8>
//       CHECK:   %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<1x64x16x2xi32>
//       CHECK:   %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<1x64x16x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<1x64x16x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[LHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<64x2xi8>
//       CHECK:   %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [2] into %[[INIT_RHS_PACK]] : tensor<128xi8> -> tensor<64x2xi8>
//       CHECK:   %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<64x2xi32>
//       CHECK:   %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<64x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<64x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[RHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_FILL:.+]] = tensor.empty() : tensor<1x16xi32>
//       CHECK:   %[[EXPAND_RHS:.+]] = tensor.expand_shape %[[RHS_EXT]] {{\[}}[0, 1], [2, 3]] output_shape [1, 64, 1, 2] : tensor<64x2xi32> into tensor<1x64x1x2xi32>
//       CHECK:   %[[EXPAND_INIT:.+]] = tensor.expand_shape %[[INIT_FILL:.+]] {{\[}}[0, 1], [2, 3]] output_shape [1, 1, 1, 16] : tensor<1x16xi32> into tensor<1x1x1x16xi32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EXPAND_INIT]] : tensor<1x1x1x16xi32>) -> tensor<1x1x1x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXPAND_RHS]], %[[LHS_EXT]]  : tensor<1x64x1x2xi32>, tensor<1x64x16x2xi32>) outs(%[[FILL]] : tensor<1x1x1x16xi32>) -> tensor<1x1x1x16xi32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0, 1], [2, 3]] : tensor<1x1x1x16xi32> into tensor<1x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<15xi32>
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[COLLAPSED]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<1x16xi32> -> tensor<15xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 16, 16>>
func.func @batch_vecmat(%arg0: tensor<32x128xi8>, %arg1: tensor<32x128x11008xi8>) -> tensor<32x11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
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
//       CHECK:   %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [2] into %[[INIT_LHS_PACK]] : tensor<32x128xi8> -> tensor<32x64x2xi8>
//       CHECK:   %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<32x64x2xi32>
//       CHECK:   %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<32x64x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<32x64x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[LHS_EXT_OP]] : i32
//       CHECK:   %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<32x688x64x16x2xi8>
//       CHECK:   %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %[[INIT_RHS_PACK]] : tensor<32x128x11008xi8> -> tensor<32x688x64x16x2xi8>
//       CHECK:   %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<32x688x64x16x2xi32>
//       CHECK:   %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<32x688x64x16x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<32x688x64x16x2xi32>) {
//  CHECK-NEXT:       ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
//  CHECK-NEXT:       %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
//  CHECK-NEXT:       linalg.yield %[[RHS_EXT_OP]] : i32
//   CHECK-DAG:   %[[INIT_FILL:.+]] = tensor.empty() : tensor<32x688x16xi32>
//   CHECK-DAG:   %[[EXPAND_LHS:.+]] = tensor.expand_shape %[[LHS_EXT]] {{\[}}[0], [1, 2], [3, 4]] output_shape [32, 1, 64, 1, 2] : tensor<32x64x2xi32> into tensor<32x1x64x1x2xi32>
//   CHECK-DAG:   %[[EXPAND_INIT:.+]] = tensor.expand_shape %[[INIT_FILL:.+]] {{\[}}[0], [1, 2], [3, 4]] output_shape [32, 1, 688, 1, 16] : tensor<32x688x16xi32> into tensor<32x1x688x1x16xi32>
//   CHECK-DAG:   %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EXPAND_INIT]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//       CHECK:   %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[EXPAND_LHS]], %[[RHS_EXT]] : tensor<32x1x64x1x2xi32>, tensor<32x688x64x16x2xi32>) outs(%[[FILL]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0], [1, 2], [3, 4]] : tensor<32x1x688x1x16xi32> into tensor<32x688x16xi32>
//       CHECK:   %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<32x11008xi32>
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[COLLAPSED]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<32x688x16xi32> -> tensor<32x11008xi32>
//       CHECK:   return %[[UNPACK]]

// -----

#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], round_dims_to = array<i64: 16, 1, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], round_dims_to = array<i64: 16, 1, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], round_dims_to = array<i64: 16, 1, 16>>
func.func @batch_matvec(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<32x11008x128xi8>
  %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<32x128xi8>
  %2 = hal.tensor.import %arg2 "input2" : !hal.buffer_view -> tensor<32x11008xi32>
  %3 = iree_encoding.set_encoding %0 : tensor<32x11008x128xi8> -> tensor<32x11008x128xi8, #encoding_lhs>
  %4 = iree_encoding.set_encoding %1 : tensor<32x128xi8> -> tensor<32x128xi8, #encoding_rhs>
  %5 = iree_encoding.set_encoding %2 : tensor<32x11008xi32> -> tensor<32x11008xi32, #encoding_result>
  %6 = linalg.batch_matvec ins(%3, %4 : tensor<32x11008x128xi8, #encoding_lhs>, tensor<32x128xi8, #encoding_rhs>) outs(%5 : tensor<32x11008xi32, #encoding_result>) -> tensor<32x11008xi32, #encoding_result>
  %7 = iree_encoding.unset_encoding %6 : tensor<32x11008xi32, #encoding_result> -> tensor<32x11008xi32>
  %8 = hal.tensor.export %7 "output0" : tensor<32x11008xi32> -> !hal.buffer_view
  func.return %8 : !hal.buffer_view
}

// CHECK-LABEL: func.func @batch_matvec(

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_transpose_a_f32f32f32(%arg0: tensor<256x128xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c512 = arith.constant 512 : index
  %6 = iree_encoding.set_encoding %arg0 : tensor<256x128xf32> -> tensor<256x128xf32, #encoding_lhs>
  %10 = iree_encoding.set_encoding %arg1 : tensor<256x512xf32> -> tensor<256x512xf32, #encoding_rhs>
  %14 = iree_encoding.set_encoding %arg2 : tensor<128x512xf32> -> tensor<128x512xf32, #encoding_result>
  %15 = linalg.matmul_transpose_a ins(%6, %10 : tensor<256x128xf32, #encoding_lhs>, tensor<256x512xf32, #encoding_rhs>) outs(%14 : tensor<128x512xf32, #encoding_result>) -> tensor<128x512xf32, #encoding_result>
  %16 = iree_encoding.unset_encoding %15 : tensor<128x512xf32, #encoding_result> -> tensor<128x512xf32>
  return %16 : tensor<128x512xf32>
}

// CHECK-LABEL: func.func @matmul_transpose_a_f32f32f32(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<256x128xf32>, %[[RHS:.+]]: tensor<256x512xf32>, %[[RESULT:.+]]: tensor<128x512xf32>) -> tensor<128x512xf32>
//       CHECK:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<16x256x8x1xf32>
//       CHECK:   %[[PACK_LHS:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<256x128xf32> -> tensor<16x256x8x1xf32>
//       CHECK:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<128x256x4x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<256x512xf32> -> tensor<128x256x4x1xf32>
//       CHECK:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<16x128x8x4xf32>
//       CHECK:   %[[PACK_RES:.+]] = tensor.pack %[[RESULT]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<128x512xf32> -> tensor<16x128x8x4xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
//  CHECK-SAME:       outs(%[[PACK_RES]] :
//   CHECK-DAG:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<128x512xf32>
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<16x128x8x4xf32> -> tensor<128x512xf32>
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @matmul_transpose_b_f32f32f32(%arg0: tensor<128x256xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c512 = arith.constant 512 : index
  %6 = iree_encoding.set_encoding %arg0 : tensor<128x256xf32> -> tensor<128x256xf32, #encoding_lhs>
  %10 = iree_encoding.set_encoding %arg1 : tensor<512x256xf32> -> tensor<512x256xf32, #encoding_rhs>
  %14 = iree_encoding.set_encoding %arg2 : tensor<128x512xf32> -> tensor<128x512xf32, #encoding_result>
  %15 = linalg.matmul_transpose_b ins(%6, %10 : tensor<128x256xf32, #encoding_lhs>, tensor<512x256xf32, #encoding_rhs>) outs(%14 : tensor<128x512xf32, #encoding_result>) -> tensor<128x512xf32, #encoding_result>
  %16 = iree_encoding.unset_encoding %15 : tensor<128x512xf32, #encoding_result> -> tensor<128x512xf32>
  return %16 : tensor<128x512xf32>
}

// CHECK-LABEL: func.func @matmul_transpose_b_f32f32f32(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<128x256xf32>, %[[RHS:.+]]: tensor<512x256xf32>, %[[RESULT:.+]]: tensor<128x512xf32>) -> tensor<128x512xf32>
//       CHECK:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<16x256x8x1xf32>
//       CHECK:   %[[PACK_LHS:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<128x256xf32> -> tensor<16x256x8x1xf32>
//       CHECK:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<128x256x4x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<512x256xf32> -> tensor<128x256x4x1xf32>
//       CHECK:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<16x128x8x4xf32>
//       CHECK:   %[[PACK_RES:.+]] = tensor.pack %[[RESULT]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<128x512xf32> -> tensor<16x128x8x4xf32>
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
//  CHECK-SAME:       outs(%[[PACK_RES]] :
//       CHECK:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<128x512xf32>
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<16x128x8x4xf32> -> tensor<128x512xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @batch_matmul_transpose_a_f32f32f32(%arg0: tensor<2x256x128xf32>, %arg1: tensor<2x256x512xf32>, %arg2: tensor<2x128x512xf32>) -> tensor<2x128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c2 = arith.constant 2 : index
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c512 = arith.constant 512 : index
  %7 = iree_encoding.set_encoding %arg0 : tensor<2x256x128xf32> -> tensor<2x256x128xf32, #encoding_lhs>
  %12 = iree_encoding.set_encoding %arg1 : tensor<2x256x512xf32> -> tensor<2x256x512xf32, #encoding_rhs>
  %17 = iree_encoding.set_encoding %arg2 : tensor<2x128x512xf32> -> tensor<2x128x512xf32, #encoding_result>
  %18 = linalg.batch_matmul_transpose_a ins(%7, %12 : tensor<2x256x128xf32, #encoding_lhs>, tensor<2x256x512xf32, #encoding_rhs>) outs(%17 : tensor<2x128x512xf32, #encoding_result>) -> tensor<2x128x512xf32, #encoding_result>
  %19 = iree_encoding.unset_encoding %18 : tensor<2x128x512xf32, #encoding_result> -> tensor<2x128x512xf32>
  return %19 : tensor<2x128x512xf32>
}

// CHECK-LABEL: func.func @batch_matmul_transpose_a_f32f32f32(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<2x256x128xf32>, %[[RHS:.+]]: tensor<2x256x512xf32>, %[[RESULT:.+]]: tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
//       CHECK:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<2x16x256x8x1xf32>
//       CHECK:   %[[PACK_LHS:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<2x256x128xf32> -> tensor<2x16x256x8x1xf32>
//       CHECK:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<2x128x256x4x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<2x256x512xf32> -> tensor<2x128x256x4x1xf32>
//       CHECK:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<2x16x128x8x4xf32>
//       CHECK:   %[[PACK_RES:.+]] = tensor.pack %[[RESULT]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<2x128x512xf32> -> tensor<2x16x128x8x4xf32>
//       CHECK:   %[[BATCH_MMT4D:.+]] = linalg.batch_mmt4d
//       CHECK:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
//       CHECK:       outs(%[[PACK_RES]] :
//       CHECK:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<2x128x512xf32>
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[BATCH_MMT4D]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<2x16x128x8x4xf32> -> tensor<2x128x512xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
func.func @batch_matmul_transpose_b_f32f32f32(%arg0: tensor<2x128x256xf32>, %arg1: tensor<2x512x256xf32>, %arg2: tensor<2x128x512xf32>) -> tensor<2x128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c512 = arith.constant 512 : index
  %7 = iree_encoding.set_encoding %arg0 : tensor<2x128x256xf32> -> tensor<2x128x256xf32, #encoding_lhs>
  %12 = iree_encoding.set_encoding %arg1 : tensor<2x512x256xf32> -> tensor<2x512x256xf32, #encoding_rhs>
  %17 = iree_encoding.set_encoding %arg2 : tensor<2x128x512xf32> -> tensor<2x128x512xf32, #encoding_result>
  %18 = linalg.batch_matmul_transpose_b ins(%7, %12 : tensor<2x128x256xf32, #encoding_lhs>, tensor<2x512x256xf32, #encoding_rhs>) outs(%17 : tensor<2x128x512xf32, #encoding_result>) -> tensor<2x128x512xf32, #encoding_result>
  %19 = iree_encoding.unset_encoding %18 : tensor<2x128x512xf32, #encoding_result> -> tensor<2x128x512xf32>
  return %19 : tensor<2x128x512xf32>
}

// CHECK-LABEL: func.func @batch_matmul_transpose_b_f32f32f32(
//  CHECK-SAME:   %[[LHS:.+]]: tensor<2x128x256xf32>, %[[RHS:.+]]: tensor<2x512x256xf32>, %[[RESULT:.+]]: tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
//       CHECK:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<2x16x256x8x1xf32>
//       CHECK:   %[[PACK_LHS:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<2x128x256xf32> -> tensor<2x16x256x8x1xf32>
//       CHECK:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<2x128x256x4x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<2x512x256xf32> -> tensor<2x128x256x4x1xf32>
//       CHECK:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<2x16x128x8x4xf32>
//       CHECK:   %[[PACK_RES:.+]] = tensor.pack %[[RESULT]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<2x128x512xf32> -> tensor<2x16x128x8x4xf32>
//       CHECK:   %[[BATCH_MMT4D:.+]] = linalg.batch_mmt4d
//  CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
//  CHECK-SAME:       outs(%[[PACK_RES]] :
//   CHECK-DAG:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<2x128x512xf32>
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[BATCH_MMT4D]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<2x16x128x8x4xf32> -> tensor<2x128x512xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 32, 32>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 32, 32>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 32, 32>>
func.func @generic_batch_vecmat_transposed_i16u4i32(%arg0: tensor<32x128xi16>, %arg1: tensor<4096x32x128xi4>, %arg2: tensor<4096x32xi32>) -> tensor<4096x32xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c0_i32 = arith.constant 0 : i32
  %c0_i4 = arith.constant 0 : i4
  %c4096 = arith.constant 4096 : index
  %c0_i16 = arith.constant 0 : i16
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
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
//   CHECK-DAG:   %[[PACK_LHS:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [8] into %[[PACK_LHS_DEST]] : tensor<32x128xi16> -> tensor<32x16x8xi16>
//   CHECK-DAG:   %[[EXPAND_LHS:.+]] = tensor.expand_shape %[[PACK_LHS]] {{.*}} output_shape [32, 1, 16, 1, 8] : tensor<32x16x8xi16> into tensor<32x1x16x1x8xi16>
//   CHECK-DAG:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<32x128x16x32x8xi4>
//   CHECK-DAG:   %[[PACK_RHS:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [1, 0, 2] inner_dims_pos = [0, 2] inner_tiles = [32, 8] into %[[PACK_RHS_DEST]] : tensor<4096x32x128xi4> -> tensor<32x128x16x32x8xi4>
//   CHECK-DAG:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<32x128x32xi32>
//   CHECK-DAG:   %[[PACK_RES:.+]] = tensor.pack %[[RESULT]] outer_dims_perm = [1, 0] inner_dims_pos = [0] inner_tiles = [32] into %[[PACK_RES_DEST]] : tensor<4096x32xi32> -> tensor<32x128x32xi32>
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
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[COLLAPSE]] outer_dims_perm = [1, 0] inner_dims_pos = [0] inner_tiles = [32] into %[[UNPACK_DEST]] : tensor<32x128x32xi32> -> tensor<4096x32xi32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], bcast_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, round_dims_to = array<i64: 16, 16, 16>>
#encoding_bcast = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], bcast_map = affine_map<(d0, d1, d2) -> (d0, d2)>, round_dims_to = array<i64: 16, 16, 16>>
func.func @dequantization() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128x64xi8, #encoding>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x64xf32, #encoding_bcast>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x64xf32, #encoding_bcast>>
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x128x64xf32, #encoding>>
  %7 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 128, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x64xi8, #encoding>> -> tensor<2x128x64xi8, #encoding>
  %8 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x64xf32, #encoding_bcast>> -> tensor<2x64xf32, #encoding_bcast>
  %9 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [2, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x64xf32, #encoding_bcast>> -> tensor<2x64xf32, #encoding_bcast>
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %8, %9 : tensor<2x128x64xi8, #encoding>, tensor<2x64xf32, #encoding_bcast>, tensor<2x64xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %21 = arith.extui %in : i8 to i32
    %22 = arith.uitofp %21 : i32 to f32
    %23 = arith.subf %22, %in_1 : f32
    %24 = arith.mulf %23, %in_0 : f32
    linalg.yield %24 : f32
  } -> tensor<2x128x64xf32, #encoding>
  flow.dispatch.tensor.store %14, %6, offsets = [0, 0, 0], sizes = [2, 128, 64], strides = [1, 1, 1] : tensor<2x128x64xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<2x128x64xf32, #encoding>>
  return
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
// CHECK-LABEL: func.func @dequantization()
//   CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<2x4x128x16x1xi8>>
//   CHECK-DAG:   %[[LHS_SCALES_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !flow.dispatch.tensor<readonly:tensor<2x4x16xf32>>
//   CHECK-DAG:   %[[LHS_ZPS_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(2) {{.*}} : !flow.dispatch.tensor<readonly:tensor<2x4x16xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(3) {{.*}} : !flow.dispatch.tensor<writeonly:tensor<2x4x128x16x1xf32>>
//   CHECK-DAG:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]], offsets = [0, 0, 0, 0, 0], sizes = [2, 4, 128, 16, 1], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x128x16x1xi8>> -> tensor<2x4x128x16x1xi8>
//   CHECK-DAG:   %[[LHS_SCALES:.+]] = flow.dispatch.tensor.load %[[LHS_SCALES_BINDING]], offsets = [0, 0, 0], sizes = [2, 4, 16], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x16xf32>> -> tensor<2x4x16xf32>
//   CHECK-DAG:   %[[LHS_ZPS:.+]] = flow.dispatch.tensor.load %[[LHS_ZPS_BINDING]], offsets = [0, 0, 0], sizes = [2, 4, 16], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x16xf32>> -> tensor<2x4x16xf32>
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
//       CHECK:   flow.dispatch.tensor.store %[[LHS_DEQUANT]], %[[RESULT_BINDING]], offsets = [0, 0, 0, 0, 0], sizes = [2, 4, 128, 16, 1], strides = [1, 1, 1, 1, 1] : tensor<2x4x128x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x4x128x16x1xf32>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], bcast_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, round_dims_to = array<i64: 16, 16, 16>>
#encoding_bcast = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], bcast_map = affine_map<(d0, d1, d2) -> (d1, d2)>, round_dims_to = array<i64: 16, 16, 16>>
func.func @broadcast_batch() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x64xf32, #encoding_bcast>>
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x128x64xf32, #encoding>>
  %8 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x64xf32, #encoding_bcast>> -> tensor<128x64xf32, #encoding_bcast>
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<128x64xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x128x64xf32, #encoding>
  flow.dispatch.tensor.store %14, %6, offsets = [0, 0, 0], sizes = [2, 128, 64], strides = [1, 1, 1] : tensor<2x128x64xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<2x128x64xf32, #encoding>>
  return
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @broadcast_batch()
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<4x128x16x1xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !flow.dispatch.tensor<writeonly:tensor<2x4x128x16x1xf32>>
//   CHECK-DAG:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]], offsets = [0, 0, 0, 0], sizes = [4, 128, 16, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x128x16x1xf32>> -> tensor<4x128x16x1xf32>
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x4x128x16x1xf32>
//   CHECK-DAG:   %[[BROADCAST:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<4x128x16x1xf32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<2x4x128x16x1xf32>)
//       CHECK:   flow.dispatch.tensor.store %[[BROADCAST]], %[[RESULT_BINDING]], offsets = [0, 0, 0, 0, 0], sizes = [2, 4, 128, 16, 1], strides = [1, 1, 1, 1, 1] : tensor<2x4x128x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x4x128x16x1xf32>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], bcast_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, round_dims_to = array<i64: 16, 16, 16>>
#encoding_bcast = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], bcast_map = affine_map<(d0, d1, d2) -> (d0, d1)>, round_dims_to = array<i64: 16, 16, 16>>
func.func @broadcast_M() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x128xf32, #encoding_bcast>>
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x128x64xf32, #encoding>>
  %8 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128xf32, #encoding_bcast>> -> tensor<2x128xf32, #encoding_bcast>
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<2x128xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x128x64xf32, #encoding>
  flow.dispatch.tensor.store %14, %6, offsets = [0, 0, 0], sizes = [2, 128, 64], strides = [1, 1, 1] : tensor<2x128x64xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<2x128x64xf32, #encoding>>
  return
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @broadcast_M()
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<2x128x1xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !flow.dispatch.tensor<writeonly:tensor<2x4x128x16x1xf32>>
//   CHECK-DAG:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]], offsets = [0, 0, 0], sizes = [2, 128, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x128x1xf32>> -> tensor<2x128x1xf32>
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x4x128x16x1xf32>
//   CHECK-DAG:   %[[BROADCAST:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<2x128x1xf32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<2x4x128x16x1xf32>)
//       CHECK:   flow.dispatch.tensor.store %[[BROADCAST]], %[[RESULT_BINDING]], offsets = [0, 0, 0, 0, 0], sizes = [2, 4, 128, 16, 1], strides = [1, 1, 1, 1, 1] : tensor<2x4x128x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x4x128x16x1xf32>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], bcast_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, round_dims_to = array<i64: 16, 16, 16>>
#encoding_bcast = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], bcast_map = affine_map<(d0, d1, d2) -> (d0, d2)>, round_dims_to = array<i64: 16, 16, 16>>
func.func @broadcast_N() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x64xf32, #encoding_bcast>>
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x128x64xf32, #encoding>>
  %8 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x64xf32, #encoding_bcast>> -> tensor<2x64xf32, #encoding_bcast>
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<2x64xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x128x64xf32, #encoding>
  flow.dispatch.tensor.store %14, %6, offsets = [0, 0, 0], sizes = [2, 128, 64], strides = [1, 1, 1] : tensor<2x128x64xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<2x128x64xf32, #encoding>>
  return
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @broadcast_N()
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<2x64x1xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !flow.dispatch.tensor<writeonly:tensor<2x8x64x16x1xf32>>
//   CHECK-DAG:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]], offsets = [0, 0, 0], sizes = [2, 64, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x64x1xf32>> -> tensor<2x64x1xf32>
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x8x64x16x1xf32>
//   CHECK-DAG:   %[[BROADCAST:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<2x64x1xf32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<2x8x64x16x1xf32>)
//       CHECK:   flow.dispatch.tensor.store %[[BROADCAST]], %[[RESULT_BINDING]], offsets = [0, 0, 0, 0, 0], sizes = [2, 8, 64, 16, 1], strides = [1, 1, 1, 1, 1] : tensor<2x8x64x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x8x64x16x1xf32>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], bcast_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, round_dims_to = array<i64: 16, 16, 16>>
#encoding_bcast = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], bcast_map = affine_map<(d0, d1, d2) -> (d0, d2)>, round_dims_to = array<i64: 16, 16, 16>>
func.func @broadcast_K() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x64xf32, #encoding_bcast>>
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x128x64xf32, #encoding>>
  %8 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x64xf32, #encoding_bcast>> -> tensor<2x64xf32, #encoding_bcast>
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<2x64xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x128x64xf32, #encoding>
  flow.dispatch.tensor.store %14, %6, offsets = [0, 0, 0], sizes = [2, 128, 64], strides = [1, 1, 1] : tensor<2x128x64xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<2x128x64xf32, #encoding>>
  return
}

//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @broadcast_K()
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<2x4x16xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !flow.dispatch.tensor<writeonly:tensor<2x4x128x16x1xf32>>
//   CHECK-DAG:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]], offsets = [0, 0, 0], sizes = [2, 4, 16], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x16xf32>> -> tensor<2x4x16xf32>
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x4x128x16x1xf32>
//   CHECK-DAG:   %[[BROADCAST:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[INPUT]] : tensor<2x4x16xf32>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<2x4x128x16x1xf32>)
//       CHECK:   flow.dispatch.tensor.store %[[BROADCAST]], %[[RESULT_BINDING]], offsets = [0, 0, 0, 0, 0], sizes = [2, 4, 128, 16, 1], strides = [1, 1, 1, 1, 1] : tensor<2x4x128x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x4x128x16x1xf32>>

// -----

//----------------------------------------------------------------------------//
// Test suite for encodings with resolved layouts.
//----------------------------------------------------------------------------//

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f", encoding = #iree_cpu.cpu_encoding_layout<>}>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], layouts = [#iree_cpu.cpu_encoding_layout<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [1, 1], outerDimsPerm = [0, 1]}}>]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding1 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 32, 32>>
func.func @set_encoding_LHS_with_layout() attributes {
  hal.executable.target = #executable_target
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<1x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<1x256xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x256xf32>> -> tensor<1x256xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<1x256xf32> -> tensor<1x256xf32, #encoding1>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : tensor<1x256xf32, #encoding1> -> !flow.dispatch.tensor<writeonly:tensor<1x256xf32, #encoding>>
  return
}
// CHECK-LABEL: func.func @set_encoding_LHS_with_layout
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<1x256xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !flow.dispatch.tensor<writeonly:tensor<1x256x1x1xf32>>
//       CHECK:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]]
//       CHECK:   %[[PACK:.+]] = tensor.pack %[[INPUT]]
//  CHECK-SAME:     outer_dims_perm = [0, 1]
//  CHECK-SAME:     inner_dims_pos = [0, 1]
//  CHECK-SAME:     inner_tiles = [1, 1]
//  CHECK-SAME:     tensor<1x256xf32> -> tensor<1x256x1x1xf32>
//       CHECK:   flow.dispatch.tensor.store %[[PACK]], %[[RESULT_BINDING]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], layouts = [#iree_cpu.cpu_encoding_layout<configuration = {encoding_info = {innerDimsPos = [1, 0], innerTileSizes = [16, 1], outerDimsPerm = [1, 0]}}>]>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f", encoding = #iree_cpu.cpu_encoding_layout<>}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 32, 32>>
func.func @set_encoding_RHS_with_layout() attributes {
  hal.executable.target = #executable_target
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x10xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<256x10xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 10], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x10xf32>> -> tensor<256x10xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<256x10xf32> -> tensor<256x10xf32, #encoding1>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [256, 10], strides = [1, 1] : tensor<256x10xf32, #encoding1> -> !flow.dispatch.tensor<writeonly:tensor<256x10xf32, #encoding>>
  return
}
// CHECK-LABEL: func.func @set_encoding_RHS_with_layout
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<256x10xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !flow.dispatch.tensor<writeonly:tensor<1x256x16x1xf32>>
//   CHECK-DAG:   %[[PAD_VALUE:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]]
//       CHECK:   %[[PACK:.+]] = tensor.pack %[[INPUT]]
//  CHECK-SAME:     padding_value(%[[PAD_VALUE]] : f32)
//  CHECK-SAME:     outer_dims_perm = [1, 0]
//  CHECK-SAME:     inner_dims_pos = [1, 0]
//  CHECK-SAME:     inner_tiles = [16, 1]
//  CHECK-SAME:     tensor<256x10xf32> -> tensor<1x256x16x1xf32>
//       CHECK:   flow.dispatch.tensor.store %[[PACK]], %[[RESULT_BINDING]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], layouts = [#iree_cpu.cpu_encoding_layout<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [1, 16], outerDimsPerm = [0, 1]}}>]>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx512f", encoding = #iree_cpu.cpu_encoding_layout<>, target_triple = "x86_64-xyz-xyz"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#encoding1 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 32, 32>>
func.func @unset_encoding_RES_with_layout() attributes {
  hal.executable.target = #executable_target
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<1x10xf32, #encoding>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<1x10xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 10], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x10xf32, #encoding>> -> tensor<1x10xf32, #encoding1>
  %3 = iree_encoding.unset_encoding %2 : tensor<1x10xf32, #encoding1> -> tensor<1x10xf32>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1, 10], strides = [1, 1] : tensor<1x10xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x10xf32>>
  return
}
// CHECK-LABEL: func.func @unset_encoding_RES_with_layout
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<1x1x1x16xf32>>
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : !flow.dispatch.tensor<writeonly:tensor<1x10xf32>>
//       CHECK:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]]
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[INPUT]]
//  CHECK-SAME:     outer_dims_perm = [0, 1]
//  CHECK-SAME:     inner_dims_pos = [0, 1]
//  CHECK-SAME:     inner_tiles = [1, 16]
//  CHECK-SAME:     tensor<1x1x1x16xf32> -> tensor<1x10xf32>
//       CHECK:   flow.dispatch.tensor.store %[[UNPACK]], %[[RESULT_BINDING]]
