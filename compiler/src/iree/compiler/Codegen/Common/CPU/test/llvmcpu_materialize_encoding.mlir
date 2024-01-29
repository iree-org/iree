// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-cpu-materialize-encoding),canonicalize,cse)" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @set_encoding_7x7x7_matmul_LHS() attributes {
   hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma"}>
} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.index_castui %2 : i32 to index
  %7 = arith.index_castui %3 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<7x7xf32>>
  %9 = flow.dispatch.workload.ordinal %6, 2 : index
  %10 = flow.dispatch.workload.ordinal %7, 3 : index
  %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<7x7xf32>, user_indexing_maps = [#map, #map1, #map2]>>>{%9, %10}
  %12 = flow.dispatch.workload.ordinal %4, 0 : index
  %13 = flow.dispatch.workload.ordinal %5, 1 : index
  %14 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [7, 7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7xf32>> -> tensor<7x7xf32>
  %15 = affine.apply affine_map<()[s0] -> ((7 ceildiv s0) * s0 - 7)>()[%12]
  %16 = affine.apply affine_map<()[s0] -> ((7 ceildiv s0) * s0 - 7)>()[%13]
  %padded = tensor.pad %14 low[0, 0] high[%15, %16] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<7x7xf32> to tensor<?x?xf32>
  %17 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<7x7xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %17, %11, offsets = [0, 0], sizes = [%9, %10], strides = [1, 1] : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<7x7xf32>, user_indexing_maps = [#map, #map1, #map2]>> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<7x7xf32>, user_indexing_maps = [#map, #map1, #map2]>>>{%9, %10}
  return
}
// CHECK:    func @set_encoding_7x7x7_matmul_LHS(
// CHECK-DAG:  %[[CST:.+]] = arith.constant 0.0
// CHECK:      %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} !flow.dispatch.tensor<readonly:tensor<7x7xf32>>
// CHECK:      %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan {{.*}} !flow.dispatch.tensor<writeonly:tensor<1x7x8x1xf32>>
// CHECK:      %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]], offsets = [0, 0], sizes = [7, 7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7xf32>> -> tensor<7x7xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<1x7x8x1xf32>
// CHECK:      %[[PACK:.+]] = tensor.pack %[[INPUT]] padding_value(%[[CST]] : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<7x7xf32> -> tensor<1x7x8x1xf32>
// CHECK:      flow.dispatch.tensor.store %[[PACK]], %[[OUTPUT_BINDING]], offsets = [0, 0, 0, 0], sizes = [1, 7, 8, 1], strides = [1, 1, 1, 1] : tensor<1x7x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x7x8x1xf32>>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @set_encoding_128x80x32_batch_matmul_LHS() attributes {
   hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma"}>
} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.index_castui %2 : i32 to index
  %7 = arith.index_castui %3 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x80x32xf32>>
  %9 = flow.dispatch.workload.ordinal %6, 2 : index
  %10 = flow.dispatch.workload.ordinal %7, 3 : index
  %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<128x80x32xf32>, user_indexing_maps = [#map, #map1, #map2]>>>{%9, %10}
  %12 = flow.dispatch.workload.ordinal %4, 0 : index
  %13 = flow.dispatch.workload.ordinal %5, 1 : index
  %14 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0], sizes = [128, 80, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x80x32xf32>> -> tensor<128x80x32xf32>
  %15 = affine.apply affine_map<()[s0] -> ((32 ceildiv s0) * s0 - 32)>()[%12]
  %16 = affine.apply affine_map<()[s0] -> ((80 ceildiv s0) * s0 - 80)>()[%13]
  %padded = tensor.pad %14 low[0, 0, 0] high[0, %16, %15] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<128x80x32xf32> to tensor<128x?x?xf32>
  %17 = iree_linalg_ext.set_encoding %padded : tensor<128x?x?xf32> -> tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<128x80x32xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %17, %11, offsets = [0, 0, 0], sizes = [128, %9, %10], strides = [1, 1, 1]
    : tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<128x80x32xf32>, user_indexing_maps = [#map, #map1, #map2]>>
    -> !flow.dispatch.tensor<writeonly:tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], original_type = tensor<128x80x32xf32>, user_indexing_maps = [#map, #map1, #map2]>>>{%9, %10}
  return
}
// CHECK:    func @set_encoding_128x80x32_batch_matmul_LHS(
// CHECK:      %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) {{.*}} !flow.dispatch.tensor<readonly:tensor<128x80x32xf32>>
// CHECK:      %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1) {{.*}} !flow.dispatch.tensor<writeonly:tensor<128x10x32x8x1xf32>>
// CHECK:      %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]], offsets = [0, 0, 0], sizes = [128, 80, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x80x32xf32>> -> tensor<128x80x32xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<128x10x32x8x1xf32>
// CHECK:      %[[PACK:.+]] = tensor.pack %[[INPUT]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 1] into %[[EMPTY]] : tensor<128x80x32xf32> -> tensor<128x10x32x8x1xf32>
// CHECK:      flow.dispatch.tensor.store %[[PACK]], %[[OUTPUT_BINDING]], offsets = [0, 0, 0, 0, 0], sizes = [128, 10, 32, 8, 1], strides = [1, 1, 1, 1, 1] : tensor<128x10x32x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x10x32x8x1xf32>>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @set_encoding_128x32x320_batch_matmul_RHS() attributes {
   hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma"}>
} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = hal.interface.constant.load[4] : i32
  %5 = arith.index_castui %0 {stream.alignment = 64 : index} : i32 to index
  %6 = arith.index_castui %1 : i32 to index
  %7 = arith.index_castui %2 : i32 to index
  %8 = arith.index_castui %3 : i32 to index
  %9 = arith.index_castui %4 : i32 to index
  %10 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x32x320xf32>>
  %11 = flow.dispatch.workload.ordinal %8, 2 : index
  %12 = flow.dispatch.workload.ordinal %9, 3 : index
  %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<128x32x320xf32>, user_indexing_maps = [#map, #map1, #map2]>>>{%11, %12}
  %14 = flow.dispatch.workload.ordinal %6, 0 : index
  %15 = flow.dispatch.workload.ordinal %7, 1 : index
  %16 = flow.dispatch.tensor.load %10, offsets = [0, 0, 0], sizes = [128, 32, 320], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x32x320xf32>> -> tensor<128x32x320xf32>
  %17 = affine.apply affine_map<()[s0] -> ((320 ceildiv s0) * s0 - 320)>()[%14]
  %18 = affine.apply affine_map<()[s0] -> ((32 ceildiv s0) * s0 - 32)>()[%15]
  %padded = tensor.pad %16 low[0, 0, 0] high[0, %18, %17] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<128x32x320xf32> to tensor<128x?x?xf32>
  %19 = iree_linalg_ext.set_encoding %padded : tensor<128x?x?xf32> -> tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<128x32x320xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %19, %13, offsets = [0, 0, 0], sizes = [128, %11, %12], strides = [1, 1, 1]
    : tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<128x32x320xf32>, user_indexing_maps = [#map, #map1, #map2]>>
    -> !flow.dispatch.tensor<writeonly:tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], original_type = tensor<128x32x320xf32>, user_indexing_maps = [#map, #map1, #map2]>>>{%11, %12}
  return
}
// CHECK:    func @set_encoding_128x32x320_batch_matmul_RHS(
// CHECK:      %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) {{.*}} !flow.dispatch.tensor<readonly:tensor<128x32x320xf32>>
// CHECK:      %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1) {{.*}} !flow.dispatch.tensor<writeonly:tensor<128x40x32x8x1xf32>>
// CHECK:      %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]], offsets = [0, 0, 0], sizes = [128, 32, 320], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x32x320xf32>> -> tensor<128x32x320xf32>
// CHECK:      %[[EMPTY:.+]] = tensor.empty() : tensor<128x40x32x8x1xf32>
// CHECK:      %[[PACK:.+]] = tensor.pack %[[INPUT]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [8, 1] into %[[EMPTY]] : tensor<128x32x320xf32> -> tensor<128x40x32x8x1xf32>
// CHECK:      flow.dispatch.tensor.store %[[PACK]], %[[OUTPUT_BINDING]], offsets = [0, 0, 0, 0, 0], sizes = [128, 40, 32, 8, 1], strides = [1, 1, 1, 1, 1] : tensor<128x40x32x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x40x32x8x1xf32>>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @unset_encoding_128x80x320_batch_matmul_RESULT() attributes {
   hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = arith.index_castui %0 : i32 to index
  %4 = arith.index_castui %1 : i32 to index
  %5 = arith.index_castui %2 : i32 to index
  %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x80x320xf32>>
  %7 = flow.dispatch.workload.ordinal %4, 0 : index
  %8 = flow.dispatch.workload.ordinal %5, 1 : index
  %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<128x80x320xf32>, user_indexing_maps = [#map, #map1, #map2]>>>{%7, %8}
  %10 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0], sizes = [128, %7, %8], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<128x80x320xf32>, user_indexing_maps = [#map, #map1, #map2]>>>{%7, %8}
      -> tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<128x80x320xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %11 = iree_linalg_ext.unset_encoding %10 : tensor<128x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], original_type = tensor<128x80x320xf32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<128x?x?xf32>
  %extracted_slice = tensor.extract_slice %11[0, 0, 0] [128, 80, 320] [1, 1, 1] : tensor<128x?x?xf32> to tensor<128x80x320xf32>
  flow.dispatch.tensor.store %extracted_slice, %6, offsets = [0, 0, 0], sizes = [128, 80, 320], strides = [1, 1, 1] : tensor<128x80x320xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x80x320xf32>>
  return
}
//       CHECK: func @unset_encoding_128x80x320_batch_matmul_RESULT()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load[0]
//       CHECK:   %[[CAST:.+]] = arith.index_castui %[[D0]] : i32 to index
//       CHECK:   %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[C0]])
//  CHECK-SAME:       : !flow.dispatch.tensor<writeonly:tensor<128x80x320xf32>>
//       CHECK:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[CAST]])
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
func.func @pack_gemm_fill_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx,+avx2,+fma"}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = iree_linalg_ext.unset_encoding %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @pack_gemm_fill_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[OUT_D0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
//  CHECK-DAG:   %[[OUT_D1:.+]] = affine.apply #[[MAP0]]()[%[[D1]]]
//  CHECK-DAG:   %[[PACK_LHS:.+]] = tensor.pack {{.*}}%[[ARG0]]
//      CHECK:   %[[PACK_RHS:.+]] = tensor.pack
// CHECK-SAME:     %[[ARG1]]
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[OUT_D0]], %[[OUT_D1]]) : tensor<?x?x8x8xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]]
//      CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f32f32f32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @matmul_lowering_f32f32f32_aarch64()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matvec_lowering_f32f32f32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<16x16xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<16x1xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<16x1xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<16x16xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>>
      -> tensor<16x16xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<16x1xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>>
      -> tensor<16x1xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<16x1xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>>
      -> tensor<16x1xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<16x16xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<16x1xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<16x1xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<16x1xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : tensor<16x1xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<16x1xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], matmul_narrow_N = 1 : index, user_indexing_maps = [#map, #map1, #map2]>>>
  return
}
//      CHECK: func @matvec_lowering_f32f32f32_aarch64()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<2x16x8x1xf32>>
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<1x16x1x1xf32>>
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<2x1x8x1xf32>>
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [2, 16, 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 16, 1, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [2, 1, 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [2, 1, 8, 1], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f16f16f16_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @matmul_lowering_f16f16f16_aarch64()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf16>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f32f32f32_x86_64() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @matmul_lowering_f32f32f32_x86_64()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP1]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x4x1xf32>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x4xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 4, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 4], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f32f32f32_x86_64_avx2() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @matmul_lowering_f32f32f32_x86_64_avx2()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f32f32f32_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//      CHECK: func @matmul_lowering_f32f32f32_x86_64_avx512f()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f16f16f32_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//      CHECK: func @matmul_lowering_f16f16f32_x86_64_avx512f()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf16>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf16>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f16f16f16_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//      CHECK: func @matmul_lowering_f16f16f16_x86_64_avx512f()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf16>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf16>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf16>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_bf16bf16f32_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//      CHECK: func @matmul_lowering_bf16bf16f32_x86_64_avx512f()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xbf16>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xbf16>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_bf16bf16bf16_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//      CHECK: func @matmul_lowering_bf16bf16bf16_x86_64_avx512f()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xbf16>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xbf16>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xbf16>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_bf16bf16f32_x86_64_avx512bf16() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f,+avx512bf16"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//      CHECK: func @matmul_lowering_bf16bf16f32_x86_64_avx512bf16()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xbf16>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xbf16>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_bf16bf16bf16_x86_64_avx512bf16() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f,+avx512bf16"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #iree_linalg_ext.encoding<role = LHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RHS, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xbf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [bf16, bf16, bf16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//      CHECK: func @matmul_lowering_bf16bf16bf16_x86_64_avx512bf16()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xbf16>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xbf16>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xbf16>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f32f16f16_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %lhs_f32 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %rhs = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %dest = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>

  %empty = tensor.empty(%M, %K) : tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %lhs_f16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
     ins(%lhs_f32 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>)
     outs(%empty : tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: f32, %out: f16):
    %17 = arith.truncf %in : f32 to f16
    linalg.yield %17 : f16
  } -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%lhs_f16, %rhs : tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%dest : tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
// CHECK-DAG: #[[MAP_CEILDIV_8:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-DAG: #[[MAP_IDENTITY_4D:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:     func.func @matmul_lowering_f32f16f16_aarch64()
// CHECK-DAG: %[[M:.+]] = hal.interface.constant.load[0] : index
// CHECK-DAG: %[[N:.+]] = hal.interface.constant.load[1] : index
// CHECK-DAG: %[[K:.+]] = hal.interface.constant.load[2] : index
// CHECK-DAG: %[[M_CEILDIV_8:.+]] = affine.apply #[[MAP_CEILDIV_8]]()[%[[M]]]
// CHECK-DAG: %[[N_CEILDIV_8:.+]] = affine.apply #[[MAP_CEILDIV_8]]()[%[[N]]]
// CHECK-DAG: %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[M_CEILDIV_8]], %[[K]]}
// CHECK-DAG: %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[N_CEILDIV_8]], %[[K]]}
// CHECK-DAG: %[[OUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2) {{.*}} : !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf16>>{%[[M_CEILDIV_8]], %[[N_CEILDIV_8]]}
// CHECK-DAG: %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M_CEILDIV_8]], %[[K]], 8, 1], {{.*}} -> tensor<?x?x8x1xf32>
// CHECK-DAG: %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[N_CEILDIV_8]], %[[K]], 8, 1], {{.*}} -> tensor<?x?x8x1xf16>
// CHECK-DAG: %[[OUT:.+]] = flow.dispatch.tensor.load %[[OUT_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M_CEILDIV_8]], %[[N_CEILDIV_8]], 8, 8], {{.*}} -> tensor<?x?x8x8xf16>
// CHECK-DAG: %[[EMPTY:.+]] = tensor.empty(%[[M_CEILDIV_8]], %[[K]]) : tensor<?x?x8x1xf16>
// CHECK-DAG: %[[LHS_F16:.+]] = linalg.generic {indexing_maps = [#[[MAP_IDENTITY_4D]], #[[MAP_IDENTITY_4D]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS]] : tensor<?x?x8x1xf32>) outs(%[[EMPTY]] : tensor<?x?x8x1xf16>) {
// CHECK-DAG: %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS_F16]], %[[RHS]] : tensor<?x?x8x1xf16>, tensor<?x?x8x1xf16>) outs(%[[OUT]] : tensor<?x?x8x8xf16>)
// CHECK: flow.dispatch.tensor.store %[[MMT4D]], %[[OUT_BINDING]],

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f32f16f16_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f,+avx512bf16"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %lhs_f32 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %rhs = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %dest = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>

  %empty = tensor.empty(%M, %K) : tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %lhs_f16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
     ins(%lhs_f32 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>)
     outs(%empty : tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: f32, %out: f16):
    %17 = arith.truncf %in : f32 to f16
    linalg.yield %17 : f16
  } -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%lhs_f16, %rhs : tensor<?x?xf16, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf16, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%dest : tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf16, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}

// CHECK-DAG: #[[MAP_CEILDIV_16:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-DAG: #[[MAP_IDENTITY_4D:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:     func.func @matmul_lowering_f32f16f16_x86_64_avx512f()
// CHECK-DAG: %[[M:.+]] = hal.interface.constant.load[0] : index
// CHECK-DAG: %[[N:.+]] = hal.interface.constant.load[1] : index
// CHECK-DAG: %[[K:.+]] = hal.interface.constant.load[2] : index
// CHECK-DAG: %[[M_CEILDIV_16:.+]] = affine.apply #[[MAP_CEILDIV_16]]()[%[[M]]]
// CHECK-DAG: %[[N_CEILDIV_16:.+]] = affine.apply #[[MAP_CEILDIV_16]]()[%[[N]]]
// CHECK-DAG: %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%[[M_CEILDIV_16]], %[[K]]}
// CHECK-DAG: %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf16>>{%[[N_CEILDIV_16]], %[[K]]}
// CHECK-DAG: %[[OUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2) {{.*}} : !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf16>>{%[[M_CEILDIV_16]], %[[N_CEILDIV_16]]}
// CHECK-DAG: %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M_CEILDIV_16]], %[[K]], 16, 1], {{.*}} -> tensor<?x?x16x1xf32>
// CHECK-DAG: %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[N_CEILDIV_16]], %[[K]], 16, 1], {{.*}} -> tensor<?x?x16x1xf16>
// CHECK-DAG: %[[OUT:.+]] = flow.dispatch.tensor.load %[[OUT_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M_CEILDIV_16]], %[[N_CEILDIV_16]], 16, 16], {{.*}} -> tensor<?x?x16x16xf16>
// CHECK-DAG: %[[EMPTY:.+]] = tensor.empty(%[[M_CEILDIV_16]], %[[K]]) : tensor<?x?x16x1xf16>
// CHECK-DAG: %[[LHS_F16:.+]] = linalg.generic {indexing_maps = [#[[MAP_IDENTITY_4D]], #[[MAP_IDENTITY_4D]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS]] : tensor<?x?x16x1xf32>) outs(%[[EMPTY]] : tensor<?x?x16x1xf16>) {
// CHECK-DAG: %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS_F16]], %[[RHS]] : tensor<?x?x16x1xf16>, tensor<?x?x16x1xf16>) outs(%[[OUT]] : tensor<?x?x16x16xf16>)
// CHECK: flow.dispatch.tensor.store %[[MMT4D]], %[[OUT_BINDING]],

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_i8i8i32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @matmul_lowering_i8i8i32_aarch64()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xi8>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xi8>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_i8i8i32_aarch64_dotprod() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @matmul_lowering_i8i8i32_aarch64_dotprod()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_i8i8i32_aarch64_i8mm() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+i8mm"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @matmul_lowering_i8i8i32_aarch64_i8mm()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP0]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f32f32f32_aarch64_sve(%lhs : tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {cpu_features = "+sve", target_triple="aarch64-xyz-xyz"}>
} {
  %0 = iree_linalg_ext.set_encoding %lhs : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %1 = iree_linalg_ext.set_encoding %rhs : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %2 = iree_linalg_ext.set_encoding %acc : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%2 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// Targets that has "sve" features does not implement data-tiling yet.
// CHECK: func @matmul_lowering_f32f32f32_aarch64_sve
// CHECK:   %[[RES:.+]] = linalg.matmul
// CHECK:   return %[[RES]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_f32f32f32_riscv(%lhs : tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="riscv32-xyz-xyz"}>
} {
  %0 = iree_linalg_ext.set_encoding %lhs : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %1 = iree_linalg_ext.set_encoding %rhs : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %2 = iree_linalg_ext.set_encoding %acc : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%2 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
// RISC-V targets does not implement data-tiling yet.
// CHECK: func @matmul_lowering_f32f32f32_riscv
// CHECK:   %[[RES:.+]] = linalg.matmul
// CHECK:   return %[[RES]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_i8i8i32_riscv32_ukernel() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="riscv32-xyz-xyz", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @matmul_lowering_i8i8i32_riscv32_ukernel()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_i8i8i32_x86_64_avx2() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx2"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//      CHECK: func @matmul_lowering_i8i8i32_x86_64_avx2()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x2xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_i8i8i32_x86_64_avx512bw() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512bw"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//      CHECK: func @matmul_lowering_i8i8i32_x86_64_avx512bw()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_i8i8i32_x86_64_avx512vnni() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//      CHECK: func @matmul_lowering_i8i8i32_x86_64_avx512vnni()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @extend_batch_vecmat_explicit_unit_dim(%arg0: tensor<32x1x128xi8>, %arg1: tensor<32x128x11008xi8>) -> tensor<32x1x11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c11008 = arith.constant 11008 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %padded = tensor.pad %arg0 low[0, 0, 0] high[%c0, %c0, %c0] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):
    tensor.yield %c0_i8 : i8
  } : tensor<32x1x128xi8> to tensor<?x?x?xi8>
  %4 = iree_linalg_ext.set_encoding %padded : tensor<?x?x?xi8> -> tensor<?x?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %5 = tensor.empty(%c32, %c1, %c128) : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<?x?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%5 : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %padded_0 = tensor.pad %arg1 low[0, 0, 0] high[%c0, %c0, %c0] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):
    tensor.yield %c0_i8 : i8
  } : tensor<32x128x11008xi8> to tensor<?x?x?xi8>
  %7 = iree_linalg_ext.set_encoding %padded_0 : tensor<?x?x?xi8> -> tensor<?x?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %8 = tensor.empty(%c32, %c128, %c11008) : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<?x?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%8 : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %10 = tensor.empty(%c32, %c1, %c11008) : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %12 = linalg.batch_matmul ins(%6, %9 : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%11 : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %13 = iree_linalg_ext.unset_encoding %12 : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x1x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?x?xi32>
  %extracted_slice = tensor.extract_slice %13[0, 0, 0] [32, 1, 11008] [1, 1, 1] : tensor<?x?x?xi32> to tensor<32x1x11008xi32>
  return %extracted_slice : tensor<32x1x11008xi32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//      CHECK: func @extend_batch_vecmat_explicit_unit_dim(%[[LHS:.+]]: tensor<32x1x128xi8>, %[[RHS:.+]]: tensor<32x128x11008xi8>) -> tensor<32x1x11008xi32>
//  CHECK-DAG: %[[C0_I32:.+]] = arith.constant 0 : i32
//      CHECK: %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<32x1x64x1x2xi8>
//      CHECK: %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [1, 2] into %[[INIT_LHS_PACK]] : tensor<32x1x128xi8> -> tensor<32x1x64x1x2xi8>
//      CHECK: %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<32x1x64x1x2xi32>
//      CHECK: %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<32x1x64x1x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<32x1x64x1x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[LHS_EXT_OP]] : i32
//      CHECK: %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<32x688x64x16x2xi8>
//      CHECK: %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %[[INIT_RHS_PACK]] : tensor<32x128x11008xi8> -> tensor<32x688x64x16x2xi8>
//      CHECK: %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<32x688x64x16x2xi32>
//      CHECK: %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<32x688x64x16x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<32x688x64x16x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[RHS_EXT_OP]] : i32
//      CHECK: %[[INIT_FILL:.+]] = tensor.empty() : tensor<32x1x688x1x16xi32>
//      CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[INIT_FILL]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//      CHECK: %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[LHS_EXT]], %[[RHS_EXT]] : tensor<32x1x64x1x2xi32>, tensor<32x688x64x16x2xi32>) outs(%[[FILL]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//      CHECK: %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<32x1x11008xi32>
//      CHECK: %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [1, 16] into %[[INIT_UNPACK]] : tensor<32x1x688x1x16xi32> -> tensor<32x1x11008xi32>
//      CHECK: return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_i16i16i32_x86_64_avx2() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx2"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #iree_linalg_ext.encoding<role = LHS, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #iree_linalg_ext.encoding<role = LHS, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xi16, #iree_linalg_ext.encoding<role = LHS, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xi16, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi16, #iree_linalg_ext.encoding<role = LHS, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xi16, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, i16, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//      CHECK: func @matmul_lowering_i16i16i32_x86_64_avx2()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x2xi16>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x2xi16>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_lowering_i16ui4i32_x86_64_avx512vnni() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #iree_linalg_ext.encoding<role = LHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
  %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi4, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
  %out_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi16, #iree_linalg_ext.encoding<role = LHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %K}
      -> tensor<?x?xi16, #iree_linalg_ext.encoding<role = LHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %rhs_i4 = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi4, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%K, %N}
      -> tensor<?x?xi4, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %empty = tensor.empty(%K, %N) : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %rhs_i32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
     ins(%rhs_i4 : tensor<?x?xi4, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>) outs(%empty : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i4, %out: i32):
    %17 = arith.extui %in : i4 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %out = flow.dispatch.tensor.load %out_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>
  %result = linalg.matmul
      ins(%lhs, %rhs_i32 : tensor<?x?xi16, #iree_linalg_ext.encoding<role = LHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>,
                   tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%out : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>
  flow.dispatch.tensor.store %result, %out_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i16, ui4, i32], user_indexing_maps = [#map, #map1, #map2]>>>{%M, %N}
  return
}

// CHECK-DAG: #[[MAP_CEILDIV_8:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-DAG: #[[MAP_CEILDIV_32:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
// CHECK-DAG: #[[MAP_IDENTITY_4D:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:     func.func @matmul_lowering_i16ui4i32_x86_64_avx512vnni()
// CHECK-DAG: %[[M:.+]] = hal.interface.constant.load[0] : index
// CHECK-DAG: %[[N:.+]] = hal.interface.constant.load[1] : index
// CHECK-DAG: %[[K:.+]] = hal.interface.constant.load[2] : index
// CHECK-DAG: %[[K_CEILDIV_8:.+]] = affine.apply #[[MAP_CEILDIV_8]]()[%[[K]]]
// CHECK-DAG: %[[N_CEILDIV_32:.+]] = affine.apply #[[MAP_CEILDIV_32]]()[%[[N]]]
// CHECK-DAG: %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x1x8xi16>>{%[[M]], %[[K_CEILDIV_8]]}
// CHECK-DAG: %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1) {{.*}} : !flow.dispatch.tensor<readonly:tensor<?x?x32x8xi4>>{%[[N_CEILDIV_32]], %[[K_CEILDIV_8]]}
// CHECK-DAG: %[[OUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2) {{.*}} : !flow.dispatch.tensor<readwrite:tensor<?x?x1x32xi32>>{%[[M]], %[[N_CEILDIV_32]]}
// CHECK-DAG: %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M]], %[[K_CEILDIV_8]], 1, 8], {{.*}} -> tensor<?x?x1x8xi16>
// CHECK-DAG: %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[N_CEILDIV_32]], %[[K_CEILDIV_8]], 32, 8], {{.*}} -> tensor<?x?x32x8xi4>
// CHECK-DAG: %[[OUT:.+]] = flow.dispatch.tensor.load %[[OUT_BINDING]], offsets = [0, 0, 0, 0], sizes = [%[[M]], %[[N_CEILDIV_32]], 1, 32], {{.*}} -> tensor<?x?x1x32xi32>
// CHECK-DAG: %[[EMPTY:.+]] = tensor.empty(%[[N_CEILDIV_32]], %[[K_CEILDIV_8]]) : tensor<?x?x32x8xi32>
// CHECK-DAG: %[[RHS_I32:.+]] = linalg.generic {indexing_maps = [#[[MAP_IDENTITY_4D]], #[[MAP_IDENTITY_4D]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS]] : tensor<?x?x32x8xi4>) outs(%[[EMPTY]] : tensor<?x?x32x8xi32>) {
// CHECK-DAG: %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS]], %[[RHS_I32]] : tensor<?x?x1x8xi16>, tensor<?x?x32x8xi32>) outs(%[[OUT]] : tensor<?x?x1x32xi32>) -> tensor<?x?x1x32xi32>
// CHECK: flow.dispatch.tensor.store %[[MMT4D]], %[[OUT_BINDING]],

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @vecmat(%arg0: tensor<128xi8>, %arg1: tensor<128x11008xi8>) -> tensor<11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c11008 = arith.constant 11008 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %padded = tensor.pad %arg0 low[0] high[%c0] {
  ^bb0(%arg2: index):
    tensor.yield %c0_i8 : i8
  } : tensor<128xi8> to tensor<?xi8>
  %4 = iree_linalg_ext.set_encoding %padded : tensor<?xi8> -> tensor<?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %5 = tensor.empty(%c128) : tensor<?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4 : tensor<?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%5 : tensor<?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %padded_0 = tensor.pad %arg1 low[0, 0] high[%c0, %c0] {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %c0_i8 : i8
  } : tensor<128x11008xi8> to tensor<?x?xi8>
  %7 = iree_linalg_ext.set_encoding %padded_0 : tensor<?x?xi8> -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %8 = tensor.empty(%c128, %c11008) : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%8 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %10 = tensor.empty(%c11008) : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %12 = linalg.vecmat ins(%6, %9 : tensor<?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%11 : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %13 = iree_linalg_ext.unset_encoding %12 : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?xi32>
  %extracted_slice = tensor.extract_slice %13[0] [11008] [1] : tensor<?xi32> to tensor<11008xi32>
  return %extracted_slice : tensor<11008xi32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func.func @vecmat(%[[LHS:.+]]: tensor<128xi8>, %[[RHS:.+]]: tensor<128x11008xi8>) -> tensor<11008xi32>
//  CHECK-DAG: %[[C0_I32:.+]] = arith.constant 0 : i32
//      CHECK: %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<64x2xi8>
//      CHECK: %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [2] into %[[INIT_LHS_PACK]] : tensor<128xi8> -> tensor<64x2xi8>
//      CHECK: %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<64x2xi32>
//      CHECK: %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<64x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<64x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[LHS_EXT_OP]] : i32
//      CHECK: %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<688x64x16x2xi8>
//      CHECK: %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %[[INIT_RHS_PACK]] : tensor<128x11008xi8> -> tensor<688x64x16x2xi8>
//      CHECK: %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<688x64x16x2xi32>
//      CHECK: %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<688x64x16x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<688x64x16x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[RHS_EXT_OP]] : i32
//      CHECK: %[[INIT_FILL:.+]] = tensor.empty() : tensor<688x16xi32>
//      CHECK: %[[EXPAND_LHS:.+]] = tensor.expand_shape %[[LHS_EXT]] {{\[}}[0, 1], [2, 3]] : tensor<64x2xi32> into tensor<1x64x1x2xi32>
//      CHECK: %[[EXPAND_INIT:.+]] = tensor.expand_shape %[[INIT_FILL:.+]] {{\[}}[0, 1], [2, 3]] : tensor<688x16xi32> into tensor<1x688x1x16xi32>
//      CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EXPAND_INIT]] : tensor<1x688x1x16xi32>) -> tensor<1x688x1x16xi32>
//      CHECK: %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXPAND_LHS]], %[[RHS_EXT]] : tensor<1x64x1x2xi32>, tensor<688x64x16x2xi32>) outs(%[[FILL]] : tensor<1x688x1x16xi32>) -> tensor<1x688x1x16xi32>
//      CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0, 1], [2, 3]] : tensor<1x688x1x16xi32> into tensor<688x16xi32>
//      CHECK: %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<11008xi32>
//      CHECK: %[[UNPACK:.+]] = tensor.unpack %[[COLLAPSED]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<688x16xi32> -> tensor<11008xi32>
//      CHECK: return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @matvec(%arg0: tensor<11008x128xi8>, %arg1: tensor<128xi8>) -> tensor<11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c11008 = arith.constant 11008 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %padded = tensor.pad %arg0 low[0, 0] high[%c0, %c0] {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %c0_i8 : i8
  } : tensor<11008x128xi8> to tensor<?x?xi8>
  %4 = iree_linalg_ext.set_encoding %padded : tensor<?x?xi8> -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %5 = tensor.empty(%c11008, %c128) : tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %padded_0 = tensor.pad %arg1 low[0] high[%c0] {
  ^bb0(%arg2: index):
    tensor.yield %c0_i8 : i8
  } : tensor<128xi8> to tensor<?xi8>
  %7 = iree_linalg_ext.set_encoding %padded_0 : tensor<?xi8> -> tensor<?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %8 = tensor.empty(%c128) : tensor<?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7 : tensor<?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%8 : tensor<?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %10 = tensor.empty(%c11008) : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %12 = linalg.matvec ins(%6, %9 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%11 : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %13 = iree_linalg_ext.unset_encoding %12 : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<11008xi32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?xi32>
  %extracted_slice = tensor.extract_slice %13[0] [11008] [1] : tensor<?xi32> to tensor<11008xi32>
  return %extracted_slice : tensor<11008xi32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: func.func @matvec(%[[LHS:.+]]: tensor<11008x128xi8>, %[[RHS:.+]]: tensor<128xi8>) -> tensor<11008xi32>
//  CHECK-DAG: %[[C0_I32:.+]] = arith.constant 0 : i32
//      CHECK: %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<688x64x16x2xi8>
//      CHECK: %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 2] into %[[INIT_LHS_PACK]] : tensor<11008x128xi8> -> tensor<688x64x16x2xi8>
//      CHECK: %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<688x64x16x2xi32>
//      CHECK: %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<688x64x16x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<688x64x16x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[LHS_EXT_OP]] : i32
//      CHECK: %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<64x2xi8>
//      CHECK: %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [2] into %[[INIT_RHS_PACK]] : tensor<128xi8> -> tensor<64x2xi8>
//      CHECK: %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<64x2xi32>
//      CHECK: %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<64x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<64x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[RHS_EXT_OP]] : i32
//      CHECK: %[[INIT_FILL:.+]] = tensor.empty() : tensor<688x16xi32>
//      CHECK: %[[EXPAND_RHS:.+]] = tensor.expand_shape %[[RHS_EXT]] {{\[}}[0, 1], [2, 3]] : tensor<64x2xi32> into tensor<1x64x1x2xi32>
//      CHECK: %[[EXPAND_INIT:.+]] = tensor.expand_shape %[[INIT_FILL:.+]] {{\[}}[0, 1], [2, 3]] : tensor<688x16xi32> into tensor<688x1x16x1xi32>
//      CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EXPAND_INIT]] : tensor<688x1x16x1xi32>) -> tensor<688x1x16x1xi32>
//      CHECK: %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS_EXT]], %[[EXPAND_RHS]]  : tensor<688x64x16x2xi32>, tensor<1x64x1x2xi32>) outs(%[[FILL]] : tensor<688x1x16x1xi32>) -> tensor<688x1x16x1xi32>
//      CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0, 1], [2, 3]] : tensor<688x1x16x1xi32> into tensor<688x16xi32>
//      CHECK: %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<11008xi32>
//      CHECK: %[[UNPACK:.+]] = tensor.unpack %[[COLLAPSED]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<688x16xi32> -> tensor<11008xi32>
//      CHECK: return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @matvec_with_narrow_M(%arg0: tensor<15x128xi8>, %arg1: tensor<128xi8>) -> tensor<15xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c15 = arith.constant 15 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %padded = tensor.pad %arg0 low[0, 0] high[%c0, %c0] {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %c0_i8 : i8
  } : tensor<15x128xi8> to tensor<?x?xi8>
  %4 = iree_linalg_ext.set_encoding %padded : tensor<?x?xi8> -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %5 = tensor.empty(%c15, %c128) : tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %padded_0 = tensor.pad %arg1 low[0] high[%c0] {
  ^bb0(%arg2: index):
    tensor.yield %c0_i8 : i8
  } : tensor<128xi8> to tensor<?xi8>
  %7 = iree_linalg_ext.set_encoding %padded_0 : tensor<?xi8> -> tensor<?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %8 = tensor.empty(%c128) : tensor<?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7 : tensor<?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%8 : tensor<?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %10 = tensor.empty(%c15) : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %12 = linalg.matvec ins(%6, %9 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%11 : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %13 = iree_linalg_ext.unset_encoding %12 : tensor<?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 15 : index, matmul_narrow_N = 1 : index, original_type = tensor<15xi32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?xi32>
  %extracted_slice = tensor.extract_slice %13[0] [15] [1] : tensor<?xi32> to tensor<15xi32>
  return %extracted_slice : tensor<15xi32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: func @matvec_with_narrow_M(%[[LHS:.+]]: tensor<15x128xi8>, %[[RHS:.+]]: tensor<128xi8>) -> tensor<15xi32>
//  CHECK-DAG: %[[C0_I8:.+]] = arith.constant 0 : i8
//  CHECK-DAG: %[[C0_I32:.+]] = arith.constant 0 : i32
//      CHECK: %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<1x64x16x2xi8>
//      CHECK: %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] padding_value(%[[C0_I8]] : i8) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 2] into %[[INIT_LHS_PACK]] : tensor<15x128xi8> -> tensor<1x64x16x2xi8>
//      CHECK: %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<1x64x16x2xi32>
//      CHECK: %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<1x64x16x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<1x64x16x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[LHS_EXT_OP]] : i32
//      CHECK: %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<64x2xi8>
//      CHECK: %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [2] into %[[INIT_RHS_PACK]] : tensor<128xi8> -> tensor<64x2xi8>
//      CHECK: %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<64x2xi32>
//      CHECK: %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<64x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<64x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[RHS_EXT_OP]] : i32
//      CHECK: %[[INIT_FILL:.+]] = tensor.empty() : tensor<1x16xi32>
//      CHECK: %[[EXPAND_RHS:.+]] = tensor.expand_shape %[[RHS_EXT]] {{\[}}[0, 1], [2, 3]] : tensor<64x2xi32> into tensor<1x64x1x2xi32>
//      CHECK: %[[EXPAND_INIT:.+]] = tensor.expand_shape %[[INIT_FILL:.+]] {{\[}}[0, 1], [2, 3]] : tensor<1x16xi32> into tensor<1x1x16x1xi32>
//      CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EXPAND_INIT]] : tensor<1x1x16x1xi32>) -> tensor<1x1x16x1xi32>
//      CHECK: %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS_EXT]], %[[EXPAND_RHS]]  : tensor<1x64x16x2xi32>, tensor<1x64x1x2xi32>) outs(%[[FILL]] : tensor<1x1x16x1xi32>) -> tensor<1x1x16x1xi32>
//      CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0, 1], [2, 3]] : tensor<1x1x16x1xi32> into tensor<1x16xi32>
//      CHECK: %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<15xi32>
//      CHECK: %[[UNPACK:.+]] = tensor.unpack %[[COLLAPSED]] outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<1x16xi32> -> tensor<15xi32>
//      CHECK: return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @batch_vecmat(%arg0: tensor<32x128xi8>, %arg1: tensor<32x128x11008xi8>) -> tensor<32x11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c11008 = arith.constant 11008 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %padded = tensor.pad %arg0 low[0, 0] high[%c0, %c0] {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %c0_i8 : i8
  } : tensor<32x128xi8> to tensor<?x?xi8>
  %4 = iree_linalg_ext.set_encoding %padded : tensor<?x?xi8> -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %5 = tensor.empty(%c32, %c128) : tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %padded_0 = tensor.pad %arg1 low[0, 0, 0] high[%c0, %c0, %c0] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):
    tensor.yield %c0_i8 : i8
  } : tensor<32x128x11008xi8> to tensor<?x?x?xi8>
  %7 = iree_linalg_ext.set_encoding %padded_0 : tensor<?x?x?xi8> -> tensor<?x?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %8 = tensor.empty(%c32, %c128, %c11008) : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<?x?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%8 : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %10 = tensor.empty(%c32, %c11008) : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %12 = linalg.batch_vecmat ins(%6, %9 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x128x11008xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%11 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %13 = iree_linalg_ext.unset_encoding %12 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_M = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xi32>
  %extracted_slice = tensor.extract_slice %13[0, 0] [32, 11008] [1, 1] : tensor<?x?xi32> to tensor<32x11008xi32>
  return %extracted_slice : tensor<32x11008xi32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//      CHECK: func.func @batch_vecmat(%[[LHS:.+]]: tensor<32x128xi8>, %[[RHS:.+]]: tensor<32x128x11008xi8>) -> tensor<32x11008xi32>
//  CHECK-DAG: %[[C0_I32:.+]] = arith.constant 0 : i32
//      CHECK: %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<32x64x2xi8>
//      CHECK: %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [2] into %[[INIT_LHS_PACK]] : tensor<32x128xi8> -> tensor<32x64x2xi8>
//      CHECK: %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<32x64x2xi32>
//      CHECK: %[[LHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<32x64x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<32x64x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[LHS_EXT_OP]] : i32
//      CHECK: %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<32x688x64x16x2xi8>
//      CHECK: %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %[[INIT_RHS_PACK]] : tensor<32x128x11008xi8> -> tensor<32x688x64x16x2xi8>
//      CHECK: %[[INIT_RHS_EXT:.+]] = tensor.empty() : tensor<32x688x64x16x2xi32>
//      CHECK: %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<32x688x64x16x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<32x688x64x16x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[RHS_EXT_OP]] : i32
//      CHECK: %[[INIT_FILL:.+]] = tensor.empty() : tensor<32x688x16xi32>
//      CHECK: %[[EXPAND_LHS:.+]] = tensor.expand_shape %[[LHS_EXT]] {{\[}}[0], [1, 2], [3, 4]] : tensor<32x64x2xi32> into tensor<32x1x64x1x2xi32>
//      CHECK: %[[EXPAND_INIT:.+]] = tensor.expand_shape %[[INIT_FILL:.+]] {{\[}}[0], [1, 2], [3, 4]] : tensor<32x688x16xi32> into tensor<32x1x688x1x16xi32>
//      CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EXPAND_INIT]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//      CHECK: %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[EXPAND_LHS]], %[[RHS_EXT]] : tensor<32x1x64x1x2xi32>, tensor<32x688x64x16x2xi32>) outs(%[[FILL]] : tensor<32x1x688x1x16xi32>) -> tensor<32x1x688x1x16xi32>
//      CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0], [1, 2], [3, 4]] : tensor<32x1x688x1x16xi32> into tensor<32x688x16xi32>
//      CHECK: %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<32x11008xi32>
//      CHECK: %[[UNPACK:.+]] = tensor.unpack %[[COLLAPSED]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<32x688x16xi32> -> tensor<32x11008xi32>
//      CHECK: return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @batch_matvec(%arg0: tensor<32x11008x128xi8>, %arg1: tensor<32x128xi8>) -> tensor<32x11008xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c11008 = arith.constant 11008 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %padded = tensor.pad %arg0 low[0, 0, 0] high[%c0, %c0, %c0] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):
    tensor.yield %c0_i8 : i8
  } : tensor<32x11008x128xi8> to tensor<?x?x?xi8>
  %4 = iree_linalg_ext.set_encoding %padded : tensor<?x?x?xi8> -> tensor<?x?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %5 = tensor.empty(%c32, %c11008, %c128) : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<?x?x?xi8, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%5 : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %padded_0 = tensor.pad %arg1 low[0, 0] high[%c0, %c0] {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %c0_i8 : i8
  } : tensor<32x128xi8> to tensor<?x?xi8>
  %7 = iree_linalg_ext.set_encoding %padded_0 : tensor<?x?xi8> -> tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %8 = tensor.empty(%c32, %c128) : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?x?xi8, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%8 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) {
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    linalg.yield %17 : i32
  } -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>
  %10 = tensor.empty(%c32, %c11008) : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %12 = linalg.batch_matvec ins(%6, %9 : tensor<?x?x?xi32, #iree_linalg_ext.encoding<role = LHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?xi32, #iree_linalg_ext.encoding<role = RHS, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x128xi8>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%11 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>>
  %13 = iree_linalg_ext.unset_encoding %12 : tensor<?x?xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], matmul_narrow_N = 1 : index, original_type = tensor<32x11008xi32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xi32>
  %extracted_slice = tensor.extract_slice %13[0, 0] [32, 11008] [1, 1] : tensor<?x?xi32> to tensor<32x11008xi32>
  return %extracted_slice : tensor<32x11008xi32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func.func @batch_matvec(%[[LHS:.+]]: tensor<32x11008x128xi8>, %[[RHS:.+]]: tensor<32x128xi8>) -> tensor<32x11008xi32>
//  CHECK-DAG: %[[C0_I32:.+]] = arith.constant 0 : i32
//      CHECK: %[[INIT_LHS_PACK:.+]] = tensor.empty() : tensor<32x688x64x16x2xi8>
//      CHECK: %[[LHS_PACK:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [16, 2] into %[[INIT_LHS_PACK]] : tensor<32x11008x128xi8> -> tensor<32x688x64x16x2xi8>
//      CHECK: %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<32x688x64x16x2xi32>
//      CHECK: %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS_PACK]] : tensor<32x688x64x16x2xi8>) outs(%[[INIT_LHS_EXT]] : tensor<32x688x64x16x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[LHS_EXT_ARG_IN:.+]]: i8, %[[LHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[LHS_EXT_OP:.+]] = arith.extsi %[[LHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[LHS_EXT_OP]] : i32
//      CHECK: %[[INIT_RHS_PACK:.+]] = tensor.empty() : tensor<32x64x2xi8>
//      CHECK: %[[RHS_PACK:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [2] into %[[INIT_RHS_PACK]] : tensor<32x128xi8> -> tensor<32x64x2xi8>
//      CHECK: %[[INIT_LHS_EXT:.+]] = tensor.empty() : tensor<32x64x2xi32>
//      CHECK: %[[RHS_EXT:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[RHS_PACK]] : tensor<32x64x2xi8>) outs(%[[INIT_RHS_EXT]] : tensor<32x64x2xi32>) {
// CHECK-NEXT:     ^bb0(%[[RHS_EXT_ARG_IN:.+]]: i8, %[[RHS_EXT_ARG_OUT:.+]]: i32):
// CHECK-NEXT:     %[[RHS_EXT_OP:.+]] = arith.extsi %[[RHS_EXT_ARG_IN]] : i8 to i32
// CHECK-NEXT:     linalg.yield %[[RHS_EXT_OP]] : i32
//      CHECK: %[[INIT_FILL:.+]] = tensor.empty() : tensor<32x688x16xi32>
//      CHECK: %[[EXPAND_LHS:.+]] = tensor.expand_shape %[[RHS_EXT]] {{\[}}[0], [1, 2], [3, 4]] : tensor<32x64x2xi32> into tensor<32x1x64x1x2xi32>
//      CHECK: %[[EXPAND_INIT:.+]] = tensor.expand_shape %[[INIT_FILL:.+]] {{\[}}[0], [1, 2], [3, 4]] : tensor<32x688x16xi32> into tensor<32x688x1x16x1xi32>
//      CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EXPAND_INIT]] : tensor<32x688x1x16x1xi32>) -> tensor<32x688x1x16x1xi32>
//      CHECK: %[[MMT4D:.+]] = linalg.batch_mmt4d ins(%[[LHS_EXT]], %[[EXPAND_RHS]] : tensor<32x688x64x16x2xi32>, tensor<32x1x64x1x2xi32>) outs(%[[FILL]] : tensor<32x688x1x16x1xi32>) -> tensor<32x688x1x16x1xi32>
//      CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMT4D]] {{\[}}[0], [1, 2], [3, 4]] : tensor<32x688x1x16x1xi32> into tensor<32x688x16xi32>
//      CHECK: %[[INIT_UNPACK:.+]] = tensor.empty() : tensor<32x11008xi32>
//      CHECK: %[[UNPACK:.+]] = tensor.unpack %[[COLLAPSED]] outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [16] into %[[INIT_UNPACK]] : tensor<32x688x16xi32> -> tensor<32x11008xi32>
//      CHECK: return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
func.func @matmul_transpose_a_f32f32f32(%arg0: tensor<256x128xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c512 = arith.constant 512 : index
  %3:2 = iree_linalg_ext.upper_bound_tile_size tensor<256x128xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  %4 = affine.apply #map3()[%3#0, %c256]
  %5 = affine.apply #map3()[%3#1, %c128]
  %padded = tensor.pad %arg0 low[0, 0] high[%4, %5] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<256x128xf32> to tensor<?x?xf32>
  %6 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<256x128xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %7:2 = iree_linalg_ext.upper_bound_tile_size tensor<256x512xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  %8 = affine.apply #map3()[%7#0, %c256]
  %9 = affine.apply #map3()[%7#1, %c512]
  %padded_0 = tensor.pad %arg1 low[0, 0] high[%8, %9] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<256x512xf32> to tensor<?x?xf32>
  %10 = iree_linalg_ext.set_encoding %padded_0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<256x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %11:2 = iree_linalg_ext.upper_bound_tile_size tensor<128x512xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  %12 = affine.apply #map3()[%11#0, %c128]
  %13 = affine.apply #map3()[%11#1, %c512]
  %padded_1 = tensor.pad %arg2 low[0, 0] high[%12, %13] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<128x512xf32> to tensor<?x?xf32>
  %14 = iree_linalg_ext.set_encoding %padded_1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %15 = linalg.matmul_transpose_a ins(%6, %10 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<256x128xf32>, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<256x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%14 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %16 = iree_linalg_ext.unset_encoding %15 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xf32>
  %extracted_slice = tensor.extract_slice %16[0, 0] [128, 512] [1, 1] : tensor<?x?xf32> to tensor<128x512xf32>
  return %extracted_slice : tensor<128x512xf32>
}

//      CHECK: func.func @matmul_transpose_a_f32f32f32(%[[LHS:.+]]: tensor<256x128xf32>, %[[RHS:.+]]: tensor<256x512xf32>, %[[RESULT:.+]]: tensor<128x512xf32>) -> tensor<128x512xf32>
//  CHECK-DAG:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<16x256x8x1xf32>
//  CHECK-DAG:   %[[PACK_LHS:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<256x128xf32> -> tensor<16x256x8x1xf32>
//  CHECK-DAG:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<128x256x4x1xf32>
//  CHECK-DAG:   %[[PACK_RHS:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<256x512xf32> -> tensor<128x256x4x1xf32>
//  CHECK-DAG:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<16x128x8x4xf32>
//  CHECK-DAG:   %[[PACK_RES:.+]] = tensor.pack %[[RESULT]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<128x512xf32> -> tensor<16x128x8x4xf32>
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[PACK_RES]] :
//  CHECK-DAG:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<128x512xf32>
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<16x128x8x4xf32> -> tensor<128x512xf32>
//      CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
func.func @matmul_transpose_b_f32f32f32(%arg0: tensor<128x256xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c512 = arith.constant 512 : index
  %3:2 = iree_linalg_ext.upper_bound_tile_size tensor<128x256xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  %4 = affine.apply #map3()[%3#0, %c128]
  %5 = affine.apply #map3()[%3#1, %c256]
  %padded = tensor.pad %arg0 low[0, 0] high[%4, %5] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<128x256xf32> to tensor<?x?xf32>
  %6 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<128x256xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %7:2 = iree_linalg_ext.upper_bound_tile_size tensor<512x256xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  %8 = affine.apply #map3()[%7#0, %c512]
  %9 = affine.apply #map3()[%7#1, %c256]
  %padded_0 = tensor.pad %arg1 low[0, 0] high[%8, %9] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<512x256xf32> to tensor<?x?xf32>
  %10 = iree_linalg_ext.set_encoding %padded_0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<512x256xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %11:2 = iree_linalg_ext.upper_bound_tile_size tensor<128x512xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  %12 = affine.apply #map3()[%11#0, %c128]
  %13 = affine.apply #map3()[%11#1, %c512]
  %padded_1 = tensor.pad %arg2 low[0, 0] high[%12, %13] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<128x512xf32> to tensor<?x?xf32>
  %14 = iree_linalg_ext.set_encoding %padded_1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %15 = linalg.matmul_transpose_b ins(%6, %10 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<128x256xf32>, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<512x256xf32>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%14 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %16 = iree_linalg_ext.unset_encoding %15 : tensor<?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xf32>
  %extracted_slice = tensor.extract_slice %16[0, 0] [128, 512] [1, 1] : tensor<?x?xf32> to tensor<128x512xf32>
  return %extracted_slice : tensor<128x512xf32>
}

//      CHECK: func.func @matmul_transpose_b_f32f32f32(%[[LHS:.+]]: tensor<128x256xf32>, %[[RHS:.+]]: tensor<512x256xf32>, %[[RESULT:.+]]: tensor<128x512xf32>) -> tensor<128x512xf32>
//  CHECK-DAG:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<16x256x8x1xf32>
//  CHECK-DAG:   %[[PACK_LHS:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<128x256xf32> -> tensor<16x256x8x1xf32>
//  CHECK-DAG:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<128x256x4x1xf32>
//  CHECK-DAG:   %[[PACK_RHS:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<512x256xf32> -> tensor<128x256x4x1xf32>
//  CHECK-DAG:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<16x128x8x4xf32>
//  CHECK-DAG:   %[[PACK_RES:.+]] = tensor.pack %[[RESULT]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<128x512xf32> -> tensor<16x128x8x4xf32>
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[PACK_RES]] :
//  CHECK-DAG:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<128x512xf32>
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[MMT4D]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<16x128x8x4xf32> -> tensor<128x512xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
func.func @batch_matmul_transpose_a_f32f32f32(%arg0: tensor<2x256x128xf32>, %arg1: tensor<2x256x512xf32>, %arg2: tensor<2x128x512xf32>) -> tensor<2x128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c2 = arith.constant 2 : index
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c512 = arith.constant 512 : index
  %3:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x256x128xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index, index
  %4 = affine.apply #map3()[%3#0, %c2]
  %5 = affine.apply #map3()[%3#1, %c256]
  %6 = affine.apply #map3()[%3#2, %c128]
  %padded = tensor.pad %arg0 low[0, 0, 0] high[%4, %5, %6] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst : f32
  } : tensor<2x256x128xf32> to tensor<?x?x?xf32>
  %7 = iree_linalg_ext.set_encoding %padded : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<2x256x128xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %8:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x256x512xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index, index
  %9 = affine.apply #map3()[%8#0, %c2]
  %10 = affine.apply #map3()[%8#1, %c256]
  %11 = affine.apply #map3()[%8#2, %c512]
  %padded_0 = tensor.pad %arg1 low[0, 0, 0] high[%9, %10, %11] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst : f32
  } : tensor<2x256x512xf32> to tensor<?x?x?xf32>
  %12 = iree_linalg_ext.set_encoding %padded_0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<2x256x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %13:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x128x512xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index, index
  %14 = affine.apply #map3()[%13#0, %c2]
  %15 = affine.apply #map3()[%13#1, %c128]
  %16 = affine.apply #map3()[%13#2, %c512]
  %padded_1 = tensor.pad %arg2 low[0, 0, 0] high[%14, %15, %16] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst : f32
  } : tensor<2x128x512xf32> to tensor<?x?x?xf32>
  %17 = iree_linalg_ext.set_encoding %padded_1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<2x128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %18 = linalg.batch_matmul_transpose_a ins(%7, %12 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<2x256x128xf32>, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<2x256x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%17 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<2x128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<2x128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %19 = iree_linalg_ext.unset_encoding %18 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<2x128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?x?xf32>
  %extracted_slice = tensor.extract_slice %19[0, 0, 0] [2, 128, 512] [1, 1, 1] : tensor<?x?x?xf32> to tensor<2x128x512xf32>
  return %extracted_slice : tensor<2x128x512xf32>
}

//      CHECK: func.func @batch_matmul_transpose_a_f32f32f32(%[[LHS:.+]]: tensor<2x256x128xf32>, %[[RHS:.+]]: tensor<2x256x512xf32>, %[[RESULT:.+]]: tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
//  CHECK-DAG:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<2x16x256x8x1xf32>
//  CHECK-DAG:   %[[PACK_LHS:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<2x256x128xf32> -> tensor<2x16x256x8x1xf32>
//  CHECK-DAG:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<2x128x256x4x1xf32>
//  CHECK-DAG:   %[[PACK_RHS:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<2x256x512xf32> -> tensor<2x128x256x4x1xf32>
//  CHECK-DAG:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<2x16x128x8x4xf32>
//  CHECK-DAG:   %[[PACK_RES:.+]] = tensor.pack %[[RESULT]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<2x128x512xf32> -> tensor<2x16x128x8x4xf32>
//      CHECK:   %[[BATCH_MMT4D:.+]] = linalg.batch_mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[PACK_RES]] :
//  CHECK-DAG:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<2x128x512xf32>
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[BATCH_MMT4D]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<2x16x128x8x4xf32> -> tensor<2x128x512xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
func.func @batch_matmul_transpose_b_f32f32f32(%arg0: tensor<2x128x256xf32>, %arg1: tensor<2x512x256xf32>, %arg2: tensor<2x128x512xf32>) -> tensor<2x128x512xf32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c512 = arith.constant 512 : index
  %3:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x128x256xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index, index
  %4 = affine.apply #map3()[%3#0, %c2]
  %5 = affine.apply #map3()[%3#1, %c128]
  %6 = affine.apply #map3()[%3#2, %c256]
  %padded = tensor.pad %arg0 low[0, 0, 0] high[%4, %5, %6] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst : f32
  } : tensor<2x128x256xf32> to tensor<?x?x?xf32>
  %7 = iree_linalg_ext.set_encoding %padded : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<2x128x256xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %8:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x512x256xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index, index
  %9 = affine.apply #map3()[%8#0, %c2]
  %10 = affine.apply #map3()[%8#1, %c512]
  %11 = affine.apply #map3()[%8#2, %c256]
  %padded_0 = tensor.pad %arg1 low[0, 0, 0] high[%9, %10, %11] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst : f32
  } : tensor<2x512x256xf32> to tensor<?x?x?xf32>
  %12 = iree_linalg_ext.set_encoding %padded_0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<2x512x256xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %13:3 = iree_linalg_ext.upper_bound_tile_size tensor<2x128x512xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index, index
  %14 = affine.apply #map3()[%13#0, %c2]
  %15 = affine.apply #map3()[%13#1, %c128]
  %16 = affine.apply #map3()[%13#2, %c512]
  %padded_1 = tensor.pad %arg2 low[0, 0, 0] high[%14, %15, %16] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst : f32
  } : tensor<2x128x512xf32> to tensor<?x?x?xf32>
  %17 = iree_linalg_ext.set_encoding %padded_1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<2x128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %18 = linalg.batch_matmul_transpose_b ins(%7, %12 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  LHS, element_types = [f32, f32, f32], original_type = tensor<2x128x256xf32>, user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RHS, element_types = [f32, f32, f32], original_type = tensor<2x512x256xf32>, user_indexing_maps = [#map, #map1, #map2]>>) outs(%17 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<2x128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<2x128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>>
  %19 = iree_linalg_ext.unset_encoding %18 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role =  RESULT, element_types = [f32, f32, f32], original_type = tensor<2x128x512xf32>, user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?x?xf32>
  %extracted_slice = tensor.extract_slice %19[0, 0, 0] [2, 128, 512] [1, 1, 1] : tensor<?x?x?xf32> to tensor<2x128x512xf32>
  return %extracted_slice : tensor<2x128x512xf32>
}

//      CHECK: func.func @batch_matmul_transpose_b_f32f32f32(%[[LHS:.+]]: tensor<2x128x256xf32>, %[[RHS:.+]]: tensor<2x512x256xf32>, %[[RESULT:.+]]: tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
//  CHECK-DAG:   %[[PACK_LHS_DEST:.+]] = tensor.empty() : tensor<2x16x256x8x1xf32>
//  CHECK-DAG:   %[[PACK_LHS:.+]] = tensor.pack %[[LHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 1] into %[[PACK_LHS_DEST]] : tensor<2x128x256xf32> -> tensor<2x16x256x8x1xf32>
//  CHECK-DAG:   %[[PACK_RHS_DEST:.+]] = tensor.empty() : tensor<2x128x256x4x1xf32>
//  CHECK-DAG:   %[[PACK_RHS:.+]] = tensor.pack %[[RHS]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [4, 1] into %[[PACK_RHS_DEST]] : tensor<2x512x256xf32> -> tensor<2x128x256x4x1xf32>
//  CHECK-DAG:   %[[PACK_RES_DEST:.+]] = tensor.empty() : tensor<2x16x128x8x4xf32>
//  CHECK-DAG:   %[[PACK_RES:.+]] = tensor.pack %[[RESULT]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[PACK_RES_DEST]] : tensor<2x128x512xf32> -> tensor<2x16x128x8x4xf32>
//      CHECK:   %[[BATCH_MMT4D:.+]] = linalg.batch_mmt4d
// CHECK-SAME:       ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:       outs(%[[PACK_RES]] :
//  CHECK-DAG:   %[[UNPACK_DEST:.+]] = tensor.empty() : tensor<2x128x512xf32>
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[BATCH_MMT4D]] outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 4] into %[[UNPACK_DEST]] : tensor<2x16x128x8x4xf32> -> tensor<2x128x512xf32>
