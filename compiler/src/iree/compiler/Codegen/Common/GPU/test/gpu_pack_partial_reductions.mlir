// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-pack-partial-reductions))" | FileCheck %s

// Pack reduction dim 2 (size 16384) with inner_tile=8 (thread[2]).
// partial_reduction=512, thread=8 → outer: partial_reduction=64, thread=1;
//                                   inner: partial_reduction=0, thread=8.
func.func @pack_reduction_dim_matvec(%a: tensor<4x16384xf16>, %b: tensor<1x16384xf16>) -> tensor<4x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x1xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x1xf32>) -> tensor<4x1xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<4x16384xf16>, tensor<1x16384xf16>)
    outs(%fill : tensor<4x1xf32>)
    attrs = {
      lowering_config = #iree_gpu.lowering_config<{
      lane_basis = [[1, 1, 64], [0, 1, 2]],
      partial_reduction = [0, 0, 512],
      subgroup_basis = [[1, 1, 1], [0, 1, 2]],
      thread = [0, 0, 8],
      workgroup = [4, 1, 0]}>} {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.extf %in_0 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// CHECK-LABEL: func.func @pack_reduction_dim_matvec
// CHECK:         linalg.pack %{{.*}} inner_dims_pos = [1] inner_tiles = [8]
// CHECK:         linalg.pack %{{.*}} inner_dims_pos = [1] inner_tiles = [8]
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:      lane_basis = {{\[}}[1, 1, 64, 1], [0, 1, 2, 3]{{\]}}
// CHECK-SAME:      partial_reduction = [0, 0, 64, 0]
// CHECK-SAME:      thread = [0, 0, 1, 8]
// CHECK-SAME:      workgroup = [4, 1, 0, 0]

// -----

// No packing when thread tile is 1 on reduction dim.
func.func @no_pack_thread_1(%a: tensor<?x4096xf16>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %d = tensor.dim %a, %c0 : tensor<?x4096xf16>
  %empty = tensor.empty(%d) : tensor<?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]}
    ins(%a : tensor<?x4096xf16>)
    outs(%fill : tensor<?xf32>)
    attrs = {
      lowering_config = #iree_gpu.lowering_config<{
      lane_basis = [[1, 64], [0, 1]],
      partial_reduction = [0, 4096],
      subgroup_basis = [[1, 8], [0, 1]],
      thread = [0, 1],
      workgroup = [1, 0]}>} {
  ^bb0(%in: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.addf %0, %out : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// CHECK-LABEL: func.func @no_pack_thread_1
// CHECK-NOT:     linalg.pack
// CHECK:         linalg.generic

// -----

// Check that linalg.index on a packed dim is replaced with outer * T + inner.
func.func @pack_with_linalg_index(%a: tensor<4x128xf32>) -> tensor<4xi32> {
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<4xi32>
  %fill = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]}
    ins(%a : tensor<4x128xf32>)
    outs(%fill : tensor<4xi32>)
    attrs = {
      lowering_config = #iree_gpu.lowering_config<{
      lane_basis = [[1, 32], [0, 1]],
      partial_reduction = [0, 128],
      subgroup_basis = [[1, 1], [0, 1]],
      thread = [0, 4],
      workgroup = [1, 0]}>} {
  ^bb0(%in: f32, %out: i32):
    %idx = linalg.index 1 : index
    %idx_i32 = arith.index_cast %idx : index to i32
    %sum = arith.addi %idx_i32, %out : i32
    linalg.yield %sum : i32
  } -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// CHECK-LABEL: func.func @pack_with_linalg_index
// CHECK:         linalg.generic
//       CHECK:     %[[OUTER:.+]] = linalg.index 1 : index
//       CHECK:     %[[INNER:.+]] = linalg.index 2 : index
//       CHECK:     affine.apply #[[$INDEX_MAP:.+]]()[%[[OUTER]], %[[INNER]]]
