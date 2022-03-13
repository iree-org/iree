// RUN: iree-dialects-opt -linalg-interp-transforms -split-input-file %s 
// TODO: enable once https://reviews.llvm.org/D121369 lands
// | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Check that vectorization applies after interchange+tiling.

// CHECK-LABEL: @matmul_021
// CHECK-NOT: linalg.generic
// CHECK: vector.contract
func public @matmul_021(%arg0: tensor<39x154xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg1: tensor<154x5xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg2: tensor<39x5xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true}) -> tensor<39x5xf32> attributes {passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<39x154xf32>, tensor<154x5xf32>) outs(%arg2 : tensor<39x5xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %1 = arith.mulf %arg3, %arg4 : f32
    %2 = arith.addf %arg5, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<39x5xf32>
  return %0 : tensor<39x5xf32>
}

pdl.pattern @target_pattern : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg.generic"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  apply_native_constraint "nestedInFunc" [@matmul_021](%2 : !pdl.operation)
  rewrite %2 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @target_pattern
  %1 = tile %0 {interchange = [0, 2, 1], sizes = [3, 5, 14]}
  %2 = tile %1 {sizes = [3, 5, 2]}
  %3 = vectorize %2 {vectorize_padding = true}
}


// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Check that vectorization applies after interchange+tiling.

// CHECK-LABEL: @matmul_210
// CHECK-NOT: linalg.generic
// CHECK: vector.contract
func public @matmul_210(%arg0: tensor<39x154xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg1: tensor<154x5xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg2: tensor<39x5xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true}) -> tensor<39x5xf32> attributes {passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<39x154xf32>, tensor<154x5xf32>) outs(%arg2 : tensor<39x5xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %1 = arith.mulf %arg3, %arg4 : f32
    %2 = arith.addf %arg5, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<39x5xf32>
  return %0 : tensor<39x5xf32>
}

pdl.pattern @target_pattern : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg.generic"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  apply_native_constraint "nestedInFunc" [@matmul_210](%2 : !pdl.operation)
  rewrite %2 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @target_pattern
  %1 = tile %0 {interchange = [2, 1, 0], sizes = [3, 5, 14]}
  %2 = tile %1 {sizes = [3, 5, 2]}
  %3 = vectorize %2 {vectorize_padding = true}
}
