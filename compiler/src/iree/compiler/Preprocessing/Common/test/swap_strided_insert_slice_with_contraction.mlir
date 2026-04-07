// RUN: iree-opt --split-input-file --mlir-print-local-scope --iree-preprocessing-swap-strided-insert-slice-with-contraction %s | FileCheck %s

// Swapped: 1x1 backward conv with stride-2 scatter feeding a contraction
// followed by truncf. The contraction moves before the scatter and operates
// on the small (un-scattered) source.
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @swap_1x1_with_truncf(
    %src: tensor<2x8x8x16xf16>,
    %filter: tensor<16x1x1x4xf16>) -> tensor<2x18x18x4xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x18x18x16xf16>
  %inserted = tensor.insert_slice %src into %cst[0, 0, 0, 0] [2, 8, 8, 16] [1, 2, 2, 1]
    : tensor<2x8x8x16xf16> into tensor<2x18x18x16xf16>
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %empty_f32 = tensor.empty() : tensor<2x18x18x4xf32>
  %fill = linalg.fill ins(%cst_f32 : f32) outs(%empty_f32 : tensor<2x18x18x4xf32>) -> tensor<2x18x18x4xf32>
  %conv = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%inserted, %filter : tensor<2x18x18x16xf16>, tensor<16x1x1x4xf16>)
    outs(%fill : tensor<2x18x18x4xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.extf %in_0 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<2x18x18x4xf32>
  %empty_f16 = tensor.empty() : tensor<2x18x18x4xf16>
  %trunc = linalg.generic {
    indexing_maps = [#map3, #map3],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv : tensor<2x18x18x4xf32>) outs(%empty_f16 : tensor<2x18x18x4xf16>) {
  ^bb0(%in: f32, %out: f16):
    %0 = arith.truncf %in : f32 to f16
    linalg.yield %0 : f16
  } -> tensor<2x18x18x4xf16>
  util.return %trunc : tensor<2x18x18x4xf16>
}

// CHECK-LABEL: @swap_1x1_with_truncf
// CHECK-SAME:      %[[SRC:.*]]: tensor<2x8x8x16xf16>, %[[FILTER:.*]]: tensor<16x1x1x4xf16>
// CHECK-NOT:   tensor.insert_slice
// CHECK:       %[[CONV:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[SRC]], %[[FILTER]] : tensor<2x8x8x16xf16>, tensor<16x1x1x4xf16>)
// CHECK-SAME:      outs({{.*}} : tensor<2x8x8x4xf32>)
// CHECK:       %[[TRUNC:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[CONV]] : tensor<2x8x8x4xf32>)
// CHECK:       } -> tensor<2x8x8x4xf16>
// CHECK:       %[[OUT:.*]] = tensor.insert_slice %[[TRUNC]] into
// CHECK-SAME:      [0, 0, 0, 0] [2, 8, 8, 4] [1, 2, 2, 1]
// CHECK-SAME:      tensor<2x8x8x4xf16> into tensor<2x18x18x4xf16>
// CHECK:       util.return %[[OUT]]

// -----

// Swapped: matmul-like contraction without consumer chain (no truncf).
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
util.func public @swap_matmul_no_consumer(
    %src: tensor<4x10x10x32xf16>,
    %filter: tensor<8x32xf16>) -> tensor<4x22x22x8xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<4x22x22x32xf16>
  %inserted = tensor.insert_slice %src into %cst[0, 0, 0, 0] [4, 10, 10, 32] [1, 2, 2, 1]
    : tensor<4x10x10x32xf16> into tensor<4x22x22x32xf16>
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x22x22x8xf32>
  %fill = linalg.fill ins(%cst_f32 : f32) outs(%empty : tensor<4x22x22x8xf32>) -> tensor<4x22x22x8xf32>
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
  } ins(%inserted, %filter : tensor<4x22x22x32xf16>, tensor<8x32xf16>)
    outs(%fill : tensor<4x22x22x8xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.extf %in_0 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<4x22x22x8xf32>
  util.return %result : tensor<4x22x22x8xf32>
}

// CHECK-LABEL: @swap_matmul_no_consumer
// CHECK-SAME:      %[[SRC:.*]]: tensor<4x10x10x32xf16>, %[[FILTER:.*]]: tensor<8x32xf16>
// CHECK-NOT:   tensor.insert_slice
// CHECK:       %[[MATMUL:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[SRC]], %[[FILTER]] : tensor<4x10x10x32xf16>, tensor<8x32xf16>)
// CHECK-SAME:      outs({{.*}} : tensor<4x10x10x8xf32>)
// CHECK:       %[[OUT:.*]] = tensor.insert_slice %[[MATMUL]] into
// CHECK-SAME:      [0, 0, 0, 0] [4, 10, 10, 8] [1, 2, 2, 1]
// CHECK-SAME:      tensor<4x10x10x8xf32> into tensor<4x22x22x8xf32>
// CHECK:       util.return %[[OUT]]

// -----

// No transformation: 3x3 conv has reduction dims with loop bound > 1.
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @no_swap_3x3_conv(
    %src: tensor<1x4x4x8xf16>,
    %filter: tensor<8x3x3x4xf16>) -> tensor<1x8x8x4xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x10x10x8xf16>
  %inserted = tensor.insert_slice %src into %cst[0, 1, 1, 0] [1, 4, 4, 8] [1, 2, 2, 1]
    : tensor<1x4x4x8xf16> into tensor<1x10x10x8xf16>
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1x8x8x4xf32>
  %fill = linalg.fill ins(%cst_f32 : f32) outs(%empty : tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%inserted, %filter : tensor<1x10x10x8xf16>, tensor<8x3x3x4xf16>)
    outs(%fill : tensor<1x8x8x4xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.extf %in_0 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<1x8x8x4xf32>
  util.return %result : tensor<1x8x8x4xf32>
}

// CHECK-LABEL: @no_swap_3x3_conv
// CHECK:       tensor.insert_slice
// CHECK:       linalg.generic
// CHECK-SAME:      ins({{.*}} : tensor<1x10x10x8xf16>, tensor<8x3x3x4xf16>)

// -----

// No transformation: destination is not a zero constant.
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
util.func public @no_swap_nonzero_dest(
    %src: tensor<4x8xf32>,
    %dest: tensor<8x8xf32>,
    %filter: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %inserted = tensor.insert_slice %src into %dest[0, 0] [4, 8] [2, 1]
    : tensor<4x8xf32> into tensor<8x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x4xf32>) -> tensor<8x4xf32>
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%inserted, %filter : tensor<8x8xf32>, tensor<4x8xf32>)
    outs(%fill : tensor<8x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.mulf %in, %in_0 : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<8x4xf32>
  util.return %result : tensor<8x4xf32>
}

// CHECK-LABEL: @no_swap_nonzero_dest
// CHECK:       tensor.insert_slice %{{.*}} into %{{.*}}
// CHECK:       linalg.generic
// CHECK-SAME:      ins({{.*}} : tensor<8x8xf32>, tensor<4x8xf32>)

// -----

// No transformation: all strides are 1.
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
util.func public @no_swap_unit_strides(
    %src: tensor<4x8xf32>,
    %filter: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %cst_dest = arith.constant dense<0.000000e+00> : tensor<8x8xf32>
  %inserted = tensor.insert_slice %src into %cst_dest[0, 0] [4, 8] [1, 1]
    : tensor<4x8xf32> into tensor<8x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x4xf32>) -> tensor<8x4xf32>
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%inserted, %filter : tensor<8x8xf32>, tensor<4x8xf32>)
    outs(%fill : tensor<8x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.mulf %in, %in_0 : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<8x4xf32>
  util.return %result : tensor<8x4xf32>
}

// CHECK-LABEL: @no_swap_unit_strides
// CHECK:       tensor.insert_slice
// CHECK:       linalg.generic
// CHECK-SAME:      ins({{.*}} : tensor<8x8xf32>, tensor<4x8xf32>)

// -----

// No transformation: strided dim has a scaled indexing expression (3*d0).
#map_scaled0 = affine_map<(d0, d1, d2) -> (3 * d0, d2)>
#map_scaled1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_scaled2 = affine_map<(d0, d1, d2) -> (d0, d1)>
util.func public @no_swap_scaled_dim(
    %src: tensor<4x8xf32>,
    %filter: tensor<4x8xf32>) -> tensor<2x4xf32> {
  %cst_dest = arith.constant dense<0.000000e+00> : tensor<8x8xf32>
  %inserted = tensor.insert_slice %src into %cst_dest[0, 0] [4, 8] [2, 1]
    : tensor<4x8xf32> into tensor<8x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>
  %result = linalg.generic {
    indexing_maps = [#map_scaled0, #map_scaled1, #map_scaled2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%inserted, %filter : tensor<8x8xf32>, tensor<4x8xf32>)
    outs(%fill : tensor<2x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.mulf %in, %in_0 : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<2x4xf32>
  util.return %result : tensor<2x4xf32>
}

// CHECK-LABEL: @no_swap_scaled_dim
// CHECK:       tensor.insert_slice
// CHECK:       linalg.generic
// CHECK-SAME:      ins({{.*}} : tensor<8x8xf32>, tensor<4x8xf32>)

// -----

// No transformation: strided dim maps to sum of two parallel dims (d0 + d1).
#map_sum0 = affine_map<(d0, d1, d2, d3) -> (d0 + d1, d3)>
#map_sum1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map_sum2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
util.func public @no_swap_parallel_sum(
    %src: tensor<4x8xf32>,
    %filter: tensor<4x8xf32>) -> tensor<3x3x4xf32> {
  %cst_dest = arith.constant dense<0.000000e+00> : tensor<8x8xf32>
  %inserted = tensor.insert_slice %src into %cst_dest[0, 0] [4, 8] [2, 1]
    : tensor<4x8xf32> into tensor<8x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<3x3x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<3x3x4xf32>) -> tensor<3x3x4xf32>
  %result = linalg.generic {
    indexing_maps = [#map_sum0, #map_sum1, #map_sum2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%inserted, %filter : tensor<8x8xf32>, tensor<4x8xf32>)
    outs(%fill : tensor<3x3x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.mulf %in, %in_0 : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<3x3x4xf32>
  util.return %result : tensor<3x3x4xf32>
}

// CHECK-LABEL: @no_swap_parallel_sum
// CHECK:       tensor.insert_slice
// CHECK:       linalg.generic
// CHECK-SAME:      ins({{.*}} : tensor<8x8xf32>, tensor<4x8xf32>)
