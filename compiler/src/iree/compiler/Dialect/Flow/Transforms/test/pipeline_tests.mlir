// TODO(hanchung): Split the transformation pipeline tests into two mlir files.
// RUN: iree-opt --iree-global-optimization-transformation-pipeline --iree-flow-transformation-pipeline --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> ()>
module {
  util.func public @main(%arg0: tensor<833xi32>, %arg1: tensor<833x833xf32>, %arg2: tensor<f32>) -> tensor<f32> {
    %cst = arith.constant 5.66893432E-4 : f32
    %0 = tensor.empty() : tensor<833x833xf32>
    %1 = linalg.generic {
        indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]}
        ins(%arg1, %arg2 : tensor<833x833xf32>, tensor<f32>)
        outs(%0 : tensor<833x833xf32>) {
      ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
        %2 = arith.divf %b0, %b1 : f32
        linalg.yield %2 : f32
      } -> tensor<833x833xf32>
    %4 = linalg.generic {
        indexing_maps = [#map, #map1, #map2, #map2], iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg0, %1 : tensor<833xi32>, tensor<833xi32>, tensor<833x833xf32>)
        outs(%0 : tensor<833x833xf32>) {
      ^bb0(%b0 : i32, %b1 : i32, %b2 : f32, %b3 : f32):
        %5 = arith.cmpi eq, %b0, %b1 : i32
        %6 = arith.select %5, %b2, %cst : f32
        linalg.yield %6 : f32
      } -> tensor<833x833xf32>
    %7 = tensor.empty() : tensor<f32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<f32>) -> tensor<f32>
    %9 = linalg.generic {
        indexing_maps = [#map2, #map3], iterator_types = ["reduction", "reduction"]}
        ins(%4 : tensor<833x833xf32>) outs(%7 : tensor<f32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %10 = arith.addf %b1, %b0 : f32
        linalg.yield %10 : f32
      } -> tensor<f32>
    util.return %9 : tensor<f32>
  }
}
// Check that the linalg op with two reduction loops get folded into a single reduction
// which then prevents the parallel ops to be folded into it.
// See https://github.com/openxla/iree/issues/13285
//       CHECK:   flow.executable private @[[EXECUTABLE0:[a-zA-Z0-9_]+]]
//       CHECK:     func.func @[[FUNC0:[a-zA-Z0-9_x]+]]
//       CHECK:       linalg.generic
//  CHECK-SAME:         ["reduction", "reduction"]
//   CHECK-NOT:       linalg.generic
//       CHECK:   util.func public @main(
//       CHECK:     %[[T0:.+]] = flow.dispatch @[[EXECUTABLE0]]::@[[FUNC0]]
//       CHECK:     util.return %[[T0]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module {
  util.func public @grouped_quantized_matmul(%arg0: tensor<4096x32x128xi4>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4096xf32>
    %1 = tensor.empty() : tensor<4096x32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi4>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i4 to i32
      %6 = arith.uitofp %5 : i32 to f32
      %7 = arith.subf %6, %in_1 : f32
      %8 = arith.mulf %7, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4096x32x128xf32>
    %4 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<1x1x4096xf32>
    util.return %4 : tensor<1x1x4096xf32>
  }
}
// Check that the two linalg.generic ops are fused into the same dispatch
//       CHECK:   flow.executable private @[[EXECUTABLE0:[a-zA-Z0-9_]+]]
//       CHECK:   func.func @[[FUNC0:[a-zA-Z0-9_x]+]]
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:         ["parallel", "parallel", "parallel"]
//       CHECK:   arith.extui
//       CHECK:   arith.uitofp
//       CHECK:   arith.subf
//       CHECK:   arith.mulf
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       ["parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(
//  CHECK-SAME:       %[[GEN0]]
//  CHECK-SAME:       outs(
//       CHECK:   arith.mulf
//       CHECK:   arith.addf
//       CHECK:   flow.dispatch.tensor.store %[[GEN1]]
//       CHECK:   util.func public @grouped_quantized_matmul(
//       CHECK:     %[[T0:.+]] = flow.dispatch @[[EXECUTABLE0]]::@[[FUNC0]]
//       CHECK:     %[[RS:.+]] = flow.tensor.reshape %[[T0]] : tensor<4096xf32> -> tensor<1x1x4096xf32>
//       CHECK:     util.return %[[RS]]
