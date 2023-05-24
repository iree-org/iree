// RUN: iree-opt --iree-flow-transformation-pipeline --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> ()>
module {
  func.func @main(%arg0: tensor<833xi32>, %arg1: tensor<833x833xf32>, %arg2: tensor<f32>) -> tensor<f32> {
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
    return %9 : tensor<f32>
  }
}
// Check that the linalg op with two reduction loops get folded into a single reduction
// which then prevents the parallel ops to be folded into it.
// See https://github.com/openxla/iree/issues/13285
//       CHECK:   flow.executable private @[[EXECUTABLE0:[a-zA-Z0-9_]+]]
//       CHECK:     func.func @[[FUNC0:[a-zA-Z0-9_x]+]]
//       CHECK:       linalg.generic
//  CHECK-SAME:         ["parallel", "parallel"]
//       CHECK:   flow.executable private @[[EXECUTABLE1:[a-zA-Z0-9_]+]]
//       CHECK:     func.func @[[FUNC1:[a-zA-Z0-9_x]+]]
//       CHECK:       linalg.generic
//  CHECK-SAME:         ["reduction"]
//       CHECK:   func.func @main(
//       CHECK:     %[[T0:.+]] = flow.dispatch @[[EXECUTABLE0]]::@[[FUNC0]]
//       CHECK:     %[[T1:.+]] = flow.tensor.reshape %[[T0]] : tensor<833x833xf32> -> tensor<693889xf32>
//       CHECK:     %[[T2:.+]] = flow.dispatch @[[EXECUTABLE1]]::@[[FUNC1]](%[[T1]])
//       CHECK:     return %[[T2]]
