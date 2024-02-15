// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-flow-form-scalar-dispatches))" --split-input-file %s | FileCheck %s

#map = affine_map<() -> ()>
util.func public @simpleDAG(
    %arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>, %arg3 : tensor<f32>)
    -> (tensor<f32>, tensor<f32>) {
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
      ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
      %2 = arith.addf %b0, %b1 : f32
      linalg.yield %2 : f32
    } -> tensor<f32>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
      ins(%1, %arg3 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
      %4 = arith.mulf %b0, %b1 : f32
      linalg.yield %4 : f32
    } -> tensor<f32>
  %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
      ins(%arg2, %3 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
      %6 = arith.subf %b1, %b0 : f32
      linalg.yield %6 : f32
    } -> tensor<f32>
  util.return %1, %5 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL: util.func public @simpleDAG(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<f32>)
//       CHECK:   %[[RESULT:.+]]:2 = flow.dispatch.region
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0]], %[[ARG1]] :
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC1]], %[[ARG3]] :
//       CHECK:     %[[GENERIC3:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG2]], %[[GENERIC2]] :
//       CHECK:     flow.return %[[GENERIC3]], %[[GENERIC1]]
//       CHECK:     count() -> (index, index, index)
//  CHECK-NEXT:       %[[C1:.+]] = arith.constant 1 : index
//  CHECK-NEXT:       flow.return %[[C1]], %[[C1]], %[[C1]]
//       CHECK:   util.return %[[RESULT]]#1, %[[RESULT]]#0

// -----

#map = affine_map<() -> ()>
util.func public @simpleHorizontal(
    %arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>, %arg3 : tensor<f32>)
    -> (tensor<f32>, tensor<f32>) {
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
      ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
      %2 = arith.addf %b0, %b1 : f32
      linalg.yield %2 : f32
    } -> tensor<f32>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
      ins(%1, %arg2 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
      %4 = arith.mulf %b0, %b1 : f32
      linalg.yield %4 : f32
    } -> tensor<f32>
  %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []}
      ins(%arg3 : tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32) :
      %6 = arith.addf %b0, %b0 : f32
      linalg.yield %6 : f32
    } -> tensor<f32>
  util.return %3, %5 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL: util.func public @simpleHorizontal
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<f32>
//       CHECK:   %[[RESULT:.+]]:2 = flow.dispatch.region
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0]], %[[ARG1]] :
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC1]], %[[ARG2]] :
//       CHECK:     %[[GENERIC3:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG3]] :
//       CHECK:     flow.return %[[GENERIC3]], %[[GENERIC2]]
//       CHECK:     count() -> (index, index, index)
//  CHECK-NEXT:       %[[C1:.+]] = arith.constant 1 : index
//  CHECK-NEXT:       flow.return %[[C1]], %[[C1]], %[[C1]]
//       CHECK:   util.return %[[RESULT]]#1, %[[RESULT]]#0

// -----

#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0) -> (d0)>
util.func public @interleaving(
      %arg0 : tensor<1x1xf32>, %arg1 : tensor<1xf32>, %arg2 : tensor<f32>, %arg3 : tensor<f32>)
      -> (tensor<f32>, tensor<1xf32>) {
    %cst = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "reduction"]}
        ins(%arg0, %arg1 : tensor<1x1xf32>, tensor<1xf32>) outs(%1 : tensor<1xf32>) {
      ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
        %3 = arith.mulf %b0, %b1 : f32
        %4 = arith.addf %3, %b2 : f32
        linalg.yield %4 : f32
      } -> tensor<1xf32>
    %5 = tensor.empty() : tensor<f32>
    %6 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []}
        ins(%arg2, %arg3 : tensor<f32>, tensor<f32>) outs(%5 : tensor<f32>) {
      ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
        %7 = arith.subf %b1, %b0 : f32
        linalg.yield %7 : f32
      } -> tensor<f32>
      cf.br ^b1
  ^b1:
    %7 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]}
        ins(%2, %arg1 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) {
      ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
        %8 = arith.mulf %b0, %b1 : f32
        linalg.yield %8 : f32
      } -> tensor<1xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []}
        ins(%6, %arg3 : tensor<f32>, tensor<f32>) outs(%5 : tensor<f32>) {
      ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
        %10 = arith.divf %b1, %b0 : f32
        linalg.yield %10 : f32
      } -> tensor<f32>
    util.return %9, %7 : tensor<f32>, tensor<1xf32>
}
// CHECK-LABEL: util.func public @interleaving(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<1x1xf32>,
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<1xf32>,
//  CHECK-SAME:     %[[ARG2:.+]]: tensor<f32>,
//  CHECK-SAME:     %[[ARG3:.+]]: tensor<f32>)
//       CHECK:     %[[EMPTY0:.+]] = tensor.empty() : tensor<1xf32>
//       CHECK:     %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:         outs(%[[EMPTY0]] :
//       CHECK:     %[[EMPTY1:.+]] = tensor.empty() : tensor<f32>
//       CHECK:   %[[DISPATCH0:.+]]:2 = flow.dispatch.region
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0]], %[[ARG1]] :
//  CHECK-SAME:         outs(%[[FILL]] :
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:         outs(%[[EMPTY1]] :
//       CHECK:    flow.return %[[GENERIC1]], %[[GENERIC0]]
//       CHECK: ^bb1:
//   CHECK-DAG:   %[[DISPATCH1:.+]]:2 = flow.dispatch.region
//       CHECK:     %[[GENERIC2:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[DISPATCH0]]#1, %[[ARG1]] :
//  CHECK-SAME:         outs(%[[EMPTY0]] :
//       CHECK:     %[[GENERIC3:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[DISPATCH0]]#0, %[[ARG3]] :
//  CHECK-SAME:         outs(%[[EMPTY1]] :
//       CHECK:    flow.return %[[GENERIC3]], %[[GENERIC2]]
//       CHECK:  util.return %[[DISPATCH1]]#0, %[[DISPATCH1]]#1
