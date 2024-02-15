// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-flow-fold-unit-extent-dims))" %s | FileCheck %s

util.func public @no_fold_unit_dims_in_dispatches(%arg0 : tensor<1x1x10xf32>) -> tensor<1x1x10xf32> {
  %0 = tensor.empty() : tensor<1x1x10xf32>
  %1 = flow.dispatch.region[] -> (tensor<1x1x10xf32>) {
    %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<1x1x10xf32>) outs(%0 : tensor<1x1x10xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %3 = arith.addf %b0, %b0 : f32
      linalg.yield %3 : f32
    } -> tensor<1x1x10xf32>
    flow.return %2 : tensor<1x1x10xf32>
  }
  util.return %1 : tensor<1x1x10xf32>
}
//      CHECK: util.func public @no_fold_unit_dims_in_dispatches(%[[ARG0:.+]]: tensor<1x1x10xf32>)
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//      CHECK:     %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[ARG0]] : tensor<1x1x10xf32>)
//      CHECK:     flow.return %[[GENERIC]]
//      CHECK:   util.return %[[DISPATCH]]
