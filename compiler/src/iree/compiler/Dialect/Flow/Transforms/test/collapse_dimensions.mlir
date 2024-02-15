// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-flow-collapse-dimensions))" %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
util.func public @do_not_collapse_cst_in_place(%arg0: tensor<1x1x2304xf32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x1x2304xf32>
  %0 = tensor.empty() : tensor<1x1x2304xf32>
  %1 = flow.dispatch.region -> (tensor<1x1x2304xf32>) {
    %2 = tensor.empty() : tensor<1x1x2304xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst : tensor<1x1x2304xf32>, tensor<1x1x2304xf32>) outs(%2 : tensor<1x1x2304xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<1x1x2304xf32>
    flow.return %3 : tensor<1x1x2304xf32>
  }
  util.return
}
// CHECK-LABEL: util.func public @do_not_collapse_cst_in_place
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]]]
// CHECK-DAG:     %[[CST:.+]] = arith.constant
// CHECK-DAG:     %[[COLLAPSED_ARG0:.+]] = tensor.collapse_shape %[[ARG0]]
// CHECK-DAG:     %[[COLLAPSED_CST:.+]] = tensor.collapse_shape %[[CST]]
// CHECK:         %{{.+}} = flow.dispatch.region
// CHECK:            %[[RES:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[COLLAPSED_ARG0]], %[[COLLAPSED_CST]]
// CHECK:            flow.return %[[RES]]
