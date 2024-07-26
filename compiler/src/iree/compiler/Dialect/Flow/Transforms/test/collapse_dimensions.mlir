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


// -----
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
util.func public @unpack_collapse(%arg0: tensor<2x320x128x128xf32>, %arg1: tensor<320xf32>, %arg2: tensor<320xf32>, %arg3: tensor<1x5x2x64xf32>) -> tensor<2x320x128x128xf16> {
  %dispatch = flow.dispatch.region -> (tensor<2x320x128x128xf16>) {
    %0 = tensor.empty() : tensor<2x320xf32>
    %unpack = tensor.unpack %arg3 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [2, 64] into %0 : tensor<1x5x2x64xf32> -> tensor<2x320xf32>
    %1 = tensor.empty() : tensor<2x320x128x128xf16>
    %2 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map1, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }
      ins(%arg0, %arg1, %unpack, %arg2 : tensor<2x320x128x128xf32>, tensor<320xf32>, tensor<2x320xf32>, tensor<320xf32>)
      outs(%1 : tensor<2x320x128x128xf16>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f16):
      %3 = arith.addf %in_1, %in_2 : f32
      %4 = arith.addf %in, %in_0 : f32
      %5 = arith.truncf %3 : f32 to f16
      %6 = arith.truncf %4 : f32 to f16
      %7 = arith.addf %6, %5 : f16
      linalg.yield %7 : f16
    } -> tensor<2x320x128x128xf16>
    flow.return %2 : tensor<2x320x128x128xf16>
  }
  util.return %dispatch : tensor<2x320x128x128xf16>
}

// CHECK-LABEL:  util.func public @unpack_collapse
//       CHECK:    %[[GEN:.+]] = linalg.generic
//  CHECK-SAME:      tensor<2x320x16384xf32>, tensor<320xf32>, tensor<2x320xf32>, tensor<320xf32>
