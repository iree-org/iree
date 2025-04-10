// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fuse-encoding-ops-into-dispatch-regions-pass))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.testing_encoding<>
util.func public @parallel_fusion(%arg0: tensor<2x11008x128xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x11008x128xf32>
  %1 = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
    %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg0 : tensor<2x11008x128xf32>, tensor<2x11008x128xf32>)
        outs(%0 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<2x11008x128xf32>
    flow.return %3 : tensor<2x11008x128xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
  util.return %2 : tensor<2x11008x128xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK-LABEL: @parallel_fusion
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32, #[[$ENCODING]]>)
// CHECK:         %[[ADD:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH0]] : tensor<2x11008x128xf32, #[[$ENCODING]]>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.testing_encoding<>
util.func public @reduction_fusion(%arg0: tensor<2x11008x128x16xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %0 = tensor.empty() : tensor<2x11008x128xf32>
  %1 = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
    %5 = linalg.generic {
        indexing_maps = [#map, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0 : tensor<2x11008x128x16xf32>)
        outs(%0 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<2x11008x128xf32>
    flow.return %5 : tensor<2x11008x128xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
  util.return %2 : tensor<2x11008x128xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK-LABEL: @reduction_fusion
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32>)
// CHECK:         %[[REDUCTION:.+]] = linalg.generic
// CHECK:         flow.return %[[REDUCTION]] :
// CHECK:       }
// CHECK:       %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[DISPATCH]]
// CHECK:       util.return %[[SET_ENCODING]] : tensor<2x11008x128xf32, #[[$ENCODING]]>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#encoding = #iree_encoding.testing_encoding<>
util.func public @transpose_fusion(%arg0: tensor<2x128x11008xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %0 = tensor.empty() : tensor<2x11008x128xf32>
  %1 = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
    %5 = linalg.generic {
        indexing_maps = [#map, #map1],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<2x128x11008xf32>)
        outs(%0 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x11008x128xf32>
    flow.return %5 : tensor<2x11008x128xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
  util.return %2 : tensor<2x11008x128xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK-LABEL: @transpose_fusion
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32, #[[$ENCODING]]>
// CHECK:         %[[TRANSPOSE:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[TRANSPOSE]]
// CHECK:         flow.return %[[SET_ENCODING]]
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]] : tensor<2x11008x128xf32, #[[$ENCODING]]>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.testing_encoding<>
util.func public @fusion_dynamic(%arg0: tensor<?x?x?xf32>, %d0: index, %d1: index, %d2: index) -> tensor<?x?x?xf32, #encoding> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %1 = flow.dispatch.region -> (tensor<?x?x?xf32>{%d0, %d1, %d2}) {
    %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg0 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%0 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?x?xf32>
    flow.return %3 : tensor<?x?x?xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding>
  util.return %2 : tensor<?x?x?xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK-LABEL: @fusion_dynamic
// CHECK-SAME:    {{.+}}: tensor<?x?x?xf32>, %[[D0:.+]]: index, %[[D1:.+]]: index, %[[D2:.+]]: index)
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region -> (tensor<?x?x?xf32, #[[$ENCODING]]>
// CHECK-SAME:      {%[[D0]], %[[D1]], %[[D2]]}
// CHECK:         %[[ADD:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH0]] : tensor<?x?x?xf32, #[[$ENCODING]]>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.testing_encoding<>
util.func public @multi_encoding_fusion_dynamic(%arg0: tensor<?x?x?xf32>, %d0: index, %d1: index, %d2: index) -> (tensor<?x?x?xf32, #encoding>, tensor<?x?x?xf32, #encoding>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %1 = flow.dispatch.region -> (tensor<?x?x?xf32>{%d0, %d1, %d2}) {
    %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg0 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%0 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?x?xf32>
    flow.return %3 : tensor<?x?x?xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding>
  %3 = iree_encoding.set_encoding %1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding>
  util.return %2, %3 : tensor<?x?x?xf32, #encoding>, tensor<?x?x?xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK-LABEL: @multi_encoding_fusion_dynamic
// CHECK-SAME:    {{.+}}: tensor<?x?x?xf32>, %[[D0:.+]]: index, %[[D1:.+]]: index, %[[D2:.+]]: index)
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<?x?x?xf32, #[[$ENCODING]]>
// CHECK-SAME:      {%[[D0]], %[[D1]], %[[D2]]}
// CHECK:         %[[ADD:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]], %[[DISPATCH]]
