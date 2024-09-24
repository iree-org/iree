// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fuse-encoding-ops-into-dispatch-regions-pass),canonicalize)" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>
module {
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
}

// CHECK-LABEL: @parallel_fusion
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32, #iree_encoding.encoding
// CHECK:         %[[ADD:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH0]] : tensor<2x11008x128xf32, #iree_encoding.encoding

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>
module {
  util.func public @reduction_fusion(%arg0: tensor<2x11008x128x16xf32>) -> tensor<2x11008x128xf32, #encoding> {
    %0 = tensor.empty() : tensor<2x11008x128xf32>
    %1 = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
      %5 = linalg.generic {
          indexing_maps = [#map, #map4],
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
}

// CHECK-LABEL: @reduction_fusion
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32, #iree_encoding.encoding
// CHECK:         %[[REDUCTION:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[REDUCTION]]
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]] : tensor<2x11008x128xf32, #iree_encoding.encoding

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>
module {
  util.func public @transpose_fusion(%arg0: tensor<2x128x11008xf32>) -> tensor<2x11008x128xf32, #encoding> {
    %0 = tensor.empty() : tensor<2x11008x128xf32>
    %1 = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
      %5 = linalg.generic {
          indexing_maps = [#map, #map4],
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
}

// CHECK-LABEL: @transpose_fusion
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32, #iree_encoding.encoding
// CHECK:         %[[TRANSPOSE:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[TRANSPOSE]]
// CHECK:         flow.return %[[SET_ENCODING]]
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]] : tensor<2x11008x128xf32, #iree_encoding.encoding

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>
module {
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
}

// CHECK-LABEL: @fusion_dynamic
// CHECK-SAME:    {{.+}}: tensor<?x?x?xf32>, %[[D0:.+]]: index, %[[D1:.+]]: index, %[[D2:.+]]: index)
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region -> (tensor<?x?x?xf32, #iree_encoding.encoding
// CHECK-SAME:      {%[[D0]], %[[D1]], %[[D2]]}
// CHECK:         %[[ADD:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH0]] : tensor<?x?x?xf32, #iree_encoding.encoding
