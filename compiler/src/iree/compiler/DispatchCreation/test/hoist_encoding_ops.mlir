// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-hoist-encoding-ops))" --split-input-file %s | FileCheck %s

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#lhs_encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x128x64xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>
#rhs_encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>
#result_encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>
module {
  util.func public @hoist_matmul_encodings(%arg0: tensor<2x128x64xf32>, %arg1: tensor<2x11008x128xf32>) -> tensor<2x11008x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %2 = flow.dispatch.region -> (tensor<2x11008x64xf32>) {
      %3 = iree_encoding.set_encoding %arg0 : tensor<2x128x64xf32> -> tensor<2x128x64xf32, #lhs_encoding>
      %4 = iree_encoding.set_encoding %arg1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #rhs_encoding>
      %5 = tensor.empty() : tensor<2x11008x64xf32, #result_encoding>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x11008x64xf32, #result_encoding>) -> tensor<2x11008x64xf32, #result_encoding>
      %7 = linalg.generic {
          indexing_maps = [#map1, #map2, #map3],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%3, %4 : tensor<2x128x64xf32, #lhs_encoding>, tensor<2x11008x128xf32, #rhs_encoding>)
          outs(%6 : tensor<2x11008x64xf32, #result_encoding>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %9 = arith.mulf %in, %in_0 : f32
        %10 = arith.addf %9, %out : f32
        linalg.yield %10 : f32
      } -> tensor<2x11008x64xf32, #result_encoding>
      %8 = iree_encoding.unset_encoding %7 : tensor<2x11008x64xf32, #result_encoding> -> tensor<2x11008x64xf32>
      flow.return %8 : tensor<2x11008x64xf32>
    }
    util.return %2 : tensor<2x11008x64xf32>
  }
}

// CHECK-LABEL: @hoist_matmul_encodings
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<2x128x64xf32>, %[[ARG1:.+]]: tensor<2x11008x128xf32>)
// CHECK-DAG:   %[[SET_ENCODING0:.+]] = iree_encoding.set_encoding %[[ARG0]] : tensor<2x128x64xf32> -> tensor<2x128x64xf32, #iree_encoding.encoding
// CHECK-DAG:   %[[SET_ENCODING1:.+]] = iree_encoding.set_encoding %[[ARG1]] : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #iree_encoding.encoding
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<2x11008x64xf32>) {
// CHECK:         %[[MATMUL:.+]] = linalg.generic {{.*}} ins(%[[SET_ENCODING0]], %[[SET_ENCODING1]]
// CHECK:         %[[UNSET_ENCODING1:.+]] = iree_encoding.unset_encoding %[[MATMUL]] : tensor<2x11008x64xf32, #iree_encoding.encoding
// CHECK:         flow.return %[[UNSET_ENCODING1]] : tensor<2x11008x64xf32>
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]] : tensor<2x11008x64xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], round_dims_to = array<i64: 32, 32, 32>>
util.func public @bubble_through_dequant(
    %arg0: tensor<2x11008x128xi8>, %arg1: tensor<2x11008xf32>, %arg2: tensor<2x11008xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %6 = flow.dispatch.region -> (tensor<2x11008x128xf32, #encoding>) {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map, #map1, #map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg1, %arg2 : tensor<2x11008x128xi8>, tensor<2x11008xf32>, tensor<2x11008xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %18 = arith.extui %in : i8 to i32
      %19 = arith.uitofp %18 : i32 to f32
      %20 = arith.subf %19, %in_1 : f32
      %21 = arith.mulf %20, %in_0 : f32
      linalg.yield %21 : f32
    } -> tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    flow.return %13 : tensor<2x11008x128xf32, #encoding>
  }
  util.return %6 : tensor<2x11008x128xf32, #encoding>
}

// CHECK-DAG:   #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG:   #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:   #[[$MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: @bubble_through_dequant
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x11008x128xi8>,
// CHECK-SAME:    %[[ARG1:.+]]: tensor<2x11008xf32>, %[[ARG2:.+]]: tensor<2x11008xf32>
// CHECK-DAG:   %[[SET_ENCODING0:.+]] = iree_encoding.set_encoding %[[ARG0]] : {{.*}} bcast_map = #[[$MAP4]]
// CHECK-DAG:   %[[SET_ENCODING1:.+]] = iree_encoding.set_encoding %[[ARG1]] : {{.*}} bcast_map = #[[$MAP3]]
// CHECK-DAG:   %[[SET_ENCODING2:.+]] = iree_encoding.set_encoding %[[ARG2]] : {{.*}} bcast_map = #[[$MAP3]]
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<2x11008x128xf32, #iree_encoding.encoding
// CHECK:         %[[DEQUANT:.+]] = linalg.generic {{.*}} ins(%[[SET_ENCODING0]], %[[SET_ENCODING1]], %[[SET_ENCODING2]] : {{.*}} outs(%[[INIT]] :
// CHECK:         flow.return %[[DEQUANT]]
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], round_dims_to = array<i64: 32, 32, 32>>
util.func public @bubble_through_broadcast(
    %arg0: tensor<11008x128xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %6 = flow.dispatch.region -> (tensor<2x11008x128xf32, #encoding>) {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<11008x128xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    flow.return %13 : tensor<2x11008x128xf32, #encoding>
  }
  util.return %6 : tensor<2x11008x128xf32, #encoding>
}

// CHECK-DAG:   #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG:   #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:   #[[$MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: @bubble_through_broadcast
// CHECK-SAME:    %[[ARG0:.+]]: tensor<11008x128xf32>
// CHECK-DAG:   %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[ARG0]] : {{.*}} bcast_map = #[[$MAP3]]
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<2x11008x128xf32, #iree_encoding.encoding
// CHECK:         %[[BROADCAST:.+]] = linalg.generic {{.*}} ins(%[[SET_ENCODING]] : {{.*}} outs(%[[INIT]] :
// CHECK:         flow.return %[[BROADCAST]]
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
module {
  util.func public @hoist_below(%arg0: tensor<2x11008x128xf32>) -> tensor<2x11008x128xf32, #encoding> {
    %0 = flow.dispatch.region -> (tensor<2x11008x128xf32, #encoding>) {
      %1 = tensor.empty() : tensor<2x11008x128xf32>
      %2 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg0 : tensor<2x11008x128xf32>, tensor<2x11008x128xf32>) outs(%1 : tensor<2x11008x128xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %4 = arith.addf %in, %in_0 : f32
        linalg.yield %4 : f32
      } -> tensor<2x11008x128xf32>
      %3 = iree_encoding.set_encoding %2 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
      flow.return %3 : tensor<2x11008x128xf32, #encoding>
    }
    util.return %0 : tensor<2x11008x128xf32, #encoding>
  }
}

// CHECK-LABEL: @hoist_below
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x11008x128xf32>
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<2x11008x128xf32>
// CHECK:         %[[ADD:.+]] = linalg.generic {{.*}} ins(%[[ARG0]], %[[ARG0]] : {{.*}} outs(%[[INIT]] :
// CHECK:         flow.return %[[ADD]]
// CHECK:       }
// CHECK:       %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[DISPATCH]]
// CHECK:       util.return %[[SET_ENCODING]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<?x?x?xf32>, user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
module {
  util.func public @hoist_dynamic(%arg0: tensor<?x?x?xf32>, %d0: index, %d1: index, %d2: index) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32, #encoding>) {
    %0:2 = flow.dispatch.region -> (tensor<?x?x?xf32>{%d0, %d1, %d2}, tensor<?x?x?xf32, #encoding>{%d0, %d1, %d2}) {
      %1 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
      %2 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg0 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%1 : tensor<?x?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %4 = arith.addf %in, %in_0 : f32
        linalg.yield %4 : f32
      } -> tensor<?x?x?xf32>
      %3 = iree_encoding.set_encoding %2 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding>
      flow.return %2, %3 : tensor<?x?x?xf32>, tensor<?x?x?xf32, #encoding>
    }
    util.return %0#0, %0#1 : tensor<?x?x?xf32>, tensor<?x?x?xf32, #encoding>
  }
}

// CHECK-LABEL: @hoist_dynamic
// CHECK-SAME:    %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[D0:.+]]: index, %[[D1:.+]]: index, %[[D2:.+]]: index)
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<?x?x?xf32>{%[[D0]], %[[D1]], %[[D2]]})
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]]) : tensor<?x?x?xf32>
// CHECK:         %[[ADD:.+]] = linalg.generic {{.*}} ins(%[[ARG0]], %[[ARG0]] : {{.*}} outs(%[[INIT]] :
// CHECK:         flow.return %[[ADD]]
// CHECK:       }
// CHECK:       %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[DISPATCH]]
// CHECK:       util.return %[[DISPATCH]], %[[SET_ENCODING]]
