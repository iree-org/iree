// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-flow-fuse-encoding-ops-into-dispatch-regions-pass),canonicalize)" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  util.func public @quantized_matmul(%arg0: tensor<2x11008x128xf32>, %arg1: tensor<2x128x64xf32>) -> tensor<2x11008x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x11008x128xf32>
    %1 = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
      %5 = linalg.generic {
          indexing_maps = [#map, #map, #map],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%arg0, %arg0 : tensor<2x11008x128xf32>, tensor<2x11008x128xf32>)
          outs(%0 : tensor<2x11008x128xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.addf %in, %in_0 : f32
        linalg.yield %6 : f32
      } -> tensor<2x11008x128xf32>
      flow.return %5 : tensor<2x11008x128xf32>
    }
    %2 = iree_encoding.set_encoding %1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>
    %3 = iree_encoding.set_encoding %arg1 : tensor<2x128x64xf32> -> tensor<2x128x64xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x128x64xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>
    %4 = flow.dispatch.region -> (tensor<2x11008x64xf32>) {
      %5 = tensor.empty() : tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>) -> tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>
      %7 = linalg.generic {
          indexing_maps = [#map1, #map2, #map3],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%3, %2 : tensor<2x128x64xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x128x64xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>, tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>)
          outs(%6 : tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %9 = arith.mulf %in, %in_0 : f32
        %10 = arith.addf %9, %out : f32
        linalg.yield %10 : f32
      } -> tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>
      %8 = iree_encoding.unset_encoding %7 : tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>> -> tensor<2x11008x64xf32>
      flow.return %8 : tensor<2x11008x64xf32>
    }
    util.return %4 : tensor<2x11008x64xf32>
  }
}

// CHECK-LABEL: @quantized_matmul
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32, #iree_encoding.encoding
// CHECK:         %[[ADD:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING0:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING0]] :
// CHECK:       }
// CHECK:       %[[DISPATCH1:.+]] = flow.dispatch.region -> (tensor<2x128x64xf32, #iree_encoding.encoding
// CHECK:         %[[SET_ENCODING1:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING1]] :
// CHECK:       }
// CHECK:       %[[DISPATCH2:.+]] = flow.dispatch.region -> (tensor<2x11008x64xf32>) {
// CHECK:         %[[MATMUL:.+]] = linalg.generic {{.*}} ins(%[[DISPATCH1]], %[[DISPATCH0]]
// CHECK:         %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
// CHECK:         flow.return %[[UNSET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH2]] : tensor<2x11008x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  util.func public @reduction_fusion(%arg0: tensor<2x11008x128x16xf32>) -> tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>> {
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
    %2 = iree_encoding.set_encoding %1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>
    util.return %2 : tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>
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
module {
  util.func public @transpose_no_fusion(%arg0: tensor<2x128x11008xf32>) -> tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>> {
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
    %2 = iree_encoding.set_encoding %1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>
    util.return %2 : tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 32, 32, 32>>>
  }
}

// CHECK-LABEL: @transpose_no_fusion
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
// CHECK:         %[[TRANSPOSE:.+]] = linalg.generic
// CHECK:         flow.return %[[TRANSPOSE]]
// CHECK:       }
// CHECK:       %[[DISPATCH1:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32, #iree_encoding.encoding
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[DISPATCH0]]
// CHECK:         flow.return %[[SET_ENCODING]]
// CHECK:       }
// CHECK:       util.return %[[DISPATCH1]] : tensor<2x11008x128xf32, #iree_encoding.encoding
