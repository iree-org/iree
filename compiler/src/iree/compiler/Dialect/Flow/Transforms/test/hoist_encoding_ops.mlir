// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-flow-hoist-encoding-ops))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
util.func public @quantized_matmul(
    %arg0: tensor<2x11008x128xi8>, %arg1: tensor<2x128x64xi8>,
    %arg2: tensor<2x11008xf32>, %arg3: tensor<2x11008xf32>,
    %arg4: tensor<2x64xf32>, %arg5: tensor<2x64xf32>) -> tensor<2x11008x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %6 = flow.dispatch.region -> (tensor<2x11008x64xf32>) {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %9 = tensor.empty() : tensor<2x128x64xf32>
    %10 = linalg.generic
        {indexing_maps = [#map, #map1, #map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg1, %arg4, %arg5 : tensor<2x128x64xi8>, tensor<2x64xf32>, tensor<2x64xf32>)
        outs(%9 : tensor<2x128x64xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %18 = arith.extui %in : i8 to i32
      %19 = arith.uitofp %18 : i32 to f32
      %20 = arith.subf %19, %in_1 : f32
      %21 = arith.mulf %20, %in_0 : f32
      linalg.yield %21 : f32
    } -> tensor<2x128x64xf32>
    %11 = linalg.generic
        {indexing_maps = [#map, #map2, #map2, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg2, %arg3 : tensor<2x11008x128xi8>, tensor<2x11008xf32>, tensor<2x11008xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %18 = arith.extui %in : i8 to i32
      %19 = arith.uitofp %18 : i32 to f32
      %20 = arith.subf %19, %in_1 : f32
      %21 = arith.mulf %20, %in_0 : f32
      linalg.yield %21 : f32
    } -> tensor<2x11008x128xf32>
    %12 = iree_encoding.set_encoding %10 : tensor<2x128x64xf32> -> tensor<2x128x64xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x128x64xf32>, user_indexing_maps = [#map3, #map4, #map5], round_dims_to = array<i64: 32, 32, 32>>>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map3, #map4, #map5], round_dims_to = array<i64: 32, 32, 32>>>
    %14 = tensor.empty() : tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map3, #map4, #map5], round_dims_to = array<i64: 32, 32, 32>>>
    %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map3, #map4, #map5], round_dims_to = array<i64: 32, 32, 32>>>) -> tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map3, #map4, #map5], round_dims_to = array<i64: 32, 32, 32>>>
    %16 = linalg.generic
        {indexing_maps = [#map3, #map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%12, %13 : tensor<2x128x64xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x128x64xf32>, user_indexing_maps = [#map3, #map4, #map5], round_dims_to = array<i64: 32, 32, 32>>>, tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#map3, #map4, #map5], round_dims_to = array<i64: 32, 32, 32>>>)
        outs(%15 : tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map3, #map4, #map5], round_dims_to = array<i64: 32, 32, 32>>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %18 = arith.mulf %in, %in_0 : f32
      %19 = arith.addf %18, %out : f32
      linalg.yield %19 : f32
    } -> tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map3, #map4, #map5], round_dims_to = array<i64: 32, 32, 32>>>
    %17 = iree_encoding.unset_encoding %16 : tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#map3, #map4, #map5], round_dims_to = array<i64: 32, 32, 32>>> -> tensor<2x11008x64xf32>
    flow.return %17 : tensor<2x11008x64xf32>
  }
  util.return %6 : tensor<2x11008x64xf32>
}

// CHECK-DAG:   #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG:   #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:   #[[$MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-LABEL: @quantized_matmul
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x11008x128xi8>, %[[ARG1:.+]]: tensor<2x128x64xi8>,
// CHECK-SAME:    %[[ARG2:.+]]: tensor<2x11008xf32>, %[[ARG3:.+]]: tensor<2x11008xf32>,
// CHECK-SAME:    %[[ARG4:.+]]: tensor<2x64xf32>, %[[ARG5:.+]]: tensor<2x64xf32>
// CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   %[[ENCODING0:.+]] = iree_encoding.set_encoding %[[ARG0]] : tensor<2x11008x128xi8> -> tensor<2x11008x128xi8, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]], bcast_map = #[[$MAP4]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK-DAG:   %[[ENCODING1:.+]] = iree_encoding.set_encoding %[[ARG1]] : tensor<2x128x64xi8> -> tensor<2x128x64xi8, #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x128x64xf32>, user_indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]], bcast_map = #[[$MAP4]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK-DAG:   %[[ENCODING2:.+]] = iree_encoding.set_encoding %[[ARG2]] : tensor<2x11008xf32> -> tensor<2x11008xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]], bcast_map = #[[$MAP3]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK-DAG:   %[[ENCODING3:.+]] = iree_encoding.set_encoding %[[ARG3]] : tensor<2x11008xf32> -> tensor<2x11008xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]], bcast_map = #[[$MAP3]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK-DAG:   %[[ENCODING4:.+]] = iree_encoding.set_encoding %[[ARG4]] : tensor<2x64xf32> -> tensor<2x64xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x128x64xf32>, user_indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]], bcast_map = #[[$MAP5]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK-DAG:   %[[ENCODING5:.+]] = iree_encoding.set_encoding %[[ARG5]] : tensor<2x64xf32> -> tensor<2x64xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x128x64xf32>, user_indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]], bcast_map = #[[$MAP5]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<2x11008x64xf32>) {
// CHECK:         %[[INIT0:.+]] = tensor.empty() : tensor<2x128x64xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x128x64xf32>, user_indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK:         %[[DEQUANT0:.+]] = linalg.generic {{.*}} ins(%[[ENCODING1]], %[[ENCODING4]], %[[ENCODING5]] : {{.*}} outs(%[[INIT0]] :
// CHECK:         %[[INIT1:.+]] = tensor.empty() : tensor<2x11008x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x128xf32>, user_indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK:         %[[DEQUANT1:.+]] = linalg.generic {{.*}} ins(%[[ENCODING0]], %[[ENCODING2]], %[[ENCODING3]] : {{.*}} outs(%[[INIT1]] :
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<2x11008x64xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], original_type = tensor<2x11008x64xf32>, user_indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] :
// CHECK:         %[[MATMUL:.+]] = linalg.generic {{.*}} ins(%[[DEQUANT0]], %[[DEQUANT1]] : {{.*}} outs(%[[FILL]] :
// CHECK:         %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[MATMUL]] {{.*}} -> tensor<2x11008x64xf32>
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]] : tensor<2x11008x64xf32>
