// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-flow-set-encoding))" %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
util.func public @broadcasting_dequant_op(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> tensor<?x?x?xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
  %2 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<?x?xi8>{%0, %1}
  %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
  %4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
  %5 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[2] : index
  %6 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<?x?x?xi32>{%3, %4, %5}
  %7 = flow.dispatch.region -> (tensor<?x?x?xi32>{%3, %0, %4}) {
    %9 = tensor.empty(%3, %0, %4) : tensor<?x?x?xi32>
    %c0_i32_0 = arith.constant 0 : i32
    %10 = tensor.empty(%3, %0, %1) : tensor<?x?x?xi32>
    %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<?x?xi8>) outs(%10 : tensor<?x?x?xi32>) {
    ^bb0(%in: i8, %out: i32):
      %14 = arith.extui %in : i8 to i32
      linalg.yield %14 : i32
    } -> tensor<?x?x?xi32>
    %12 = linalg.fill ins(%c0_i32_0 : i32) outs(%9 : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
    %13 = linalg.batch_matmul_transpose_b ins(%11, %6 : tensor<?x?x?xi32>, tensor<?x?x?xi32>) outs(%12 : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
    flow.return %13 : tensor<?x?x?xi32>
  }
  util.return %7 : tensor<?x?x?xi32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:      util.func public @broadcasting_dequant_op(
// CHECK:      %{{.+}} = flow.dispatch.region
// CHECK:        %[[BCAST:.+]] = linalg.generic
// CHECK:        %[[LHS:.+]] = iree_encoding.set_encoding %[[BCAST]] : tensor<?x?x?xi32>
// CHECK-SAME:    -> tensor<?x?x?xi32, #iree_encoding.encoding
// CHECK-SAME:      operand_index = 0 : index
// CHECK-SAME:      element_types = [i8, i32, i32]
// CHECK-SAME:      original_type = tensor<?x?x?xi32>
// CHECK-SAME:      user_indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME:      round_dims_to = array<i64: 16, 16, 16, 16>
// CHECK:        %[[RHS:.+]] = iree_encoding.set_encoding %{{.+}} : tensor<?x?x?xi32>
// CHECK-SAME:      operand_index = 1 : index
// CHECK-SAME:      element_types = [i8, i32, i32]
// CHECK-SAME:      original_type = tensor<?x?x?xi32>
// CHECK-SAME:      user_indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME:      round_dims_to = array<i64: 16, 16, 16, 16>
// CHECK:        %[[INIT:.+]] = tensor.empty({{.+}}) : tensor<?x?x?xi32, #iree_encoding.encoding
// CHECK-SAME:      operand_index = 2 : index
// CHECK-SAME:      element_types = [i8, i32, i32]
// CHECK-SAME:      original_type = tensor<?x?x?xi32>
// CHECK-SAME:      user_indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME:      round_dims_to = array<i64: 16, 16, 16, 16>
// CHECK:        %[[FILL:.+]] = linalg.fill ins({{.+}}) outs(%[[INIT]]
// CHECK:        %[[GEMM:.+]] = linalg.batch_matmul_transpose_b
// CHECK-SAME:     ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:    outs(%[[FILL]]
// CHECK:        %[[UNSET:.+]] = iree_encoding.unset_encoding %[[GEMM]]
// CHECK:        %[[SLICE:.+]] = tensor.extract_slice %[[UNSET]]
// CHECK:        flow.return %[[SLICE]]
