// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-remove-tensor-barriers))" %s | FileCheck %s

util.func public @simple_barrier(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  %1 = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %0 : tensor<4x8xf32> -> tensor<4x8xf32>
  util.return %1 : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @simple_barrier
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//   CHECK-NOT:   compute_barrier
//       CHECK:   util.return %[[ARG0]]

// -----

util.func public @barrier_with_compute(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%0 : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x8xf32>
  %2 = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %1 : tensor<4x8xf32> -> tensor<4x8xf32>
  util.return %2 : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @barrier_with_compute
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//   CHECK-NOT:   compute_barrier
//       CHECK:   %[[EMPTY:.+]] = tensor.empty
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]] :
//   CHECK-NOT:   compute_barrier
//       CHECK:   util.return %[[GENERIC]]

// -----

util.func public @multiple_barriers(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<4xf32> -> tensor<4xf32>
  %1 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg1 : tensor<4xf32> -> tensor<4xf32>
  %2 = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %0 : tensor<4xf32> -> tensor<4xf32>
  %3 = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %1 : tensor<4xf32> -> tensor<4xf32>
  util.return %2, %3 : tensor<4xf32>, tensor<4xf32>
}
// CHECK-LABEL: util.func public @multiple_barriers
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
//   CHECK-NOT:   compute_barrier
//       CHECK:   util.return %[[ARG0]], %[[ARG1]]

// -----

util.func public @barrier_with_dynamic_dims(%arg0: tensor<?x?xf32>, %dim0: index, %dim1: index) -> tensor<?x?xf32> {
  %0 = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<?x?xf32> {%dim0, %dim1} -> tensor<?x?xf32>
  %1 = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %0 : tensor<?x?xf32> {%dim0, %dim1} -> tensor<?x?xf32>
  util.return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @barrier_with_dynamic_dims
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//   CHECK-NOT:   compute_barrier
//       CHECK:   util.return %[[ARG0]]
