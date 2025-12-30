// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fold-reshapes-into-tensor-barriers))" %s | FileCheck %s

util.func public @move_expand_shape_above_barrier_start(%arg0: tensor<32xf32>) -> tensor<4x8xf32> {
  %start = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<32xf32> -> tensor<32xf32>
  %expanded = tensor.expand_shape %start [[0, 1]] output_shape [4, 8] : tensor<32xf32> into tensor<4x8xf32>
  util.return %expanded : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @move_expand_shape_above_barrier_start
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-SAME:     tensor<32xf32> into tensor<4x8xf32>
//       CHECK:   %[[START:.+]] = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %[[EXPAND]]
//  CHECK-SAME:     tensor<4x8xf32> -> tensor<4x8xf32>
//       CHECK:   util.return %[[START]]

// -----

util.func public @move_collapse_shape_above_barrier_start(%arg0: tensor<4x8xf32>) -> tensor<32xf32> {
  %start = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>
  %collapsed = tensor.collapse_shape %start [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  util.return %collapsed : tensor<32xf32>
}
// CHECK-LABEL: util.func public @move_collapse_shape_above_barrier_start
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]]
//  CHECK-SAME:     tensor<4x8xf32> into tensor<32xf32>
//       CHECK:   %[[START:.+]] = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %[[COLLAPSE]]
//  CHECK-SAME:     tensor<32xf32> -> tensor<32xf32>
//       CHECK:   util.return %[[START]]

// -----

util.func public @move_expand_shape_below_barrier_end(%arg0: tensor<32xf32>) -> tensor<4x8xf32> {
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [4, 8] : tensor<32xf32> into tensor<4x8xf32>
  %end = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %expanded : tensor<4x8xf32> -> tensor<4x8xf32>
  util.return %end : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @move_expand_shape_below_barrier_end
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %[[ARG0]]
//  CHECK-SAME:     tensor<32xf32> -> tensor<32xf32>
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[END]]
//  CHECK-SAME:     tensor<32xf32> into tensor<4x8xf32>
//       CHECK:   util.return %[[EXPAND]]

// -----

util.func public @move_collapse_shape_below_barrier_end(%arg0: tensor<4x8xf32>) -> tensor<32xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %end = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %collapsed : tensor<32xf32> -> tensor<32xf32>
  util.return %end : tensor<32xf32>
}
// CHECK-LABEL: util.func public @move_collapse_shape_below_barrier_end
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %[[ARG0]]
//  CHECK-SAME:     tensor<4x8xf32> -> tensor<4x8xf32>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[END]]
//  CHECK-SAME:     tensor<4x8xf32> into tensor<32xf32>
//       CHECK:   util.return %[[COLLAPSE]]

// -----

util.func public @move_expand_shape_above_barrier_start_dynamic(%arg0: tensor<?xf32>) -> tensor<?x8xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %c8 = arith.constant 8 : index
  %div = arith.divsi %dim, %c8 : index
  %start = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %arg0 : tensor<?xf32>{%dim} -> tensor<?xf32>
  %expanded = tensor.expand_shape %start [[0, 1]] output_shape [%div, 8] : tensor<?xf32> into tensor<?x8xf32>
  util.return %expanded : tensor<?x8xf32>
}
// CHECK-LABEL: util.func public @move_expand_shape_above_barrier_start_dynamic
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-SAME:     tensor<?xf32> into tensor<?x8xf32>
//       CHECK:   %[[START:.+]] = iree_tensor_ext.compute_barrier<up, "AllowExpand|AllowCollapse"> %[[EXPAND]]
//  CHECK-SAME:     tensor<?x8xf32>{{.*}} -> tensor<?x8xf32>
//       CHECK:   util.return %[[START]]

// -----

util.func public @move_expand_shape_below_barrier_end_dynamic(%arg0: tensor<?xf32>) -> tensor<?x8xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %c8 = arith.constant 8 : index
  %div = arith.divsi %dim, %c8 : index
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [%div, 8] : tensor<?xf32> into tensor<?x8xf32>
  %end = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %expanded : tensor<?x8xf32>{%dim} -> tensor<?x8xf32>
  util.return %end : tensor<?x8xf32>
}
// CHECK-LABEL: util.func public @move_expand_shape_below_barrier_end_dynamic
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %[[ARG0]]
//  CHECK-SAME:     tensor<?xf32>{{.*}} -> tensor<?xf32>
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[END]]
//  CHECK-SAME:     tensor<?xf32> into tensor<?x8xf32>
//       CHECK:   util.return %[[EXPAND]]

// -----

util.func public @move_collapse_shape_below_barrier_end_dynamic(%arg0: tensor<?x8xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x8xf32>
  %c8 = arith.constant 8 : index
  %size = arith.muli %dim, %c8 : index
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x8xf32> into tensor<?xf32>
  %end = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %collapsed : tensor<?xf32>{%size} -> tensor<?xf32>
  util.return %end : tensor<?xf32>
}
// CHECK-LABEL: util.func public @move_collapse_shape_below_barrier_end_dynamic
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier<down, "AllowExpand|AllowCollapse"> %[[ARG0]]
//  CHECK-SAME:     tensor<?x8xf32>{{.*}} -> tensor<?x8xf32>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[END]]
//  CHECK-SAME:     tensor<?x8xf32> into tensor<?xf32>
//       CHECK:   util.return %[[COLLAPSE]]
