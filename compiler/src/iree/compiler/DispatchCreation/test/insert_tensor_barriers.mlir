// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-insert-tensor-barriers))" %s | FileCheck %s

util.func public @simple_linalg(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x8xf32>
  util.return %0 : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @simple_linalg
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[START:.+]] = iree_tensor_ext.compute_barrier.start %[[ARG0]]
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[START]] :
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier.end %[[GENERIC]]
//       CHECK:   util.return %[[END]]

// -----

util.func public @multiple_compute_ops(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %empty0 = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%empty0 : tensor<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4xf32>
  %empty1 = tensor.empty() : tensor<4xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%0 : tensor<4xf32>) outs(%empty1 : tensor<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4xf32>
  util.return %1 : tensor<4xf32>
}
// CHECK-LABEL: util.func public @multiple_compute_ops
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[START:.+]] = iree_tensor_ext.compute_barrier.start %[[ARG0]]
//       CHECK:   %[[GENERIC0:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[START]] :
//       CHECK:   %[[GENERIC1:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[GENERIC0]] :
//   CHECK-NOT:   compute_barrier.start
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier.end %[[GENERIC1]]
//       CHECK:   util.return %[[END]]

// -----

util.func public @with_reshapes(%arg0: tensor<32xf32>) -> tensor<4x8xf32> {
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [4, 8] : tensor<32xf32> into tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%expanded : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x8xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
  %result = tensor.expand_shape %collapsed [[0, 1]] output_shape [4, 8] : tensor<32xf32> into tensor<4x8xf32>
  util.return %result : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @with_reshapes
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[START:.+]] = iree_tensor_ext.compute_barrier.start %[[ARG0]]
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[START]]
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[EXPANDED]] :
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[GENERIC]]
//       CHECK:   %[[RESULT:.+]] = tensor.expand_shape %[[COLLAPSED]]
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier.end %[[RESULT]]
//       CHECK:   util.return %[[END]]

// -----

util.func public @tensor_reshape_only(%arg0: tensor<32xf32>) -> tensor<4x8xf32> {
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [4, 8] : tensor<32xf32> into tensor<4x8xf32>
  util.return %expanded : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @tensor_reshape_only
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[START:.+]] = iree_tensor_ext.compute_barrier.start %[[ARG0]]
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[START]]
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier.end %[[EXPANDED]]
//       CHECK:   util.return %[[END]]

// -----

util.func public @with_hal_ops(%arg0: !hal.buffer_view, %arg1: !hal.fence) -> !hal.buffer_view {
  %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%0 : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x8xf32>
  %2 = hal.tensor.export %1 : tensor<4x8xf32> -> !hal.buffer_view
  util.return %2 : !hal.buffer_view
}
// CHECK-LABEL: util.func public @with_hal_ops
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: !hal.buffer_view
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: !hal.fence
//       CHECK:   %[[IMPORT:.+]] = hal.tensor.import wait(%[[ARG1]]) => %[[ARG0]]
//       CHECK:   %[[START:.+]] = iree_tensor_ext.compute_barrier.start %[[IMPORT]]
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[START]] :
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier.end %[[GENERIC]]
//       CHECK:   %[[EXPORT:.+]] = hal.tensor.export %[[END]]
//       CHECK:   util.return %[[EXPORT]]

// -----

util.func public @with_hal_barrier_dynamic(%arg0: tensor<?xf32>, %arg1: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view) {
  %0:2 = hal.tensor.barrier join(%arg0, %arg0 : tensor<?xf32>, tensor<?xf32>) => %arg1 : !hal.fence
  %c0 = arith.constant 0 : index
  %dim0 = tensor.dim %0#0, %c0 : tensor<?xf32>
  %dim1 = tensor.dim %0#1, %c0 : tensor<?xf32>
  %1 = hal.tensor.export %0#0 : tensor<?xf32>{%dim0} -> !hal.buffer_view
  %2 = hal.tensor.export %0#1 : tensor<?xf32>{%dim1} -> !hal.buffer_view
  util.return %1, %2 : !hal.buffer_view, !hal.buffer_view
}
// Verifies that tensor.dim doesn't trigger barrier insertion (metadata ops).
// CHECK-LABEL: util.func public @with_hal_barrier_dynamic
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: !hal.fence
//   CHECK-NOT:   compute_barrier.start
//       CHECK:   %[[BARRIER:.+]]:2 = hal.tensor.barrier join(%[[ARG0]], %[[ARG0]]
//       CHECK:   %[[DIM0:.+]] = tensor.dim %[[BARRIER]]#0
//       CHECK:   %[[DIM1:.+]] = tensor.dim %[[BARRIER]]#1
//       CHECK:   %[[EXPORT0:.+]] = hal.tensor.export %[[BARRIER]]#0
//       CHECK:   %[[EXPORT1:.+]] = hal.tensor.export %[[BARRIER]]#1
//   CHECK-NOT:   compute_barrier.end
//       CHECK:   util.return %[[EXPORT0]], %[[EXPORT1]]

// -----

util.func public @linalg_with_dynamic_dims(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %empty = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<?x?xf32>
  util.return %result : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @linalg_with_dynamic_dims
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//       CHECK:   %[[START:.+]] = iree_tensor_ext.compute_barrier.start %[[ARG0]]
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[START]] :
//       CHECK:   %[[END:.+]] = iree_tensor_ext.compute_barrier.end %[[GENERIC]]
//       CHECK:   util.return %[[END]]
