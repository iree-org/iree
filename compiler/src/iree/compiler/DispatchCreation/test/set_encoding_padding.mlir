// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-set-encoding{encoding-option=padding}))" --split-input-file --mlir-print-local-scope %s | FileCheck %s

util.func @simple_test(%lhs : tensor<?x?xf32>, %rhs0 : tensor<?x?xf32>,
    %rhs1 : tensor<?x?xf32>, %M : index, %K1 : index, %K2 : index,
    %N: index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = flow.dispatch.region[%M, %K2, %K1] -> (tensor<?x?xf32>{%M, %K2}) {
    %1 = tensor.empty(%M, %K2) : tensor<?x?xf32>
    %2 = linalg.fill ins(%c0 : f32) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %3 = linalg.matmul ins(%lhs, %rhs0 : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %3 : tensor<?x?xf32>
  } count(%w0: index, %w1 : index, %w2 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %w0, %w1, %w2
    flow.return %x, %y, %z : index, index, index
  }
  %1 = flow.dispatch.region[%M, %N, %K2] -> (tensor<?x?xf32>{%M, %N}) {
    %2 = tensor.empty(%M, %N) : tensor<?x?xf32>
    %3 = linalg.fill ins(%c0 : f32) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %4 = linalg.matmul ins(%0, %rhs1 : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %4 : tensor<?x?xf32>
  } count(%w0: index, %w1 : index, %w2 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %w0, %w1, %w2
    flow.return %x, %y, %z : index, index, index
  }
  util.return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @simple_test(
//  CHECK-SAME:     %[[LHS:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[RHS0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[RHS1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[M:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[K1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[K2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[N:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.region
//  CHECK-SAME:       [%[[M]], %[[K2]], %[[K1]]]
//  CHECK-SAME:       -> (tensor<?x?xf32, #iree_encoding.pad_encoding_layout<[0, ?]>>{%[[M]], %[[K2]]})
//       CHECK:     %[[OP0:.+]] = linalg.matmul
//       CHECK:     %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[OP0]]
//       CHECK:     flow.return %[[SET_ENCODING]]
//       CHECK:     count(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index, %[[B2:[a-zA-Z0-9]+]]: index)
//       CHECK:       %[[X:[a-zA-Z0-9_]+]], %[[Y:[a-zA-Z0-9_]+]], %[[Z:[a-zA-Z0-9_]+]] = iree_tensor_ext.dispatch.workgroup_count_from_slice %[[B0]], %[[B1]], %[[B2]]
//       CHECK:       flow.return %[[X]], %[[Y]], %[[Z]]
//       CHECK:   flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[DISPATCH0]]
//  CHECK-SAME:         tensor<?x?xf32>{%[[M]], %[[K2]]}
//       CHECK:     linalg.matmul
//  CHECK-SAME:         ins(%[[UNSET_ENCODING]],

// -----

// Check that padding encoding can be set across collapse_shape ops.
util.func @encoding_across_collapse(%lhs : tensor<?x?xf32>, %rhs : tensor<?x?xf32>,
    %M : index, %N : index, %K1 : index, %K2 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = flow.dispatch.region[%M, %K1, %K2] -> (tensor<?x?x?xf32>{%M, %K1, %K2}) {
    %1 = tensor.empty(%M, %K1, %K2) : tensor<?x?x?xf32>
    flow.return %1 : tensor<?x?x?xf32>
  } count(%w0: index, %w1 : index, %w2 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %w0, %w1, %w2
    flow.return %x, %y, %z : index, index, index
  }
  %1 = tensor.collapse_shape %0 [[0], [1, 2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
  %2 = arith.muli %K1, %K2 : index
  %3 = flow.dispatch.region[%M, %N, %2] -> (tensor<?x?xf32>{%M, %N}) {
    %4 = tensor.empty(%M, %N) : tensor<?x?xf32>
    %5 = linalg.fill ins(%c0 : f32) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %6 = linalg.matmul ins(%1, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %6 : tensor<?x?xf32>
  } count(%w0: index, %w1 : index, %w2 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %w0, %w1, %w2
    flow.return %x, %y, %z : index, index, index
  }
  util.return %3 : tensor<?x?xf32>
}
// CHECK-LABEL: @encoding_across_collapse
//  CHECK-SAME:     %[[M:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[N:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[K1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[K2:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[K:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[K1]], %[[K2]]]
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.region
//  CHECK-SAME:       -> (tensor<?x?xf32, #iree_encoding.pad_encoding_layout<[0, ?]>>{%[[M]], %[[K]]}
//       CHECK:     %[[EMPTY:.+]] = tensor.empty
//       CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EMPTY]]
//       CHECK:     %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[COLLAPSE]]
//       CHECK:     flow.return %[[SET_ENCODING]]
//       CHECK:   flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding
//  CHECK-SAME:         -> tensor<?x?xf32>{%[[M]], %[[K]]}
//       CHECK:     linalg.matmul ins(%[[UNSET_ENCODING]],

// -----

// Check that padding encoding can be set across sequence of operations.
util.func @encoding_across_collapse(%lhs : tensor<?x?xf32>, %rhs : tensor<?x?xf32>,
    %M : index, %N : index, %K1 : index, %K2 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = flow.dispatch.region[%M, %K1] -> (tensor<?x?xf32>{%M, %K1}) {
    %1 = tensor.empty(%M, %K1) : tensor<?x?xf32>
    flow.return %1 : tensor<?x?xf32>
  } count(%w0: index, %w1 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %w0, %w1
    flow.return %x, %y, %z : index, index, index
  }
  %1 = arith.divsi %K1, %K2 : index
  %2 = tensor.expand_shape %0 [[0], [1, 2]] output_shape[%M, %1, %K2] : tensor<?x?xf32> into tensor<?x?x?xf32>
  %3 = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
  %4 = arith.muli %M, %1 : index
  %5 = flow.dispatch.region[%4, %N, %K2] -> (tensor<?x?xf32>{%4, %K2}) {
    %6 = tensor.empty(%4, %N) : tensor<?x?xf32>
    %7 = linalg.fill ins(%c0 : f32) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %8 = linalg.matmul ins(%3, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%7 : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %8 : tensor<?x?xf32>
  } count(%w0: index, %w1 : index, %w2 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %w0, %w1, %w2
    flow.return %x, %y, %z : index, index, index
  }
  util.return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: @encoding_across_collapse
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//  CHECK-SAME:       #iree_encoding.pad_encoding_layout<[0, ?]>
//       CHECK:     %[[EMPTY:.+]] = tensor.empty
//       CHECK:     %[[EXPAND:.+]] = tensor.expand_shape %[[EMPTY]]
//       CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]]
//       CHECK:     %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[COLLAPSE]]
//       CHECK:     flow.return %[[SET_ENCODING]]
//       CHECK:   flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[DISPATCH]]
//       CHECK:     linalg.matmul ins(%[[UNSET_ENCODING]],
