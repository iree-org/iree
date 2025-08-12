// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-set-encoding{encoding-option=padding}))" --split-input-file --mlir-print-local-scope %s | FileCheck %s

util.func @simple_test(%lhs : tensor<?x?xf32>, %rhs0 : tensor<?x2048xf32>,
    %rhs1 : tensor<2048x?xf32>, %M : index, %K : index, %N: index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = flow.dispatch.region[%M, %K] -> (tensor<?x2048xf32>{%M}) {
    %1 = tensor.empty(%M) : tensor<?x2048xf32>
    %2 = linalg.fill ins(%c0 : f32) outs(%1 : tensor<?x2048xf32>) -> tensor<?x2048xf32>
    %3 = linalg.matmul ins(%lhs, %rhs0 : tensor<?x?xf32>, tensor<?x2048xf32>)
        outs(%2 : tensor<?x2048xf32>) -> tensor<?x2048xf32>
    flow.return %3 : tensor<?x2048xf32>
  } count(%w0: index, %w1 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0, %w1)
    flow.return %x, %y, %z : index, index, index
  }
  %1 = flow.dispatch.region[%M, %N] -> (tensor<?x?xf32>{%M, %N}) {
    %2 = tensor.empty(%M, %N) : tensor<?x?xf32>
    %3 = linalg.fill ins(%c0 : f32) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %4 = linalg.matmul ins(%0, %rhs1 : tensor<?x2048xf32>, tensor<2048x?xf32>)
        outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %4 : tensor<?x?xf32>
  } count(%w0: index, %w1 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0, %w1)
    flow.return %x, %y, %z : index, index, index
  }
  util.return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @simple_test(
//  CHECK-SAME:     %[[LHS:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[RHS0:[a-zA-Z0-9_]+]]: tensor<?x2048xf32>
//  CHECK-SAME:     %[[RHS1:[a-zA-Z0-9_]+]]: tensor<2048x?xf32>
//  CHECK-SAME:     %[[M:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[K:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[N:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.region
//  CHECK-SAME:       [%[[M]], %[[K]]]
//  CHECK-SAME:       -> (tensor<?x2048xf32, #iree_encoding.padding<[0, ?]>>{%[[M]]})
//       CHECK:     %[[OP0:.+]] = linalg.matmul
//       CHECK:     %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[OP0]]
//       CHECK:     flow.return %[[SET_ENCODING]]
//       CHECK:     count(%[[B0:[a-zA-Z0-9]+]]: index, %[[B1:[a-zA-Z0-9]+]]: index)
//       CHECK:       %[[X:[a-zA-Z0-9_]+]], %[[Y:[a-zA-Z0-9_]+]], %[[Z:[a-zA-Z0-9_]+]] = iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[B0]], %[[B1]])
//       CHECK:       flow.return %[[X]], %[[Y]], %[[Z]]
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[DISPATCH0]]
//  CHECK-SAME:         tensor<?x2048xf32>{%[[M]]}
//       CHECK:     linalg.matmul
//  CHECK-SAME:         ins(%[[UNSET_ENCODING]],
//       CHECK:   return %[[DISPATCH1]]

// -----

// Check that padding encoding can be set across collapse_shape ops.
util.func @encoding_across_collapse(%lhs : tensor<?x2048xf32>, %rhs : tensor<2048x?xf32>,
    %M : index, %N : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = flow.dispatch.region[%M] -> (tensor<?x128x16xf32>{%M}) {
    %1 = tensor.empty(%M) : tensor<?x128x16xf32>
    flow.return %1 : tensor<?x128x16xf32>
  } count(%w0: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0)
    flow.return %x, %y, %z : index, index, index
  }
  %1 = tensor.collapse_shape %0 [[0], [1, 2]] : tensor<?x128x16xf32> into tensor<?x2048xf32>
  %3 = flow.dispatch.region[%M, %N] -> (tensor<?x?xf32>{%M, %N}) {
    %4 = tensor.empty(%M, %N) : tensor<?x?xf32>
    %5 = linalg.fill ins(%c0 : f32) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %6 = linalg.matmul ins(%1, %rhs : tensor<?x2048xf32>, tensor<2048x?xf32>)
        outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %6 : tensor<?x?xf32>
  } count(%w0: index, %w1 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0, %w1)
    flow.return %x, %y, %z : index, index, index
  }
  util.return %3 : tensor<?x?xf32>
}
// CHECK-LABEL: @encoding_across_collapse
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.region
//  CHECK-SAME:       -> (tensor<?x2048xf32, #iree_encoding.padding<[0, ?]>>
//       CHECK:     %[[EMPTY:.+]] = tensor.empty
//       CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EMPTY]]
//       CHECK:     %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[COLLAPSE]]
//       CHECK:     flow.return %[[SET_ENCODING]]
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding
//  CHECK-SAME:         -> tensor<?x2048xf32>
//       CHECK:     linalg.matmul ins(%[[UNSET_ENCODING]],
//       CHECK:   return %[[DISPATCH1]]

// -----

// Check that padding encoding can be set across sequence of operations.
util.func @encoding_across_collapse_expand(%lhs : tensor<?x2048xf32>, %rhs : tensor<1024x?xf32>,
    %M : index, %N : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c2 = arith.constant 2 : index
  %0 = flow.dispatch.region[%M] -> (tensor<?x2048xf32>{%M}) {
    %1 = tensor.empty(%M) : tensor<?x2048xf32>
    flow.return %1 : tensor<?x2048xf32>
  } count(%w0: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0)
    flow.return %x, %y, %z : index, index, index
  }
  %2 = tensor.expand_shape %0 [[0], [1, 2]] output_shape[%M, 2, 1024] : tensor<?x2048xf32> into tensor<?x2x1024xf32>
  %3 = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<?x2x1024xf32> into tensor<?x1024xf32>
  %4 = arith.muli %M, %c2 : index
  %5 = flow.dispatch.region[%4, %N] -> (tensor<?x?xf32>{%4, %N}) {
    %6 = tensor.empty(%4, %N) : tensor<?x?xf32>
    %7 = linalg.fill ins(%c0 : f32) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %8 = linalg.matmul ins(%3, %rhs : tensor<?x1024xf32>, tensor<1024x?xf32>)
        outs(%7 : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %8 : tensor<?x?xf32>
  } count(%w0: index, %w1 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0, %w1)
    flow.return %x, %y, %z : index, index, index
  }
  util.return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: @encoding_across_collapse_expand(
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//  CHECK-SAME:       #iree_encoding.padding<[0, ?]>
//       CHECK:     %[[EMPTY:.+]] = tensor.empty
//       CHECK:     %[[EXPAND:.+]] = tensor.expand_shape %[[EMPTY]]
//       CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]]
//       CHECK:     %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[COLLAPSE]]
//       CHECK:     flow.return %[[SET_ENCODING]]
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[DISPATCH]]
//       CHECK:     linalg.matmul ins(%[[UNSET_ENCODING]],
//       CHECK:   return %[[DISPATCH1]]

// -----

// Check that dynamic reduction dimensions are not padded since it is unsupported right now.
util.func @no_pad_dynamic_reduction_dims(%lhs : tensor<?x?xf32>, %rhs0 : tensor<?x?xf32>,
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
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0, %w1, %w2)
    flow.return %x, %y, %z : index, index, index
  }
  %1 = flow.dispatch.region[%M, %N] -> (tensor<?x?xf32>{%M, %N}) {
    %2 = tensor.empty(%M, %N) : tensor<?x?xf32>
    %3 = linalg.fill ins(%c0 : f32) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %4 = linalg.matmul ins(%0, %rhs1 : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %4 : tensor<?x?xf32>
  } count(%w0: index, %w1 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0, %w1)
    flow.return %x, %y, %z : index, index, index
  }
  util.return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @no_pad_dynamic_reduction_dims
//   CHECK-NOT: #iree_encoding.padding

// -----

// Check matvecs/vecmats and skinny matmuls arent padded

util.func @no_pad_skinny_matmuls(%lhs : tensor<?x?xf32>, %rhs0 : tensor<?x2048xf32>,
    %rhs1 : tensor<2048x4xf32>, %K : index) -> tensor<8x4xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = flow.dispatch.region[%K] -> (tensor<8x2048xf32>) {
    %1 = tensor.empty() : tensor<8x2048xf32>
    %2 = linalg.fill ins(%c0 : f32) outs(%1 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %3 = linalg.matmul ins(%lhs, %rhs0 : tensor<?x?xf32>, tensor<?x2048xf32>)
        outs(%2 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    flow.return %3 : tensor<8x2048xf32>
  } count(%w0: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0)
    flow.return %x, %y, %z : index, index, index
  }
  %1 = flow.dispatch.region -> (tensor<8x4xf32>) {
    %2 = tensor.empty() : tensor<8x4xf32>
    %3 = linalg.fill ins(%c0 : f32) outs(%2 : tensor<8x4xf32>) -> tensor<8x4xf32>
    %4 = linalg.matmul ins(%0, %rhs1 : tensor<8x2048xf32>, tensor<2048x4xf32>)
        outs(%3 : tensor<8x4xf32>) -> tensor<8x4xf32>
    flow.return %4 : tensor<8x4xf32>
  } count() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  util.return %1 : tensor<8x4xf32>
}
// CHECK-LABEL: @no_pad_skinny_matmuls
//   CHECK-NOT: #iree_encoding.padding

// -----

// Check threshold of 64 for setting the padding

util.func @check_padding_threshold(%lhs : tensor<?x?xf32>, %rhs0 : tensor<?x2048xf32>,
    %rhs1 : tensor<2048x8xf32>, %K : index) -> tensor<8x8xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = flow.dispatch.region[%K] -> (tensor<8x2048xf32>) {
    %1 = tensor.empty() : tensor<8x2048xf32>
    %2 = linalg.fill ins(%c0 : f32) outs(%1 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %3 = linalg.matmul ins(%lhs, %rhs0 : tensor<?x?xf32>, tensor<?x2048xf32>)
        outs(%2 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    flow.return %3 : tensor<8x2048xf32>
  } count(%w0: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0)
    flow.return %x, %y, %z : index, index, index
  }
  %1 = flow.dispatch.region -> (tensor<8x8xf32>) {
    %2 = tensor.empty() : tensor<8x8xf32>
    %3 = linalg.fill ins(%c0 : f32) outs(%2 : tensor<8x8xf32>) -> tensor<8x8xf32>
    %4 = linalg.matmul ins(%0, %rhs1 : tensor<8x2048xf32>, tensor<2048x8xf32>)
        outs(%3 : tensor<8x8xf32>) -> tensor<8x8xf32>
    flow.return %4 : tensor<8x8xf32>
  } count() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  util.return %1 : tensor<8x8xf32>
}
// CHECK-LABEL: @check_padding_threshold(
//       CHECK:   %[[DISPATCH0:.+]] = flow.dispatch.region
//  CHECK-SAME:       -> (tensor<8x2048xf32, #iree_encoding.padding<[0, ?]>>)
//       CHECK:     %[[OP0:.+]] = linalg.matmul
//       CHECK:     %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[OP0]]
//       CHECK:     flow.return %[[SET_ENCODING]]
//       CHECK:   %[[DISPATCH1:.+]] = flow.dispatch.region
//       CHECK:     %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[DISPATCH0]]
//  CHECK-SAME:         tensor<8x2048xf32>
//       CHECK:     linalg.matmul
//  CHECK-SAME:         ins(%[[UNSET_ENCODING]],
//       CHECK:   return %[[DISPATCH1]]

// -----

// Check that a matmul with attention producer isn't padded.

util.func public @no_attention_producer(%arg0: tensor<4x8x4x?x32x2x64xf32>, %arg1: tensor<4x?x32x8x2x64xf32>, %arg2: tensor<4x?x32x8x128xf32>, %arg3: tensor<?x32x?x32xf32>, %arg4: tensor<4096x8xf32>, %arg5: f32, %arg6: index) -> tensor<?x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %0 = flow.dispatch.region -> (tensor<4x8x4x?x32x128xf32>{%arg6}) {
    %3 = tensor.empty(%arg6) : tensor<4x8x4x?x32x128xf32>
    %4 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d8, d9, d1, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d8, d9, d1, d5)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d3, d4, d8, d9)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>]} ins(%arg0, %arg1, %arg2, %arg5, %arg3 : tensor<4x8x4x?x32x2x64xf32>, tensor<4x?x32x8x2x64xf32>, tensor<4x?x32x8x128xf32>, f32, tensor<?x32x?x32xf32>) outs(%3 : tensor<4x8x4x?x32x128xf32>) {
    ^bb0(%arg7: f32):
      iree_linalg_ext.yield %arg7 : f32
    } -> tensor<4x8x4x?x32x128xf32>
    flow.return %4 : tensor<4x8x4x?x32x128xf32>
  }
  %1 = arith.muli %arg6, %c128 : index
  %collapsed = tensor.collapse_shape %0 [[0, 1, 2, 3], [4, 5]] : tensor<4x8x4x?x32x128xf32> into tensor<?x4096xf32>
  %2 = flow.dispatch.region -> (tensor<?x8xf32>{%1}) {
    %3 = tensor.empty(%1) : tensor<?x8xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<?x8xf32>) -> tensor<?x8xf32>
    %5 = linalg.matmul ins(%collapsed, %arg4 : tensor<?x4096xf32>, tensor<4096x8xf32>) outs(%4 : tensor<?x8xf32>) -> tensor<?x8xf32>
    flow.return %5 : tensor<?x8xf32>
  }
  util.return %2 : tensor<?x8xf32>
}
// CHECK-LABEL: @no_attention_producer(
//   CHECK-NOT: #iree_encoding.pad_encoding_layout
