// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

func.func @shuffle_tensor(%init: tensor<6x6xf32>, %source: tensor<2x3xf32>, %x: index) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %source[%x, 0] [2, 3] [1, 1] to %init {
  ^bb0(%intermediate: tensor<6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, %x] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
    iree_gpu.yield %slice : tensor<3x2xf32>
  } : tensor<2x3xf32> -> tensor<6x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_shuffle_tensor
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @shuffle_tensor
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9]+]]: tensor<6x6xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<2x3xf32>
//  CHECK-SAME:   %[[X:[A-Za-z0-9]+]]: index

//       CHECK:   %[[IN:.+]] = tensor.insert_slice %[[ARG1]] into %[[INIT]][%[[X]], 0] [2, 3] [1, 1] : tensor<2x3xf32> into tensor<6x6xf32>
//       CHECK:   gpu.barrier
//       CHECK:   %[[OUT:.+]] = tensor.extract_slice %[[IN]][0, %[[X]]] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
//       CHECK:   gpu.barrier
//       CHECK:   return %[[OUT]] : tensor<3x2xf32>

// -----

func.func @rank_reducing_shuffle_tensor(%init: tensor<1x6x6xf32>, %source: tensor<2x3xf32>, %x: index, %y: index) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %source[0, %x, %y] [1, 2, 3] [1, 1, 1] to %init {
  ^bb0(%intermediate: tensor<1x6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, %y, %x] [1, 3, 2] [1, 1, 1] : tensor<1x6x6xf32> to tensor<3x2xf32>
    iree_gpu.yield %slice : tensor<3x2xf32>
  } : tensor<2x3xf32> -> tensor<1x6x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_shuffle_tensor
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @rank_reducing_shuffle_tensor
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9]+]]: tensor<1x6x6xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<2x3xf32>
//  CHECK-SAME:   %[[X:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[Y:[A-Za-z0-9]+]]: index

//       CHECK:   %[[IN:.+]] = tensor.insert_slice %[[ARG1]] into %[[INIT]][0, %[[X]], %[[Y]]] [1, 2, 3] [1, 1, 1] : tensor<2x3xf32> into tensor<1x6x6xf32>
//       CHECK:   gpu.barrier
//       CHECK:   %[[OUT:.+]] = tensor.extract_slice %[[IN]][0, %[[Y]], %[[X]]] [1, 3, 2] [1, 1, 1] : tensor<1x6x6xf32> to tensor<3x2xf32>
//       CHECK:   gpu.barrier
//       CHECK:   return %[[OUT]]

// -----

func.func @reshape_shuffle_tensor(%init: tensor<12x12xf32>, %source: tensor<2x3xf32>) -> tensor<2x1x3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %source[0, 0] [2, 3] [1, 1] to %init {
  ^bb0(%intermediate: tensor<12x12xf32>):
    %expand = tensor.expand_shape %intermediate [[0, 1], [2, 3]] output_shape [4, 3, 3, 4] : tensor<12x12xf32> into tensor<4x3x3x4xf32>
    %slice = tensor.extract_slice %expand[0, 0, 0, 0] [2, 1, 3, 2] [1, 1, 1, 1] : tensor<4x3x3x4xf32> to tensor<2x1x3x2xf32>
    iree_gpu.yield %slice : tensor<2x1x3x2xf32>
  } : tensor<2x3xf32> -> tensor<12x12xf32> -> tensor<2x1x3x2xf32>
  return %0 : tensor<2x1x3x2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_shuffle_tensor
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @reshape_shuffle_tensor
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9]+]]: tensor<12x12xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<2x3xf32>

//       CHECK:   %[[IN:.+]] = tensor.insert_slice %[[ARG1]] into %[[INIT]][0, 0] [2, 3] [1, 1] : tensor<2x3xf32> into tensor<12x12xf32>
//       CHECK:   gpu.barrier
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[IN]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [4, 3, 3, 4] : tensor<12x12xf32> into tensor<4x3x3x4xf32>
//       CHECK:   tensor.extract_slice %[[EXPAND]][0, 0, 0, 0] [2, 1, 3, 2] [1, 1, 1, 1] : tensor<4x3x3x4xf32> to tensor<2x1x3x2xf32>
//       CHECK:   gpu.barrier
