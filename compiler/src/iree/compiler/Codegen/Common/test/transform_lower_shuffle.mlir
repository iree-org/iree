// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

func.func @shuffle_tensor(%init: tensor<6x6xf32>, %arg0: tensor<2x3xf32>, %x: index) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %arg0[%x, 0] [2, 3] [1, 1] to %init[0, %x] [3, 2] [1, 1] : tensor<2x3xf32> -> tensor<6x6xf32> -> tensor<3x2xf32>
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
//       CHECK:   return %[[OUT]] : tensor<3x2xf32>

// -----

func.func @rank_reducing_shuffle_tensor(%init: tensor<1x6x6xf32>, %arg0: tensor<2x3xf32>, %x: index, %y: index) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %arg0[0, %x, %y] [1, 2, 3] [1, 1, 1] to %init[0, %y, %x] [1, 3, 2] [1, 1, 1] : tensor<2x3xf32> -> tensor<1x6x6xf32> -> tensor<3x2xf32>
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
//       CHECK:   tensor.extract_slice %[[IN]][0, %[[Y]], %[[X]]] [1, 3, 2] [1, 1, 1] : tensor<1x6x6xf32> to tensor<3x2xf32>
