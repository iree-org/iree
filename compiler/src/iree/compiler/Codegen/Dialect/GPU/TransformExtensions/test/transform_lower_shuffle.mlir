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
//       CHECK:   %[[WRITE_BARRIER:.+]] = iree_gpu.value_barrier %[[IN]]
//       CHECK:   %[[OUT:.+]] = tensor.extract_slice %[[WRITE_BARRIER]][0, %[[X]]] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
//       CHECK:   %[[READ_BARRIER:.+]] = iree_gpu.value_barrier %[[OUT]]
//       CHECK:   return %[[READ_BARRIER]] : tensor<3x2xf32>

// -----

func.func @rank_reducing_shuffle_tensor(%init: tensor<1x6x6xf32>, %source: tensor<2x3xf32>, %x: index, %y: index) -> vector<3x2xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = iree_gpu.shuffle_tensor %source[0, %x, %y] [1, 2, 3] [1, 1, 1] to %init {
  ^bb0(%intermediate: tensor<1x6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, %y, %x] [1, 3, 2] [1, 1, 1] : tensor<1x6x6xf32> to tensor<3x2xf32>
    %read = vector.transfer_read %slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<3x2xf32>, vector<3x2xf32>
    iree_gpu.yield %read : vector<3x2xf32>
  } : tensor<2x3xf32> -> tensor<1x6x6xf32> -> vector<3x2xf32>
  return %0 : vector<3x2xf32>
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
//       CHECK:   %[[WRITE_BARRIER:.+]] = iree_gpu.value_barrier %[[IN]]
//       CHECK:   %[[OUT:.+]] = tensor.extract_slice %[[WRITE_BARRIER]][0, %[[Y]], %[[X]]] [1, 3, 2] [1, 1, 1] : tensor<1x6x6xf32> to tensor<3x2xf32>
//       CHECK:   %[[VEC_OUT:.+]] = vector.transfer_read %[[OUT]]
//       CHECK:   %[[READ_BARRIER:.+]] = iree_gpu.value_barrier %[[VEC_OUT]]
//       CHECK:   return %[[READ_BARRIER]]

// -----

func.func @reshape_shuffle_tensor(%init: tensor<12x12xf32>, %source: tensor<2x3xf32>) -> vector<2x1x3x2xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = iree_gpu.shuffle_tensor %source[0, 0] [2, 3] [1, 1] to %init {
  ^bb0(%intermediate: tensor<12x12xf32>):
    %expand = tensor.expand_shape %intermediate [[0, 1], [2, 3]] output_shape [4, 3, 3, 4] : tensor<12x12xf32> into tensor<4x3x3x4xf32>
    %read = vector.transfer_read %expand[%c0, %c0, %c0, %c0], %cst : tensor<4x3x3x4xf32>, vector<2x1x3x2xf32>
    iree_gpu.yield %read : vector<2x1x3x2xf32>
  } : tensor<2x3xf32> -> tensor<12x12xf32> -> vector<2x1x3x2xf32>
  return %0 : vector<2x1x3x2xf32>
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
//       CHECK:   %[[WRITE_BARRIER:.+]] = iree_gpu.value_barrier %[[IN]]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[WRITE_BARRIER]]
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[EXPAND]]
//       CHECK:   %[[READ_BARRIER:.+]] = iree_gpu.value_barrier %[[READ]]
