// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

func.func @barrier_region(%init: tensor<6x6xf32>, %x: index) -> tensor<3x2xf32> {
  %0 = iree_gpu.barrier_region %init {
  ^bb0(%intermediate: tensor<6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, %x] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
    iree_gpu.yield %slice : tensor<3x2xf32>
  } : tensor<6x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_barrier_region
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @barrier_region
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9]+]]: tensor<6x6xf32>
//  CHECK-SAME:   %[[X:[A-Za-z0-9]+]]: index

//       CHECK:   %[[WRITE_BARRIER:.+]] = iree_gpu.value_barrier %[[INIT]]
//       CHECK:   %[[OUT:.+]] = tensor.extract_slice %[[WRITE_BARRIER]][0, %[[X]]] [3, 2] [1, 1]
//       CHECK:   %[[READ_BARRIER:.+]] = iree_gpu.value_barrier %[[OUT]]
//       CHECK:   return %[[READ_BARRIER]] : tensor<3x2xf32>

// -----

func.func @reshape_barrier_region(%init: tensor<12x12xf32>) -> vector<2x1x3x2xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = iree_gpu.barrier_region %init {
  ^bb0(%intermediate: tensor<12x12xf32>):
    %expand = tensor.expand_shape %intermediate [[0, 1], [2, 3]] output_shape [4, 3, 3, 4] : tensor<12x12xf32> into tensor<4x3x3x4xf32>
    %read = vector.transfer_read %expand[%c0, %c0, %c0, %c0], %cst : tensor<4x3x3x4xf32>, vector<2x1x3x2xf32>
    iree_gpu.yield %read : vector<2x1x3x2xf32>
  } : tensor<12x12xf32> -> vector<2x1x3x2xf32>
  return %0 : vector<2x1x3x2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_barrier_region
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @reshape_barrier_region
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9]+]]: tensor<12x12xf32>

//       CHECK:   %[[WRITE_BARRIER:.+]] = iree_gpu.value_barrier %[[INIT]]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[WRITE_BARRIER]]
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[EXPAND]]
//       CHECK:   %[[READ_BARRIER:.+]] = iree_gpu.value_barrier %[[READ]]
