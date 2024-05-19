// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

func.func @main(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  return %arg0 : tensor<?xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.return"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.iree.copy_tensor_operand %0 [0] : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: @main
//  CHECK-SAME:   (%[[ARG:.+]]: tensor<?xf32>)
//       CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG]], %c0 : tensor<?xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[DIM]]) : tensor<?xf32>
//       CHECK:   %[[COPY:.+]] = linalg.copy
//  CHECK-SAME:     ins(%[[ARG]] : tensor<?xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<?xf32>)
//       CHECK:   return %[[COPY]] : tensor<?xf32>
