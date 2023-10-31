// RUN: iree-opt --split-input-file --torch-iree-tm-tensor-to-linalg-ext %s | FileCheck %s

// -----
func.func @scan(%in: tensor<128xi32>, %out: tensor<128xi32>, %acc: tensor<i32>) -> (tensor<128xi32>, tensor<i32>) {
  %ret_out, %ret_acc = tm_tensor.scan dimension(0) inclusive(true)
    ins(%in : tensor<128xi32>) outs(%out, %acc: tensor<128xi32>, tensor<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      tm_tensor.yield %sum : i32
  } -> tensor<128xi32>, tensor<i32>
  return %ret_out, %ret_acc: tensor<128xi32>, tensor<i32>
}
// CHECK-LABEL:   func.func @scan(
// CHECK-SAME:            %[[IN:.*]]: tensor<128xi32>, %[[OUT:.*]]: tensor<128xi32>,
// CHECK-SAME:            %[[ACC:.*]]: tensor<i32>) -> (tensor<128xi32>, tensor<i32>) {
// CHECK:           %[[SCAN:.*]]:2 = iree_linalg_ext.scan
// CHECK-SAME:            dimension(0)
// CHECK-SAME:            inclusive(true)
// CHECK-SAME:            ins(%[[IN]] : tensor<128xi32>)
// CHECK-SAME:            outs(%[[OUT]], %[[ACC]] : tensor<128xi32>, tensor<i32>) {
// CHECK:           ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
// CHECK:             %[[ADD:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
// CHECK:             iree_linalg_ext.yield %[[ADD]] : i32
// CHECK:           } -> tensor<128xi32>, tensor<i32>
// CHECK:           return %[[SCAN:.*]]#0, %[[SCAN]]#1 : tensor<128xi32>, tensor<i32>
