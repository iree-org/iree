// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(torch-iree-tm-tensor-to-linalg-ext))" %s | FileCheck %s

func.func @scatter_update(
    %original: tensor<8xi32>, %indices: tensor<3x1xi32>,
    %updates: tensor<3xi32>) -> tensor<8xi32> {
  %0 = tm_tensor.scatter {dimension_map = array<i64: 0>} unique_indices(true)
    ins(%updates, %indices : tensor<3xi32>, tensor<3x1xi32>)
    outs(%original : tensor<8xi32>)  {
  ^bb0(%update: i32, %orig: i32):  // no predecessors
    tm_tensor.yield %update: i32
  } -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL:   func.func @scatter_update(
// CHECK-SAME:            %[[ORIGINAL:.*]]: tensor<8xi32>,
// CHECK-SAME:            %[[INDICES:.*]]: tensor<3x1xi32>,
// CHECK-SAME:            %[[UPDATES:.*]]: tensor<3xi32>) -> tensor<8xi32> {
// CHECK:           %[[SCATTER:.*]] = iree_linalg_ext.scatter
// CHECK-SAME:            dimension_map = [0]
// CHECK-SAME:            unique_indices(true)
// CHECK-SAME:            ins(%[[UPDATES]], %[[INDICES]] : tensor<3xi32>, tensor<3x1xi32>)
// CHECK-SAME:            outs(%[[ORIGINAL]] : tensor<8xi32>) {
// CHECK:           ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
// CHECK:             iree_linalg_ext.yield %[[ARG0]] : i32
// CHECK:           } -> tensor<8xi32>
// CHECK:           return %[[SCATTER:.*]] : tensor<8xi32>

// -----
func.func @scatter_add(
    %original: tensor<8xi32>, %indices: tensor<3x1xi32>,
    %updates: tensor<3xi32>) -> tensor<8xi32> {
  %0 = tm_tensor.scatter {dimension_map = array<i64: 0>} unique_indices(true)
    ins(%updates, %indices : tensor<3xi32>, tensor<3x1xi32>)
    outs(%original : tensor<8xi32>)  {
  ^bb0(%update: i32, %orig: i32):  // no predecessors
    %add = arith.addi %orig, %update: i32
    tm_tensor.yield %add: i32
  } -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
// CHECK-LABEL:   func.func @scatter_add(
// CHECK-SAME:            %[[ORIGINAL:.*]]: tensor<8xi32>,
// CHECK-SAME:            %[[INDICES:.*]]: tensor<3x1xi32>,
// CHECK-SAME:            %[[UPDATES:.*]]: tensor<3xi32>) -> tensor<8xi32> {
// CHECK:           %[[SCATTER:.*]] = iree_linalg_ext.scatter
// CHECK-SAME:            unique_indices(true)
// CHECK-SAME:            ins(%[[UPDATES]], %[[INDICES]] : tensor<3xi32>, tensor<3x1xi32>)
// CHECK-SAME:            outs(%[[ORIGINAL]] : tensor<8xi32>) {
// CHECK:           ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
// CHECK:             %[[ADD:.*]] = arith.addi %[[ARG1]], %[[ARG0]] : i32
// CHECK:             iree_linalg_ext.yield %[[ADD]] : i32
// CHECK:           } -> tensor<8xi32>
// CHECK:           return %[[SCATTER:.*]] : tensor<8xi32>

// -----
