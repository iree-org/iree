// RUN: iree-opt --split-input-file --iree-tm-tensor-to-linalg-ext %s | FileCheck %s

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

// -----
func.func @scatter_update(
    %original: tensor<8xi32>, %indices: tensor<3x1xi32>,
    %updates: tensor<3xi32>) -> tensor<8xi32> {
  %0 = tm_tensor.scatter unique_indices(true)
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
  %0 = tm_tensor.scatter unique_indices(true)
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
func.func @sort(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xi64>) -> (tensor<3x4xf32>, tensor<3x4xi64>) {
  %0:2 = tm_tensor.sort dimension(1) outs(%arg0, %arg1 : tensor<3x4xf32>, tensor<3x4xi64>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
    %1 = arith.cmpf ole, %arg2, %arg3 : f32
    tm_tensor.yield %1 : i1
  } -> tensor<3x4xf32>, tensor<3x4xi64>
  return %0#0, %0#1 : tensor<3x4xf32>, tensor<3x4xi64>
}
// CHECK-LABEL:   func.func @sort(
// CHECK-SAME:            %[[INPUT:.*]]: tensor<3x4xf32>, %[[INDICES:.*]]: tensor<3x4xi64>) -> (tensor<3x4xf32>, tensor<3x4xi64>) {
// CHECK:          %[[SORT:.*]]:2 = iree_linalg_ext.sort
// CHECK-SAME:            dimension(1)
// CHECK-SAME:            outs(%[[INPUT]], %[[INDICES]] : tensor<3x4xf32>, tensor<3x4xi64>) {
// CHECK:          ^bb0(%[[INP1:.*]]: f32, %[[INP2:.*]]: f32, %{{.*}}: i64, %{{.*}}: i64):
// CHECK:            %[[PREDICATE:.*]] = arith.cmpf ole, %[[INP1]], %[[INP2]] : f32
// CHECK:            iree_linalg_ext.yield %[[PREDICATE]] : i1
// CHECK:          } -> tensor<3x4xf32>, tensor<3x4xi64>
// CHECK:          return %[[SORT]]#0, %[[SORT]]#1 : tensor<3x4xf32>, tensor<3x4xi64>

