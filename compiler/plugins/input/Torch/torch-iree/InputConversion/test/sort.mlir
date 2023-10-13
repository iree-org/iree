// RUN: iree-opt --split-input-file --torch-iree-tm-tensor-to-linalg-ext %s | FileCheck %s

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

// -----
