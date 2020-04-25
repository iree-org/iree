// RUN: iree-opt -iree-linalg-fusion %s | IreeFileCheck %s

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @scalar_const_fusion
func @scalar_const_fusion(%arg0 : tensor<5x5xi32>) -> tensor<5x5xi32> {
  // CHECK: %[[CONST:.+]] = constant 2 : i32
  %0 = constant dense<2> : tensor<i32>
  // CHECK: linalg.generic
  // CHECK-SAME: args_in = 1
  // CHECK-SAME: args_out = 1
  // CHECK-SAME: indexing_maps = [#[[MAP]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK-SAME: %arg0 {
  // CHECK-NEXT: ^{{.+}}(%[[ARG:.+]]: i32):
  // CHECK-NEXT: %[[RESULT:.+]] = muli %[[ARG]], %[[CONST]] : i32
  // CHECK-NEXT: linalg.yield %[[RESULT]]
  %1 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> ()>],
    iterator_types = ["parallel", "parallel"]} %arg0, %0 {
  ^bb0(%arg1 : i32, %arg2 : i32) :
    %2 = muli %arg1, %arg2 : i32
    linalg.yield %2 : i32
  } : tensor<5x5xi32>, tensor<i32> -> tensor<5x5xi32>
  return %1 : tensor<5x5xi32>
}
