// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: func @sort
// CHECK:         linalg_plus.sort
// CHECK:           linalg_plus.yield
func @sort(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  %0 = "linalg_plus.sort"(%arg0) ( {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %1 = cmpi sgt, %arg1, %arg2 : i32
    linalg_plus.yield %1 : i1
  }) {dimension = 0 : i64} : (tensor<128xi32>) -> (tensor<128xi32>)
  return %0 : tensor<128xi32>
}
