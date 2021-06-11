// RUN: iree-opt -split-input-file -iree-convert-to-hal %s | IreeFileCheck %s

// CHECK-LABEL: func @tensorIO(%arg0: !hal.buffer) -> !hal.buffer
func @tensorIO(%arg0 : tensor<1x1xi32>) -> tensor<1x1xi32> {
  // CHECK-NEXT: br ^bb1(%arg0 : !hal.buffer)
  br ^bb1(%arg0 : tensor<1x1xi32>)
// CHECK-NEXT: ^bb1(%[[BB0:.+]]: !hal.buffer)
^bb1(%0 : tensor<1x1xi32>):
  // CHECK-NEXT: return %[[BB0]] : !hal.buffer
  return %0 : tensor<1x1xi32>
}

// -----

// CHECK-LABEL: func @select(%arg0: i1, %arg1: !hal.buffer, %arg2: !hal.buffer) -> !hal.buffer
func @select(%arg0: i1, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  // CHECK: %[[RES:.+]] = select %arg0, %arg1, %arg2 : !hal.buffer
  %0 = select %arg0, %arg1, %arg2 : tensor<i32>
  // CHECK: return %[[RES]] : !hal.buffer
  return %0 : tensor<i32>
}
