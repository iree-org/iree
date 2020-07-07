// RUN: iree-opt -split-input-file -test-iree-flow-dispatchability %s | IreeFileCheck %s

// CHECK-LABEL: @empty
// CHECK-SAME: dispatchable = true
func @empty() {
  return
}

// -----

// CHECK-LABEL: @customOp
// CHECK-SAME: dispatchable = false
func @customOp() {
  "do.foo"() : () -> ()
  return
}

// -----

// CHECK-LABEL: @simpleMath
// CHECK-SAME: dispatchable = true
func @simpleMath(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @stdElementwiseOps
// CHECK-SAME: dispatchable = true
func @stdElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = addf %arg0, %arg0 : tensor<4xf32>
  %1 = subf %0, %arg0 : tensor<4xf32>
  %2 = mulf %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @hloElementwiseOps
// CHECK-SAME: dispatchable = true
func @hloElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  %1 = mhlo.subtract %0, %arg0 : tensor<4xf32>
  %2 = mhlo.multiply %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @interleavedDot
// CHECK-SAME: dispatchable = false
func @interleavedDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mhlo.add %arg0, %arg0 : tensor<4x4xf32>
  %1 = "mhlo.dot"(%0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mhlo.multiply %1, %arg0 : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @caller
// CHECK-SAME: dispatchable = true
func @caller(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  %1 = call @callee(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = mhlo.multiply %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}
// CHECK-LABEL: func @callee
// CHECK-SAME: dispatchable = true
func @callee(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.multiply %arg0, %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @dotCaller
// CHECK-SAME: dispatchable = false
func @dotCaller(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mhlo.add %arg0, %arg0 : tensor<4x4xf32>
  %1 = call @dotCallee(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mhlo.multiply %1, %arg0 : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
// CHECK-LABEL: func @dotCallee
// CHECK-SAME: dispatchable = false
func @dotCallee(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "mhlo.dot"(%arg0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
